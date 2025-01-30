use std::str::FromStr;

use crate::{client::Client, document_to_json, json_to_document};
use aws_sdk_bedrockruntime::{
    operation::converse::ConverseError,
    types::{
        ContentBlock, ConversationRole, ConverseOutput, InferenceConfiguration, Message,
        SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema, ToolSpecification,
    },
};
use rig::completion::{self, CompletionError};

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    model_id: &'static str,
}

impl CompletionModel {
    pub fn new(client: Client, model_id: &'static str) -> Self {
        Self { client, model_id }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = ConverseOutput;

    async fn completion(
        &self,
        mut completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<ConverseOutput>, CompletionError> {
        let mut full_history = Vec::new();
        full_history.append(&mut completion_request.chat_history);
        full_history.push(completion::Message {
            role: "user".into(),
            content: completion_request.prompt_with_context(),
        });

        let prompt_with_history = full_history
            .into_iter()
            .filter_map(|m| {
                let role = ConversationRole::from_str(&m.role).unwrap_or(ConversationRole::User);
                Message::builder()
                    .role(role)
                    .content(ContentBlock::Text(completion_request.prompt_with_context()))
                    .build()
                    .ok()
            })
            .collect::<Vec<_>>();

        let mut converse_builder = self.client.aws_client.converse().model_id(self.model_id);
        let mut inference_configuration = InferenceConfiguration::builder();

        if let Some(params) = completion_request.additional_params {
            converse_builder = converse_builder
                .set_additional_model_request_fields(Some(json_to_document(params)));
        }

        if let Some(temperature) = completion_request.temperature {
            inference_configuration =
                inference_configuration.set_temperature(Some(temperature as f32));
        }

        if let Some(max_tokens) = completion_request.max_tokens {
            inference_configuration =
                inference_configuration.set_max_tokens(Some(max_tokens as i32));
        }

        converse_builder =
            converse_builder.set_inference_config(Some(inference_configuration.build()));

        let mut tools = vec![];
        for tool_definition in completion_request.tools.iter() {
            let document = json_to_document(tool_definition.parameters.clone());
            let schema = ToolInputSchema::Json(document);
            let tool = Tool::ToolSpec(
                ToolSpecification::builder()
                    .name(tool_definition.name.clone())
                    .set_description(Some(tool_definition.description.clone()))
                    .set_input_schema(Some(schema))
                    .build()
                    .map_err(|e| CompletionError::RequestError(e.into()))?,
            );
            tools.push(tool);
        }

        if !tools.is_empty() {
            let config = ToolConfiguration::builder()
                .set_tools(Some(tools))
                .build()
                .map_err(|e| CompletionError::RequestError(e.into()))?;

            converse_builder = converse_builder.set_tool_config(Some(config));
        }

        if let Some(system_prompt) = completion_request.preamble {
            converse_builder =
                converse_builder.set_system(Some(vec![SystemContentBlock::Text(system_prompt)]));
        }

        let model_response = converse_builder
            .set_messages(Some(prompt_with_history))
            .send()
            .await;

        let response = model_response
            .map_err(|sdk_error| if let Some(service_error) = sdk_error.as_service_error() {
                let err: String = match service_error {
                    ConverseError::ModelTimeoutException(e) => e.to_owned().message.unwrap_or("The request took too long to process. Processing time exceeded the model timeout length.".into()),
                    ConverseError::AccessDeniedException(e) => e.to_owned().message.unwrap_or("The request is denied because you do not have sufficient permissions to perform the requested action.".into()),
                    ConverseError::ResourceNotFoundException(e) => e.to_owned().message.unwrap_or("The specified resource ARN was not found.".into()),
                    ConverseError::ThrottlingException(e) => e.to_owned().message.unwrap_or("Your request was denied due to exceeding the account quotas for AWS Bedrock.".into()),
                    ConverseError::ServiceUnavailableException(e) => e.to_owned().message.unwrap_or("The service isn't currently available.".into()),
                    ConverseError::InternalServerException(e) => e.to_owned().message.unwrap_or("An internal server error occurred.".into()),
                    ConverseError::ValidationException(e) => e.to_owned().message.unwrap_or("The input fails to satisfy the constraints specified by AWS Bedrock.".into()),
                    ConverseError::ModelNotReadyException(e) => e.to_owned().message.unwrap_or("The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.".into()),
                    ConverseError::ModelErrorException(e) => e.to_owned().message.unwrap_or("The request failed due to an error while processing the model.".into()),
                    _ => String::from("An unexpected error occurred (e.g., invalid JSON returned by the service or an unknown error code).")
                };
                CompletionError::ProviderError(err)
            } else {
                CompletionError::ProviderError(format!("{:?}", sdk_error))
            })?;

        let response = response.output.ok_or(CompletionError::ProviderError(
            "Model didn't return any converse output".into(),
        ))?;

        let content_blocks = response
            .as_message()
            .map_err(|_| {
                CompletionError::ProviderError(
                    "Failed to extract message from converse output".into(),
                )
            })?
            .content();

        if let Some(tool_use) = content_blocks.iter().find_map(|content| match content {
            ContentBlock::ToolUse(tool_use_block) => Some(tool_use_block.to_owned()),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::ToolCall(
                    tool_use.name,
                    tool_use.tool_use_id,
                    document_to_json(tool_use.input),
                ),
                raw_response: response,
            });
        }

        if let Some(text) = content_blocks.iter().find_map(|content| match content {
            ContentBlock::Text(text) => Some(text.to_owned()),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(text),
                raw_response: response,
            });
        }

        Err(CompletionError::ResponseError(
            "Response did not contain a message or tool call".into(),
        ))
    }
}
