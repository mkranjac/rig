use aws_sdk_bedrockruntime::operation::invoke_model::*;
use aws_smithy_types::Blob;
use rig::embeddings::{self, Embedding, EmbeddingError};
use serde::{Deserialize, Serialize};

use crate::client::Client;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingRequest {
    pub input_text: String,
    pub dimensions: usize,
    pub normalize: bool,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResponse {
    pub embedding: Vec<f64>,
    pub input_text_token_count: usize,
}

#[derive(Clone)]
pub enum BedrockEmbeddingModel {
    TitanTextEmbeddingsV2(usize),
    Custom(&'static str, usize),
}

impl BedrockEmbeddingModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            BedrockEmbeddingModel::TitanTextEmbeddingsV2(_) => "amazon.titan-embed-text-v2:0",
            BedrockEmbeddingModel::Custom(str, _) => str,
        }
    }
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    model: BedrockEmbeddingModel,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: BedrockEmbeddingModel) -> Self {
        Self { client, model }
    }

    pub async fn document_to_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let input_document = serde_json::to_string(&request).map_err(EmbeddingError::JsonError)?;

        let model_response = self
            .client
            .aws_client
            .invoke_model()
            .model_id(self.model.as_str())
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(input_document))
            .send()
            .await;

        let response = model_response
                    .map_err(|sdk_error| if let Some(service_error) = sdk_error.as_service_error() {
                        let err: String = match service_error {
                            InvokeModelError::ModelTimeoutException(e) => e.to_owned().message.unwrap_or("The request took too long to process. Processing time exceeded the model timeout length.".into()),
                            InvokeModelError::AccessDeniedException(e) => e.to_owned().message.unwrap_or("The request is denied because you do not have sufficient permissions to perform the requested action.".into()),
                            InvokeModelError::ResourceNotFoundException(e) => e.to_owned().message.unwrap_or("The specified resource ARN was not found.".into()),
                            InvokeModelError::ThrottlingException(e) => e.to_owned().message.unwrap_or("Your request was denied due to exceeding the account quotas for Amazon Bedrock.".into()),
                            InvokeModelError::ServiceUnavailableException(e) => e.to_owned().message.unwrap_or("The service isn't currently available.".into()),
                            InvokeModelError::InternalServerException(e) => e.to_owned().message.unwrap_or("An internal server error occurred.".into()),
                            InvokeModelError::ValidationException(e) => e.to_owned().message.unwrap_or("The input fails to satisfy the constraints specified by Amazon Bedrock.".into()),
                            InvokeModelError::ModelNotReadyException(e) => e.to_owned().message.unwrap_or("The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.".into()),
                            InvokeModelError::ModelErrorException(e) => e.to_owned().message.unwrap_or("The request failed due to an error while processing the model.".into()),
                            InvokeModelError::ServiceQuotaExceededException(e) => e.to_owned().message.unwrap_or("Your request exceeds the service quota for your account.".into()),
                            _ => String::from("An unexpected error occurred (e.g., invalid JSON returned by the service or an unknown error code)."),
                        };
                        EmbeddingError::ProviderError(err)
                    } else {
                        EmbeddingError::ProviderError(format!("{:?}", sdk_error))
                    })?;

        let response_str = String::from_utf8(response.body.into_inner())
            .map_err(|e| EmbeddingError::ResponseError(e.to_string()))?;

        let result: EmbeddingResponse =
            serde_json::from_str(&response_str).map_err(EmbeddingError::JsonError)?;

        Ok(result)
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        match self.model {
            BedrockEmbeddingModel::TitanTextEmbeddingsV2(ndims) => ndims,
            BedrockEmbeddingModel::Custom(_, ndims) => ndims,
        }
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let documents: Vec<_> = documents.into_iter().collect();

        let mut results = Vec::new();
        let mut errors = Vec::new();

        let mut iterator = documents.into_iter();
        while let Some(embedding) = iterator.next().map(|doc| async move {
            let request = EmbeddingRequest {
                input_text: doc.to_owned(),
                dimensions: self.ndims(),
                normalize: true,
            };
            self.document_to_embeddings(request)
                .await
                .map(|embeddings| Embedding {
                    document: doc.to_owned(),
                    vec: embeddings.embedding,
                })
        }) {
            match embedding.await {
                Ok(embedding) => results.push(embedding),
                Err(err) => errors.push(err),
            }
        }

        match errors.as_slice() {
            [] => Ok(results),
            [err, ..] => Err(EmbeddingError::ResponseError(err.to_string())),
        }
    }
}
