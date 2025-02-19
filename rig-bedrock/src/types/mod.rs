pub mod errors;

use aws_sdk_bedrockruntime::types as aws_bedrock;
use errors::IntegrationError;
use rig::{
    message::{
        AssistantContent, ContentFormat, Document, DocumentMediaType, Image, ImageMediaType,
        Message, MimeType, Text, ToolCall, ToolFunction, ToolResult, ToolResultContent,
        UserContent,
    },
    OneOrMany,
};

use crate::{document_to_json, json_to_document};
use base64::{prelude::BASE64_STANDARD, Engine};

pub fn from_user_content(
    content: UserContent,
) -> Result<aws_bedrock::ContentBlock, IntegrationError> {
    match content {
        UserContent::Text(text) => Ok(aws_bedrock::ContentBlock::Text(text.text)),
        UserContent::ToolResult(tool_result) => {
            let builder = aws_bedrock::ToolResultBlock::builder()
                .tool_use_id(tool_result.id)
                .set_content(Some(
                    tool_result
                        .content
                        .into_iter()
                        .filter_map(|tool| from_tool_result(tool).ok())
                        .collect(),
                ))
                .build()
                .map_err(IntegrationError::BuildError)?;
            Ok(aws_bedrock::ContentBlock::ToolResult(builder))
        }
        UserContent::Image(image) => {
            let image = from_image(image)?;
            Ok(aws_bedrock::ContentBlock::Image(image))
        }
        UserContent::Document(document) => {
            let doc = from_document(document)?;
            Ok(aws_bedrock::ContentBlock::Document(doc))
        }
        UserContent::Audio(_) => Err(IntegrationError::UnsupportedFeature("Audio")),
    }
}

pub fn from_assistent_content(
    content: AssistantContent,
) -> Result<aws_bedrock::ContentBlock, IntegrationError> {
    match content {
        AssistantContent::Text(text) => Ok(aws_bedrock::ContentBlock::Text(text.text)),
        AssistantContent::ToolCall(tool_call) => Ok(aws_bedrock::ContentBlock::ToolUse(
            aws_bedrock::ToolUseBlock::builder()
                .tool_use_id(tool_call.id)
                .name(tool_call.function.name)
                .input(json_to_document(tool_call.function.arguments))
                .build()
                .map_err(IntegrationError::BuildError)?,
        )),
    }
}

pub fn from_message(message: Message) -> Result<aws_bedrock::Message, IntegrationError> {
    let result = match message {
        Message::User { content } => aws_bedrock::Message::builder()
            .role(aws_bedrock::ConversationRole::User)
            .set_content(Some(
                content
                    .into_iter()
                    .filter_map(|content| from_user_content(content).ok())
                    .collect(),
            ))
            .build()
            .map_err(IntegrationError::BuildError)?,
        Message::Assistant { content } => aws_bedrock::Message::builder()
            .role(aws_bedrock::ConversationRole::Assistant)
            .set_content(Some(
                content
                    .into_iter()
                    .filter_map(|content| from_assistent_content(content).ok())
                    .collect(),
            ))
            .build()
            .map_err(IntegrationError::BuildError)?,
    };
    Ok(result)
}

pub fn into_message(message: aws_bedrock::Message) -> Result<Message, IntegrationError> {
    match message.role {
        aws_bedrock::ConversationRole::Assistant => {
            let content = OneOrMany::many(
                message
                    .content
                    .into_iter()
                    .filter_map(|c| into_assistent_content(c).ok()),
            )
            .map_err(|_| {
                IntegrationError::UnsupportedFeature("Message returned invalid response")
            })?;
            Ok(Message::Assistant { content })
        }
        aws_bedrock::ConversationRole::User => {
            let content = OneOrMany::many(
                message
                    .content
                    .into_iter()
                    .filter_map(|c| into_user_content(c).ok()),
            )
            .map_err(|_| {
                IntegrationError::UnsupportedFeature("Message returned invalid response")
            })?;
            Ok(Message::User { content })
        }
        _ => Err(IntegrationError::UnsupportedFeature(
            "AWS Bedrock returned unsupported ConversationRole",
        )),
    }
}

pub fn into_assistent_content(
    content_block: aws_bedrock::ContentBlock,
) -> Result<AssistantContent, IntegrationError> {
    match content_block {
        aws_bedrock::ContentBlock::Text(text) => Ok(AssistantContent::Text(Text { text })),
        aws_bedrock::ContentBlock::ToolUse(tool_use_block) => {
            Ok(AssistantContent::ToolCall(ToolCall {
                id: tool_use_block.tool_use_id,
                function: ToolFunction {
                    name: tool_use_block.name,
                    arguments: document_to_json(tool_use_block.input),
                },
            }))
        }
        _ => Err(IntegrationError::UnsupportedFeature(
            "AWS Bedrock returned unsupported ContentBlock",
        )),
    }
}

pub fn into_user_content(
    content_block: aws_bedrock::ContentBlock,
) -> Result<UserContent, IntegrationError> {
    match content_block {
        aws_bedrock::ContentBlock::Text(text) => Ok(UserContent::Text(Text { text })),
        aws_bedrock::ContentBlock::ToolResult(tool_result) => {
            let tool_results = OneOrMany::many(
                tool_result
                    .content
                    .into_iter()
                    .filter_map(|t| into_tool_result(t).ok()),
            )
            .map_err(|_| {
                IntegrationError::UnsupportedFeature("ToolResult returned invalid response")
            })?;
            Ok(UserContent::ToolResult(ToolResult {
                id: tool_result.tool_use_id,
                content: tool_results,
            }))
        }
        aws_bedrock::ContentBlock::Document(document) => {
            let doc = into_document(document)?;
            Ok(UserContent::Document(doc))
        }
        aws_bedrock::ContentBlock::Image(image) => {
            let image = into_image(image)?;
            Ok(UserContent::Image(image))
        }
        _ => Err(IntegrationError::UnsupportedFeature(
            "ToolResultContentBlock contains unsupported variant",
        )),
    }
}

pub fn into_tool_result(
    tool_result_content_block: aws_bedrock::ToolResultContentBlock,
) -> Result<ToolResultContent, IntegrationError> {
    match tool_result_content_block {
        aws_bedrock::ToolResultContentBlock::Image(image) => {
            let image = into_image(image)?;
            Ok(ToolResultContent::Image(image))
        }
        aws_bedrock::ToolResultContentBlock::Json(document) => Ok(ToolResultContent::Text(Text {
            text: document_to_json(document).to_string(),
        })),
        aws_bedrock::ToolResultContentBlock::Text(text) => {
            Ok(ToolResultContent::Text(Text { text }))
        }
        _ => Err(IntegrationError::UnsupportedFeature(
            "ToolResultContentBlock contains unsupported variant",
        )),
    }
}

pub fn from_tool_result(
    tool_result_content: ToolResultContent,
) -> Result<aws_bedrock::ToolResultContentBlock, IntegrationError> {
    match tool_result_content {
        ToolResultContent::Text(text) => Ok(aws_bedrock::ToolResultContentBlock::Text(text.text)),
        ToolResultContent::Image(image) => {
            let image = from_image(image)?;
            Ok(aws_bedrock::ToolResultContentBlock::Image(image))
        }
    }
}

pub fn into_image(image: aws_bedrock::ImageBlock) -> Result<Image, IntegrationError> {
    let media_type = match image.format {
        aws_bedrock::ImageFormat::Gif => Ok(ImageMediaType::GIF),
        aws_bedrock::ImageFormat::Jpeg => Ok(ImageMediaType::JPEG),
        aws_bedrock::ImageFormat::Png => Ok(ImageMediaType::PNG),
        aws_bedrock::ImageFormat::Webp => Ok(ImageMediaType::WEBP),
        e => Err(IntegrationError::UnsupportedFormat(e.to_string())),
    };
    let data = match image.source {
        Some(aws_bedrock::ImageSource::Bytes(blob)) => {
            let encoded_img = BASE64_STANDARD.encode(blob.into_inner());
            Ok(encoded_img)
        }
        _ => Err(IntegrationError::ModelError("Image source is missing")),
    }?;
    Ok(Image {
        data,
        format: Some(ContentFormat::Base64),
        media_type: media_type.ok(),
        detail: None,
    })
}

pub fn from_image(image: Image) -> Result<aws_bedrock::ImageBlock, IntegrationError> {
    let format = image
        .media_type
        .map(|f| match f {
            ImageMediaType::JPEG => Ok(aws_bedrock::ImageFormat::Jpeg),
            ImageMediaType::PNG => Ok(aws_bedrock::ImageFormat::Png),
            ImageMediaType::GIF => Ok(aws_bedrock::ImageFormat::Gif),
            ImageMediaType::WEBP => Ok(aws_bedrock::ImageFormat::Webp),
            e => Err(IntegrationError::UnsupportedFormat(e.to_mime_type().into())),
        })
        .and_then(|img| img.ok());

    let img_data = BASE64_STANDARD
        .decode(image.data)
        .map_err(|e| IntegrationError::ConversionError(e.to_string()))?;
    let blob = aws_smithy_types::Blob::new(img_data);
    let result = aws_bedrock::ImageBlock::builder()
        .set_format(format)
        .source(aws_bedrock::ImageSource::Bytes(blob))
        .build()
        .map_err(IntegrationError::BuildError)?;
    Ok(result)
}

pub fn into_document(document: aws_bedrock::DocumentBlock) -> Result<Document, IntegrationError> {
    let media_type = match document.format {
        aws_bedrock::DocumentFormat::Csv => Ok(DocumentMediaType::CSV),
        aws_bedrock::DocumentFormat::Html => Ok(DocumentMediaType::HTML),
        aws_bedrock::DocumentFormat::Md => Ok(DocumentMediaType::MARKDOWN),
        aws_bedrock::DocumentFormat::Pdf => Ok(DocumentMediaType::PDF),
        aws_bedrock::DocumentFormat::Txt => Ok(DocumentMediaType::TXT),
        e => Err(IntegrationError::UnsupportedFormat(e.to_string())),
    };
    let data = match document.source {
        Some(aws_bedrock::DocumentSource::Bytes(blob)) => {
            let encoded_data = BASE64_STANDARD.encode(blob.into_inner());
            Ok(encoded_data)
        }
        _ => Err(IntegrationError::ModelError("Document source is missing")),
    }?;
    Ok(Document {
        data,
        format: Some(ContentFormat::Base64),
        media_type: media_type.ok(),
    })
}

pub fn from_document(document: Document) -> Result<aws_bedrock::DocumentBlock, IntegrationError> {
    let format = document
        .media_type
        .map(|doc| match doc {
            DocumentMediaType::PDF => Ok(aws_bedrock::DocumentFormat::Pdf),
            DocumentMediaType::TXT => Ok(aws_bedrock::DocumentFormat::Txt),
            DocumentMediaType::HTML => Ok(aws_bedrock::DocumentFormat::Html),
            DocumentMediaType::MARKDOWN => Ok(aws_bedrock::DocumentFormat::Md),
            DocumentMediaType::CSV => Ok(aws_bedrock::DocumentFormat::Csv),
            e => Err(IntegrationError::UnsupportedFeature(e.to_mime_type())),
        })
        .and_then(|doc| doc.ok());

    let document_data = BASE64_STANDARD
        .decode(document.data)
        .map_err(|e| IntegrationError::ConversionError(e.to_string()))?;
    let data = aws_smithy_types::Blob::new(document_data);
    let document_source = aws_bedrock::DocumentSource::Bytes(data);

    let result = aws_bedrock::DocumentBlock::builder()
        .source(document_source)
        .set_format(format)
        .build()
        .map_err(IntegrationError::BuildError)?;

    Ok(result)
}
