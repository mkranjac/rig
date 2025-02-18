use aws_sdk_bedrockruntime::types::{DocumentSource, ImageSource, ToolUseBlock};
use rig::{
    message::{
        AssistantContent, ContentFormat, Document, DocumentMediaType, Image, ImageMediaType,
        Message, Text, ToolCall, ToolFunction, ToolResult, ToolResultContent, UserContent,
    },
    OneOrMany,
};

use crate::{document_to_json, json_to_document};

pub fn from_user_content(content: UserContent) -> aws_sdk_bedrockruntime::types::ContentBlock {
    match content {
        UserContent::Text(text) => aws_sdk_bedrockruntime::types::ContentBlock::Text(text.text),
        UserContent::ToolResult(tool_result) => {
            aws_sdk_bedrockruntime::types::ContentBlock::ToolResult(
                aws_sdk_bedrockruntime::types::ToolResultBlock::builder()
                    .tool_use_id(tool_result.id)
                    .set_content(Some(
                        tool_result
                            .content
                            .into_iter()
                            .map(from_tool_result)
                            .collect(),
                    ))
                    .build()
                    .unwrap(),
            )
        }
        UserContent::Image(image) => {
            aws_sdk_bedrockruntime::types::ContentBlock::Image(from_image(image))
        }
        UserContent::Document(document) => {
            aws_sdk_bedrockruntime::types::ContentBlock::Document(from_document(document))
        }
        _ => unimplemented!("AWS Bedrock does not support given UserContent"),
    }
}

pub fn from_assistent_content(
    content: AssistantContent,
) -> aws_sdk_bedrockruntime::types::ContentBlock {
    match content {
        AssistantContent::Text(text) => {
            aws_sdk_bedrockruntime::types::ContentBlock::Text(text.text)
        }
        AssistantContent::ToolCall(tool_call) => {
            aws_sdk_bedrockruntime::types::ContentBlock::ToolUse(
                ToolUseBlock::builder()
                    .tool_use_id(tool_call.id)
                    .name(tool_call.function.name)
                    .input(json_to_document(tool_call.function.arguments))
                    .build()
                    .unwrap(),
            )
        }
    }
}

pub fn from_message(message: Message) -> aws_sdk_bedrockruntime::types::Message {
    match message {
        Message::User { content } => aws_sdk_bedrockruntime::types::Message::builder()
            .role(aws_sdk_bedrockruntime::types::ConversationRole::User)
            .set_content(Some(content.into_iter().map(from_user_content).collect()))
            .build()
            .unwrap(),
        Message::Assistant { content } => aws_sdk_bedrockruntime::types::Message::builder()
            .role(aws_sdk_bedrockruntime::types::ConversationRole::Assistant)
            .set_content(Some(
                content.into_iter().map(from_assistent_content).collect(),
            ))
            .build()
            .unwrap(),
    }
}

pub fn into_message(message: aws_sdk_bedrockruntime::types::Message) -> Message {
    match message.role {
        aws_sdk_bedrockruntime::types::ConversationRole::Assistant => {
            let content =
                OneOrMany::many(message.content.into_iter().map(into_assistent_content)).unwrap();
            Message::Assistant { content }
        }
        aws_sdk_bedrockruntime::types::ConversationRole::User => {
            let content =
                OneOrMany::many(message.content.into_iter().map(into_user_content)).unwrap();
            Message::User { content }
        }
        _ => unimplemented!("AWS Bedrock has unsupported ConversationRole"),
    }
}

pub fn into_assistent_content(
    content_block: aws_sdk_bedrockruntime::types::ContentBlock,
) -> AssistantContent {
    match content_block {
        aws_sdk_bedrockruntime::types::ContentBlock::Text(text) => {
            AssistantContent::Text(Text { text })
        }
        aws_sdk_bedrockruntime::types::ContentBlock::ToolUse(tool_use_block) => {
            AssistantContent::ToolCall(ToolCall {
                id: tool_use_block.tool_use_id,
                function: ToolFunction {
                    name: tool_use_block.name,
                    arguments: document_to_json(tool_use_block.input),
                },
            })
        }
        _ => unimplemented!("AWS Bedrock sent unsupported ContentBlock"),
    }
}

pub fn into_user_content(
    content_block: aws_sdk_bedrockruntime::types::ContentBlock,
) -> UserContent {
    match content_block {
        aws_sdk_bedrockruntime::types::ContentBlock::Text(text) => UserContent::Text(Text { text }),
        aws_sdk_bedrockruntime::types::ContentBlock::ToolResult(tool_result) => {
            let tool_results =
                OneOrMany::many(tool_result.content.into_iter().map(into_tool_result)).unwrap();
            UserContent::ToolResult(ToolResult {
                id: tool_result.tool_use_id,
                content: tool_results,
            })
        }
        aws_sdk_bedrockruntime::types::ContentBlock::Document(document) => {
            UserContent::Document(into_document(document))
        }
        aws_sdk_bedrockruntime::types::ContentBlock::Image(image) => {
            UserContent::Image(into_image(image))
        }
        _ => unimplemented!("AWS Bedrock sent unsupported ContentBlock"),
    }
}

pub fn into_tool_result(
    tool_result_content_block: aws_sdk_bedrockruntime::types::ToolResultContentBlock,
) -> ToolResultContent {
    match tool_result_content_block {
        aws_sdk_bedrockruntime::types::ToolResultContentBlock::Image(image) => {
            ToolResultContent::Image(into_image(image))
        }
        aws_sdk_bedrockruntime::types::ToolResultContentBlock::Json(document) => {
            ToolResultContent::Text(Text {
                text: document_to_json(document).to_string(),
            })
        }
        aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(text) => {
            ToolResultContent::Text(Text { text })
        }
        _ => unimplemented!("AWS Bedrock sent unsupported ToolResultContentBlock"),
    }
}

pub fn from_tool_result(
    tool_result_content: ToolResultContent,
) -> aws_sdk_bedrockruntime::types::ToolResultContentBlock {
    match tool_result_content {
        ToolResultContent::Text(text) => {
            aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(text.text)
        }
        ToolResultContent::Image(image) => {
            aws_sdk_bedrockruntime::types::ToolResultContentBlock::Image(from_image(image))
        }
    }
}

pub fn into_image(image: aws_sdk_bedrockruntime::types::ImageBlock) -> Image {
    let media_type = match image.format {
        aws_sdk_bedrockruntime::types::ImageFormat::Gif => ImageMediaType::GIF,
        aws_sdk_bedrockruntime::types::ImageFormat::Jpeg => ImageMediaType::JPEG,
        aws_sdk_bedrockruntime::types::ImageFormat::Png => ImageMediaType::PNG,
        aws_sdk_bedrockruntime::types::ImageFormat::Webp => ImageMediaType::WEBP,
        _ => unimplemented!("AWS Bedrock sent unsupported ImageBlock"),
    };
    let data = match image.source {
        Some(ImageSource::Bytes(bytes)) => String::from_utf8(bytes.into_inner()).unwrap(),
        _ => unimplemented!("AWS Bedrock sent unsupported ImageBlock data"),
    };
    Image {
        data,
        format: Some(ContentFormat::Base64),
        media_type: Some(media_type),
        detail: None,
    }
}

pub fn from_image(image: Image) -> aws_sdk_bedrockruntime::types::ImageBlock {
    let format = image.media_type.map(|f| match f {
        ImageMediaType::JPEG => aws_sdk_bedrockruntime::types::ImageFormat::Jpeg,
        ImageMediaType::PNG => aws_sdk_bedrockruntime::types::ImageFormat::Png,
        ImageMediaType::GIF => aws_sdk_bedrockruntime::types::ImageFormat::Gif,
        ImageMediaType::WEBP => aws_sdk_bedrockruntime::types::ImageFormat::Webp,
        _ => unimplemented!("AWS Bedrock doesn't support given image format"),
    });
    let data = aws_smithy_types::Blob::new(image.data);
    aws_sdk_bedrockruntime::types::ImageBlock::builder()
        .set_format(format)
        .source(aws_sdk_bedrockruntime::types::ImageSource::Bytes(data))
        .build()
        .unwrap()
}

pub fn into_document(document: aws_sdk_bedrockruntime::types::DocumentBlock) -> Document {
    let media_type = match document.format {
        aws_sdk_bedrockruntime::types::DocumentFormat::Csv => DocumentMediaType::CSV,
        aws_sdk_bedrockruntime::types::DocumentFormat::Html => DocumentMediaType::HTML,
        aws_sdk_bedrockruntime::types::DocumentFormat::Md => DocumentMediaType::MARKDOWN,
        aws_sdk_bedrockruntime::types::DocumentFormat::Pdf => DocumentMediaType::PDF,
        aws_sdk_bedrockruntime::types::DocumentFormat::Txt => DocumentMediaType::TXT,
        _ => unimplemented!("AWS Bedrock sent unsupported DocumentBlock format"),
    };
    let data = match document.source {
        Some(DocumentSource::Bytes(blob)) => String::from_utf8(blob.into_inner()).unwrap(),
        _ => unimplemented!("AWS Bedrock sent unsupported DocumentBlock data"),
    };
    Document {
        data,
        format: Some(ContentFormat::Base64),
        media_type: Some(media_type),
    }
}

pub fn from_document(document: Document) -> aws_sdk_bedrockruntime::types::DocumentBlock {
    let format = document.media_type.map(|doc| match doc {
        DocumentMediaType::PDF => aws_sdk_bedrockruntime::types::DocumentFormat::Pdf,
        DocumentMediaType::TXT => aws_sdk_bedrockruntime::types::DocumentFormat::Txt,
        DocumentMediaType::HTML => aws_sdk_bedrockruntime::types::DocumentFormat::Html,
        DocumentMediaType::MARKDOWN => aws_sdk_bedrockruntime::types::DocumentFormat::Md,
        DocumentMediaType::CSV => aws_sdk_bedrockruntime::types::DocumentFormat::Csv,
        _ => unimplemented!("AWS Bedrock sent unsupported Document type"),
    });

    let data = aws_smithy_types::Blob::new(document.data);
    let document_source = DocumentSource::Bytes(data);

    aws_sdk_bedrockruntime::types::DocumentBlock::builder()
        .source(document_source)
        .set_format(format)
        .build()
        .unwrap()
}
