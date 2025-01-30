use aws_config::{BehaviorVersion, Region};
use rig::{agent::AgentBuilder, embeddings, extractor::ExtractorBuilder, Embed};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    completion::CompletionModel,
    embedding::{BedrockEmbeddingModel, EmbeddingModel},
};

pub enum BedrockModel {
    NovaLite,
    Mistral8x7BInstruct,
    Custom(&'static str),
}
// ================================================================
// All supported models: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
// ================================================================
impl BedrockModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            BedrockModel::NovaLite => "amazon.nova-lite-v1:0",
            BedrockModel::Mistral8x7BInstruct => "mistral.mixtral-8x7b-instruct-v0:1",
            BedrockModel::Custom(str) => str,
        }
    }
}

// Important: make sure to verify model and region compatibility: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html
pub const DEFAULT_AWS_REGION: &str = "us-east-1";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    region: &'a str,
}

/// Create a new Bedrock client using the builder
///
/// #(Make sure you have permissions to access Amazon Bedrock foundation model)
/// [https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html]
impl<'a> ClientBuilder<'a> {
    pub fn new() -> Self {
        Self {
            region: DEFAULT_AWS_REGION,
        }
    }

    pub fn region(mut self, region: &'a str) -> Self {
        self.region = region;
        self
    }

    pub async fn build(self) -> Client {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(String::from(self.region)))
            .load()
            .await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);
        Client { aws_client: client }
    }
}

impl<'a> Default for ClientBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct Client {
    pub(crate) aws_client: aws_sdk_bedrockruntime::Client,
}

impl Client {
    pub fn completion_model(&self, model: BedrockModel) -> CompletionModel {
        CompletionModel::new(self.clone(), model.as_str())
    }

    pub fn agent(&self, model: BedrockModel) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn embedding_model(&self, model: BedrockEmbeddingModel) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model)
    }

    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: BedrockModel,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }

    pub fn embeddings<D: Embed>(
        &self,
        model: BedrockEmbeddingModel,
    ) -> embeddings::EmbeddingsBuilder<EmbeddingModel, D> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model))
    }
}
