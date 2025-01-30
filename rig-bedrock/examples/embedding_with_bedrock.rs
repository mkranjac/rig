use rig::Embed;
use rig_bedrock::client::ClientBuilder;
use rig_bedrock::embedding::BedrockEmbeddingModel;
use tracing::info;

#[derive(rig_derive::Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = ClientBuilder::new().build().await;

    let embeddings = client
        .embeddings(BedrockEmbeddingModel::TitanTextEmbeddingsV2(256))
        .document(Greetings {
            message: "aa".to_string(),
        })?
        .document(Greetings {
            message: "bb".to_string(),
        })?
        .build()
        .await?;

    info!("{:?}", embeddings);

    Ok(())
}
