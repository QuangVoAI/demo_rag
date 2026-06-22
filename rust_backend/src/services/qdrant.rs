//! Qdrant service - truy vấn vector database

use anyhow::Result;
use qdrant_client::Qdrant;
use tracing::info;

use crate::config::AppConfig;

#[derive(Clone)]
pub struct QdrantService {
    client: Qdrant,
    collection: String,
}

impl QdrantService {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let mut builder = Qdrant::from_url(&config.qdrant_grpc_url);
        if config.qdrant_skip_compat_check {
            builder = builder.skip_compatibility_check();
        }
        let client = builder.build()?;
        info!(
            "Connected to Qdrant: {} (REST config: {})",
            config.qdrant_grpc_url,
            config.qdrant_url
        );

        Ok(Self {
            client,
            collection: config.qdrant_collection.clone(),
        })
    }

    /// Kiểm tra kết nối
    pub async fn health_check(&self) -> Result<bool> {
        let info = self.client.collection_info(&self.collection).await?;
        Ok(info.result.is_some())
    }
}
