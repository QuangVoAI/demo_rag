//! Cấu hình cho backend server.
//! Đọc biến môi trường từ .env

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub host: String,
    pub port: u16,
    // Kafka
    pub kafka_brokers: String,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Self {
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_collection: std::env::var("QDRANT_LISTINGS_COLLECTION")
                .or_else(|_| std::env::var("QDRANT_COLLECTION"))
                .unwrap_or_else(|_| "listings_v1".to_string()),
            host: std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "8083".to_string())
                .parse()
                .unwrap_or(8083),
            kafka_brokers: std::env::var("KAFKA_BROKERS")
                .unwrap_or_else(|_| "localhost:9092".to_string()),
        })
    }
}
