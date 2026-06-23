//! Cấu hình cho backend server.
//! Đọc biến môi trường từ .env

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub qdrant_url: String,
    pub qdrant_grpc_url: String,
    pub qdrant_skip_compat_check: bool,
    pub qdrant_collection: String,
    pub host: String,
    pub port: u16,
    // Kafka
    pub kafka_brokers: String,
    pub kafka_response_consumer_group_id: String,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Self {
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_grpc_url: qdrant_grpc_url(),
            qdrant_skip_compat_check: env_bool("QDRANT_SKIP_COMPAT_CHECK", true),
            qdrant_collection: std::env::var("QDRANT_ROOMS_COLLECTION")
                .or_else(|_| std::env::var("QDRANT_COLLECTION"))
                .unwrap_or_else(|_| "rooms_v1".to_string()),
            host: std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "8083".to_string())
                .parse()
                .unwrap_or(8083),
            kafka_brokers: std::env::var("KAFKA_BROKERS")
                .unwrap_or_else(|_| "localhost:9092".to_string()),
            kafka_response_consumer_group_id: response_consumer_group_id(),
        })
    }
}

fn response_consumer_group_id() -> String {
    if let Ok(group_id) = std::env::var("KAFKA_RESPONSE_CONSUMER_GROUP_ID") {
        let trimmed = group_id.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    format!("rust-backend-response-consumer-{}", uuid::Uuid::new_v4())
}

fn qdrant_grpc_url() -> String {
    if let Ok(url) = std::env::var("QDRANT_GRPC_URL") {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let rest_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6333".to_string());
    if rest_url.ends_with(":6333") {
        return format!("{}:6334", rest_url.trim_end_matches(":6333"));
    }
    rest_url
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}
