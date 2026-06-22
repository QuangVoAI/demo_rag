//! Nhatrovn Assistant - Rust Backend
//!
//! Architecture: Rust Gateway + Kafka + Python read-only room assistant
//! - WebSocket /ws/chat → Kafka → Python workflow → Kafka → WebSocket
//! - REST API /api/sessions → SQLite Chat History

mod config;
mod errors;
mod models;
mod routes;
mod services;

use actix_cors::Cors;
use actix_files as fs;
use actix_web::{middleware, web, App, HttpServer};
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::config::AppConfig;
use crate::services::database::Database;
use crate::services::kafka::KafkaService;
use crate::services::qdrant::QdrantService;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // ─── Logging ─────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // ─── Configuration ───────────────────────────────────
    let config = AppConfig::from_env().expect("Failed to load configuration");
    let bind_addr = format!("{}:{}", config.host, config.port);

    info!("Nhatrovn Assistant Backend (read-only + History) starting...");
    info!("📍 Binding to: {}", bind_addr);

    // ─── Database ────────────────────────────────────────
    let db = Database::new("chat_history.db").expect("Failed to initialize SQLite database");
    let db = Arc::new(db);
    info!("✅ SQLite database ready");

    // ─── Services ────────────────────────────────────────
    let qdrant_service = QdrantService::new(&config)
        .await
        .expect("Failed to connect to Qdrant");
    info!("✅ Qdrant service initialized");

    // Kafka service
    let kafka_service = match KafkaService::new(&config) {
        Ok(ks) => {
            info!(
                "✅ Kafka service initialized (brokers: {})",
                config.kafka_brokers
            );
            let ks = Arc::new(ks);
            // Spawn background consumer for query.response (streaming)
            KafkaService::spawn_response_consumer(ks.clone());
            ks
        }
        Err(e) => {
            tracing::warn!("⚠️ Kafka not available ({}). WebSocket will not work.", e);
            tracing::warn!("   Start Kafka with: docker-compose up -d redpanda");
            Arc::new(
                KafkaService::new(&AppConfig {
                    kafka_brokers: "localhost:9092".to_string(),
                    ..config.clone()
                })
                .unwrap_or_else(|_| panic!("Cannot create even a fallback Kafka service")),
            )
        }
    };

    // ─── Server ──────────────────────────────────────────
    info!("🚀 Server ready at http://{}", bind_addr);
    info!("WebSocket at ws://{}/ws/chat", bind_addr);
    info!("📚 History API at http://{}/api/sessions", bind_addr);

    let kafka_for_server = kafka_service.clone();
    let db_for_server = db.clone();

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(cors)
            .wrap(middleware::Logger::default())
            .wrap(tracing_actix_web::TracingLogger::default())
            .app_data(web::Data::new(qdrant_service.clone()))
            .app_data(web::Data::new(kafka_for_server.clone()))
            .app_data(web::Data::new(db_for_server.clone()))
            // Health check
            .service(routes::health::health_check)
            // Chat History REST API
            .route(
                "/api/sessions",
                web::get().to(routes::history::list_sessions),
            )
            .route(
                "/api/sessions/{session_id}/messages",
                web::get().to(routes::history::get_session_messages),
            )
            .route(
                "/api/sessions/{session_id}",
                web::delete().to(routes::history::delete_session),
            )
            // WebSocket endpoint (real-time streaming via Kafka)
            .route("/ws/chat", web::get().to(routes::ws::ws_chat_handler))
            // Static files (frontend)
            .service(fs::Files::new("/", "../frontend").index_file("index.html"))
    })
    .bind(&bind_addr)?
    .run()
    .await
}
