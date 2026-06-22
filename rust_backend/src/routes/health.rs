//! Health check endpoint

use crate::services::qdrant::QdrantService;
use actix_web::{get, web, HttpResponse};

#[get("/health")]
pub async fn health_check(qdrant: web::Data<QdrantService>) -> HttpResponse {
    let qdrant_ok = qdrant.health_check().await.unwrap_or(false);

    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "services": {
            "qdrant": if qdrant_ok { "connected" } else { "disconnected" },
            "assistant": "read_only",
        },
        "version": env!("CARGO_PKG_VERSION"),
    }))
}
