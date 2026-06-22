//! Custom error types cho backend

use actix_web::{HttpResponse, ResponseError};
use std::fmt;

#[derive(Debug)]
pub enum AppError {
    Internal(String),
    BadRequest(String),
    NotFound(String),
    Qdrant(String),
    Kafka(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal(msg) => write!(f, "Internal error: {msg}"),
            Self::BadRequest(msg) => write!(f, "Bad request: {msg}"),
            Self::NotFound(msg) => write!(f, "Not found: {msg}"),
            Self::Qdrant(msg) => write!(f, "Qdrant error: {msg}"),
            Self::Kafka(msg) => write!(f, "Kafka error: {msg}"),
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let (status, message) = match self {
            Self::Internal(msg) => (
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                msg.clone(),
            ),
            Self::BadRequest(msg) => (actix_web::http::StatusCode::BAD_REQUEST, msg.clone()),
            Self::NotFound(msg) => (actix_web::http::StatusCode::NOT_FOUND, msg.clone()),
            Self::Qdrant(msg) => (
                actix_web::http::StatusCode::SERVICE_UNAVAILABLE,
                msg.clone(),
            ),
            Self::Kafka(msg) => (actix_web::http::StatusCode::BAD_GATEWAY, msg.clone()),
        };

        HttpResponse::build(status).json(serde_json::json!({
            "error": message,
            "status": status.as_u16(),
        }))
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal(err.to_string())
    }
}
