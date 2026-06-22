//! WebSocket endpoint — Real-time streaming query.
//!
//! Flow:
//! 1. Client connects via WebSocket `/ws/chat`
//! 2. Client sends JSON query message
//! 3. Server publishes to Kafka `query.request`
//! 4. Python LangGraph worker streams tokens to `query.response` (is_final=false)
//! 5. Server pushes each token to client via WebSocket
//! 6. Final response with sources/trace arrives (is_final=true)

use actix_web::{rt, web, HttpRequest, HttpResponse};
use actix_ws::Message;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use tracing::{error, info, warn};

use crate::services::database::Database;
use crate::services::kafka::{ChatHistoryMsg, KafkaService, QueryRequestEvent};

/// WebSocket query message từ client
#[derive(Debug, Deserialize)]
struct WsQueryMessage {
    question: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    history: Vec<WsChatHistory>,
}

#[derive(Debug, Deserialize, Clone)]
struct WsChatHistory {
    role: String,
    content: String,
}

fn default_top_k() -> usize {
    5
}

/// WebSocket response messages gửi về client
#[derive(Debug, Serialize)]
struct WsStreamMessage {
    #[serde(rename = "type")]
    msg_type: String,
    session_id: String,
    token: Option<String>,
}

#[derive(Debug, Serialize)]
struct WsAnswerMessage {
    #[serde(rename = "type")]
    msg_type: String,
    session_id: String,
    answer: String,
    intent: Option<String>,
    session_state: serde_json::Value,
    listings: Vec<serde_json::Value>,
    cost_estimate: Option<serde_json::Value>,
    comparison: Option<serde_json::Value>,
    suggested_questions: Vec<String>,
    retrieval_confidence: Option<f64>,
    retrieval_low_confidence: Option<bool>,
    retrieval_feedback_retry_count: usize,
    retrieval_attempts: Vec<serde_json::Value>,
    sources: Vec<serde_json::Value>,
    agent_trace: serde_json::Value,
    processing_time_ms: u64,
    status: String,
}

#[derive(Debug, Serialize)]
struct WsStatusMessage {
    #[serde(rename = "type")]
    msg_type: String,
    session_id: Option<String>,
    status: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct WsErrorMessage {
    #[serde(rename = "type")]
    msg_type: String,
    session_id: Option<String>,
    message: String,
}

/// WebSocket handler
pub async fn ws_chat_handler(
    req: HttpRequest,
    stream: web::Payload,
    kafka: web::Data<Arc<KafkaService>>,
    db: web::Data<Arc<Database>>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, mut session, mut msg_stream) = actix_ws::handle(&req, stream)?;

    let kafka_service = kafka.get_ref().clone();
    let db_service = db.get_ref().clone();

    rt::spawn(async move {
        info!("🔌 WebSocket client connected");

        loop {
            let msg = msg_stream.next().await;
            match msg {
                Some(Ok(Message::Text(text))) => {
                    match serde_json::from_str::<WsQueryMessage>(&text) {
                        Ok(query) => {
                            let session_id = query
                                .session_id
                                .clone()
                                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                            info!(
                                "📨 WS Query: '{}' (session: {})",
                                &query.question[..query.question.len().min(50)],
                                &session_id[..8]
                            );

                            // ─── Save to Database ───
                            // Create session + save user message
                            if let Err(e) = db_service.create_session(&session_id) {
                                warn!("DB create_session error: {}", e);
                            }
                            // Auto-title: first 50 chars of question
                            let title = query.question.chars().take(50).collect::<String>();
                            if let Err(e) = db_service.update_session_title(&session_id, &title) {
                                warn!("DB update_title error: {}", e);
                            }

                            let server_history =
                                match db_service.get_recent_messages(&session_id, 8) {
                                    Ok(messages) => messages
                                        .into_iter()
                                        .map(|m| ChatHistoryMsg {
                                            role: m.role,
                                            content: m.content,
                                        })
                                        .collect::<Vec<_>>(),
                                    Err(e) => {
                                        warn!("DB recent history error: {}", e);
                                        Vec::new()
                                    }
                                };
                            let history_for_worker =
                                if server_history.is_empty() && !query.history.is_empty() {
                                    query
                                        .history
                                        .iter()
                                        .rev()
                                        .take(8)
                                        .cloned()
                                        .collect::<Vec<_>>()
                                        .into_iter()
                                        .rev()
                                        .map(|h| ChatHistoryMsg {
                                            role: h.role,
                                            content: h.content,
                                        })
                                        .collect::<Vec<_>>()
                                } else {
                                    server_history
                                };

                            if let Err(e) =
                                db_service.save_message(&session_id, "user", &query.question)
                            {
                                warn!("DB save_message(user) error: {}", e);
                            }

                            // Send status: processing
                            let status_msg = WsStatusMessage {
                                msg_type: "status".to_string(),
                                session_id: Some(session_id.clone()),
                                status: "processing".to_string(),
                                message: "Đang phân tích câu hỏi...".to_string(),
                            };
                            if let Ok(json) = serde_json::to_string(&status_msg) {
                                let _ = session.text(json).await;
                            }

                            // Register response channel
                            let mut rx = kafka_service.register_session(&session_id).await;

                            // Publish query to Kafka
                            let kafka_event = QueryRequestEvent {
                                session_id: session_id.clone(),
                                question: query.question,
                                history: history_for_worker,
                                top_k: query.top_k,
                            };

                            if let Err(e) = kafka_service.publish_query_request(kafka_event).await {
                                error!("❌ Failed to publish query: {}", e);
                                let err_msg = WsErrorMessage {
                                    msg_type: "error".to_string(),
                                    session_id: Some(session_id.clone()),
                                    message: format!("Kafka error: {}", e),
                                };
                                if let Ok(json) = serde_json::to_string(&err_msg) {
                                    let _ = session.text(json).await;
                                }
                                kafka_service.unregister_session(&session_id).await;
                                continue;
                            }

                            // ═══════════════════════════════════════════
                            // STREAMING LOOP: receive tokens until final
                            // ═══════════════════════════════════════════
                            // 300s timeout cho cold start (lần đầu load 3 model)
                            let timeout_deadline =
                                tokio::time::Instant::now() + Duration::from_secs(300);

                            loop {
                                let remaining = timeout_deadline
                                    .saturating_duration_since(tokio::time::Instant::now());

                                if remaining.is_zero() {
                                    warn!("⏰ Query timeout for session {}", &session_id[..8]);
                                    let timeout_msg = WsErrorMessage {
                                        msg_type: "error".to_string(),
                                        session_id: Some(session_id.clone()),
                                        message: "Quá thời gian chờ phản hồi. Vui lòng thử lại."
                                            .to_string(),
                                    };
                                    if let Ok(json) = serde_json::to_string(&timeout_msg) {
                                        let _ = session.text(json).await;
                                    }
                                    break;
                                }

                                match tokio::time::timeout(remaining, rx.recv()).await {
                                    Ok(Some(response_event)) => {
                                        if response_event.is_final {
                                            // ─── Final answer ───
                                            let answer_for_db = response_event.answer.clone();
                                            let answer_msg = WsAnswerMessage {
                                                msg_type: "answer".to_string(),
                                                session_id: session_id.clone(),
                                                answer: response_event.answer,
                                                intent: response_event.intent,
                                                session_state: response_event.session_state,
                                                listings: response_event.listings,
                                                cost_estimate: response_event.cost_estimate,
                                                comparison: response_event.comparison,
                                                suggested_questions: response_event
                                                    .suggested_questions,
                                                retrieval_confidence: response_event
                                                    .retrieval_confidence,
                                                retrieval_low_confidence: response_event
                                                    .retrieval_low_confidence,
                                                retrieval_feedback_retry_count: response_event
                                                    .retrieval_feedback_retry_count,
                                                retrieval_attempts: response_event
                                                    .retrieval_attempts,
                                                sources: response_event.sources,
                                                agent_trace: response_event.agent_trace,
                                                processing_time_ms: response_event
                                                    .processing_time_ms,
                                                status: "complete".to_string(),
                                            };
                                            if let Ok(json) = serde_json::to_string(&answer_msg) {
                                                let _ = session.text(json).await;
                                            }

                                            // ─── Save AI answer to DB ───
                                            if let Err(e) = db_service.save_message(
                                                &session_id,
                                                "assistant",
                                                &answer_for_db,
                                            ) {
                                                warn!("DB save_message(assistant) error: {}", e);
                                            }

                                            break;
                                        } else {
                                            // ─── Streaming token ───
                                            let stream_msg = WsStreamMessage {
                                                msg_type: "stream".to_string(),
                                                session_id: session_id.clone(),
                                                token: Some(response_event.answer),
                                            };
                                            if let Ok(json) = serde_json::to_string(&stream_msg) {
                                                let _ = session.text(json).await;
                                            }
                                        }
                                    }
                                    Ok(None) => {
                                        warn!(
                                            "⚠️ Response channel closed for session {}",
                                            &session_id[..8]
                                        );
                                        break;
                                    }
                                    Err(_) => {
                                        warn!("⏰ Query timeout for session {}", &session_id[..8]);
                                        let timeout_msg = WsErrorMessage {
                                            msg_type: "error".to_string(),
                                            session_id: Some(session_id.clone()),
                                            message:
                                                "Quá thời gian chờ phản hồi. Vui lòng thử lại."
                                                    .to_string(),
                                        };
                                        if let Ok(json) = serde_json::to_string(&timeout_msg) {
                                            let _ = session.text(json).await;
                                        }
                                        break;
                                    }
                                }
                            }

                            // Cleanup
                            kafka_service.unregister_session(&session_id).await;
                        }
                        Err(e) => {
                            warn!("⚠️ Invalid WS message format: {}", e);
                            let err_msg = serde_json::json!({
                                "type": "error",
                                "message": format!("Invalid message format: {}", e),
                            });
                            let _ = session.text(err_msg.to_string()).await;
                        }
                    }
                }
                Some(Ok(Message::Ping(bytes))) => {
                    let _ = session.pong(&bytes).await;
                }
                Some(Ok(Message::Close(reason))) => {
                    info!("🔌 WebSocket client disconnected: {:?}", reason);
                    break;
                }
                Some(Err(e)) => {
                    warn!("⚠️ WebSocket error: {}", e);
                    break;
                }
                None => break,
                _ => {}
            }
        }

        info!("🔌 WebSocket connection closed");
    });

    Ok(response)
}
