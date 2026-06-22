//! Kafka Service — Producer/Consumer cho event-driven architecture.
//!
//! Producer: Publish events (query.request)
//! Consumer: Lắng nghe responses (query.response)

use anyhow::Result;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::Message;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info, warn};

use crate::config::AppConfig;

// ─── Topics ──────────────────────────────────────────────────

pub const TOPIC_QUERY_REQUEST: &str = "query.request";
pub const TOPIC_QUERY_RESPONSE: &str = "query.response";
pub const TOPIC_LISTING_CHANGED: &str = "listing.changed";
pub const TOPIC_LISTING_INDEX_DLQ: &str = "listing.index.dlq";

// ─── Message Types ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequestEvent {
    pub session_id: String,
    pub question: String,
    pub history: Vec<ChatHistoryMsg>,
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatHistoryMsg {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponseEvent {
    pub session_id: String,
    pub answer: String,
    #[serde(default)]
    pub intent: Option<String>,
    #[serde(default)]
    pub session_state: serde_json::Value,
    #[serde(default)]
    pub listings: Vec<serde_json::Value>,
    #[serde(default)]
    pub cost_estimate: Option<serde_json::Value>,
    #[serde(default)]
    pub comparison: Option<serde_json::Value>,
    #[serde(default)]
    pub suggested_questions: Vec<String>,
    pub sources: Vec<serde_json::Value>,
    pub agent_trace: serde_json::Value,
    pub processing_time_ms: u64,
    pub is_final: bool,
    #[serde(default)]
    pub chunk_type: Option<String>, // "token" | "status" | null
}

// ─── Kafka Service ───────────────────────────────────────────

#[derive(Clone)]
pub struct KafkaService {
    producer: FutureProducer,
    brokers: String,
    /// Channel senders: session_id → mpsc sender for response routing
    response_channels: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<QueryResponseEvent>>>>,
}

impl KafkaService {
    pub fn new(config: &AppConfig) -> Result<Self> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka_brokers)
            .set("message.timeout.ms", "10000")
            .set("queue.buffering.max.ms", "100")
            .create()
            .map_err(|e| anyhow::anyhow!("Failed to create Kafka producer: {}", e))?;

        info!("✅ Kafka producer connected to: {}", config.kafka_brokers);

        Ok(Self {
            producer,
            brokers: config.kafka_brokers.clone(),
            response_channels: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    // ─── Produce Events ──────────────────────────────────

    /// Publish query request event
    pub async fn publish_query_request(&self, event: QueryRequestEvent) -> Result<()> {
        let payload = serde_json::to_string(&event)?;
        let key = event.session_id.clone();

        self.producer
            .send(
                FutureRecord::to(TOPIC_QUERY_REQUEST)
                    .key(&key)
                    .payload(&payload),
                Duration::from_secs(5),
            )
            .await
            .map_err(|(e, _)| anyhow::anyhow!("Failed to publish query.request: {}", e))?;

        info!("📤 Published query.request: session={}", key);
        Ok(())
    }

    // ─── Response Channel Management ─────────────────────

    /// Register a channel to receive response for a specific session
    pub async fn register_session(
        &self,
        session_id: &str,
    ) -> mpsc::UnboundedReceiver<QueryResponseEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut channels = self.response_channels.lock().await;
        channels.insert(session_id.to_string(), tx);
        rx
    }

    /// Unregister session channel
    pub async fn unregister_session(&self, session_id: &str) {
        let mut channels = self.response_channels.lock().await;
        channels.remove(session_id);
    }

    // ─── Background Consumer ─────────────────────────────

    /// Start background consumer for query.response topic.
    /// Routes responses to the correct session via registered channels.
    pub fn spawn_response_consumer(self: Arc<Self>) {
        let brokers = self.brokers.clone();
        let channels = self.response_channels.clone();

        tokio::spawn(async move {
            info!("🔄 Starting query.response consumer...");

            let consumer: StreamConsumer = match ClientConfig::new()
                .set("bootstrap.servers", &brokers)
                .set("group.id", "rust-backend-response-consumer")
                .set("auto.offset.reset", "latest")
                .set("enable.auto.commit", "true")
                .create()
            {
                Ok(c) => c,
                Err(e) => {
                    error!("❌ Failed to create response consumer: {}", e);
                    return;
                }
            };

            if let Err(e) = consumer.subscribe(&[TOPIC_QUERY_RESPONSE]) {
                error!("❌ Failed to subscribe to {}: {}", TOPIC_QUERY_RESPONSE, e);
                return;
            }

            info!("✅ Listening on topic: {}", TOPIC_QUERY_RESPONSE);

            use futures::StreamExt;
            let mut stream = consumer.stream();

            while let Some(result) = stream.next().await {
                match result {
                    Ok(message) => {
                        if let Some(payload) = message.payload() {
                            match serde_json::from_slice::<QueryResponseEvent>(payload) {
                                Ok(event) => {
                                    let session_id = event.session_id.clone();
                                    let channels_guard = channels.lock().await;

                                    if let Some(sender) = channels_guard.get(&session_id) {
                                        if let Err(e) = sender.send(event) {
                                            warn!(
                                                "⚠️ Failed to route response to session {}: {}",
                                                session_id, e
                                            );
                                        }
                                    } else {
                                        warn!(
                                            "⚠️ No registered channel for session: {}",
                                            session_id
                                        );
                                    }
                                }
                                Err(e) => {
                                    warn!("⚠️ Failed to parse query.response: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("❌ Kafka consumer error: {}", e);
                    }
                }
            }
        });
    }
}
