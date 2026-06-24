"""Kafka worker for continuous room indexing into Qdrant."""

from __future__ import annotations

import json
import logging
import signal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Consumer, KafkaError, Producer

from kafka_workers.kafka_config import (
    BROKERS,
    GROUP_ROOM_INDEXER,
    TOPIC_ROOM_CHANGED,
    TOPIC_ROOM_INDEX_DLQ,
    deserialize,
    serialize,
)
from room_assistant.indexing import (
    BgeEmbeddingProvider,
    RoomIndexingService,
    PermanentIndexingError,
    RetryableIndexingError,
    make_dlq_record,
)
from room_assistant.qdrant_index import QdrantRoomVectorIndex
from room_assistant.repository import create_room_repository


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("room_index_worker")
running = True


def _log(event: str, **fields):
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))


def _stop(sig, frame):
    global running
    running = False
    _log("shutdown_requested", signal=sig)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def create_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": BROKERS,
        "group.id": GROUP_ROOM_INDEXER,
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    })


def create_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BROKERS,
        "acks": "all",
    })


def _build_service() -> RoomIndexingService:
    return RoomIndexingService(
        repository=create_room_repository(),
        vector_index=QdrantRoomVectorIndex(),
        embedding_provider=BgeEmbeddingProvider(),
    )


def run_worker():
    consumer = create_consumer()
    producer = create_producer()
    service = _build_service()
    consumer.subscribe([TOPIC_ROOM_CHANGED])
    _log("worker_started", topic=TOPIC_ROOM_CHANGED, group=GROUP_ROOM_INDEXER)

    while running:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                _log("kafka_error", error=str(msg.error()))
            continue

        attempts = 3
        raw_event = {}
        try:
            raw_event = deserialize(msg.value())
            result = service.process_with_retry(raw_event, attempts=attempts)
            consumer.commit(message=msg, asynchronous=False)
            _log(
                "index_success",
                event_id=raw_event.get("event_id"),
                room_id=raw_event.get("room_id"),
                operation=raw_event.get("operation"),
                source_version=raw_event.get("source_version"),
                result=result.get("result"),
                attempts=result.get("attempts"),
            )
        except (PermanentIndexingError, RetryableIndexingError, Exception) as exc:
            actual_attempts = 1 if isinstance(exc, PermanentIndexingError) else attempts
            dlq = make_dlq_record(raw_event, exc, attempts=actual_attempts)
            producer.produce(
                TOPIC_ROOM_INDEX_DLQ,
                key=str(raw_event.get("room_id", "unknown")).encode("utf-8"),
                value=serialize(dlq),
            )
            producer.flush()
            consumer.commit(message=msg, asynchronous=False)
            _log(
                "index_dlq",
                event_id=raw_event.get("event_id"),
                room_id=raw_event.get("room_id"),
                error_type=dlq["error_type"],
                attempts=actual_attempts,
            )

    consumer.close()
    producer.flush()
    _log("worker_stopped")


if __name__ == "__main__":
    run_worker()
