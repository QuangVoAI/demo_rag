"""Continuous room indexing contract and service."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

from .repository import RoomRepository


VALID_OPERATIONS = {"upsert", "delete", "publish", "unpublish"}
RETRYABLE_ERRORS = {"mongo_timeout", "network_timeout", "embedding_timeout", "qdrant_timeout", "inconsistent_data"}

_MONGO_CDC_OPERATION_MAP = {
    "insert": "upsert",
    "update": "upsert",
    "replace": "upsert",
    "upsert": "upsert",
    "delete": "delete",
    "publish": "publish",
    "unpublish": "unpublish",
}


def coerce_room_changed_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy `{action, room_id}` payloads into the full indexing schema."""
    if not isinstance(raw, dict):
        raise PermanentIndexingError("invalid_schema", "Event must be a dict")

    required = ("event_id", "room_id", "operation", "source_version", "occurred_at", "producer")
    if all(field in raw for field in required):
        merged = dict(raw)
        operation = str(merged.get("operation") or merged.get("action") or "").strip().lower()
        mapped = _MONGO_CDC_OPERATION_MAP.get(operation, operation)
        merged["operation"] = mapped
        return merged

    room_id = raw.get("room_id")
    action = raw.get("action") or raw.get("operation")
    if not room_id or not action:
        raise PermanentIndexingError("invalid_schema", "Missing room_id or operation/action")

    now = datetime.now(timezone.utc).isoformat()
    operation = _MONGO_CDC_OPERATION_MAP.get(str(action).strip().lower(), str(action).strip().lower())
    return {
        "event_id": str(raw.get("event_id") or f"legacy-{room_id}-{int(time.time() * 1000)}"),
        "room_id": str(room_id),
        "operation": operation,
        "source_version": int(raw.get("source_version") or 0),
        "occurred_at": str(raw.get("occurred_at") or now),
        "producer": str(raw.get("producer") or "mongo-cdc"),
    }


def source_version_from_change(change: dict[str, Any], doc: dict[str, Any] | None = None) -> int:
    if doc and doc.get("source_version") is not None:
        try:
            return int(doc["source_version"])
        except Exception:
            pass
    cluster_time = change.get("clusterTime")
    if cluster_time is not None:
        try:
            return int(getattr(cluster_time, "time", cluster_time))
        except Exception:
            pass
    return int(time.time())


def build_mongo_cdc_event(
    *,
    room_id: str,
    mongo_operation: str,
    source_version: int | None = None,
    producer: str = "mongo-cdc",
) -> dict[str, Any]:
    operation = _MONGO_CDC_OPERATION_MAP.get(str(mongo_operation).strip().lower(), "upsert")
    now = datetime.now(timezone.utc).isoformat()
    version = int(source_version or 0)
    return {
        "event_id": f"cdc-{room_id}-{int(time.time() * 1000)}",
        "room_id": str(room_id),
        "operation": operation,
        "source_version": version,
        "occurred_at": now,
        "producer": producer,
    }


class RoomVectorIndex(Protocol):
    def get_payload(self, room_id: str, chunk_type: str = "room_summary") -> dict | None:
        ...

    def upsert_room_chunk(
        self,
        room_id: str,
        chunk_type: str,
        text: str,
        embedding: Any,
        payload: dict,
    ) -> str:
        ...

    def delete_room(self, room_id: str) -> None:
        ...


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Any:
        ...


class CacheInvalidator(Protocol):
    def invalidate_room(self, room_id: str) -> None:
        ...


class NoopCacheInvalidator:
    def invalidate_room(self, room_id: str) -> None:
        return


class BgeEmbeddingProvider:
    def embed(self, text: str) -> np.ndarray:
        from agents.model_registry import get_embed_model

        model = get_embed_model()
        return model.encode(text, normalize_embeddings=True)


class PermanentIndexingError(Exception):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class RetryableIndexingError(Exception):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


def validate_room_changed_event(event: dict[str, Any]) -> dict[str, Any]:
    event = coerce_room_changed_event(event)
    required = ("event_id", "room_id", "operation", "source_version", "occurred_at", "producer")
    missing = [field for field in required if field not in event]
    if missing:
        raise PermanentIndexingError("invalid_schema", f"Missing fields: {', '.join(missing)}")
    if event["operation"] not in VALID_OPERATIONS:
        raise PermanentIndexingError("unsupported_operation", str(event["operation"]))
    try:
        source_version = int(event["source_version"])
    except Exception as exc:
        raise PermanentIndexingError("invalid_schema", "source_version must be an integer") from exc
    return {
        "event_id": str(event["event_id"]),
        "room_id": str(event["room_id"]),
        "operation": str(event["operation"]),
        "source_version": source_version,
        "occurred_at": str(event["occurred_at"]),
        "producer": str(event["producer"]),
    }


def build_canonical_embedding_text(room: dict[str, Any]) -> str:
    """Build embedding text for a room document.

    If the room already has embedding_text from the rooms collection, use it and
    append ``tien_ich_xq`` when present (nearby is stored separately in Mongo).
  Otherwise, build from individual fields.
    """
    from .landmark_aliases import merge_nearby_into_embedding_text

    if room.get("embedding_text") or room.get("tien_ich_xq"):
        return merge_nearby_into_embedding_text(room)

    fields = [
        ("Tiêu đề", room.get("title")),
        ("Mô tả", room.get("description") or room.get("house_remark")),
        ("Tiện ích xung quanh", room.get("tien_ich_xq")),
        ("Khu vực", room.get("address")),
        ("Tiện ích", ", ".join(room.get("amenities") or [])),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RoomIndexingService:
    def __init__(
        self,
        repository: RoomRepository,
        vector_index: RoomVectorIndex,
        embedding_provider: EmbeddingProvider | None = None,
        cache: CacheInvalidator | None = None,
        embedding_model: str | None = None,
        embedding_version: int | None = None,
    ) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider or BgeEmbeddingProvider()
        self.cache = cache or NoopCacheInvalidator()
        self.embedding_model = embedding_model or _config_value("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.embedding_version = int(embedding_version or _config_value("EMBEDDING_VERSION", 1))

    def process_event(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        event = validate_room_changed_event(raw_event)
        operation = event["operation"]
        room_id = event["room_id"]

        existing_payload = self.vector_index.get_payload(room_id) or {}
        existing_version = int(existing_payload.get("source_version") or -1)
        if existing_version > event["source_version"]:
            return {"result": "skipped_old_event", "room_id": room_id, "source_version": event["source_version"]}

        if operation in {"delete", "unpublish"}:
            self.vector_index.delete_room(room_id)
            self.cache.invalidate_room(room_id)
            return {"result": "deleted", "room_id": room_id, "operation": operation}

        room = self.repository.get_by_id(room_id)
        if not room:
            raise RetryableIndexingError("inconsistent_data", f"Room not found: {room_id}")

        latest_version = int(room.get("source_version") or 0)
        if latest_version < event["source_version"]:
            raise RetryableIndexingError("inconsistent_data", "Repository version is older than event")
        if latest_version < existing_version:
            return {"result": "skipped_old_repository_version", "room_id": room_id}

        text = build_canonical_embedding_text(room)
        new_hash = content_hash(text)
        if (
            existing_payload.get("content_hash") == new_hash
            and int(existing_payload.get("source_version") or -1) >= latest_version
            and int(existing_payload.get("embedding_version") or -1) == self.embedding_version
        ):
            self.cache.invalidate_room(room_id)
            return {"result": "skipped_unchanged", "room_id": room_id, "content_hash": new_hash}

        embedding = self.embedding_provider.embed(text)
        payload = {
            "room_id": room_id,
            "house_id": room.get("house_id"),
            "chunk_type": "room_summary",
            "source_version": latest_version,
            "content_hash": new_hash,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "status": room.get("status", "active"),
            "district": room.get("district"),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        point_id = self.vector_index.upsert_room_chunk(
            room_id=room_id,
            chunk_type="room_summary",
            text=text,
            embedding=embedding,
            payload=payload,
        )
        self.cache.invalidate_room(room_id)
        return {
            "result": "upserted",
            "room_id": room_id,
            "point_id": point_id,
            "source_version": latest_version,
            "content_hash": new_hash,
        }

    def process_with_retry(self, raw_event: dict[str, Any], attempts: int = 3, base_delay_seconds: float = 0.2) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = self.process_event(raw_event)
                result["attempts"] = attempt
                return result
            except PermanentIndexingError:
                raise
            except RetryableIndexingError as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                time.sleep(min(base_delay_seconds * (2 ** (attempt - 1)), 5))
            except Exception as exc:
                last_error = RetryableIndexingError("network_timeout", str(exc))
                if attempt >= attempts:
                    raise last_error
                time.sleep(min(base_delay_seconds * (2 ** (attempt - 1)), 5))
        raise last_error or RetryableIndexingError("unknown", "Indexing failed")


def make_dlq_record(original_event: dict[str, Any], error: Exception, attempts: int) -> dict[str, Any]:
    error_type = getattr(error, "error_type", error.__class__.__name__)
    return {
        "original_event": original_event,
        "error_type": error_type,
        "error_message": str(error),
        "attempts": attempts,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }


def _config_value(name: str, default: Any) -> Any:
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default
