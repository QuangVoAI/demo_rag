"""Continuous listing indexing contract and service."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Protocol

from .repository import ListingRepository


VALID_OPERATIONS = {"upsert", "delete", "publish", "unpublish"}
RETRYABLE_ERRORS = {"mongo_timeout", "network_timeout", "embedding_timeout", "qdrant_timeout", "inconsistent_data"}


class ListingVectorIndex(Protocol):
    def get_payload(self, listing_id: str, chunk_type: str = "listing_summary") -> dict | None:
        ...

    def upsert_listing_chunk(
        self,
        listing_id: str,
        chunk_type: str,
        text: str,
        embedding: Any,
        payload: dict,
    ) -> str:
        ...

    def delete_listing(self, listing_id: str) -> None:
        ...


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Any:
        ...


class CacheInvalidator(Protocol):
    def invalidate_listing(self, listing_id: str) -> None:
        ...


class NoopCacheInvalidator:
    def invalidate_listing(self, listing_id: str) -> None:
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


def validate_listing_changed_event(event: dict[str, Any]) -> dict[str, Any]:
    required = ("event_id", "listing_id", "operation", "source_version", "occurred_at", "producer")
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
        "listing_id": str(event["listing_id"]),
        "operation": str(event["operation"]),
        "source_version": source_version,
        "occurred_at": str(event["occurred_at"]),
        "producer": str(event["producer"]),
    }


def build_canonical_embedding_text(listing: dict[str, Any]) -> str:
    fields = [
        ("Tiêu đề", listing.get("title")),
        ("Mô tả", listing.get("description")),
        ("Đặc điểm không gian", listing.get("space_features") or listing.get("area_m2")),
        ("Tiện ích", ", ".join(listing.get("amenities") or [])),
        ("Khu vực xung quanh", listing.get("neighborhood") or listing.get("address")),
        ("Quy định dạng văn bản", listing.get("rules")),
        ("Điểm nổi bật", listing.get("highlights")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ListingIndexingService:
    def __init__(
        self,
        repository: ListingRepository,
        vector_index: ListingVectorIndex,
        embedding_provider: EmbeddingProvider,
        cache: CacheInvalidator | None = None,
        embedding_model: str | None = None,
        embedding_version: int | None = None,
    ) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.cache = cache or NoopCacheInvalidator()
        self.embedding_model = embedding_model or _config_value("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.embedding_version = int(embedding_version or _config_value("EMBEDDING_VERSION", 1))

    def process_event(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        event = validate_listing_changed_event(raw_event)
        operation = event["operation"]
        listing_id = event["listing_id"]

        if operation in {"delete", "unpublish"}:
            self.vector_index.delete_listing(listing_id)
            self.cache.invalidate_listing(listing_id)
            return {"result": "deleted", "listing_id": listing_id, "operation": operation}

        existing_payload = self.vector_index.get_payload(listing_id) or {}
        existing_version = int(existing_payload.get("source_version") or -1)
        if existing_version > event["source_version"]:
            return {"result": "skipped_old_event", "listing_id": listing_id, "source_version": event["source_version"]}

        listing = self.repository.get_by_id(listing_id)
        if not listing:
            raise RetryableIndexingError("inconsistent_data", f"Listing not found: {listing_id}")

        latest_version = int(listing.get("source_version") or 0)
        if latest_version < event["source_version"]:
            raise RetryableIndexingError("inconsistent_data", "Repository version is older than event")
        if latest_version < existing_version:
            return {"result": "skipped_old_repository_version", "listing_id": listing_id}

        text = build_canonical_embedding_text(listing)
        new_hash = content_hash(text)
        if (
            existing_payload.get("content_hash") == new_hash
            and int(existing_payload.get("source_version") or -1) >= latest_version
            and int(existing_payload.get("embedding_version") or -1) == self.embedding_version
        ):
            self.cache.invalidate_listing(listing_id)
            return {"result": "skipped_unchanged", "listing_id": listing_id, "content_hash": new_hash}

        embedding = self.embedding_provider.embed(text)
        payload = {
            "listing_id": listing_id,
            "chunk_type": "listing_summary",
            "source_version": latest_version,
            "content_hash": new_hash,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "status": listing.get("status", "active"),
            "district": listing.get("district"),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        point_id = self.vector_index.upsert_listing_chunk(
            listing_id=listing_id,
            chunk_type="listing_summary",
            text=text,
            embedding=embedding,
            payload=payload,
        )
        self.cache.invalidate_listing(listing_id)
        return {
            "result": "upserted",
            "listing_id": listing_id,
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
