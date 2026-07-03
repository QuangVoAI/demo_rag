"""Structured session state storage.

Redis is the primary implementation when configured. In-memory storage is a
graceful local fallback for tests and developer runs without Redis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from copy import deepcopy
from typing import Any, Protocol

_logger = logging.getLogger(__name__)

from .schemas import (
    ALLOWED_OPERATION_PATHS,
    LIST_OPERATION_PATHS,
    OP_TYPES,
    default_session_state,
    utc_now_iso,
)


class SessionStore(Protocol):
    def get(self, session_id: str) -> dict[str, Any] | None:
        ...

    def save(self, session_id: str, state: dict[str, Any], ttl_seconds: int) -> None:
        ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, session_id: str) -> dict[str, Any] | None:
        item = self._data.get(session_id)
        if not item:
            return None
        expires_at, state = item
        if expires_at and expires_at < time.time():
            self._data.pop(session_id, None)
            return None
        return deepcopy(state)

    def save(self, session_id: str, state: dict[str, Any], ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds else 0
        self._data[session_id] = (expires_at, deepcopy(state))


class RedisSessionStore:
    def __init__(self, url: str, token: str) -> None:
        from upstash_redis import Redis

        self._redis = Redis(url=url, token=token)

    @staticmethod
    def key(session_id: str) -> str:
        return f"ai:session:{session_id}"

    def get(self, session_id: str) -> dict[str, Any] | None:
        try:
            raw = self._redis.get(self.key(session_id))
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw) if isinstance(raw, str) else raw

    def save(self, session_id: str, state: dict[str, Any], ttl_seconds: int) -> None:
        value = json.dumps(state, ensure_ascii=False)
        try:
            self._redis.setex(self.key(session_id), ttl_seconds, value)
        except Exception as exc:
            _logger.warning(
                "room_assistant_session_store_redis_save_failed session_id=%s error=%s",
                session_id,
                exc,
            )
            return


def create_session_store() -> SessionStore:
    try:
        from config import (
            UPSTASH_REDIS_REST_TOKEN,
            UPSTASH_REDIS_REST_URL,
        )
    except Exception:
        UPSTASH_REDIS_REST_TOKEN = ""
        UPSTASH_REDIS_REST_URL = ""

    if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
        try:
            return RedisSessionStore(UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN)
        except Exception as exc:
            _logger.warning(
                "room_assistant_session_store_redis_failed fallback=in_memory error=%s",
                exc,
            )
            return InMemorySessionStore()
    _logger.warning(
        "room_assistant_session_store_redis_not_configured fallback=in_memory "
        "(set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN for durable session memory)",
    )
    return InMemorySessionStore()


def load_session_state(session_id: str, store: SessionStore) -> dict[str, Any]:
    state = store.get(session_id)
    if not state:
        return default_session_state(session_id)
    merged = default_session_state(session_id)
    merged.update(state)
    merged["constraints"] = _deep_merge(merged["constraints"], state.get("constraints", {}))
    return merged


def save_session_state(state: dict[str, Any], store: SessionStore, ttl_seconds: int) -> None:
    session_id = state["session_id"]
    store.save(session_id, state, ttl_seconds)


def apply_operations(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply validated patch operations to structured session state."""
    next_state = deepcopy(state)
    applied: list[dict[str, Any]] = []

    for operation in operations:
        op = operation.get("op")
        path = operation.get("path")
        if op not in OP_TYPES or path not in ALLOWED_OPERATION_PATHS:
            continue

        target = next_state.setdefault("constraints", {})
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        key = parts[-1]
        value = operation.get("value")
        changed = False

        if op == "set" or op == "replace":
            cleaned = _clean_value(value)
            if target.get(key) != cleaned:
                target[key] = cleaned
                changed = True
        elif op == "append":
            if path not in LIST_OPERATION_PATHS:
                continue
            values = target.setdefault(key, [])
            if not isinstance(values, list):
                values = []
                target[key] = values
            cleaned = _clean_value(value)
            if cleaned not in values:
                values.append(cleaned)
                changed = True
        elif op == "remove":
            cleaned = _clean_value(value)
            if path in LIST_OPERATION_PATHS:
                values = target.get(key, [])
                if isinstance(values, list):
                    filtered = [item for item in values if item != cleaned]
                    if filtered != values:
                        target[key] = filtered
                        changed = True
            else:
                if target.get(key) is not None:
                    target[key] = None
                    changed = True
        elif op == "clear":
            cleared = [] if path in LIST_OPERATION_PATHS else None
            if target.get(key) != cleared:
                target[key] = cleared
                changed = True

        if changed:
            applied.append(operation)

    if applied:
        next_state["state_version"] = int(next_state.get("state_version", 1)) + 1
        next_state["updated_at"] = utc_now_iso()

    return next_state, applied


def update_turn_state(
    state: dict[str, Any],
    intent: str,
    current_room_id: str | None,
    referenced_room_ids: list[str],
    result_ids: list[str],
) -> dict[str, Any]:
    next_state = deepcopy(state)
    if current_room_id:
        next_state["current_room_id"] = current_room_id
    if referenced_room_ids:
        next_state["selected_room_ids"] = referenced_room_ids[:3]
    elif intent == "COMPARE_ROOMS" and result_ids:
        next_state["selected_room_ids"] = result_ids[:3]
    if result_ids and intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
        # Giữ danh sách so sánh khi refine chỉ trả 1 phòng (vd. đổi quận/ngân sách).
        if intent in {"SEARCH_ROOM", "FIND_SIMILAR"} or len(result_ids) >= 2:
            next_state["last_result_ids"] = result_ids
    next_state["last_intent"] = intent
    next_state["updated_at"] = utc_now_iso()
    return next_state


def state_hash(state: dict[str, Any]) -> str:
    payload = json.dumps(state.get("constraints", {}), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
