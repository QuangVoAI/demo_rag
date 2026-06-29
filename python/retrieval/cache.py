"""
Redis Cache Layer — Upstash Serverless Redis (HTTP REST).

Caching câu trả lời RAG để bypass room_assistant workflow cho các query đã từng trả lời.
Sử dụng Upstash (serverless) → không cần cài Redis local, chỉ gọi HTTP.

Cache Strategy:
  - Dynamic room answers are disabled by default.
  - If explicitly enabled, key must include intent, normalized question,
    state hash, current room id, room source version, and index version.
  - Value: JSON {answer, sources, agent_trace, cached_at}
  - TTL: Mặc định 7 ngày (cấu hình qua REDIS_CACHE_TTL)
"""
import hashlib
import json
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    UPSTASH_REDIS_REST_TOKEN,
    UPSTASH_REDIS_REST_URL,
    REDIS_CACHE_TTL,
)
from utils.console import console

ENABLE_DYNAMIC_ROOM_ANSWER_CACHE = False


# ─── Singleton Redis Client ──────────────────────────────

_redis_client = None
_redis_available = False


def _get_redis():
    """Lazy init Upstash Redis client. Returns None if not configured."""
    global _redis_client, _redis_available

    if _redis_client is not None:
        return _redis_client if _redis_available else None

    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        console.print("[dim]  Cache: Upstash Redis not configured, skipping[/]")
        _redis_available = False
        return None

    try:
        from upstash_redis import Redis
        _redis_client = Redis(
            url=UPSTASH_REDIS_REST_URL,
            token=UPSTASH_REDIS_REST_TOKEN,
        )
        _redis_available = True
        console.print("[green]  Cache: Upstash Redis connected ✓[/]")
        return _redis_client
    except Exception as e:
        console.print(f"[yellow]  Cache: Redis init failed: {e}[/]")
        _redis_available = False
        return None


# ─── Key Generation ──────────────────────────────────────

def _make_cache_key(question: str, context: dict | None = None) -> str:
    """
    Tạo Redis key an toàn cho room động.
    """
    normalized = question.strip().lower()
    if context:
        from config import EMBEDDING_VERSION
        payload = {
            "intent": context.get("intent"),
            "query": normalized,
            "state_hash": context.get("state_hash"),
            "current_room_id": context.get("current_room_id"),
            "room_source_version": context.get("room_source_version"),
            "index_version": context.get("index_version", EMBEDDING_VERSION),
        }
        hash_digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return f"room:answer:{hash_digest}"
    hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"faq:cache:{hash_digest}"


# ─── Public API ──────────────────────────────────────────

def get_cached_answer(question: str, context: dict | None = None, dynamic_room: bool = True) -> dict | None:
    """
    Kiểm tra cache cho câu hỏi.

    Args:
        question: Câu hỏi tiếng Việt gốc

    Returns:
        dict {answer, sources, agent_trace, cached_at} nếu HIT, None nếu MISS.
    """
    if dynamic_room and (not ENABLE_DYNAMIC_ROOM_ANSWER_CACHE or not context):
        return None

    redis = _get_redis()
    if redis is None:
        return None

    try:
        key = _make_cache_key(question, context)
        data = redis.get(key)

        if data is None:
            return None

        # Upstash trả về string hoặc bytes
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        result = json.loads(data) if isinstance(data, str) else data
        console.print(f"[green]  ⚡ Cache HIT: {key}[/]")
        return result

    except Exception as e:
        console.print(f"[yellow]  Cache GET error: {e}[/]")
        return None


def set_cached_answer(
    question: str,
    answer: str,
    sources: list,
    agent_trace: dict,
    context: dict | None = None,
    dynamic_room: bool = True,
    ttl: int = REDIS_CACHE_TTL,
) -> bool:
    """
    Ghi cache cho câu trả lời.

    Args:
        question: Câu hỏi gốc
        answer: Câu trả lời đầy đủ
        sources: Danh sách sources
        agent_trace: Trace metadata
        ttl: Thời gian sống (giây), mặc định 7 ngày

    Returns:
        True nếu ghi thành công
    """
    if dynamic_room and (not ENABLE_DYNAMIC_ROOM_ANSWER_CACHE or not context):
        return False

    redis = _get_redis()
    if redis is None:
        return False

    try:
        key = _make_cache_key(question, context)
        value = json.dumps({
            "answer": answer,
            "sources": sources,
            "agent_trace": {
                **(agent_trace or {}),
                "from_cache": True,
            },
            "cached_at": time.time(),
        }, ensure_ascii=False)

        redis.setex(key, ttl, value)
        console.print(f"[dim]  Cache SET: {key} (TTL={ttl}s)[/]")
        return True

    except Exception as e:
        console.print(f"[yellow]  Cache SET error: {e}[/]")
        return False


def invalidate_room_cache(room_id: str) -> bool:
    """Placeholder invalidation hook.

    Versioned dynamic answer keys include room source/index versions, so old
    answers naturally stop matching.
    """
    return True
