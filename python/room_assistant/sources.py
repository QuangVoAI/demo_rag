"""Source attribution helpers for grounded room assistant responses."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def parse_listing_identity(room: dict[str, Any]) -> tuple[str, str]:
    """Return (listing_id, source_url) from a normalized or raw room document."""
    room_id = str(room.get("room_id") or room.get("id") or "").strip()
    source_url = str(room.get("source_url") or "").strip()

    source = room.get("source")
    if not source_url and isinstance(source, dict):
        source_url = str(source.get("url") or "").strip()

    if not source_url and "#" in room_id:
        source_url = room_id.split("#", 1)[0].strip()

    listing_id = room_id or source_url
    return listing_id, source_url


def format_updated_at(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_source_excerpt(room: dict[str, Any], query: str = "", max_chars: int = 240) -> str:
    """Pick a short grounded excerpt from embedding_text or description."""
    text = str(room.get("embedding_text") or room.get("description") or "").strip()
    if not text:
        title = str(room.get("title") or "").strip()
        district = str(room.get("district") or "").strip()
        rent = room.get("rent_price")
        parts = [part for part in (title, district, f"{rent} VND/tháng" if rent else "") if part]
        return " • ".join(parts)[:max_chars]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:max_chars]

    query_tokens = [token for token in re.findall(r"[\wÀ-ỹ]+", query.lower()) if len(token) > 2]
    if query_tokens:
        for line in lines:
            lower = line.lower()
            if any(token in lower for token in query_tokens):
                return line[:max_chars]

    for line in lines:
        if line.startswith("##"):
            continue
        if line.startswith("-") or ":" in line:
            return line.lstrip("- ").strip()[:max_chars]

    return lines[0][:max_chars]


def build_room_source(room: dict[str, Any], query: str = "", max_excerpt_chars: int = 240) -> dict[str, Any]:
    listing_id, source_url = parse_listing_identity(room)
    updated_at = format_updated_at(room.get("updated_at"))
    excerpt = build_source_excerpt(room, query=query, max_chars=max_excerpt_chars)
    detail_path = f"/tim-phong/{listing_id}/" if listing_id else None
    return {
        "type": "room",
        "listing_id": listing_id or None,
        "room_id": room.get("room_id"),
        "house_id": room.get("house_id"),
        "source_url": source_url or None,
        "updated_at": updated_at,
        "excerpt": excerpt or None,
        "detail_url": detail_path,
        "title": room.get("title"),
    }


def build_room_sources(rooms: list[dict[str, Any]], query: str = "") -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for room in rooms:
        room_id = str(room.get("room_id") or "").strip()
        if not room_id or room_id in seen:
            continue
        seen.add(room_id)
        sources.append(build_room_source(room, query=query))
    return sources
