"""Hybrid room retrieval with repository-first hard constraints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

from .repository import RoomRepository, room_matches_constraints
from retrieval.metadata_search import extract_metadata_signals, score_metadata_hit


class RoomSemanticIndex(Protocol):
    def search_rooms(
        self,
        query_text: str,
        candidate_ids: list[str],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


def search_rooms_with_hard_filters(
    query_text: str,
    constraints: dict[str, Any],
    repository: RoomRepository,
    semantic_index: RoomSemanticIndex | None = None,
    top_k: int = 20,
    candidate_limit: int = 50,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search rooms by enforcing DB constraints after semantic retrieval and batch rehydration."""
    cfg = _retrieval_config()
    signals = extract_metadata_signals(query_text)
    
    # 1. Fetch metadata candidates
    metadata_candidates = []
    if query_text.strip():
        metadata_candidates = repository.search_by_metadata(query_text, limit=candidate_limit)
        
    # 2. Fetch base constraints candidates (if any)
    base_candidates = repository.search_by_constraints(
        constraints=constraints,
        limit=candidate_limit,
        offset=0,
    )
    
    metadata_ids = [item["room_id"] for item in metadata_candidates if item.get("room_id")]
    base_ids = [item["room_id"] for item in base_candidates if item.get("room_id")]
    
    # 3. Semantic Retrieval (Hybrid)
    semantic_results = []
    if semantic_index and query_text.strip():
        # Pass base_ids to restrict semantic index search to constraint-matching candidates
        semantic_results = semantic_index.search_rooms(
            query_text=query_text,
            candidate_ids=base_ids, 
            top_k=candidate_limit,
            metadata_filter=_metadata_filter_from_constraints(constraints),
        )
    semantic_ids = [item["room_id"] for item in semantic_results if item.get("room_id")]
    
    # 4. Combine all candidate IDs
    all_candidate_ids = list(dict.fromkeys(metadata_ids + base_ids + semantic_ids))
    
    # 5. Batch Canonical Mongo Rehydration using $in
    authoritative_rooms = repository.get_many_by_ids(all_candidate_ids)
    
    # 6. Hard Filtering
    valid_rooms = []
    for room in authoritative_rooms:
        if room_matches_constraints(room, constraints):
            valid_rooms.append(room)
            
    # 7. Standard RRF Ranking (missing candidates get strictly 0.0)
    K = 60
    semantic_rank_map = {room_id: idx + 1 for idx, room_id in enumerate(semantic_ids)}
    metadata_rank_map = {room_id: idx + 1 for idx, room_id in enumerate(metadata_ids)}
    
    ranked_rooms = []
    for room in valid_rooms:
        room_id = room["room_id"]
        sem_rank = semantic_rank_map.get(room_id)
        meta_rank = metadata_rank_map.get(room_id)
        
        sem_score = 1.0 / (K + sem_rank) if sem_rank else 0.0
        meta_score = 1.0 / (K + meta_rank) if meta_rank else 0.0
        
        # Also incorporate any existing metadata hit scores if present
        meta_hit_score = score_metadata_hit(room, signals, cfg["metadata_fields"])
        
        rrf_score = sem_score + meta_score
        combined_score = rrf_score + (meta_hit_score * cfg["metadata_boost"])
        room = dict(room)
        room["rrf_score"] = rrf_score
        room["metadata_score"] = meta_score
        room["semantic_score"] = sem_score
        room["combined_score"] = combined_score
        room["retrieval_score"] = combined_score
        ranked_rooms.append(room)
        
    ranked_rooms.sort(key=lambda r: r["combined_score"], reverse=True)
    best_rooms = ranked_rooms[:top_k]
    confidence = max((r["combined_score"] for r in best_rooms), default=0.0)

    if trace is not None:
        trace.clear()
        low_confidence = len(best_rooms) < cfg.get("low_confidence_min_docs", 1) or confidence < cfg.get("low_confidence_min_score", 0.015)
        trace.update({
            "retrieval_confidence": confidence,
            "metadata_signals": signals,
            "candidate_count": len(all_candidate_ids),
            "valid_candidate_count": len(valid_rooms),
            "result_count": len(best_rooms),
            "retrieval_low_confidence": low_confidence,
            "retrieval_feedback_retry_count": 0,
            "retrieval_attempts": [{"top_room_ids": [room["room_id"] for room in best_rooms]}],
        })

    return best_rooms


def _metadata_filter_from_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    location = constraints.get("location") or {}
    metadata: dict[str, Any] = {}
    if location.get("districts"):
        metadata["district"] = location["districts"]
    return metadata


def _merge_rooms(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            room_id = item.get("room_id")
            if not room_id:
                continue
            if room_id in merged:
                merged[room_id].update({k: v for k, v in item.items() if v is not None})
            else:
                merged[room_id] = dict(item)
    return list(merged.values())


def _build_retry_query(
    query_text: str,
    constraints: dict[str, Any],
    metadata_candidates: list[dict[str, Any]],
) -> str:
    aliases: list[str] = []
    location = constraints.get("location") or {}
    aliases.extend(str(item) for item in location.get("districts", []) if item)
    aliases.extend(str(item.get("title")) for item in metadata_candidates[:3] if item.get("title"))
    aliases.extend(str(item.get("room_id")) for item in metadata_candidates[:3] if item.get("room_id"))
    suffix = " ".join(dict.fromkeys(alias for alias in aliases if alias))
    return f"{query_text} {suffix}".strip() if suffix else query_text


def _is_low_confidence(
    rooms: list[dict[str, Any]],
    confidence: float,
    min_docs: int,
    min_score: float,
) -> bool:
    return len(rooms) < min_docs or confidence < min_score


def _preferred_source_score(item: dict[str, Any]) -> float:
    return _score_value(item, "rerank_score", "combined_score", "rrf_score", "score")


def _score_value(item: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _retrieval_config() -> dict[str, Any]:
    try:
        from config import (
            ENABLE_FEEDBACK_RETRY,
            FEEDBACK_LOG_PATH,
            FEEDBACK_MAX_RETRIEVAL_RETRIES,
            LOW_CONFIDENCE_MIN_DOCS,
            LOW_CONFIDENCE_MIN_SCORE,
            METADATA_BOOST,
            METADATA_FIELDS,
            TOP_K_RETRIEVAL,
        )
        return {
            "enable_feedback_retry": ENABLE_FEEDBACK_RETRY,
            "feedback_log_path": FEEDBACK_LOG_PATH,
            "feedback_max_retries": FEEDBACK_MAX_RETRIEVAL_RETRIES,
            "low_confidence_min_docs": LOW_CONFIDENCE_MIN_DOCS,
            "low_confidence_min_score": LOW_CONFIDENCE_MIN_SCORE,
            "metadata_boost": METADATA_BOOST,
            "metadata_fields": METADATA_FIELDS,
            "top_k_retrieval": TOP_K_RETRIEVAL,
        }
    except Exception:
        return {
            "enable_feedback_retry": True,
            "feedback_log_path": None,
            "feedback_max_retries": 1,
            "low_confidence_min_docs": 1,
            "low_confidence_min_score": 0.015,
            "metadata_boost": 0.5,
            "metadata_fields": ("room_id", "district", "title", "amenities"),
            "top_k_retrieval": 6,
        }


def _write_feedback_log(query_text: str, trace: dict[str, Any]) -> None:
    cfg = _retrieval_config()
    path = cfg.get("feedback_log_path")
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": query_text,
            "retrieval_confidence": trace.get("retrieval_confidence", 0.0),
            "retrieval_low_confidence": trace.get("retrieval_low_confidence", True),
            "retrieval_feedback_retry_count": trace.get("retrieval_feedback_retry_count", 0),
            "metadata_hit_count": trace.get("metadata_hit_count", 0),
            "candidate_count": trace.get("candidate_count", 0),
            "attempts": trace.get("retrieval_attempts", []),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return
