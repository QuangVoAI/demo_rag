"""Hybrid room retrieval with repository-first hard constraints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

from .repository import (
    RoomRepository,
    room_matches_constraints,
    _room_has_canonical_amenity,
    _room_has_positive_amenity,
)
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
    top_k: int = 5,
    candidate_limit: int | None = None,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search rooms by enforcing DB constraints before semantic ranking."""
    cfg = _retrieval_config()
    if candidate_limit is None:
        candidate_limit = cfg["candidate_limit"]
    feedback_retries = cfg["feedback_max_retries"] if cfg["enable_feedback_retry"] else 0
    signals = extract_metadata_signals(query_text)
    base_candidates = repository.search_by_constraints(
        constraints=constraints,
        limit=max(candidate_limit, top_k),
        offset=0,
    )
    metadata_candidates = []
    if query_text.strip():
        metadata_candidates = repository.search_by_metadata(query_text, limit=max(top_k * 2, 10))
        metadata_candidates = [
            item for item in metadata_candidates
            if room_matches_constraints(item, constraints)
        ]

    candidates = _merge_rooms(metadata_candidates, base_candidates)
    candidate_ids = [item["room_id"] for item in candidates if item.get("room_id")]

    attempts: list[dict[str, Any]] = []
    best_rooms: list[dict[str, Any]] = []
    best_confidence = 0.0
    best_low_confidence = True
    retry_count = 0
    query_for_attempt = query_text

    if candidate_ids:
        for attempt_index in range(feedback_retries + 1):
            attempt = _rank_attempt(
                query_text=query_for_attempt,
                original_query=query_text,
                constraints=constraints,
                candidates=candidates,
                candidate_ids=candidate_ids,
                repository=repository,
                semantic_index=semantic_index,
                top_k=top_k,
            )
            attempts.append(attempt["trace"])
            confidence = attempt["confidence"]
            low_confidence = _is_low_confidence(
                attempt["rooms"],
                confidence,
                cfg["low_confidence_min_docs"],
                cfg["low_confidence_min_score"],
            )
            if (
                not best_rooms
                or confidence > best_confidence
                or (confidence == best_confidence and len(attempt["rooms"]) > len(best_rooms))
            ):
                best_rooms = attempt["rooms"]
                best_confidence = confidence
                best_low_confidence = low_confidence
            if not low_confidence or attempt_index >= feedback_retries:
                break
            retry_count += 1
            query_for_attempt = _build_retry_query(query_text, constraints, metadata_candidates)

    if trace is not None:
        trace.clear()
        trace.update({
            "retrieval_confidence": best_confidence,
            "retrieval_low_confidence": best_low_confidence,
            "retrieval_feedback_retry_count": retry_count,
            "retrieval_attempts": attempts,
            "metadata_signals": signals,
            "metadata_hit_count": len(metadata_candidates),
            "candidate_count": len(candidate_ids),
        })

    _write_feedback_log(
        query_text=query_text,
        trace=trace or {
            "retrieval_confidence": best_confidence,
            "retrieval_low_confidence": best_low_confidence,
            "retrieval_feedback_retry_count": retry_count,
            "retrieval_attempts": attempts,
            "metadata_hit_count": len(metadata_candidates),
            "candidate_count": len(candidate_ids),
        },
    )
    return best_rooms[:top_k]


def _rank_attempt(
    query_text: str,
    original_query: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    candidate_ids: list[str],
    repository: RoomRepository,
    semantic_index: RoomSemanticIndex | None,
    top_k: int,
) -> dict[str, Any]:
    cfg = _retrieval_config()
    signals = extract_metadata_signals(original_query)
    semantic_results: list[dict[str, Any]] = []
    if semantic_index and query_text.strip():
        metadata_filter = _metadata_filter_from_constraints(constraints)
        semantic_results = semantic_index.search_rooms(
            query_text=query_text,
            candidate_ids=candidate_ids,
            top_k=max(top_k, cfg["top_k_retrieval"]),
            metadata_filter=metadata_filter,
        )
    semantic_by_id = {
        str(item.get("room_id")): item
        for item in semantic_results
        if item.get("room_id") in candidate_ids
    }

    ranked = []
    candidate_position = {room_id: idx for idx, room_id in enumerate(candidate_ids)}
    for candidate in candidates:
        room_id = candidate.get("room_id")
        if not room_id:
            continue
        semantic = semantic_by_id.get(room_id, {})
        rrf_score = _score_value(semantic, "rerank_score", "combined_score", "rrf_score", "score")
        metadata_score = max(
            float(candidate.get("_metadata_score") or 0.0),
            score_metadata_hit(candidate, signals, cfg["metadata_fields"]),
        )
        preferred_boost = _preferred_amenities_boost(
            candidate,
            constraints.get("amenities_preferred") or [],
        )
        combined_score = rrf_score + metadata_score * cfg["metadata_boost"] + preferred_boost
        ranked.append({
            "room_id": room_id,
            "rrf_score": rrf_score,
            "metadata_score": metadata_score,
            "combined_score": combined_score,
            "rerank_score": semantic.get("rerank_score"),
            "position": candidate_position.get(room_id, 999999),
        })

    ranked.sort(key=lambda item: (item["combined_score"], -item["position"]), reverse=True)
    ranked = _maybe_rerank_candidates(
        query_text=query_text,
        ranked=ranked,
        candidates=candidates,
        repository=repository,
        metadata_boost=cfg["metadata_boost"],
    )
    ranked_ids = [item["room_id"] for item in ranked[:top_k]]
    authoritative = repository.get_many_by_ids(ranked_ids)
    score_by_id = {item["room_id"]: item for item in ranked}
    rooms = []
    for room in authoritative:
        if not room_matches_constraints(room, constraints):
            continue
        scores = score_by_id.get(room.get("room_id"), {})
        room = dict(room)
        room["rrf_score"] = scores.get("rrf_score", 0.0)
        room["metadata_score"] = scores.get("metadata_score", 0.0)
        room["combined_score"] = scores.get("combined_score", 0.0)
        if scores.get("rerank_score") is not None:
            room["rerank_score"] = scores.get("rerank_score")
        room["retrieval_score"] = _preferred_source_score(room)
        rooms.append(room)

    if not rooms:
        rooms = _fallback_constrained_ranked_rooms(
            candidates=candidates,
            constraints=constraints,
            repository=repository,
            signals=signals,
            top_k=top_k,
            metadata_boost=cfg["metadata_boost"],
            metadata_fields=cfg["metadata_fields"],
        )

    confidence = max((_preferred_source_score(item) for item in rooms), default=0.0)
    return {
        "rooms": rooms,
        "confidence": confidence,
        "trace": {
            "query": query_text,
            "result_count": len(rooms),
            "confidence": confidence,
            "top_room_ids": [item.get("room_id") for item in rooms],
            "semantic_result_count": len(semantic_results),
        },
    }


def _fallback_constrained_ranked_rooms(
    *,
    candidates: list[dict[str, Any]],
    constraints: dict[str, Any],
    repository: RoomRepository,
    signals: dict[str, Any],
    top_k: int,
    metadata_boost: float,
    metadata_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Khi semantic rank trống, trả phòng đã lọc cứng theo metadata score."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        room_id = candidate.get("room_id")
        if not room_id:
            continue
        room = candidate
        if room.get("rent_price") is None:
            fetched = repository.get_by_id(str(room_id))
            if fetched:
                room = fetched
        if not room_matches_constraints(room, constraints):
            continue
        metadata_score = max(
            float(candidate.get("_metadata_score") or 0.0),
            score_metadata_hit(room, signals, metadata_fields),
        )
        combined = metadata_score * metadata_boost
        item = dict(room)
        item["metadata_score"] = metadata_score
        item["combined_score"] = combined
        item["retrieval_score"] = combined
        scored.append((combined, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def _maybe_rerank_candidates(
    query_text: str,
    ranked: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    repository: RoomRepository,
    metadata_boost: float,
) -> list[dict[str, Any]]:
    try:
        from config import RERANK_CANDIDATE_POOL, TOP_K_RERANK, USE_RERANKER
    except Exception:
        return ranked
    if not USE_RERANKER or not ranked or not query_text.strip():
        return ranked

    pool_size = min(len(ranked), RERANK_CANDIDATE_POOL)
    if pool_size < 2:
        return ranked

    candidate_by_id = {
        str(item.get("room_id")): item
        for item in candidates
        if item.get("room_id")
    }
    docs: list[dict[str, Any]] = []
    for item in ranked[:pool_size]:
        room_id = str(item.get("room_id") or "")
        if not room_id:
            continue
        room = candidate_by_id.get(room_id) or repository.get_by_id(room_id) or {}
        text = str(room.get("embedding_text") or room.get("title") or room_id).strip()
        docs.append({"id": room_id, "room_id": room_id, "text": text})

    if not docs:
        return ranked

    try:
        from retrieval.reranker import rerank

        reranked_docs = rerank(query_text, docs, top_k=min(TOP_K_RERANK, len(docs)))
    except Exception:
        return ranked

    rerank_scores = {
        str(doc.get("room_id") or doc.get("id")): float(doc.get("rerank_score", 0.0))
        for doc in reranked_docs
        if doc.get("room_id") or doc.get("id")
    }
    for item in ranked:
        room_id = str(item.get("room_id") or "")
        if room_id not in rerank_scores:
            continue
        item["rerank_score"] = rerank_scores[room_id]
        item["combined_score"] = rerank_scores[room_id] + float(item.get("metadata_score") or 0.0) * metadata_boost

    ranked.sort(
        key=lambda item: (
            float(item.get("rerank_score", item.get("combined_score", 0.0))),
            -int(item.get("position", 999999)),
        ),
        reverse=True,
    )
    return ranked


def _preferred_amenities_boost(room: dict[str, Any], preferred: list[str]) -> float:
    """Soft ranking boost for amenities_preferred (not a hard filter)."""
    if not preferred:
        return 0.0
    boost = 0.0
    for amenity in preferred:
        canonical = str(amenity).strip().lower()
        if _room_has_canonical_amenity(room.get("amenities_canonical") or [], canonical):
            boost += 0.15
        elif _room_has_canonical_amenity(room.get("amenities") or [], canonical):
            boost += 0.12
        elif _room_has_positive_amenity(room, canonical):
            boost += 0.08
    return boost


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
            RETRIEVAL_CANDIDATE_LIMIT,
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
            "candidate_limit": RETRIEVAL_CANDIDATE_LIMIT,
        }
    except Exception:
        return {
            "enable_feedback_retry": True,
            "feedback_log_path": None,
            "feedback_max_retries": 1,
            "low_confidence_min_docs": 1,
            "low_confidence_min_score": 0.015,
            "metadata_boost": 0.5,
            "metadata_fields": ("room_id", "room_code", "district", "title", "amenities"),
            "top_k_retrieval": 6,
            "candidate_limit": 100,
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


def build_retrieval_explanation(
    trace: dict[str, Any] | None,
    constraints: dict[str, Any] | None,
    *,
    relaxed_fields: list[str] | None = None,
    result_count: int = 0,
) -> list[str]:
    """Tóm tắt vì sao retrieval trả kết quả như vậy (debug / audit)."""
    lines: list[str] = []
    location = (constraints or {}).get("location") or {}
    budget = (constraints or {}).get("budget") or {}

    districts = [str(item) for item in (location.get("districts") or []) if item]
    landmarks = [str(item) for item in (location.get("near_landmarks") or []) if item]
    if districts:
        lines.append(f"Lọc quận: {', '.join(districts)}")
    if landmarks:
        lines.append(f"Lọc mốc gần: {', '.join(landmarks)}")
    if budget.get("max") is not None:
        lines.append(f"Ngân sách tối đa: {budget['max']:,} VND")
    if budget.get("min") is not None:
        lines.append(f"Ngân sách tối thiểu: {budget['min']:,} VND")

    if trace:
        candidate_count = int(trace.get("candidate_count") or 0)
        metadata_hits = int(trace.get("metadata_hit_count") or 0)
        confidence = trace.get("retrieval_confidence")
        lines.append(f"Sau lọc cứng MongoDB: {candidate_count} phòng ứng viên")
        if metadata_hits:
            lines.append(f"Khớp metadata câu hỏi: {metadata_hits} phòng")
        if confidence is not None:
            lines.append(f"Độ tin cậy xếp hạng: {float(confidence):.3f}")
        if trace.get("retrieval_low_confidence"):
            lines.append("Xếp hạng ngữ nghĩa: độ tin cậy thấp")
        retry_count = int(trace.get("retrieval_feedback_retry_count") or 0)
        if retry_count:
            lines.append(f"Đã thử lại truy vấn semantic: {retry_count} lần")

    if relaxed_fields:
        lines.append(f"Đã nới điều kiện: {', '.join(relaxed_fields)}")

    lines.append(f"Kết quả trả về: {result_count} phòng")
    return lines
