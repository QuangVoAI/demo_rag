"""Hybrid room retrieval with repository-first hard constraints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

from .repository import RoomRepository, room_matches_constraints
from retrieval.metadata_search import extract_metadata_signals, score_metadata_hit

NEARBY_DISTRICTS_HCM = {
    "Quận 1": ["Quận 3", "Quận 4", "Quận 5", "Bình Thạnh", "Phú Nhuận"],
    "Quận 3": ["Quận 1", "Quận 10", "Phú Nhuận", "Tân Bình"],
    "Quận 4": ["Quận 1", "Quận 7", "Quận 8"],
    "Quận 5": ["Quận 1", "Quận 6", "Quận 8", "Quận 10", "Quận 11"],
    "Quận 6": ["Quận 5", "Quận 8", "Quận 11", "Tân Phú", "Bình Tân"],
    "Quận 7": ["Quận 4", "Quận 8", "Nhà Bè", "Bình Chánh"],
    "Quận 8": ["Quận 4", "Quận 5", "Quận 6", "Quận 7", "Bình Chánh"],
    "Quận 10": ["Quận 3", "Quận 5", "Quận 11", "Tân Bình"],
    "Quận 11": ["Quận 5", "Quận 6", "Quận 10", "Tân Bình", "Tân Phú"],
    "Tân Bình": ["Quận 3", "Quận 10", "Quận 11", "Tân Phú", "Phú Nhuận", "Gò Vấp"],
    "Tân Phú": ["Quận 6", "Quận 11", "Tân Bình", "Bình Tân", "Quận 12"],
    "Bình Tân": ["Quận 6", "Quận 8", "Tân Phú", "Bình Chánh"],
    "Gò Vấp": ["Tân Bình", "Phú Nhuận", "Bình Thạnh", "Quận 12"],
    "Phú Nhuận": ["Quận 1", "Quận 3", "Tân Bình", "Gò Vấp", "Bình Thạnh"],
    "Bình Thạnh": ["Quận 1", "Phú Nhuận", "Gò Vấp", "Quận 2", "Thủ Đức"],
    "Quận 2": ["Bình Thạnh", "Quận 1", "Quận 7", "Quận 9", "Thủ Đức"],
    "Quận 9": ["Quận 2", "Thủ Đức"],
    "Thủ Đức": ["Quận 2", "Quận 9", "Bình Thạnh", "Dĩ An"],
    "Quận 12": ["Gò Vấp", "Tân Bình", "Tân Phú", "Hóc Môn"]
}


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
    candidate_limit: int = 50,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search rooms by enforcing DB constraints before semantic ranking, with auto-relaxation."""
    cfg = _retrieval_config()
    feedback_retries = cfg["feedback_max_retries"] if cfg["enable_feedback_retry"] else 0
    signals = extract_metadata_signals(query_text)
    
    strategies = [{"name": "original", "constraints": constraints}]
    
    # Fallback 1: Relax Price
    if any(k in constraints for k in ["rent_price", "max_rent_price", "min_rent_price"]):
        c1 = dict(constraints)
        c1.pop("rent_price", None)
        c1.pop("max_rent_price", None)
        c1.pop("min_rent_price", None)
        strategies.append({"name": "relax_price", "constraints": c1})
        
    # Fallback 2: Nearby Location (Keep price)
    if "location" in constraints:
        districts = constraints["location"].get("districts") or []
        nearby = _nearby_districts(districts)
        if nearby:
            c2 = dict(constraints)
            loc = dict(c2["location"])
            loc["districts"] = list(nearby)
            c2["location"] = loc
            strategies.append({"name": "nearby_location", "constraints": c2})
            
            # Fallback 3: Relax Price + Nearby Location
            if "relax_price" in [s["name"] for s in strategies]:
                c3 = dict(c1)
                loc3 = dict(c3.get("location", {}))
                loc3["districts"] = list(nearby)
                c3["location"] = loc3
                strategies.append({"name": "relax_price_nearby_location", "constraints": c3})

        # Fallback 4: Drop Location Entirely (Global search)
        if not _has_strict_location_constraint(constraints):
            c4 = dict(constraints)
            c4.pop("location", None)
            strategies.append({"name": "relax_location", "constraints": c4})

    attempts: list[dict[str, Any]] = []
    best_rooms: list[dict[str, Any]] = []
    best_confidence = 0.0
    best_low_confidence = True
    retry_count = 0
    applied_strategy = "original"
    metadata_candidates_all = []
    candidate_ids_all = []

    for strategy in strategies:
        curr_constraints = strategy["constraints"]
        base_candidates = repository.search_by_constraints(
            constraints=curr_constraints,
            limit=max(candidate_limit, top_k),
            offset=0,
        )
        metadata_candidates = []
        if query_text.strip():
            metadata_candidates = repository.search_by_metadata(query_text, limit=max(top_k * 2, 10))
            metadata_candidates = [
                item for item in metadata_candidates
                if room_matches_constraints(item, curr_constraints)
            ]

        candidates = _merge_rooms(metadata_candidates, base_candidates)
        candidate_ids = [item["room_id"] for item in candidates if item.get("room_id")]
        
        if not metadata_candidates_all:
            metadata_candidates_all = metadata_candidates
        if not candidate_ids_all:
            candidate_ids_all = candidate_ids

        if candidate_ids:
            query_for_attempt = query_text
            for attempt_index in range(feedback_retries + 1):
                attempt = _rank_attempt(
                    query_text=query_for_attempt,
                    original_query=query_text,
                    constraints=curr_constraints,
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
                query_for_attempt = _build_retry_query(query_text, curr_constraints, metadata_candidates)
        
        if best_rooms:
            applied_strategy = strategy["name"]
            break

    if trace is not None:
        trace.clear()
        trace.update({
            "retrieval_confidence": best_confidence,
            "retrieval_low_confidence": best_low_confidence,
            "retrieval_feedback_retry_count": retry_count,
            "retrieval_attempts": attempts,
            "metadata_signals": signals,
            "metadata_hit_count": len(metadata_candidates_all),
            "candidate_count": len(candidate_ids_all),
            "fallback_strategy": applied_strategy,
            "location_constraint_mode": "strict" if _has_strict_location_constraint(constraints) else "relaxable",
        })

    _write_feedback_log(
        query_text=query_text,
        trace=trace or {
            "retrieval_confidence": best_confidence,
            "retrieval_low_confidence": best_low_confidence,
            "retrieval_feedback_retry_count": retry_count,
            "retrieval_attempts": attempts,
            "metadata_hit_count": len(metadata_candidates_all),
            "candidate_count": len(candidate_ids_all),
            "fallback_strategy": applied_strategy,
            "location_constraint_mode": "strict" if _has_strict_location_constraint(constraints) else "relaxable",
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
        combined_score = rrf_score + metadata_score * cfg["metadata_boost"]
        area_score = 0.0
        if (constraints.get("area") or {}).get("preference") == "larger":
            area = candidate.get("area_m2") or 0
            try:
                area_score = float(area)
                combined_score += area_score / 1000.0
            except (TypeError, ValueError):
                pass
        ranked.append({
            "room_id": room_id,
            "rrf_score": rrf_score,
            "metadata_score": metadata_score,
            "combined_score": combined_score,
            "area_score": area_score,
            "rerank_score": semantic.get("rerank_score"),
            "position": candidate_position.get(room_id, 999999),
        })

    ranked.sort(
        key=lambda item: (
            item["combined_score"],
            item.get("area_score", 0.0),
            -item["position"],
        ),
        reverse=True,
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
            "metadata_filter": metadata_filter if semantic_index and query_text.strip() else {},
        },
    }


def _metadata_filter_from_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    location = constraints.get("location") or {}
    metadata_filter: dict[str, Any] = {}
    districts = [_display_location_value(item) for item in location.get("districts", []) if item]
    wards = [_display_location_value(item) for item in location.get("wards", []) if item]
    if districts:
        metadata_filter["district"] = list(dict.fromkeys(districts))
    if wards:
        metadata_filter["ward"] = list(dict.fromkeys(wards))
    return metadata_filter


def _has_strict_location_constraint(constraints: dict[str, Any]) -> bool:
    location = constraints.get("location") or {}
    return bool(location.get("province") or location.get("districts") or location.get("wards"))


def _nearby_districts(districts: list[Any]) -> list[str]:
    nearby: list[str] = []
    nearby_map = {
        _normalize_location_token(key): [_normalize_location_token(item) for item in values]
        for key, values in NEARBY_DISTRICTS_HCM.items()
    }
    for district in districts:
        normalized = _normalize_location_token(district)
        nearby.extend(nearby_map.get(normalized, []))
    return list(dict.fromkeys(item for item in nearby if item))


def _normalize_location_token(value: Any) -> str:
    import re
    import unicodedata

    text = str(value or "").lower().strip()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    ).replace("đ", "d")
    text = re.sub(
        r"\b(?:thanh pho|thi xa|thi tran|quan|huyen|phuong|xa)\s+|\b(?:q|p)\.?\s*(?=\d)",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def _display_location_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = _normalize_location_token(text)
    if normalized.isdigit():
        return f"Quận {normalized}"
    return normalized.title()


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
