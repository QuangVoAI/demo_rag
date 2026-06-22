"""Hybrid listing retrieval with repository-first hard constraints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

from .repository import ListingRepository, listing_matches_constraints
from retrieval.metadata_search import extract_metadata_signals, score_metadata_hit


class ListingSemanticIndex(Protocol):
    def search_listings(
        self,
        query_text: str,
        candidate_ids: list[str],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


def search_listings_with_hard_filters(
    query_text: str,
    constraints: dict[str, Any],
    repository: ListingRepository,
    semantic_index: ListingSemanticIndex | None = None,
    top_k: int = 5,
    candidate_limit: int = 50,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search listings by enforcing DB constraints before semantic ranking."""
    cfg = _retrieval_config()
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
            if listing_matches_constraints(item, constraints)
        ]

    candidates = _merge_listings(metadata_candidates, base_candidates)
    candidate_ids = [item["listing_id"] for item in candidates if item.get("listing_id")]

    attempts: list[dict[str, Any]] = []
    best_listings: list[dict[str, Any]] = []
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
                attempt["listings"],
                confidence,
                cfg["low_confidence_min_docs"],
                cfg["low_confidence_min_score"],
            )
            if (
                not best_listings
                or confidence > best_confidence
                or (confidence == best_confidence and len(attempt["listings"]) > len(best_listings))
            ):
                best_listings = attempt["listings"]
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
    return best_listings[:top_k]


def _rank_attempt(
    query_text: str,
    original_query: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    candidate_ids: list[str],
    repository: ListingRepository,
    semantic_index: ListingSemanticIndex | None,
    top_k: int,
) -> dict[str, Any]:
    cfg = _retrieval_config()
    signals = extract_metadata_signals(original_query)
    semantic_results: list[dict[str, Any]] = []
    if semantic_index and query_text.strip():
        metadata_filter = _metadata_filter_from_constraints(constraints)
        semantic_results = semantic_index.search_listings(
            query_text=query_text,
            candidate_ids=candidate_ids,
            top_k=max(top_k, cfg["top_k_retrieval"]),
            metadata_filter=metadata_filter,
        )
    semantic_by_id = {
        str(item.get("listing_id")): item
        for item in semantic_results
        if item.get("listing_id") in candidate_ids
    }

    ranked = []
    candidate_position = {listing_id: idx for idx, listing_id in enumerate(candidate_ids)}
    for candidate in candidates:
        listing_id = candidate.get("listing_id")
        if not listing_id:
            continue
        semantic = semantic_by_id.get(listing_id, {})
        rrf_score = _score_value(semantic, "rerank_score", "combined_score", "rrf_score", "score")
        metadata_score = max(
            float(candidate.get("_metadata_score") or 0.0),
            score_metadata_hit(candidate, signals, cfg["metadata_fields"]),
        )
        combined_score = rrf_score + metadata_score * cfg["metadata_boost"]
        ranked.append({
            "listing_id": listing_id,
            "rrf_score": rrf_score,
            "metadata_score": metadata_score,
            "combined_score": combined_score,
            "rerank_score": semantic.get("rerank_score"),
            "position": candidate_position.get(listing_id, 999999),
        })

    ranked.sort(key=lambda item: (item["combined_score"], -item["position"]), reverse=True)
    ranked_ids = [item["listing_id"] for item in ranked[:top_k]]
    authoritative = repository.get_many_by_ids(ranked_ids)
    score_by_id = {item["listing_id"]: item for item in ranked}
    listings = []
    for listing in authoritative:
        if not listing_matches_constraints(listing, constraints):
            continue
        scores = score_by_id.get(listing.get("listing_id"), {})
        listing = dict(listing)
        listing["rrf_score"] = scores.get("rrf_score", 0.0)
        listing["metadata_score"] = scores.get("metadata_score", 0.0)
        listing["combined_score"] = scores.get("combined_score", 0.0)
        if scores.get("rerank_score") is not None:
            listing["rerank_score"] = scores.get("rerank_score")
        listing["retrieval_score"] = _preferred_source_score(listing)
        listings.append(listing)

    confidence = max((_preferred_source_score(item) for item in listings), default=0.0)
    return {
        "listings": listings,
        "confidence": confidence,
        "trace": {
            "query": query_text,
            "result_count": len(listings),
            "confidence": confidence,
            "top_listing_ids": [item.get("listing_id") for item in listings],
            "semantic_result_count": len(semantic_results),
        },
    }


def _metadata_filter_from_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    location = constraints.get("location") or {}
    metadata: dict[str, Any] = {"status": "active"}
    if location.get("districts"):
        metadata["district"] = location["districts"]
    return metadata


def _merge_listings(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            listing_id = item.get("listing_id")
            if not listing_id:
                continue
            if listing_id in merged:
                merged[listing_id].update({k: v for k, v in item.items() if v is not None})
            else:
                merged[listing_id] = dict(item)
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
    aliases.extend(str(item.get("listing_id")) for item in metadata_candidates[:3] if item.get("listing_id"))
    suffix = " ".join(dict.fromkeys(alias for alias in aliases if alias))
    return f"{query_text} {suffix}".strip() if suffix else query_text


def _is_low_confidence(
    listings: list[dict[str, Any]],
    confidence: float,
    min_docs: int,
    min_score: float,
) -> bool:
    return len(listings) < min_docs or confidence < min_score


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
            "metadata_fields": ("listing_id", "district", "title", "amenities"),
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
