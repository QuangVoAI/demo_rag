"""Hybrid listing retrieval with repository-first hard constraints."""

from __future__ import annotations

from typing import Any, Protocol

from .repository import ListingRepository, listing_matches_constraints


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
) -> list[dict[str, Any]]:
    """Search listings by enforcing DB constraints before semantic ranking."""
    candidates = repository.search_by_constraints(
        constraints=constraints,
        limit=max(candidate_limit, top_k),
        offset=0,
    )
    candidate_ids = [item["listing_id"] for item in candidates if item.get("listing_id")]
    if not candidate_ids:
        return []

    ranked_ids = candidate_ids[:top_k]
    if semantic_index and query_text.strip():
        metadata_filter = _metadata_filter_from_constraints(constraints)
        semantic_results = semantic_index.search_listings(
            query_text=query_text,
            candidate_ids=candidate_ids,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        semantic_ids = [
            item["listing_id"]
            for item in semantic_results
            if item.get("listing_id") in candidate_ids
        ]
        if semantic_ids:
            ranked_ids = semantic_ids[:top_k]

    authoritative = repository.get_many_by_ids(ranked_ids)
    return [
        listing for listing in authoritative
        if listing_matches_constraints(listing, constraints)
    ][:top_k]


def _metadata_filter_from_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    location = constraints.get("location") or {}
    metadata: dict[str, Any] = {"status": "active"}
    if location.get("districts"):
        metadata["district"] = location["districts"]
    return metadata

