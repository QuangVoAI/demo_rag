"""Qdrant listing semantic index adapters."""

from __future__ import annotations

import numpy as np

from retrieval.qdrant_client import QdrantWrapper


class QdrantListingSemanticIndex:
    def __init__(self, qdrant: QdrantWrapper | None = None) -> None:
        if qdrant is None:
            from config import QDRANT_LISTINGS_COLLECTION
            qdrant = QdrantWrapper(collection_name=QDRANT_LISTINGS_COLLECTION)
        self.qdrant = qdrant

    def search_listings(
        self,
        query_text: str,
        candidate_ids: list[str],
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        from agents.model_registry import get_embed_model

        model = get_embed_model()
        query_vector = np.array(model.encode(query_text, normalize_embeddings=True))
        results = self.qdrant.search_listings(
            query_vector=query_vector,
            query_text=query_text,
            candidate_ids=candidate_ids,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )
        return [
            {
                "listing_id": item.get("listing_id"),
                "score": item.get("rrf_score", item.get("score", 0)),
                "source_version": item.get("source_version", 0),
            }
            for item in results
            if item.get("listing_id")
        ]


class QdrantListingVectorIndex:
    def __init__(self, qdrant: QdrantWrapper | None = None) -> None:
        if qdrant is None:
            from config import QDRANT_LISTINGS_COLLECTION
            qdrant = QdrantWrapper(collection_name=QDRANT_LISTINGS_COLLECTION)
        self.qdrant = qdrant

    def get_payload(self, listing_id: str, chunk_type: str = "listing_summary") -> dict | None:
        return self.qdrant.get_listing_payload(listing_id, chunk_type=chunk_type)

    def upsert_listing_chunk(
        self,
        listing_id: str,
        chunk_type: str,
        text: str,
        embedding: np.ndarray,
        payload: dict,
    ) -> str:
        return self.qdrant.upsert_listing_chunk(
            listing_id=listing_id,
            chunk_type=chunk_type,
            text=text,
            embedding=embedding,
            payload=payload,
        )

    def delete_listing(self, listing_id: str) -> None:
        self.qdrant.delete_listing_points(listing_id)

