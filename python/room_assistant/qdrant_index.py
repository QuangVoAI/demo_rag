"""Qdrant room semantic index adapters."""

from __future__ import annotations

import numpy as np

from retrieval.qdrant_client import QdrantWrapper


class QdrantRoomSemanticIndex:
    def __init__(self, qdrant: QdrantWrapper | None = None) -> None:
        if qdrant is None:
            from config import QDRANT_ROOMS_COLLECTION
            qdrant = QdrantWrapper(collection_name=QDRANT_ROOMS_COLLECTION)
        self.qdrant = qdrant

    def search_rooms(
        self,
        query_text: str,
        candidate_ids: list[str],
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        from agents.model_registry import get_embed_model

        model = get_embed_model()
        query_vector = np.array(model.encode(query_text, normalize_embeddings=True))
        results = self.qdrant.search_rooms(
            query_vector=query_vector,
            query_text=query_text,
            candidate_ids=candidate_ids,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )
        return [
            {
                "room_id": item.get("room_id"),
                "house_id": item.get("house_id"),
                "score": item.get("rerank_score", item.get("combined_score", item.get("rrf_score", item.get("score", 0)))),
                "rerank_score": item.get("rerank_score"),
                "combined_score": item.get("combined_score"),
                "rrf_score": item.get("rrf_score", item.get("score", 0)),
            }
            for item in results
            if item.get("room_id")
        ]


class QdrantRoomVectorIndex:
    def __init__(self, qdrant: QdrantWrapper | None = None) -> None:
        if qdrant is None:
            from config import QDRANT_ROOMS_COLLECTION
            qdrant = QdrantWrapper(collection_name=QDRANT_ROOMS_COLLECTION)
        self.qdrant = qdrant

    def get_payload(self, room_id: str, chunk_type: str = "room_summary") -> dict | None:
        return self.qdrant.get_room_payload(room_id, chunk_type=chunk_type)

    def upsert_room_chunk(
        self,
        room_id: str,
        chunk_type: str,
        text: str,
        embedding: np.ndarray,
        payload: dict,
    ) -> str:
        return self.qdrant.upsert_room_chunk(
            room_id=room_id,
            chunk_type=chunk_type,
            text=text,
            embedding=embedding,
            payload=payload,
        )

    def delete_room(self, room_id: str) -> None:
        self.qdrant.delete_room_points(room_id)
