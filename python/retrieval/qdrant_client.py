"""
Qdrant client wrapper cho hệ thống Nhatrovn.
Quản lý collection, upsert, và search trên Qdrant vector database.
"""
import uuid
import hashlib
import math
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector, Filter,
    models,
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_DIM

from utils.console import console

# Stop-words for BM25-like sparse vector (English + Vietnamese)
STOP_WORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "not", "no", "nor", "as", "if", "then",
    "than", "so", "such", "which", "who", "whom", "what", "where",
    "when", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "any", "only", "very", "also", "just",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "while", "up", "down", "out", "off", "over", "under",
    # Vietnamese — function words / particles
    "của", "và", "là", "được", "có", "không", "cho", "với", "các",
    "trong", "từ", "đến", "để", "theo", "trên", "về", "tại", "này",
    "đó", "một", "những", "như", "thì", "mà", "nhưng", "hay", "hoặc",
    "nếu", "khi", "đã", "sẽ", "đang", "vẫn", "còn", "rất", "cũng",
    "bị", "do", "vì", "nên", "tôi", "bạn", "anh", "chị", "em",
    "mình", "chúng", "họ", "ông", "bà", "cả", "mỗi", "nào", "gì",
    "ai", "sao", "thế", "ở", "lại", "ra", "lên", "xuống", "vào",
}


class QdrantWrapper:
    """Wrapper cho Qdrant operations."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = QDRANT_COLLECTION,
    ):
        self.collection_name = collection_name

        kwargs = {"url": url, "timeout": 60}
        if api_key:
            kwargs["api_key"] = api_key

        self.client = QdrantClient(**kwargs)
        console.print(f"[green]✅ Connected to Qdrant: {url}[/]")

    def create_collection(self, recreate: bool = False):
        """Tạo collection với Hybrid Search (Dense + Sparse vectors)."""
        exists = self.client.collection_exists(self.collection_name)

        if exists and not recreate:
            info = self.client.get_collection(self.collection_name)
            console.print(
                f"[yellow]📦 Collection '{self.collection_name}' already exists "
                f"({info.points_count} points)[/]"
            )
            return

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            console.print(f"[yellow]🗑️ Deleted old collection[/]")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

        console.print(
            f"[green]✅ Created collection '{self.collection_name}' "
            f"(dense: {EMBEDDING_DIM}D + sparse BM25)[/]"
        )

    def _text_to_sparse(self, text: str) -> tuple[list[int], list[float]]:
        """
        Tạo sparse vector từ text (BM25-like).
        English stop-words removal + log-scaled TF.
        MD5 hash modulo 100k for deterministic dimension mapping.
        """
        words = text.lower().split()
        word_freq = {}
        for w in words:
            # Skip stop-words and short words
            if w in STOP_WORDS or len(w) <= 1:
                continue
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16) % 100000
            word_freq[h] = word_freq.get(h, 0) + 1

        indices = list(word_freq.keys())
        # Log-scaled TF: 1 + ln(tf)
        values = [1.0 + math.log(v) for v in word_freq.values()]

        return indices, values

    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        """Tìm kiếm Dense vector (semantic search)."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            using="dense",
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "level": r.payload.get("level", 0),
                "doc_title": r.payload.get("doc_title", ""),
                "node_id": r.payload.get("node_id", 0),
                "category": r.payload.get("metadata", {}).get("category", ""),
                "url": r.payload.get("metadata", {}).get("url", ""),
                "room_id": r.payload.get("room_id", ""),
                "house_id": r.payload.get("house_id", ""),
                "source_version": r.payload.get("source_version", 0),
            }
            for r in results.points
        ]

    def search_sparse(
        self,
        query_text: str,
        top_k: int = 20,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        """Tìm kiếm Sparse vector (keyword/BM25-like search)."""
        indices, values = self._text_to_sparse(query_text)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "level": r.payload.get("level", 0),
                "doc_title": r.payload.get("doc_title", ""),
                "node_id": r.payload.get("node_id", 0),
                "category": r.payload.get("metadata", {}).get("category", ""),
                "url": r.payload.get("metadata", {}).get("url", ""),
                "room_id": r.payload.get("room_id", ""),
                "house_id": r.payload.get("house_id", ""),
                "source_version": r.payload.get("source_version", 0),
            }
            for r in results.points
        ]

    def search_rooms(
        self,
        query_vector: np.ndarray,
        query_text: str,
        candidate_ids: list[str] | None = None,
        metadata_filter: dict | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        """Hybrid search for room chunks, constrained by room IDs/metadata."""
        q_filter = self._room_filter(candidate_ids or [], metadata_filter or {})
        dense_results = self.search_dense(query_vector, top_k=top_k * 2, query_filter=q_filter)
        sparse_results = self.search_sparse(query_text, top_k=top_k * 2, query_filter=q_filter)

        from retrieval.hybrid_search import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:top_k]

    def upsert_room_chunk(
        self,
        room_id: str,
        chunk_type: str,
        text: str,
        embedding: np.ndarray,
        payload: dict,
    ) -> str:
        """Idempotently upsert one deterministic room chunk point."""
        point_id = self.room_point_id(room_id, chunk_type)
        sparse_indices, sparse_values = self._text_to_sparse(text)
        point = PointStruct(
            id=point_id,
            vector={
                "dense": embedding.tolist(),
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={
                **payload,
                "room_id": room_id,
                "chunk_type": chunk_type,
                "text": text,
            },
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return point_id

    def delete_room_points(self, room_id: str) -> None:
        """Delete all room points for one room_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="room_id",
                            match=models.MatchValue(value=room_id),
                        )
                    ]
                )
            ),
        )

    def get_room_payload(self, room_id: str, chunk_type: str = "room_summary") -> dict | None:
        point_id = self.room_point_id(room_id, chunk_type)
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        return points[0].payload or {}

    @staticmethod
    def room_point_id(room_id: str, chunk_type: str = "room_summary") -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"room:{room_id}:{chunk_type}"))

    def _room_filter(self, candidate_ids: list[str], metadata_filter: dict) -> Filter | None:
        must = []
        if candidate_ids:
            must.append(models.FieldCondition(
                key="room_id",
                match=models.MatchAny(any=[str(item) for item in candidate_ids]),
            ))
        for key, value in metadata_filter.items():
            if value is None:
                continue
            if isinstance(value, list):
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
            else:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        if not must:
            return None
        return models.Filter(must=must)

    def get_collection_info(self) -> dict:
        """Lấy thông tin collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value,
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    wrapper = QdrantWrapper()
    info = wrapper.get_collection_info()
    console.print(f"[bold]Collection info:[/] {info}")
