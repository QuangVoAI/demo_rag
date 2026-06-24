"""
Hybrid Search: kết hợp Dense (semantic) + Sparse (BM25) search trên Qdrant.
Đảm bảo tìm được cả ngữ nghĩa tổng quát lẫn từ khóa chính xác (số hiệu luật).
"""
import numpy as np
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import TOP_K_RETRIEVAL, RRF_DENSE_WEIGHT, RRF_SPARSE_WEIGHT, RRF_K
from retrieval.qdrant_client import QdrantWrapper

from utils.console import console


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
    dense_weight: float = RRF_DENSE_WEIGHT,
    sparse_weight: float = RRF_SPARSE_WEIGHT,
) -> list[dict]:
    """
    Kết hợp kết quả Dense + Sparse bằng Reciprocal Rank Fusion (RRF).
    
    RRF score = Σ weight / (k + rank)
    
    Args:
        dense_results: Kết quả từ dense search
        sparse_results: Kết quả từ sparse search
        k: Hằng số RRF (cao hơn → ít phân biệt rank)
        dense_weight: Trọng số cho dense search
        sparse_weight: Trọng số cho sparse search
        
    Returns:
        List[dict] kết quả đã sắp xếp theo RRF score
    """
    scores = {}
    result_map = {}
    
    # Dense scores
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
        result_map[doc_id] = result
    
    # Sparse scores
    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)
        if doc_id not in result_map:
            result_map[doc_id] = result
    
    # Sắp xếp theo RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    fused = []
    for doc_id in sorted_ids:
        result = result_map[doc_id].copy()
        result["rrf_score"] = scores[doc_id]
        result["combined_score"] = scores[doc_id]
        fused.append(result)
    
    return fused


def hybrid_search(
    query_vector: np.ndarray,
    query_text: str,
    qdrant: QdrantWrapper = None,
    top_k: int = TOP_K_RETRIEVAL,
    dense_weight: float = RRF_DENSE_WEIGHT,
    sparse_weight: float = RRF_SPARSE_WEIGHT,
) -> list[dict]:
    """
    Thực hiện Hybrid Search = Dense + Sparse + RRF.
    
    Args:
        query_vector: Vector embedding của query (từ HyDE hoặc trực tiếp)
        query_text: Query text gốc (cho sparse search)
        qdrant: QdrantWrapper instance
        top_k: Số kết quả trả về
        dense_weight: Trọng số dense search
        sparse_weight: Trọng số sparse search
        
    Returns:
        Top-K kết quả đã fuse
    """
    if qdrant is None:
        qdrant = QdrantWrapper()
    
    console.print(f"[cyan]🔍 Hybrid Search (dense:{dense_weight} + sparse:{sparse_weight})...[/]")
    
    # Dense search (semantic)
    dense_results = qdrant.search_dense(query_vector, top_k=top_k * 2)
    console.print(f"[dim]  Dense: {len(dense_results)} results[/]")
    
    # Sparse search (keyword/BM25)
    sparse_results = qdrant.search_sparse(query_text, top_k=top_k * 2)
    console.print(f"[dim]  Sparse: {len(sparse_results)} results[/]")
    
    # Fuse bằng RRF
    fused = reciprocal_rank_fusion(
        dense_results, sparse_results,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    
    # Lấy top_k
    results = fused[:top_k]
    
    console.print(f"[green]✅ Hybrid Search: {len(results)} kết quả (từ {len(fused)} unique)[/]")
    
    return results


if __name__ == "__main__":
    # Test RRF fusion
    dense = [
        {"id": "a", "score": 0.95, "text": "Doc A"},
        {"id": "b", "score": 0.85, "text": "Doc B"},
        {"id": "c", "score": 0.75, "text": "Doc C"},
    ]
    sparse = [
        {"id": "b", "score": 5.2, "text": "Doc B"},
        {"id": "d", "score": 4.1, "text": "Doc D"},
        {"id": "a", "score": 3.0, "text": "Doc A"},
    ]
    
    fused = reciprocal_rank_fusion(dense, sparse)
    for r in fused:
        console.print(f"  {r['id']}: RRF={r['rrf_score']:.4f}")
