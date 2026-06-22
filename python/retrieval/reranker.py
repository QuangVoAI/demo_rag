"""
Cross-encoder Reranking sử dụng BAAI/bge-reranker-v2-m3.
Chấm điểm chéo từng tài liệu với câu hỏi gốc để lọc top-K chính xác nhất.
"""
import numpy as np
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import RERANKER_MODEL, TOP_K_RERANK
from agents.model_registry import get_reranker_model

from utils.console import console


def rerank(
    query: str,
    documents: list[dict],
    top_k: int = TOP_K_RERANK,
) -> list[dict]:
    """
    Rerank kết quả tìm kiếm bằng Cross-encoder.
    
    Quy trình:
    1. Tạo cặp (query, document) cho mỗi kết quả
    2. Cross-encoder chấm điểm relevance cho từng cặp
    3. Sắp xếp theo điểm và giữ top_k
    
    Args:
        query: Câu hỏi gốc từ người dùng
        documents: Danh sách kết quả từ Hybrid Search
        top_k: Số kết quả giữ lại
        
    Returns:
        Top-K kết quả đã rerank
    """
    if not documents:
        return []
    
    reranker = get_reranker_model()
    
    console.print(
        f"[cyan]🔀 Reranking {len(documents)} documents → top {top_k}...[/]"
    )
    
    # Tạo cặp (query, doc_text)
    pairs = [(query, doc["text"]) for doc in documents]
    
    # Chấm điểm — batch_size=len(pairs) để tránh overhead chia batch không cần thiết
    scores = reranker.predict(pairs, batch_size=len(pairs))
    
    # Gán điểm vào kết quả
    scored_docs = []
    for doc, score in zip(documents, scores):
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = float(score)
        scored_docs.append(doc_copy)
    
    # Sắp xếp giảm dần theo rerank_score
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Giữ top_k
    results = scored_docs[:top_k]
    
    console.print(
        f"[green]✅ Reranked: top {len(results)}/{len(documents)} "
        f"(best={results[0]['rerank_score']:.4f})[/]" if results else
        "[yellow]⚠️ Rerank: no results[/]"
    )
    
    return results


if __name__ == "__main__":
    # Test reranking
    test_query = "Điều kiện thành lập công ty cổ phần"
    test_docs = [
        {"text": "Hôm nay trời đẹp lắm, tôi đi chơi công viên.", "id": "1"},
        {"text": "Công ty cổ phần phải có ít nhất 3 cổ đông sáng lập theo Điều 120 Luật Doanh nghiệp.", "id": "2"},
        {"text": "Điều 111. Công ty cổ phần: Vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần.", "id": "3"},
        {"text": "Điều kiện cấp giấy phép kinh doanh bao gồm vốn pháp định và nhân sự.", "id": "4"},
    ]
    
    results = rerank(test_query, test_docs)
    console.print(f"\n[bold]Top results:[/]")
    for r in results:
        console.print(f"  Score={r['rerank_score']:.4f}: {r['text'][:80]}")
