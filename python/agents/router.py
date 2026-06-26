from __future__ import annotations

"""
Router Agent — Phân loại loại tương tác của người dùng trên nhatrovn.

3 loại:
  SEARCH    : Đang tìm / lọc phòng
  QUESTION  : Hỏi thông tin về phòng cụ thể, chi phí, thủ tục
  CASUAL    : Chào hỏi, cảm ơn, câu hỏi chung

Chiến lược:
  1. Fast classify bằng keyword (không cần model)
  2. Embedding similarity khi keyword không đủ rõ
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.console import console

# Singleton centroids cho embedding-based classification
_search_centroid: list[float] | None = None
_question_centroid: list[float] | None = None
_casual_centroid: list[float] | None = None

# ---------------------------------------------------------------------------
# Từ khóa tốc độ nhanh (không dùng embedding)
# ---------------------------------------------------------------------------
_SEARCH_FAST = [
    "tìm phòng", "tim phong", "phòng trọ", "phong tro",
    "thuê phòng", "thue phong", "cần phòng", "can phong",
    "phòng dưới", "phong duoi", "khu vực", "gần trường",
    "studio", "căn hộ", "can ho", "nhà nguyên căn",
]
_QUESTION_FAST = [
    "phòng này", "phong nay", "chi phí", "tiền cọc",
    "tổng chi phí", "so sánh", "hợp đồng", "thủ tục",
    "có nuôi mèo", "có nuôi chó", "diện tích", "tiện ích",
    "tính tiền", "mã phòng", "ưu điểm", "hạn chế",
]
_CASUAL_FAST = [
    "xin chào", "chào bạn", "hello", "hi", "hey",
    "cảm ơn", "thanks", "cám ơn", "bạn là ai",
    "tạm biệt", "bye", "oke", "ok", "được rồi",
]

# Từ khóa ngắn chỉ khớp khi câu ngắn hơn 15 ký tự
_CASUAL_SHORT = ["chào", "hi", "hey", "ok", "ừ", "vâng", "dạ", "uh"]

# Từ khóa ngữ nghĩa để tính centroid embedding
_SEARCH_SEEDS = [
    "tìm phòng trọ", "thuê nhà", "phòng dưới 5 triệu", "căn hộ quận 7",
    "phòng có máy lạnh", "gần trường đại học", "tìm nơi ở",
]
_QUESTION_SEEDS = [
    "phòng này có gì", "chi phí ban đầu", "tiền cọc bao nhiêu",
    "hợp đồng thuê nhà", "thủ tục thuê phòng", "phòng có nuôi chó không",
]
_CASUAL_SEEDS = [
    "xin chào bạn", "cảm ơn nhiều", "bạn giúp được gì",
    "tạm biệt", "mình hiểu rồi", "ok cảm ơn",
]


def get_embed_model():
    from agents.model_registry import get_embed_model as _get_embed_model

    return _get_embed_model()


def _ensure_centroids() -> None:
    global _search_centroid, _question_centroid, _casual_centroid
    if _search_centroid is not None:
        return
    model = get_embed_model()
    console.print("[dim]  Router: đang tính centroids...[/]")

    def _centroid(seeds: list[str]) -> list[float]:
        embs = model.encode(seeds, normalize_embeddings=True, batch_size=32)
        return _l2_normalize(_mean_vector(embs))

    _search_centroid   = _centroid(_SEARCH_SEEDS)
    _question_centroid = _centroid(_QUESTION_SEEDS)
    _casual_centroid   = _centroid(_CASUAL_SEEDS)
    console.print("[dim]  Router: centroids sẵn sàng[/]")


def _fast_classify(question: str) -> str | None:
    """Phân loại nhanh bằng keyword, không cần embedding."""
    q = question.lower().strip()

    # Câu rất ngắn — khả năng cao là casual
    if len(q) < 15:
        for kw in _CASUAL_SHORT:
            if q.startswith(kw) or q == kw:
                return "CASUAL"

    for kw in _SEARCH_FAST:
        if kw in q:
            return "SEARCH"
    for kw in _QUESTION_FAST:
        if kw in q:
            return "QUESTION"
    for kw in _CASUAL_FAST:
        if kw in q:
            return "CASUAL"
    return None


def classify(question: str) -> str:
    """
    Phân loại loại tương tác: SEARCH / QUESTION / CASUAL.

    Dùng fast keyword trước, embedding similarity làm fallback.
    """
    fast = _fast_classify(question)
    if fast:
        console.print(f"[dim]  Router: FAST → {fast}[/]")
        return fast

    _ensure_centroids()
    model = get_embed_model()
    q_emb = _vector_to_list(model.encode(question, normalize_embeddings=True))

    scores = {
        "SEARCH":   _dot(q_emb, _search_centroid or []),
        "QUESTION": _dot(q_emb, _question_centroid or []),
        "CASUAL":   _dot(q_emb, _casual_centroid or []),
    }
    # Ưu tiên nhẹ cho SEARCH — đây là luồng chính của nhatrovn
    scores["SEARCH"] += 0.02

    result = max(scores, key=scores.get)  # type: ignore[arg-type]
    console.print(
        f"[dim]  Router: search={scores['SEARCH']:.3f} "
        f"question={scores['QUESTION']:.3f} casual={scores['CASUAL']:.3f} → {result}[/]"
    )
    return result


def _vector_to_list(vector) -> list[float]:
    return [float(value) for value in vector]


def _mean_vector(vectors) -> list[float]:
    rows = [_vector_to_list(vector) for vector in vectors]
    if not rows:
        return []
    width = len(rows[0])
    return [
        sum(row[idx] for row in rows) / len(rows)
        for idx in range(width)
    ]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))
