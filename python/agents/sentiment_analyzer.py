from __future__ import annotations

"""
Sentiment Analyzer — Phân tích cảm xúc người dùng khi tìm phòng trọ.

Hoạt động bằng embedding cosine similarity (không tốn token LLM, ~10ms).
Nhận diện 3 trạng thái:
  - frustrated : Người dùng bực bội (không tìm được phòng, giá cao, hết phòng)
  - urgent     : Người dùng cần phòng gấp (hết hạn HĐ, chuyển nhà sớm)
  - normal     : Đang xem bình thường, chưa có dấu hiệu áp lực
"""
import re
import time
import sys
import unicodedata
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.state import NhatrovnAgentState
from utils.console import console

# Singleton centroids — tính một lần rồi cache
_centroids: dict[str, list[float]] | None = None

# Từ khóa mẫu để tính centroid cho từng cảm xúc
MOOD_CLUSTERS: dict[str, list[str]] = {
    "frustrated": [
        "không tìm được phòng", "tìm mãi không ra", "hết phòng rồi",
        "giá cao quá", "đắt quá", "quá tầm tiền", "không phù hợp",
        "tìm hoài không thấy", "chán quá", "thất vọng",
        "phòng nào cũng không ổn", "điều kiện không đáp ứng",
        "không đúng khu vực", "quá xa chỗ làm", "ô nhiễm",
        "thuê không nổi", "sinh viên sao mà thuê nổi", "mắc quá",
    ],
    "urgent": [
        "cần phòng gấp", "hết hạn hợp đồng", "bị đuổi", "chuyển nhà sớm",
        "tháng này phải dọn", "tuần sau dọn", "cần ngay", "urgent",
        "không còn chỗ ở", "tạm trú", "phải chuyển", "dọn gấp",
        "deadline", "cuối tháng dọn nhà", "hết chỗ ở rồi",
    ],
    "normal": [
        "tìm phòng", "xem thử", "đang tham khảo", "hỏi thông tin",
        "muốn biết", "cho tôi hỏi", "thắc mắc", "tư vấn giúp",
        "xem giá", "so sánh", "cần biết thêm", "hướng dẫn",
        "phòng có gì không", "chỉ xem thôi", "đang cân nhắc",
    ],
}

MOOD_CUES: dict[str, tuple[str, ...]] = {
    "frustrated": (
        "khong tim duoc", "tim mai khong ra", "tim hoai khong thay",
        "het phong", "gia cao", "dat qua", "qua tam tien",
        "khong phu hop", "chan qua", "that vong", "khong on",
        "khong dung khu vuc", "qua xa", "thue khong noi", "thue noi",
        "sinh vien sao ma thue noi", "mac qua",
    ),
    "urgent": (
        "gap", "can ngay", "het han hop dong", "bi duoi",
        "chuyen nha som", "thang nay phai don", "tuan sau don",
        "khong con cho o", "phai chuyen", "don gap", "deadline",
        "cuoi thang don",
    ),
}


def get_embed_model():
    from agents.model_registry import get_embed_model as _get_embed_model

    return _get_embed_model()


def _ensure_centroids() -> None:
    """Tính centroids một lần duy nhất khi khởi động."""
    global _centroids
    if _centroids is not None:
        return

    model = get_embed_model()
    console.print("[dim]  Sentiment Analyzer: đang tính centroids...[/]")
    _centroids = {}
    for label, keywords in MOOD_CLUSTERS.items():
        embeddings = model.encode(keywords, normalize_embeddings=True, batch_size=64)
        centroid = _l2_normalize(_mean_vector(embeddings))
        _centroids[label] = centroid
    console.print("[dim]  Sentiment Analyzer: centroids sẵn sàng[/]")


def analyze_mood(text: str) -> tuple[str, float]:
    """
    Phân tích cảm xúc người dùng.

    Returns:
        (mood_label, confidence) — mood là "frustrated" | "urgent" | "normal"
    """
    _ensure_centroids()
    model = get_embed_model()
    q_emb = _vector_to_list(model.encode(text, normalize_embeddings=True))

    scores: dict[str, float] = {}
    for label, centroid in _centroids.items():  # type: ignore[union-attr]
        scores[label] = _dot(q_emb, centroid)

    best_label = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_label]

    # Chuẩn hóa về [0, 1]
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score > min_score:
        confidence = (best_score - min_score) / (max_score - min_score)
    else:
        confidence = 0.5

    if best_label != "normal" and not _has_explicit_mood_cue(text, best_label):
        return "normal", round(1.0 - min(confidence, 0.7), 3)

    return best_label, round(confidence, 3)


def _has_explicit_mood_cue(text: str, label: str) -> bool:
    normalized = _norm(text)
    return any(_cue_matches(normalized, cue) for cue in MOOD_CUES.get(label, ()))


def _cue_matches(normalized_text: str, cue: str) -> bool:
    if cue in normalized_text:
        return True
    cue_tokens = cue.split()
    if len(cue_tokens) <= 1:
        return False
    pattern = r"\b" + r"\b(?:\s+\w+){0,2}\s+".join(re.escape(token) for token in cue_tokens) + r"\b"
    return bool(re.search(pattern, normalized_text))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


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


def sentiment_analyzer_node(state: NhatrovnAgentState) -> dict:
    """LangGraph Node: Phân tích cảm xúc người dùng trong lượt hiện tại."""
    t0 = time.time()
    question = state["question"]

    mood, score = analyze_mood(question)
    elapsed = int((time.time() - t0) * 1000)

    console.print(f"[dim]  Mood: {mood} (score={score:.3f}, {elapsed}ms)[/]")

    return {
        "user_mood": mood,
        "user_mood_score": score,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "mood": mood,
            "mood_score": score,
            "mood_ms": elapsed,
        },
    }
