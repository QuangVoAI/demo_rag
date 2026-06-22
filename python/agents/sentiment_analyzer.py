"""
Sentiment Analyzer — Phân tích cảm xúc người dùng khi tìm phòng trọ.

Hoạt động bằng embedding cosine similarity (không tốn token LLM, ~10ms).
Nhận diện 3 trạng thái:
  - frustrated : Người dùng bực bội (không tìm được phòng, giá cao, hết phòng)
  - urgent     : Người dùng cần phòng gấp (hết hạn HĐ, chuyển nhà sớm)
  - normal     : Đang xem bình thường, chưa có dấu hiệu áp lực
"""
import numpy as np
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.model_registry import get_embed_model
from agents.state import NhatrovnAgentState
from utils.console import console

# Singleton centroids — tính một lần rồi cache
_centroids: dict | None = None

# Từ khóa mẫu để tính centroid cho từng cảm xúc
MOOD_CLUSTERS: dict[str, list[str]] = {
    "frustrated": [
        "không tìm được phòng", "tìm mãi không ra", "hết phòng rồi",
        "giá cao quá", "đắt quá", "quá tầm tiền", "không phù hợp",
        "tìm hoài không thấy", "chán quá", "thất vọng",
        "phòng nào cũng không ổn", "điều kiện không đáp ứng",
        "không đúng khu vực", "quá xa chỗ làm", "ô nhiễm",
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
        centroid = np.mean(embeddings, axis=0)
        centroid /= np.linalg.norm(centroid)
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
    q_emb = model.encode(text, normalize_embeddings=True)

    scores: dict[str, float] = {}
    for label, centroid in _centroids.items():  # type: ignore[union-attr]
        scores[label] = float(np.dot(q_emb, centroid))

    best_label = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_label]

    # Chuẩn hóa về [0, 1]
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score > min_score:
        confidence = (best_score - min_score) / (max_score - min_score)
    else:
        confidence = 0.5

    return best_label, round(confidence, 3)


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
