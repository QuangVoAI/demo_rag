"""
Sentiment Analyzer — Phân tích cảm xúc khách hàng.
Embedding-based, KHÔNG dùng LLM (0 token, ~10ms).
"""
import numpy as np
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.model_registry import get_embed_model
from agents.state import AgentState
from utils.console import console

# Singleton centroids
_centroids = None

SENTIMENT_CLUSTERS = {
    "toxic": [
        "lừa đảo", "ăn cướp", "lấy tiền", "tệ hại", "rác", "ngu",
        "report", "kiện", "bóc phốt", "không bao giờ quay lại",
        "láo", "mất dạy", "vô trách nhiệm", "phẫn nộ", "tức giận",
        "chửi", "căm tức", "bất bình", "phàn nàn gay gắt",
        "tôi sẽ kiện", "báo cáo", "phê bình", "dọa", "cảnh cáo",
    ],
    "frustrated": [
        "mệt mỏi", "bực bội", "khó chịu", "bao giờ", "chờ quá lâu",
        "lần thứ ba", "vẫn chưa", "không thể chấp nhận", "chán nản",
        "tại sao", "sao lại", "ai chịu trách nhiệm", "đã nhiều lần",
        "không được giải quyết", "mất kiên nhẫn", "phiền phức",
        "rốt cuộc", "đến bao giờ", "quá nhiều", "hết chịu nổi",
    ],
    "disappointed": [
        "thất vọng", "buồn", "tiếc", "kỳ vọng", "không như mong đợi",
        "hơi buồn", "khách quen", "tin tưởng", "ủng hộ lâu năm",
        "đáng tiếc", "hy vọng", "không tốt như trước",
        "cảm thấy buồn", "lo lắng", "không hài lòng", "chưa vươn",
    ],
    "neutral": [
        "hỏi", "thắc mắc", "muốn biết", "cho tôi hỏi",
        "làm ơn", "giúp tôi", "có thể giúp", "thông tin",
        "hướng dẫn", "cách làm", "báo giá", "tư vấn",
        "xin chào", "cảm ơn", "thế nào", "tại sao",
    ],
}


def _ensure_centroids():
    """Precompute centroids cho 4 sentiment clusters."""
    global _centroids
    if _centroids is not None:
        return

    model = get_embed_model()
    console.print("[dim]  Sentiment: computing centroids...[/]")

    _centroids = {}
    for label, keywords in SENTIMENT_CLUSTERS.items():
        embeddings = model.encode(keywords, normalize_embeddings=True, batch_size=64)
        centroid = np.mean(embeddings, axis=0)
        centroid /= np.linalg.norm(centroid)
        _centroids[label] = centroid

    console.print("[dim]  Sentiment: centroids ready[/]")


def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Phân tích cảm xúc từ text.
    Returns: (sentiment_label, confidence_score)
    """
    _ensure_centroids()
    model = get_embed_model()

    q_emb = model.encode(text, normalize_embeddings=True)

    scores = {}
    for label, centroid in _centroids.items():
        scores[label] = float(np.dot(q_emb, centroid))

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # Normalize to 0-1 range
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score > min_score:
        confidence = (best_score - min_score) / (max_score - min_score)
    else:
        confidence = 0.5

    return best_label, round(confidence, 3)


def sentiment_analyzer_node(state: AgentState) -> dict:
    """LangGraph Node: Phân tích cảm xúc khách hàng."""
    t0 = time.time()
    question = state["question"]

    sentiment, score = analyze_sentiment(question)

    elapsed = int((time.time() - t0) * 1000)
    console.print(
        f"[dim]  Sentiment: {sentiment} (score={score:.3f}) ({elapsed}ms)[/]"
    )

    return {
        "sentiment": sentiment,
        "sentiment_score": score,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "sentiment": sentiment,
            "sentiment_score": score,
            "sentiment_ms": elapsed,
        },
    }
