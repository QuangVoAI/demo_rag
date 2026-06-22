"""
Router Agent — Phân loại intent cho EmpathAI.
3 intent: COMPLAINT / INQUIRY / CASUAL
Embedding-based, KHÔNG dùng LLM.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.model_registry import get_embed_model
from utils.console import console

_complaint_centroid = None
_inquiry_centroid = None
_casual_centroid = None

COMPLAINT_KEYWORDS = [
    "lỗi", "hỏng", "hư", "bể", "nát", "sai", "nhầm", "tệ",
    "lừa đảo", "ăn cướp", "bực mình", "tức giận", "thất vọng",
    "khiếu nại", "phàn nàn", "bức xúc", "mệt mỏi",
    "hoàn tiền", "đổi trả", "bồi thường", "bảo hành",
    "không được", "không hoạt động", "không phản hồi",
    "chờ quá lâu", "quá chậm", "trễ hạn", "mất hàng",
    "tính sai tiền", "trừ tiền", "không nhận được",
    "giao sai", "giao trễ", "vỡ", "rác", "ngu",
    "report", "kiện", "phốt", "đổ lỗi", "không chấp nhận",
    "quá tệ", "kinh khủng", "nguy hiểm", "dị ứng", "hư hỏng",
    "gian lận", "sale ảo", "voucher lỗi", "không áp dụng",
]

INQUIRY_KEYWORDS = [
    "hỏi", "thắc mắc", "muốn biết", "cho tôi hỏi",
    "hướng dẫn", "cách làm", "làm sao", "thế nào",
    "báo giá", "giá bao nhiêu", "có sẵn không",
    "tư vấn", "gợi ý", "khuyên", "đề xuất",
    "thời gian", "bao lâu", "khi nào",
    "chính sách", "điều kiện", "quy định",
    "ship", "giao hàng", "phí ship", "vận chuyển",
    "thanh toán", "chuyển khoản", "trả góp",
    "ưu đãi", "khuyến mãi", "giảm giá",
]

CASUAL_KEYWORDS = [
    "xin chào", "chào bạn", "hello", "hi", "hey",
    "cảm ơn", "cảm ơn bạn", "thanks", "thank you",
    "tạm biệt", "bye", "bai bai",
    "bạn là ai", "tên gì", "bạn làm được gì",
    "bạn khỏe không", "oke", "ok", "vâng", "ừ",
]

CASUAL_SHORT_ONLY = ["chào", "hi", "hey", "ok", "ừ", "vâng", "dạ"]


def _ensure_centroids():
    global _complaint_centroid, _inquiry_centroid, _casual_centroid
    if _complaint_centroid is not None:
        return

    model = get_embed_model()
    console.print("[dim]  Router: computing centroids...[/]")

    comp_emb = model.encode(COMPLAINT_KEYWORDS, normalize_embeddings=True, batch_size=64)
    _complaint_centroid = np.mean(comp_emb, axis=0)
    _complaint_centroid /= np.linalg.norm(_complaint_centroid)

    inq_emb = model.encode(INQUIRY_KEYWORDS, normalize_embeddings=True, batch_size=64)
    _inquiry_centroid = np.mean(inq_emb, axis=0)
    _inquiry_centroid /= np.linalg.norm(_inquiry_centroid)

    cas_emb = model.encode(CASUAL_KEYWORDS, normalize_embeddings=True, batch_size=64)
    _casual_centroid = np.mean(cas_emb, axis=0)
    _casual_centroid /= np.linalg.norm(_casual_centroid)

    console.print("[dim]  Router: centroids ready[/]")


COMPLAINT_FAST = [
    "khiếu nại", "phàn nàn", "bức xúc", "hoàn tiền", "đổi trả",
    "bồi thường", "bảo hành", "lừa đảo", "ăn cướp",
    "hỏng", "hư", "lỗi", "bể", "nát", "sai",
    "giao trễ", "giao sai", "mất hàng", "không nhận",
    "tính sai", "trừ tiền", "report", "kiện",
    "tệ hại", "rác", "thất vọng", "bực mình", "tức giận",
]

INQUIRY_FAST = [
    "cho tôi hỏi", "muốn biết", "làm sao", "thế nào",
    "báo giá", "giá bao nhiêu", "có sẵn không",
    "hướng dẫn", "tư vấn", "chính sách", "quy định",
    "phí ship", "thanh toán", "trả góp",
]

CASUAL_FAST = [
    "xin chào", "chào bạn", "hello", "cảm ơn",
    "bạn là ai", "bạn làm được gì", "tạm biệt", "bye",
    "bạn khỏe", "how are you",
]


def _fast_classify(question):
    q = question.lower().strip()

    # COMPLAINT first (bias an toàn cho CSKH)
    for p in COMPLAINT_FAST:
        if p in q:
            return "COMPLAINT"

    # Casual short
    if len(q) < 15:
        for p in CASUAL_SHORT_ONLY:
            if q.startswith(p) or q == p:
                return "CASUAL"

    # Casual patterns
    for p in CASUAL_FAST:
        if p in q:
            return "CASUAL"

    # Inquiry patterns
    for p in INQUIRY_FAST:
        if p in q:
            return "INQUIRY"

    return None


def classify(question):
    """Phân loại intent: COMPLAINT / INQUIRY / CASUAL."""
    fast = _fast_classify(question)
    if fast:
        console.print(f"[dim]  Router: FAST -> {fast}[/]")
        return fast

    _ensure_centroids()
    model = get_embed_model()
    q_emb = model.encode(question, normalize_embeddings=True)

    comp_sim = float(np.dot(q_emb, _complaint_centroid))
    inq_sim = float(np.dot(q_emb, _inquiry_centroid))
    cas_sim = float(np.dot(q_emb, _casual_centroid))

    # Bias toward COMPLAINT (an toàn hơn cho CSKH)
    COMPLAINT_BIAS = 0.03
    scores = {
        "COMPLAINT": comp_sim + COMPLAINT_BIAS,
        "INQUIRY": inq_sim,
        "CASUAL": cas_sim,
    }

    intent = max(scores, key=scores.get)
    console.print(
        f"[dim]  Router: comp={comp_sim:.3f} inq={inq_sim:.3f} "
        f"cas={cas_sim:.3f} -> {intent}[/]"
    )
    return intent
