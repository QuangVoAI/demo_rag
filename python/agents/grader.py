"""
Grader Agent — Đánh giá chất lượng kết quả tìm kiếm phòng.

Không dùng LLM (0 token, ~0ms).
Dựa vào semantic score từ Qdrant hoặc RRF score để phán quyết:
  GOOD : Có đủ kết quả chất lượng → tiếp tục sinh câu trả lời
  POOR : Kết quả quá ít hoặc không đủ tốt → trigger rewrite query
"""
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import GRADE_SCORE_THRESHOLD, MIN_GOOD_LISTINGS
from agents.state import NhatrovnAgentState
from utils.console import console


def grade_listings_node(state: NhatrovnAgentState) -> dict:
    """
    LangGraph Node: Đánh giá chất lượng danh sách phòng vừa tìm được.

    Kiểm tra:
      1. Có đủ listing trả về không?
      2. Listing có score đủ cao không (nếu có semantic score)?
    """
    t0 = time.time()
    listings = state.get("listings", [])
    rewrite_count = state.get("rewrite_count", 0)

    # Đếm listing có chất lượng đủ tốt
    good_listings = []
    for listing in listings:
        # Ưu tiên semantic_score; fallback về rrf_score; mặc định 1.0 nếu không có
        score = listing.get("semantic_score", listing.get("rrf_score", 1.0))
        if score >= GRADE_SCORE_THRESHOLD:
            good_listings.append(listing)

    is_sufficient = len(good_listings) >= MIN_GOOD_LISTINGS
    decision = "GOOD" if is_sufficient else "POOR"
    elapsed = int((time.time() - t0) * 1000)

    console.print(
        f"[dim]  Grader: {len(good_listings)}/{len(listings)} phòng đạt ngưỡng "
        f"score={GRADE_SCORE_THRESHOLD} → {decision} "
        f"(rewrite #{rewrite_count}, {elapsed}ms)[/]"
    )

    return {
        "grade_result": decision,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "grade_good": len(good_listings),
            "grade_total": len(listings),
            "grade_threshold": GRADE_SCORE_THRESHOLD,
            "grade_decision": decision,
            "grade_rewrite_count": rewrite_count,
            "grade_ms": elapsed,
        },
    }
