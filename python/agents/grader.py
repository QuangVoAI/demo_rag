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

from config import GRADE_SCORE_THRESHOLD, MIN_GOOD_ROOMS
from agents.state import NhatrovnAgentState
from utils.console import console


def grade_rooms_node(state: NhatrovnAgentState) -> dict:
    """
    LangGraph Node: Đánh giá chất lượng danh sách phòng vừa tìm được.

    Kiểm tra:
      1. Có đủ room trả về không?
      2. Room có score đủ cao không (nếu có semantic score)?
    """
    t0 = time.time()
    rooms = state.get("rooms", [])
    rewrite_count = state.get("rewrite_count", 0)

    # Đếm room có chất lượng đủ tốt
    good_rooms = []
    for room in rooms:
        # Ưu tiên semantic_score; fallback về rrf_score; mặc định 1.0 nếu không có
        score = room.get("semantic_score", room.get("rrf_score", 1.0))
        if score >= GRADE_SCORE_THRESHOLD:
            good_rooms.append(room)

    is_sufficient = len(good_rooms) >= MIN_GOOD_ROOMS
    decision = "GOOD" if is_sufficient else "POOR"
    elapsed = int((time.time() - t0) * 1000)

    console.print(
        f"[dim]  Grader: {len(good_rooms)}/{len(rooms)} phòng đạt ngưỡng "
        f"score={GRADE_SCORE_THRESHOLD} → {decision} "
        f"(rewrite #{rewrite_count}, {elapsed}ms)[/]"
    )

    return {
        "grade_result": decision,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "grade_good": len(good_rooms),
            "grade_total": len(rooms),
            "grade_threshold": GRADE_SCORE_THRESHOLD,
            "grade_decision": decision,
            "grade_rewrite_count": rewrite_count,
            "grade_ms": elapsed,
        },
    }
