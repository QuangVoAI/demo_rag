"""
Rewriter Agent — Viết lại / nới lỏng điều kiện tìm phòng khi tìm kiếm thất bại.

 Được gọi sau khi Grader phán quyết kết quả là POOR.
Chiến lược rewrite:
  1. Nới ngân sách thêm 20–30%
  2. Mở rộng khu vực sang quận lân cận
  3. Bỏ bớt tiện ích bắt buộc kém quan trọng
  4. Dùng LLM (Groq FAST) để diễn đạt lại query tự nhiên hơn
"""
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.state import NhatrovnAgentState
from agents.llm_client import groq_complete, GROQ_MODEL_FAST
from utils.console import console

_REWRITE_SYSTEM_PROMPT = """\
Bạn là chuyên gia viết lại câu tìm kiếm phòng trọ tại Việt Nam.
Người dùng đang tìm phòng nhưng chưa tìm được kết quả phù hợp.

Nhiệm vụ: Viết lại câu tìm kiếm theo hướng NỚI LỎNG điều kiện để tìm được phòng.

Chiến lược (chọn một hoặc kết hợp):
1. Nới ngân sách thêm 20–30%
2. Mở rộng khu vực sang quận lân cận
3. Bỏ bớt tiện ích ít quan trọng (giữ lại điều kiện cốt lõi)
4. Dùng từ ngữ tổng quát hơn

Quy tắc:
- Chỉ trả về câu query mới, KHÔNG giải thích thêm
- Ngắn gọn (1–2 câu), tiếng Việt tự nhiên
- Không bịa thêm khu vực nếu người dùng chưa đề cập
"""


async def rewrite_query_node(state: NhatrovnAgentState) -> dict:
    """LangGraph Node: Viết lại query khi tìm phòng không có kết quả."""
    t0 = time.time()
    original_query = state.get("rewritten_query") or state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    rooms = state.get("rooms", [])

    # Bổ sung context về những gì đã tìm (nếu có)
    context_hint = ""
    if rooms:
        titles = [item.get("title", "")[:60] for item in rooms[:2]]
        context_hint = (
            "\nKết quả hiện tại (chưa đạt chất lượng):\n"
            + "\n".join(f"- {t}" for t in titles if t)
            + "\nHãy rewrite để tìm phòng phù hợp hơn."
        )

    prompt = (
        f"Câu tìm kiếm gốc: {state['question']}\n"
        f"Câu tìm kiếm lần {rewrite_count + 1}: {original_query}\n"
        f"{context_hint}\n\n"
        f"Viết lại câu tìm kiếm với điều kiện nới lỏng hơn:"
    )

    try:
        rewritten = await groq_complete(
            prompt=prompt,
            system_prompt=_REWRITE_SYSTEM_PROMPT,
            model=GROQ_MODEL_FAST,
            max_tokens=120,
            temperature=0.3,
        )
        rewritten = rewritten.strip().strip('"').strip("'")
    except Exception as e:
        console.print(f"[yellow]  Rewriter lỗi: {e}, giữ nguyên query[/]")
        rewritten = original_query

    elapsed = int((time.time() - t0) * 1000)
    console.print(
        f"[yellow]  Rewrite #{rewrite_count + 1}: "
        f"'{original_query[:40]}' → '{rewritten[:40]}' ({elapsed}ms)[/]"
    )

    return {
        "rewritten_query": rewritten,
        "rewrite_count": rewrite_count + 1,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            f"rewrite_{rewrite_count + 1}_from": original_query,
            f"rewrite_{rewrite_count + 1}_to": rewritten,
            f"rewrite_{rewrite_count + 1}_ms": elapsed,
        },
    }
