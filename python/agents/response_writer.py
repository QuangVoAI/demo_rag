"""
Response Writer — Sinh câu trả lời thân thiện, đúng ngữ cảnh cho nhatro.vn.

Điều chỉnh giọng điệu theo cảm xúc người dùng (mood):
  - frustrated : Đồng cảm, gợi ý thay đổi điều kiện tìm kiếm
  - urgent     : Nhanh chóng, ưu tiên kết quả còn phòng ngay
  - normal     : Thông tin đầy đủ, thân thiện

Dual backend: Groq SMART (primary) → Groq FAST (fallback)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.llm_client import groq_chat_complete, GROQ_MODEL_SMART, GROQ_MODEL_FAST

# ---------------------------------------------------------------------------
# System prompts theo từng mood
# ---------------------------------------------------------------------------

_BASE_RULES = """\
Quy tắc bắt buộc:
- CHỈ dùng thông tin trong [DỮ LIỆU ĐÃ XÁC MINH]. Không bịa thêm giá, tiện ích, địa chỉ.
- Nếu không có dữ liệu → thành thật nói "chưa có dữ liệu".
- KHÔNG hứa hẹn: đặt lịch, nhắn chủ nhà, giữ phòng, thanh toán.
- Dùng "mình/bạn", không dùng "chúng tôi/quý khách".
- Format giá: dùng triệu (VD: 4,5 triệu/tháng).
- Ngắn gọn, dùng danh sách khi liệt kê nhiều phòng."""

_SYSTEM_PROMPTS: dict[str, str] = {
    "frustrated": (
        "Bạn là trợ lý tìm phòng nhatro.vn — thấu cảm và thực tế.\n"
        "Người dùng đang bực bội vì chưa tìm được phòng phù hợp.\n"
        "Hãy: (1) thừa nhận khó khăn của họ, (2) gợi ý điều chỉnh điều kiện "
        "cụ thể (nới ngân sách, mở rộng khu vực, bỏ bớt tiện ích), "
        "(3) đưa ra kết quả tốt nhất hiện có nếu có.\n\n"
        + _BASE_RULES
    ),
    "urgent": (
        "Bạn là trợ lý tìm phòng nhatro.vn — nhanh chóng và thiết thực.\n"
        "Người dùng cần phòng GẤP. Ưu tiên: phòng trống ngay, có thể dọn vào sớm.\n"
        "Đưa thông tin súc tích, rõ ràng. Tránh dài dòng.\n\n"
        + _BASE_RULES
    ),
    "normal": (
        "Bạn là trợ lý tìm phòng nhatro.vn — thân thiện và chuyên nghiệp.\n"
        "Trả lời đầy đủ, rõ ràng dựa trên dữ liệu đã xác minh.\n\n"
        + _BASE_RULES
    ),
}


def _remove_duplicate_lines(text: str) -> str:
    """Loại bỏ các dòng / đoạn lặp liên tiếp mà LLM đôi khi sinh ra."""
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    seen: list[str] = []
    for line in lines:
        if not any(line == s or (len(line) > 20 and line in s) for s in seen[-3:]):
            seen.append(line)
    return "\n".join(seen)


async def write_response(
    question: str,
    verified_context: str,
    history: list[dict] | None = None,
    mood: str = "normal",
) -> str:
    """
    Sinh câu trả lời dựa trên dữ liệu đã xác minh.

    Args:
        question        : Câu hỏi hiện tại của người dùng.
        verified_context: Dữ liệu listing / FAQ đã được grounding.
        history         : Lịch sử hội thoại gần nhất (tối đa 6 lượt).
        mood            : Cảm xúc người dùng (frustrated/urgent/normal).

    Returns:
        Câu trả lời tiếng Việt.
    """
    system_prompt = _SYSTEM_PROMPTS.get(mood, _SYSTEM_PROMPTS["normal"])

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Thêm lịch sử hội thoại gần nhất
    for turn in (history or [])[-6:]:
        role = turn.get("role", "user")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": str(turn.get("content", ""))[:500]})

    messages.append({
        "role": "user",
        "content": f"Câu hỏi: {question}\n\n[DỮ LIỆU ĐÃ XÁC MINH]\n{verified_context}",
    })

    # Thử model thông minh trước, fallback sang model nhanh
    for model, max_tok in [(GROQ_MODEL_SMART, 600), (GROQ_MODEL_FAST, 400)]:
        try:
            answer = await groq_chat_complete(
                messages=messages,
                model=model,
                max_tokens=max_tok,
                temperature=0.2,
            )
            if answer and len(answer.strip()) > 20:
                return _remove_duplicate_lines(answer.strip())
        except Exception:
            continue

    return ""


async def write_no_result_response(
    question: str,
    constraints: dict,
    mood: str = "normal",
) -> str:
    """
    Sinh câu trả lời khi không tìm được phòng nào phù hợp.
    Gợi ý người dùng điều chỉnh điều kiện cụ thể.
    """
    budget = constraints.get("budget") or {}
    location = constraints.get("location") or {}
    amenities = constraints.get("amenities_required") or []

    # Xây dựng gợi ý điều chỉnh dựa trên điều kiện hiện tại
    suggestions: list[str] = []
    if budget.get("max"):
        suggestions.append(f"Nới ngân sách thêm 20–30% (hiện tối đa {budget['max']:,} VND)")
    if location.get("districts"):
        suggestions.append("Mở rộng tìm kiếm sang các quận lân cận")
    if amenities:
        suggestions.append(f"Bỏ bớt tiện ích bắt buộc: {', '.join(amenities[:2])}")
    if not suggestions:
        suggestions.append("Thử mô tả nhu cầu theo cách khác")

    prompt = (
        f"Người dùng tìm phòng với điều kiện: {question}\n"
        f"Kết quả: Không tìm được phòng phù hợp.\n"
        f"Gợi ý điều chỉnh:\n" + "\n".join(f"- {s}" for s in suggestions) + "\n\n"
        f"Viết câu trả lời thân thiện, đồng cảm và đưa ra gợi ý cụ thể:"
    )
    system = _SYSTEM_PROMPTS.get(mood, _SYSTEM_PROMPTS["normal"])
    try:
        answer = await groq_chat_complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL_FAST,
            max_tokens=300,
            temperature=0.3,
        )
        if answer and len(answer.strip()) > 20:
            return answer.strip()
    except Exception:
        pass

    # Fallback template
    suggestion_str = "; ".join(suggestions)
    return (
        f"Mình chưa tìm được phòng nào khớp với điều kiện hiện tại. "
        f"Bạn thử: {suggestion_str} — rồi mình tìm lại nhé!"
    )
