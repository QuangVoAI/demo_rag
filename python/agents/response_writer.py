"""
Response Writer — Sinh câu trả lời thân thiện, đúng ngữ cảnh cho nhatrovn.

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
from config import ANSWER_MAX_TOKENS
from room_assistant.schemas import RECENT_HISTORY_TURNS

# ---------------------------------------------------------------------------
# System prompts theo từng mood
# ---------------------------------------------------------------------------

from room_assistant.prompts import (
    NO_RESULT_SYSTEM_PROMPTS as _NO_RESULT_SYSTEM_PROMPTS,
    RESPONSE_WRITER_SYSTEM_PROMPTS as _SYSTEM_PROMPTS,
)


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
        verified_context: Dữ liệu room / FAQ đã được grounding.
        history         : Lịch sử hội thoại gần nhất.
        mood            : Cảm xúc người dùng (frustrated/urgent/normal).

    Returns:
        Câu trả lời tiếng Việt.
    """
    system_prompt = _SYSTEM_PROMPTS.get(mood, _SYSTEM_PROMPTS["normal"])

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Thêm lịch sử hội thoại gần nhất
    for turn in (history or [])[-RECENT_HISTORY_TURNS:]:
        role = turn.get("role", "user")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": str(turn.get("content", ""))[:500]})

    messages.append({
        "role": "user",
        "content": f"Câu hỏi: {question}\n\n[DỮ LIỆU ĐÃ XÁC MINH]\n{verified_context}",
    })

    # Thử model thông minh trước, fallback sang model nhanh
    for model, max_tok in [
        (GROQ_MODEL_SMART, min(600, ANSWER_MAX_TOKENS)),
        (GROQ_MODEL_FAST, min(400, ANSWER_MAX_TOKENS)),
    ]:
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

    normalized_question = question.lower()
    if "xa quá" in normalized_question or "xa qua" in normalized_question:
        return (
            "Dạ em hiểu mình ngại đi xa ạ. Hiện khu mình đang chốt hơi căng điều kiện nên chưa còn căn khớp hoàn toàn. "
            "Mình chọn giúp em 1 trong 2 hướng nhé: giữ ngân sách để em lọc khu lân cận gần hơn, hoặc tăng nhẹ ngân sách để lấy căn sát nhu cầu hơn. "
            "Anh/chị ưu tiên gần hơn hay rẻ hơn ạ?"
        )
    if "đắt quá" in normalized_question or "dat qua" in normalized_question:
        return (
            "Dạ em hiểu mình đang cân đối chi phí ạ. Với mức giá hiện tại, khu và tiện ích mình chọn đang hơi khó khớp hoàn toàn. "
            "Em có thể lọc lại theo 2 hướng: giữ khu vực nhưng nới nhẹ ngân sách, hoặc giữ ngân sách và bớt 1 tiện ích bắt buộc. "
            "Anh/chị muốn em đi theo hướng nào để em lọc sát hơn ạ?"
        )

    prompt = (
        f"Người dùng tìm phòng với điều kiện: {question}\n"
        f"Kết quả: Không tìm được phòng phù hợp.\n"
        f"Gợi ý điều chỉnh:\n" + "\n".join(f"- {s}" for s in suggestions) + "\n\n"
        f"Viết câu trả lời thân thiện, đồng cảm và gợi ý cụ thể:"
    )
    system = _NO_RESULT_SYSTEM_PROMPTS.get(mood, _NO_RESULT_SYSTEM_PROMPTS["normal"])
    try:
        answer = await groq_chat_complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL_FAST,
            max_tokens=min(300, ANSWER_MAX_TOKENS),
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
