"""
Response Writer — Sinh câu trả lời thân thiện, đúng ngữ cảnh cho nhatrovn.

Điều chỉnh giọng điệu theo cảm xúc người dùng (mood):
  - frustrated : Đồng cảm, gợi ý thay đổi điều kiện tìm kiếm
  - urgent     : Nhanh chóng, ưu tiên kết quả còn phòng ngay
  - normal     : Thông tin đầy đủ, thân thiện

Dual backend: Groq SMART (primary) → Groq FAST (fallback)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Awaitable, Callable

sys.path.append(str(Path(__file__).parent.parent))

from config import ANSWER_MAX_TOKENS

from agents.prompts import RESPONSE_WRITER_SYSTEM_PROMPTS as _SYSTEM_PROMPTS


def _get_llm_client():
    from agents.llm_client import (
        GROQ_MODEL_FAST,
        GROQ_MODEL_SMART,
        groq_chat_complete,
        groq_stream_chat_complete,
    )

    return {
        "fast_model": GROQ_MODEL_FAST,
        "smart_model": GROQ_MODEL_SMART,
        "chat_complete": groq_chat_complete,
        "stream_chat_complete": groq_stream_chat_complete,
    }


def _remove_duplicate_lines(text: str) -> str:
    """Loại bỏ các dòng / đoạn lặp liên tiếp mà LLM đôi khi sinh ra."""
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    seen: list[str] = []
    for line in lines:
        if not any(line == s or (len(line) > 20 and line in s) for s in seen[-3:]):
            seen.append(line)
    return "\n".join(seen)


def _build_messages(
    question: str,
    verified_context: str,
    history: list[dict] | None = None,
    mood: str = "normal",
) -> list[dict]:
    system_prompt = _SYSTEM_PROMPTS.get(mood, _SYSTEM_PROMPTS["normal"])
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for turn in (history or [])[-6:]:
        role = turn.get("role", "user")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": str(turn.get("content", ""))[:500]})

    messages.append({
        "role": "user",
        "content": f"Câu hỏi: {question}\n\n[DỮ LIỆU ĐÃ XÁC MINH]\n{verified_context}",
    })
    return messages


async def write_response(
    question: str,
    verified_context: str,
    history: list[dict] | None = None,
    mood: str = "normal",
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """
    Sinh câu trả lời dựa trên dữ liệu đã xác minh.

    Args:
        question        : Câu hỏi hiện tại của người dùng.
        verified_context: Dữ liệu room / FAQ đã được grounding.
        history         : Lịch sử hội thoại gần nhất (tối đa 6 lượt).
        mood            : Cảm xúc người dùng (frustrated/urgent/normal).
        stream_callback : Callback stream token (nếu có).

    Returns:
        Câu trả lời tiếng Việt.
    """
    if stream_callback is not None:
        return await stream_response(
            question=question,
            verified_context=verified_context,
            history=history,
            mood=mood,
            stream_callback=stream_callback,
        )
    messages = _build_messages(question, verified_context, history=history, mood=mood)
    llm = _get_llm_client()

    # Thử model thông minh trước, fallback sang model nhanh
    for model, max_tok in [
        (llm["smart_model"], min(600, ANSWER_MAX_TOKENS)),
        (llm["fast_model"], min(400, ANSWER_MAX_TOKENS)),
    ]:
        try:
            answer = await llm["chat_complete"](
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


async def stream_response(
    question: str,
    verified_context: str,
    history: list[dict] | None = None,
    mood: str = "normal",
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Stream grounded draft tokens while collecting the full answer."""
    messages = _build_messages(question, verified_context, history=history, mood=mood)
    llm = _get_llm_client()

    for model, max_tok in [
        (llm["smart_model"], min(600, ANSWER_MAX_TOKENS)),
        (llm["fast_model"], min(400, ANSWER_MAX_TOKENS)),
    ]:
        collected: list[str] = []
        try:
            async for token in llm["stream_chat_complete"](
                messages=messages,
                model=model,
                max_tokens=max_tok,
                temperature=0.2,
            ):
                if token:
                    collected.append(token)
                    if stream_callback is not None:
                        await stream_callback(token)
            answer = _remove_duplicate_lines("".join(collected).strip())
            if len(answer) > 20:
                return answer
        except Exception:
            continue

    return await write_response(
        question=question,
        verified_context=verified_context,
        history=history,
        mood=mood,
    )


async def write_no_result_response(
    question: str,
    constraints: dict,
    mood: str = "normal",
    alt_rooms: list[dict] | None = None,
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
    *,
    sales_handoff: bool = False,
) -> str:
    """
    Sinh câu trả lời khi không tìm được phòng nào phù hợp.
    Gợi ý người dùng điều chỉnh điều kiện cụ thể hoặc đề xuất phòng lân cận.
    """
    if sales_handoff and not alt_rooms:
        from room_assistant.prompts import SEARCH_NO_RESULT_SALES_HANDOFF
        answer = SEARCH_NO_RESULT_SALES_HANDOFF.get(mood, SEARCH_NO_RESULT_SALES_HANDOFF["normal"])
        if stream_callback:
            await stream_callback(answer)
        return answer

    llm = _get_llm_client()
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
        f"Kết quả: KHÔNG TÌM ĐƯỢC PHÒNG PHÙ HỢP với các điều kiện khắt khe (ví dụ: mức giá thấp hơn thị trường).\n"
    )
    
    if alt_rooms:
        prompt += "\nDANH SÁCH CÁC PHÒNG LÂN CẬN (Mức giá / Tiện ích khác một chút) ĐỂ ĐỀ XUẤT:\n"
        for i, r in enumerate(alt_rooms):
            price_str = f"{r.get('rent_price'):,} VND" if isinstance(r.get("rent_price"), (int, float)) else str(r.get("rent_price", ""))
            addr = r.get("address", "")
            title = r.get("title", "")
            prompt += f"{i+1}. {title} - {price_str} ({addr})\n"
        prompt += "\nLệnh: Hãy viết một câu trả lời ĐỒNG CẢM, xin lỗi vì hết phòng/không có phòng theo yêu cầu. Sau đó khéo léo ĐỀ XUẤT các phòng lân cận ở trên. Phải format markdown danh sách các phòng rõ ràng. (KHÔNG tạo thêm phòng giả, chỉ dùng các phòng trong danh sách trên)"
    else:
        prompt += (
            f"Gợi ý điều chỉnh:\n" + "\n".join(f"- {s}" for s in suggestions) + "\n\n"
            f"Viết câu trả lời thân thiện, ĐỒNG CẢM (thấu cảm với khó khăn của khách khi tìm phòng khó), xin lỗi vì không có phòng phù hợp và gợi ý cụ thể để khách thay đổi điều kiện:"
        )

    system = _SYSTEM_PROMPTS.get(mood, _SYSTEM_PROMPTS["normal"])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    collected = []
    try:
        async for token in llm["stream_chat_complete"](
            messages=messages,
            model=llm["fast_model"],
            max_tokens=min(400, ANSWER_MAX_TOKENS),
            temperature=0.3,
        ):
            if token:
                collected.append(token)
                if stream_callback is not None:
                    await stream_callback(token)
        answer = "".join(collected)
        if answer and len(answer.strip()) > 20:
            return answer.strip()
    except Exception:
        pass

    # Fallback template
    suggestion_str = "; ".join(suggestions)
    return (
        f"Dạ em rà soát kỹ lắm rồi mà chưa tìm được căn nào khớp 100% điều kiện của mình ạ. "
        f"Anh/chị thử: {suggestion_str} — rồi nhắn lại để em tìm căn đẹp nhất cho mình nha!"
    )
