"""
Reviewer Agent — Kiểm duyệt chất lượng câu trả lời của Nhatrovn Assistant.

Hai tầng kiểm tra:
  1. Rule-based nhanh (0 token): phát hiện vi phạm cứng (hứa hẹn sai, hallucinate)
  2. LLM verify (Groq FAST): kiểm tra tính chính xác và tự nhiên

Chỉ trigger LLM review khi câu trả lời liên quan đến giá, cọc, hợp đồng
hoặc có dấu hiệu hallucination.
"""
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.llm_client import groq_complete, GROQ_MODEL_FAST
from utils.console import console

# Các cụm từ bị cấm — assistant không được hứa hẹn thao tác mình không làm được
_BANNED_PHRASES = [
    "mình sẽ đặt lịch",
    "mình sẽ nhắn chủ",
    "mình sẽ giữ phòng",
    "mình sẽ thanh toán",
    "mình sẽ liên hệ chủ nhà",
    "đã đặt lịch cho bạn",
    "đã giữ phòng",
    "đã thanh toán",
    "chúng tôi cam kết",
    "100% phù hợp",
    "chắc chắn còn phòng",
]

# Từ khóa trigger LLM review (rủi ro sai thông tin cao)
_REVIEW_TRIGGERS = [
    "tiền cọc", "tong chi phi", "tổng chi phí", "hợp đồng", "phí",
    "bao gồm", "tổng cộng", "cam kết", "đảm bảo", "chắc chắn",
]

_REVIEWER_SYSTEM_PROMPT = """\
Bạn là người kiểm duyệt câu trả lời cho chatbot tìm phòng trọ nhatro.vn.

Kiểm tra câu trả lời theo 4 tiêu chí:
1. KHÔNG hứa hẹn thao tác bot không làm được (đặt lịch, nhắn chủ, giữ phòng, thanh toán)
2. KHÔNG đưa số tiền/thông tin không có trong dữ liệu đã xác minh
3. Ngôn ngữ tự nhiên, thân thiện, không máy móc
4. Nếu không có dữ liệu thì thành thật nói "chưa có dữ liệu"

Trả về JSON (không giải thích thêm):
{"is_approved": true/false, "issues": ["lỗi 1", "lỗi 2"], "suggestion": "gợi ý sửa ngắn gọn"}
"""


def _check_banned_phrases(answer: str) -> list[str]:
    """Kiểm tra nhanh các cụm từ bị cấm, không cần LLM."""
    answer_lower = answer.lower()
    return [phrase for phrase in _BANNED_PHRASES if phrase in answer_lower]


def needs_review(question: str, answer: str) -> bool:
    """Xác định có cần LLM review không (tránh tốn token không cần thiết)."""
    combined = (question + " " + answer).lower()
    return any(kw in combined for kw in _REVIEW_TRIGGERS)


async def review(question: str, answer: str, listing_context: str = "") -> dict:
    """
    Kiểm duyệt câu trả lời.

    Returns:
        dict với is_approved, issues, suggestion.
    """
    # Tầng 1: Rule-based check
    banned = _check_banned_phrases(answer)
    if banned:
        return {
            "is_approved": False,
            "issues": [f"Vi phạm: '{phrase}'" for phrase in banned],
            "suggestion": "Bỏ lời hứa thao tác; chỉ tư vấn và đọc dữ liệu.",
        }

    # Nếu không trigger → approve ngay (tiết kiệm token)
    if not needs_review(question, answer):
        return {"is_approved": True, "issues": [], "suggestion": ""}

    # Tầng 2: LLM review
    prompt = (
        f"Câu hỏi người dùng: {question}\n\n"
        f"Câu trả lời của assistant:\n{answer}\n\n"
        f"Dữ liệu đã xác minh (nếu có):\n{listing_context[:1500]}\n\n"
        f"Kiểm tra và trả về JSON:"
    )
    try:
        raw = await groq_complete(
            prompt=prompt,
            system_prompt=_REVIEWER_SYSTEM_PROMPT,
            model=GROQ_MODEL_FAST,
            max_tokens=200,
            temperature=0.0,
        )
        return _parse_result(raw)
    except Exception:
        # Lỗi LLM reviewer → approve để không chặn trả lời
        return {"is_approved": True, "issues": [], "suggestion": ""}


def _parse_result(response: str) -> dict:
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
            return {
                "is_approved": bool(result.get("is_approved", True)),
                "issues": list(result.get("issues", [])),
                "suggestion": str(result.get("suggestion", "")),
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {"is_approved": True, "issues": [], "suggestion": ""}


async def review_with_retry(
    question: str,
    answer: str,
    listing_context: str = "",
    max_retries: int = 1,
) -> tuple[str, dict]:
    """
    Review + tự sửa nếu phát hiện vi phạm.

    Giới hạn max_retries để tránh tốn quá nhiều token.
    """
    current_answer = answer
    result = {"is_approved": True, "issues": [], "suggestion": ""}

    for attempt in range(max_retries + 1):
        result = await review(question, current_answer, listing_context)

        if result["is_approved"] or attempt >= max_retries:
            break

        console.print(f"[yellow]  Reviewer retry #{attempt + 1}: {result['issues']}[/]")

        # Thử sửa lại câu trả lời dựa trên gợi ý
        issues_str = "; ".join(result["issues"])
        retry_prompt = (
            f"Câu trả lời bị lỗi: {issues_str}\n"
            f"Câu hỏi gốc: {question}\n"
            f"Dữ liệu xác minh: {listing_context[:1500]}\n\n"
            f"Viết lại câu trả lời tự nhiên, đúng sự thật, không vi phạm:"
        )
        try:
            current_answer = await groq_complete(
                prompt=retry_prompt,
                system_prompt="Bạn là trợ lý tìm phòng nhatro.vn. Trả lời ngắn gọn, trung thực.",
                model=GROQ_MODEL_FAST,
                max_tokens=400,
                temperature=0.2,
            )
        except Exception:
            break  # Giữ answer cũ nếu retry lỗi

    result["retry_count"] = max_retries
    return current_answer, result
