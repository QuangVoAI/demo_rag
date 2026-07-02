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

from agents.prompts import REVIEWER_SYSTEM_PROMPT as _REVIEWER_SYSTEM_PROMPT


def _get_llm_client():
    from agents.llm_client import GROQ_MODEL_FAST, groq_complete

    return {
        "fast_model": GROQ_MODEL_FAST,
        "complete": groq_complete,
    }


def _check_banned_phrases(answer: str) -> list[str]:
    """Kiểm tra nhanh các cụm từ bị cấm, không cần LLM."""
    answer_lower = answer.lower()
    return [phrase for phrase in _BANNED_PHRASES if phrase in answer_lower]


def review_rules_only(answer: str) -> dict:
    """Kiểm tra deterministic, không gọi LLM."""
    banned = _check_banned_phrases(answer)
    if banned:
        return {
            "is_approved": False,
            "issues": [f"Vi phạm: '{phrase}'" for phrase in banned],
            "suggestion": "Bỏ lời hứa thao tác; chỉ tư vấn và đọc dữ liệu.",
        }
    return {"is_approved": True, "issues": [], "suggestion": ""}


def needs_review(question: str, answer: str) -> bool:
    """Xác định có cần LLM review không (tránh tốn token không cần thiết)."""
    combined = (question + " " + answer).lower()
    return any(kw in combined for kw in _REVIEW_TRIGGERS)


async def review(question: str, answer: str, room_context: str = "") -> dict:
    """
    Kiểm duyệt câu trả lời.

    Returns:
        dict với is_approved, issues, suggestion.
    """
    # Tầng 1: Rule-based check
    rule_result = review_rules_only(answer)
    if not rule_result["is_approved"]:
        return rule_result

    # Nếu không trigger → approve ngay (tiết kiệm token)
    if not needs_review(question, answer):
        return {"is_approved": True, "issues": [], "suggestion": ""}

    # Tầng 2: LLM review
    prompt = (
        f"Câu hỏi người dùng: {question}\n\n"
        f"Câu trả lời của assistant:\n{answer}\n\n"
        f"Dữ liệu đã xác minh (nếu có):\n{room_context[:1500]}\n\n"
        f"Kiểm tra và trả về JSON:"
    )
    llm = _get_llm_client()
    try:
        raw = await llm["complete"](
            prompt=prompt,
            system_prompt=_REVIEWER_SYSTEM_PROMPT,
            model=llm["fast_model"],
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
            if "is_approved" in result:
                return {
                    "is_approved": bool(result.get("is_approved", True)),
                    "issues": list(result.get("issues", [])),
                    "suggestion": str(result.get("suggestion", "")),
                }
            safe = result.get("safe")
            if safe is None and "violation_type" in result:
                violation = str(result.get("violation_type") or "none").lower()
                safe = violation in {"", "none"}
            feedback = str(result.get("feedback") or result.get("suggestion") or "")
            corrected = result.get("corrected_answer")
            issues = [feedback] if feedback and not safe else []
            if violation := str(result.get("violation_type") or ""):
                if violation.lower() not in {"", "none"}:
                    issues.insert(0, violation)
            return {
                "is_approved": bool(safe) if safe is not None else True,
                "issues": issues,
                "suggestion": feedback,
                "corrected_answer": corrected,
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {"is_approved": True, "issues": [], "suggestion": ""}


async def review_with_retry(
    question: str,
    answer: str,
    room_context: str = "",
    max_retries: int = 1,
) -> tuple[str, dict]:
    """
    Review and optionally replace the draft answer before it is streamed to users.

    Policy: when the reviewer returns ``corrected_answer``, that text replaces the
    draft and is treated as approved. Otherwise rejected answers may trigger one
    rewrite retry. Streaming happens only after this function returns.
    """
    current_answer = answer
    result = {"is_approved": True, "issues": [], "suggestion": ""}
    used_corrected_answer = False

    for attempt in range(max_retries + 1):
        result = await review(question, current_answer, room_context)

        if result["is_approved"] or attempt >= max_retries:
            break

        corrected = result.get("corrected_answer")
        if corrected and str(corrected).strip().lower() not in {"", "null", "none"}:
            current_answer = str(corrected).strip()
            result["is_approved"] = True
            used_corrected_answer = True
            break

        console.print(f"[yellow]  Reviewer retry #{attempt + 1}: {result['issues']}[/]")

        # Thử sửa lại câu trả lời dựa trên gợi ý
        issues_str = "; ".join(result["issues"])
        retry_prompt = (
            f"Câu trả lời bị lỗi: {issues_str}\n"
            f"Câu hỏi gốc: {question}\n"
            f"Dữ liệu xác minh: {room_context[:1500]}\n\n"
            f"Viết lại câu trả lời tự nhiên, đúng sự thật, không vi phạm:"
        )
        llm = _get_llm_client()
        try:
            current_answer = await llm["complete"](
                prompt=retry_prompt,
                system_prompt="Bạn là trợ lý tìm phòng nhatrovn. Trả lời ngắn gọn, trung thực.",
                model=llm["fast_model"],
                max_tokens=400,
                temperature=0.2,
            )
        except Exception:
            break  # Giữ answer cũ nếu retry lỗi

    result["retry_count"] = max_retries
    result["used_corrected_answer"] = used_corrected_answer
    return current_answer, result

import re
from typing import Any

def should_abstain(
    question: str,
    intent: str,
    grounding: dict[str, Any],
    tool_results: dict[str, Any],
    answer: str,
) -> tuple[bool, str]:
    """Lightweight faithfulness gate without Ragas/DeepEval runtime dependency."""
    try:
        from config import ENABLE_ABSTAIN
    except Exception:
        ENABLE_ABSTAIN = True
    if not ENABLE_ABSTAIN:
        return False, ""

    rooms = grounding.get("rooms") or []
    constraints = grounding.get("constraints") or {}
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"} and rooms:
        from room_assistant.tools import SufficiencyStatus, evaluate_room_data_sufficiency

        status, _ = evaluate_room_data_sufficiency(
            question,
            rooms[0],
            intent=intent,
            constraints=constraints,
        )
        if status == SufficiencyStatus.INSUFFICIENT:
            return True, "insufficient_verified_data"

    estimate = tool_results.get("cost_estimate") or {}
    q_lower = (question or "").lower()
    factual_intents = {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST", "COMPARE_ROOMS"}
    needs_verified_facts = any(token in q_lower for token in (
        "cọc", "tiền", "giá", "phí", "bao nhiêu", "tổng", "hợp đồng", "cam kết",
    ))

    if intent == "CALCULATE_COST" and not estimate.get("available"):
        return True, "missing_cost_estimate"
    if intent == "CALCULATE_COST" and estimate.get("available"):
        return False, ""
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"} and not rooms:
        return True, "missing_room_context"
    if intent in factual_intents and needs_verified_facts and not rooms and not estimate.get("available"):
        return True, "missing_verified_facts"

    claims: dict[str, Any] = {}
    if isinstance(estimate, dict) and estimate.get("available"):
        claims = {
            "fixed_items": estimate.get("items") or [],
            "initial_payment_options": [{"deposit": estimate.get("total_initial_cost")}],
        }
    elif rooms:
        claims = {
            "fixed_items": [
                {"amount": room.get("rent_price")}
                for room in rooms
                if room.get("rent_price")
            ],
        }

    if not claims.get("fixed_items") and not claims.get("initial_payment_options"):
        return False, ""

    issues = deterministic_claim_validator(answer, claims)
    if issues:
        return True, "unverified_claims"

    return False, ""


ABSTAIN_USER_MESSAGE = (
    "Dạ em chưa đủ dữ liệu xác minh để trả lời chính xác câu này ạ. "
    "Anh/chị có thể xem chi tiết phòng bên dưới hoặc nhắn thêm khu vực / ngân sách để em lọc lại giúp mình nha."
)

SAFE_NUMBER_PATTERNS = [
    r"\b\d+(?:[.,]\d+)?\s*(?:triệu|trieu|tr|k|nghìn|nghin|đ|d|vnd|vnđ)\b",
]

def deterministic_claim_validator(answer: str, claims: dict[str, Any]) -> list[str]:
    """
    Xác minh claims bằng Regex thay vì LLM.
    Nếu answer đề cập đến một con số có vẻ là giá/cọc mà không khớp với claims đã duyệt, trả về lỗi.
    """
    issues = []
    
    # Tìm tất cả số tiền được nhắc đến trong answer
    found_money = []
    for pattern in SAFE_NUMBER_PATTERNS:
        for match in re.finditer(pattern, answer.lower()):
            found_money.append(match.group(0))
            
    # Lấy danh sách số tiền hợp lệ từ claims (fixed_items, deposit)
    valid_amounts = set()
    for item in claims.get("fixed_items", []):
        amt = item.get("amount")
        if amt:
            valid_amounts.add(str(amt))
            # Cũng thêm dạng format, vd: 4.5
            if amt >= 1000000:
                valid_amounts.add(str(round(amt / 1000000.0, 2)).rstrip('0').rstrip('.'))
            
    for opt in claims.get("initial_payment_options", []):
        dep = opt.get("deposit")
        if dep:
            valid_amounts.add(str(dep))
            if dep >= 1000000:
                valid_amounts.add(str(round(dep / 1000000.0, 2)).rstrip('0').rstrip('.'))
                
    # Logic kiểm tra: Đảm bảo answer không nhắc đến số tiền lạ (rất đơn giản)
    # Tuy nhiên vì ngôn ngữ tự nhiên rất đa dạng, ta chỉ flag cảnh báo nếu thấy số lạ hoàn toàn
    for money_text in found_money:
        # Nếu có số liệu, ta extract digits
        digits = re.sub(r"\D", "", money_text)
        if not digits:
            continue
        # Simplistic check
        is_safe = False
        for valid in valid_amounts:
            if valid in digits or digits in valid:
                is_safe = True
                break
            # Nếu valid là '45' (4.5tr) và digits chứa '45'
            if valid.replace('.', '') in digits:
                is_safe = True
                break
                
        if not is_safe:
            issues.append(f"Số tiền '{money_text}' có thể không chính xác so với dữ liệu xác minh.")
            
    return issues

