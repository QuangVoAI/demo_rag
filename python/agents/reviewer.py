"""
Reviewer Agent — Empathy Quality Checker.
Kiểm tra phản hồi có THỰC SỰ thấu cảm không, có văn mẫu bị cấm không.
"""
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.llm_client import groq_complete, vertex_custom_complete, GROQ_MODEL_FAST
from config import EMPATHY_MODE
from utils.console import console

REVIEWER_SYSTEM_PROMPT = """\
You are an Empathy Quality Reviewer for a Vietnamese customer service AI.

TASK: Check if the AI response meets these criteria:
1. GENUINE EMPATHY — NOT robotic "We apologize for the inconvenience"
2. NO BANNED PHRASES — Must NOT contain: "Chung toi rat tiec", "Theo chinh sach", "Xin loi vi su bat tien", "Xin quy khach vui long"
3. SPECIFIC COMPENSATION — If policy allows, response must suggest specific compensation (voucher amount, refund %)
4. OPEN-ENDED QUESTION — Response should end with an open question for the customer
5. NO BLAME — Must NOT blame the customer or make excuses
6. NATURAL TONE — Should sound like a real person texting, not a corporate robot

RESPOND in JSON:
{
    "is_approved": true/false,
    "issues": ["Issue 1", "Issue 2"],
    "suggestion": "Brief suggestion if issues found"
}
"""

BANNED_PHRASES = [
    "chúng tôi rất tiếc",
    "theo chính sách",
    "xin lỗi vì sự bất tiện",
    "xin quý khách vui lòng",
    "chúng tôi sẽ chuyển",
    "vui lòng chờ",
    "hệ thống đang xử lý",
    "cảm ơn quý khách đã thông báo",
]

MAX_REVIEW_RETRIES = 1  # Keep low — each Vertex retry costs ~30s

REVIEW_TRIGGER_KEYWORDS = [
    "bồi thường", "hoàn tiền", "voucher", "đền bù",
    "lừa đảo", "ăn cướp", "kiện", "report",
    "bức xúc", "tức giận", "phẫn nộ",
    "toxic", "frustrated",
]


def needs_review(question):
    """Check xem response có cần review không."""
    q = question.lower()
    return any(kw in q for kw in REVIEW_TRIGGER_KEYWORDS)


def _check_banned_phrases(answer):
    """Quick check banned phrases (no LLM needed)."""
    answer_lower = answer.lower()
    found = [p for p in BANNED_PHRASES if p in answer_lower]
    return found


async def review(question, answer, evidence):
    """Review phản hồi."""
    # Quick ban check first
    banned = _check_banned_phrases(answer)
    if banned:
        return {
            "is_approved": False,
            "issues": [f"Sử dụng văn mẫu bị cấm: '{p}'" for p in banned],
            "suggestion": "Viết lại thấu cảm hơn, không dùng văn mẫu.",
        }

    # LLM review
    user_prompt = (
        f"CUSTOMER MESSAGE: {question}\n\n"
        f"AI RESPONSE TO CHECK:\n{answer}\n\n"
        f"POLICY CONTEXT:\n{evidence[:2000]}\n\n"
        f"Check and respond in JSON:"
    )
    messages = [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Always use Groq for review — it's fast and just needs to output JSON
    response = await groq_complete(
        prompt=user_prompt,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        model=GROQ_MODEL_FAST,
        max_tokens=256,
        temperature=0.0,
    )

    return _parse_result(response)


def _parse_result(response):
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
            return {
                "is_approved": result.get("is_approved", True),
                "issues": result.get("issues", []),
                "suggestion": result.get("suggestion", ""),
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {"is_approved": True, "issues": [], "suggestion": ""}


async def review_with_retry(question, answer, evidence):
    """Review + retry nếu fail."""
    current_answer = answer
    retry_count = 0

    for attempt in range(MAX_REVIEW_RETRIES + 1):
        result = await review(question, current_answer, evidence)

        if result["is_approved"] or attempt >= MAX_REVIEW_RETRIES:
            result["retry_count"] = retry_count
            return current_answer, result

        console.print(f"[yellow]  Reviewer retry #{attempt+1}: {result['issues']}[/]")

        issues_str = "; ".join(result["issues"])
        retry_prompt = (
            f"Phản hồi trước bị lỗi: {issues_str}\n\n"
            f"KHACH HANG: {question}\n\n"
            f"CHINH SACH: {evidence[:2000]}\n\n"
            f"Viết lại phản hồi thấu cảm, tránh các lỗi trên. "
            f"KHÔNG dùng văn mẫu, phải tự nhiên như người thật nhắn tin."
        )

        retry_messages = [
            {"role": "system", "content": "Bạn là EmpathAI. Viết phản hồi thấu cảm, tự nhiên."},
            {"role": "user", "content": retry_prompt},
        ]
        
        # Always use Groq for retries — fast, no 30s Vertex latency
        try:
            current_answer = await groq_complete(
                prompt=retry_prompt,
                system_prompt="Bạn là EmpathAI. Viết phản hồi thấu cảm, tự nhiên, ngắn gọn.",
                model=GROQ_MODEL_FAST,
                max_tokens=400,
                temperature=0.7,
            )
        except Exception as e:
            print(f"Groq retry error: {e}")
            break  # Give up and return current_answer
        retry_count += 1

    return current_answer, result
