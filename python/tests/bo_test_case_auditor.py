"""Audit runner for Bo_Test_Case_Chat.pdf scenarios."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from room_assistant.session_store import InMemorySessionStore
from room_assistant.workflow import run_room_assistant

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bo_test_case_chat.json"

NO_RESULT_MARKERS = (
    "tìm mỏi mắt",
    "chưa thấy phòng nào khớp 100%",
    "không tìm thấy phòng phù hợp",
)

SEARCH_HINTS = ("tìm", "tim", "lọc", "loc", "gợi ý", "goi y", "có phòng", "co phong", "có căn", "co can")
COMPARE_HINTS = ("so sánh", "so sanh")
DETAIL_HINTS = (
    "phòng này", "phong nay", "căn này", "can nay", "căn ở trên", "can o tren",
    "phòng kia", "phong kia", "chi tiết", "chi tiet", "giá bao nhiêu", "gia bao nhieu",
    "phòng số", "phong so",
)
COST_HINTS = ("tổng chi phí", "tong chi phi", "tính tổng", "tinh tong", "ở 6 tháng", "o 6 thang", "phụ phí", "phu phi")
ACTION_HINTS = ("đặt phòng", "dat phong", "đặt lịch", "dat lich")
OFF_TOPIC_HINTS = ("viết code", "viet code", "python")
FAQ_HINTS = ("tiền cọc", "tien coc", "quy trình", "quy trinh")

# Multi-turn PDF flows where Gửi: only captures the final user line.
CASE_SESSION_SETUP: dict[str, list[str]] = {
    "TC_CHAT_25": ["tôi muốn tìm phòng quận 7"],
    "TC_CHAT_26": ["tôi muốn tìm phòng quận 9"],
    "TC_CHAT_27": ["tôi muốn tìm phòng quận 9", "Phòng số 2 có tiện ích gì"],
    "TC_CHAT_28": ["tôi muốn tìm phòng quận 9", "Phòng số 2 có tiện ích gì"],
    "TC_CHAT_30": ["tôi muốn tìm phòng quận 9", "Phòng kia thì sao?"],
    "TC_CHAT_33": ["tôi muốn tìm phòng ở gò vấp"],
    "TC_CHAT_35": ["cho tôi phòng ở bình tân"],
    "TC_CHAT_36": ["cho tôi phòng ở bình tân"],
    "TC_CHAT_37": ["cho tôi phòng ở bình tân", "So sánh mấy phòng này giúp tôi"],
    "TC_CHAT_40": ["cho tôi phòng ở bình tân", "So sánh mấy phòng này giúp tôi"],
    "TC_CHAT_43": ["cho tôi phòng ở bình tân", "So sánh mấy phòng này giúp tôi"],
    "TC_CHAT_46": ["cho tôi phòng ở bình tân", "So sánh mấy phòng này giúp tôi"],
    "TC_CHAT_47": ["cho tôi phòng ở bình tân", "So sánh mấy phòng này giúp tôi"],
    "TC_CHAT_63": ["cho tôi phòng ở bình tân", "phòng 646750b02dac6f541f701a7f"],
}


@dataclass
class TurnAudit:
    case_id: str
    turn_index: int
    question: str
    intent: str
    room_count: int
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass
class CaseAudit:
    case_id: str
    skipped: bool = False
    skip_reason: str = ""
    turns: list[TurnAudit] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        return all(turn.passed for turn in self.turns)


def load_cases() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _normalize_turn(text: str) -> str:
    return re.sub(r"[^\w\s#]", "", _norm(text)).strip()


def _extract_user_turns_from_note(note: str) -> list[str]:
    turns: list[str] = []
    for match in re.finditer(r"User:\s*(.+?)(?=\s*Bot:|\s*User:|$)", note, flags=re.IGNORECASE):
        text = re.sub(r"\s+", " ", match.group(1)).strip()
        if text and text not in turns:
            turns.append(text)
    return turns


def _precondition_turns(case: dict[str, Any], sends: list[str]) -> list[str]:
    case_id = case.get("id") or ""
    if case_id in CASE_SESSION_SETUP:
        return list(CASE_SESSION_SETUP[case_id])

    note_turns = _extract_user_turns_from_note(case.get("expected_note") or "")
    if not note_turns or not sends:
        return []

    final_norm = _normalize_turn(sends[-1])
    for index, turn in enumerate(note_turns):
        if _normalize_turn(turn) == final_norm:
            return note_turns[:index]
    if len(note_turns) > 1:
        return note_turns[:-1]
    return []


def _resolve_test_send(case: dict[str, Any], sends: list[str]) -> str:
    if not sends:
        return ""
    question = sends[-1]
    note = case.get("expected_note") or ""
    note_turns = _extract_user_turns_from_note(note)
    final_norm = _normalize_turn(question)

    if "room a" in final_norm and "room b" in final_norm:
        for turn in reversed(note_turns):
            if "so sánh" in _norm(turn) and "#" in turn:
                return turn

    if _is_cost_question(question):
        for turn in reversed(note_turns):
            if _is_cost_question(turn) and re.search(r"[a-f0-9]{24}", turn, re.IGNORECASE):
                return turn

    if "phòng này" in final_norm or "phong nay" in final_norm:
        for turn in reversed(note_turns):
            if re.search(r"phòng\s+[a-f0-9]{24}", turn, re.IGNORECASE):
                return turn

    return question


def _is_search_question(q: str) -> bool:
    n = _norm(q)
    return any(h in n for h in SEARCH_HINTS)


def _is_compare_question(q: str) -> bool:
    return any(h in _norm(q) for h in COMPARE_HINTS)


def _is_detail_question(q: str) -> bool:
    return any(h in _norm(q) for h in DETAIL_HINTS)


def _is_cost_question(q: str) -> bool:
    return any(h in _norm(q) for h in COST_HINTS)


def _is_action_question(q: str) -> bool:
    return any(h in _norm(q) for h in ACTION_HINTS)


def _is_off_topic(q: str) -> bool:
    return any(h in _norm(q) for h in OFF_TOPIC_HINTS)


def _is_faq_question(q: str) -> bool:
    return any(h in _norm(q) for h in FAQ_HINTS)


def _expected_has_rooms(case: dict[str, Any]) -> bool:
    note = _norm(case.get("expected_note") or "")
    if _expected_expects_no_results(case):
        return False
    return "còn phòng" in note or "co phong" in note


def _expected_expects_no_results(case: dict[str, Any]) -> bool:
    note = _norm(case.get("expected_note") or "")
    return any(marker in note for marker in NO_RESULT_MARKERS)


def _abstain_wording_ok(answer: str) -> bool:
    return any(
        phrase in answer
        for phrase in (
            "chưa có dữ liệu",
            "chua co du lieu",
            "chưa đủ dữ liệu",
            "chua du du lieu",
            "chưa có thông tin",
            "chua co thong tin",
        )
    )


def audit_turn(case_id: str, turn_index: int, question: str, result: dict[str, Any], *, case: dict[str, Any]) -> TurnAudit:
    issues: list[str] = []
    answer = _norm(result.get("answer") or "")
    rooms = result.get("rooms") or []
    intent = str(result.get("intent") or "")

    if _is_off_topic(question):
        if intent != "GENERAL_HELP":
            issues.append(f"expected GENERAL_HELP, got {intent}")
        if "phòng trọ" not in answer and "phong tro" not in answer:
            issues.append("off-topic answer should mention phòng trọ")
    elif _is_action_question(question):
        if intent != "REQUEST_ACTION":
            issues.append(f"expected REQUEST_ACTION, got {intent}")
    elif _is_faq_question(question) and not _is_detail_question(question) and not _is_search_question(question):
        if intent != "REQUEST_FAQ":
            issues.append(f"expected REQUEST_FAQ, got {intent}")
    elif _is_compare_question(question):
        if intent != "COMPARE_ROOMS":
            issues.append(f"expected COMPARE_ROOMS, got {intent}")
        rows = (result.get("comparison") or {}).get("rows") or []
        if not rows and not any(token in answer for token in ("chưa đủ", "bảng so sánh", "bang so sanh")):
            issues.append("compare expected comparison rows or explicit unresolved message")
    elif _is_cost_question(question):
        if intent not in {"CALCULATE_COST", "ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
            issues.append(f"expected cost/detail intent, got {intent}")
    elif _is_detail_question(question):
        if intent not in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"}:
            issues.append(f"expected detail intent, got {intent}")
    elif _is_search_question(question):
        if intent not in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
            issues.append(f"expected search intent, got {intent}")
        if _expected_has_rooms(case):
            if not rooms and any(marker in answer for marker in NO_RESULT_MARKERS):
                issues.append("search returned no-result template but PDF expects rooms")
            elif not rooms and "sales" not in answer and "tư vấn" not in answer and "tu van" not in answer:
                issues.append("search returned empty rooms without sales handoff")
        elif _expected_expects_no_results(case):
            if rooms:
                issues.append("search returned rooms but PDF expects no-result handoff")
            elif not any(marker in answer for marker in NO_RESULT_MARKERS) and "sales" not in answer:
                issues.append("search should return no-result or sales handoff")

    if result.get("abstain") and not _abstain_wording_ok(answer):
        issues.append("abstain=true but answer missing abstain wording")

    return TurnAudit(
        case_id=case_id,
        turn_index=turn_index,
        question=question,
        intent=intent,
        room_count=len(rooms),
        issues=issues,
    )


def _run_turn(
    question: str,
    *,
    history: list[dict[str, str]],
    session_id: str,
    repository: Any,
    store: InMemorySessionStore,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    if asyncio.iscoroutinefunction(runner):
        return asyncio.run(
            runner(
                question,
                history=history,
                session_id=session_id,
                repository=repository,
                session_store=store,
                semantic_index=None,
            )
        )
    return runner(
        question,
        history=history,
        session_id=session_id,
        repository=repository,
        session_store=store,
        semantic_index=None,
    )


def run_case(
    case: dict[str, Any],
    *,
    repository: Any,
    session_id: str | None = None,
    runner: Callable[..., Any] | None = None,
) -> CaseAudit:
    sends = case.get("sends") or []
    if not sends:
        return CaseAudit(case_id=case["id"], skipped=True, skip_reason="ui_only_no_message")

    session_id = session_id or f"bo-{case['id']}"
    store = InMemorySessionStore()
    history: list[dict[str, str]] = []
    turns: list[TurnAudit] = []
    runner = runner or run_room_assistant

    setup_turns = _precondition_turns(case, sends)
    test_question = _resolve_test_send(case, sends)

    for question in setup_turns:
        result = _run_turn(
            question,
            history=history,
            session_id=session_id,
            repository=repository,
            store=store,
            runner=runner,
        )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.get("answer") or ""})

    result = _run_turn(
        test_question,
        history=history,
        session_id=session_id,
        repository=repository,
        store=store,
        runner=runner,
    )
    turns.append(audit_turn(case["id"], 0, test_question, result, case=case))

    return CaseAudit(case_id=case["id"], turns=turns)


def run_all_cases(repository: Any, *, runner: Callable[..., Any] | None = None) -> list[CaseAudit]:
    return [run_case(case, repository=repository, runner=runner) for case in load_cases()]


def summarize(audits: list[CaseAudit]) -> dict[str, Any]:
    executed = [item for item in audits if not item.skipped]
    failed = [item for item in executed if not item.passed]
    skipped = [item for item in audits if item.skipped]
    return {
        "total": len(audits),
        "executed": len(executed),
        "skipped_ui": len(skipped),
        "passed": len(executed) - len(failed),
        "failed": len(failed),
        "failures": [
            {
                "case_id": item.case_id,
                "issues": [
                    {"turn": turn.turn_index, "question": turn.question, "issues": turn.issues}
                    for turn in item.turns
                    if turn.issues
                ],
            }
            for item in failed
        ],
    }
