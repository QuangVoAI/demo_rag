"""Context Memory Audit v2 — real LLM, answer assertions, API paths (CTX-11/15)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
TESTS_ROOT = PYTHON_ROOT / "tests"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(TESTS_ROOT))
sys.path.insert(0, str(PYTHON_ROOT / "scripts"))

from django.conf import settings
from django.test import Client, override_settings

from apps.rooms.services import get_collection
from mongo_test_helpers import (
    create_mongo_repository,
    create_semantic_index,
    load_dotenv,
    mongo_is_configured,
    qdrant_is_available,
)
from room_assistant.repository import MongoRoomRepository
from room_assistant.session_store import InMemorySessionStore
from room_assistant.workflow import run_room_assistant

import context_memory_audit as v1

NO_RESULT_MARKERS = (
    "tìm mỏi mắt",
    "chưa thấy phòng nào khớp 100%",
    "không tìm thấy phòng phù hợp",
)
NO_INFO_MARKERS = (
    "không có thông tin",
    "khong co thong tin",
    "chưa có thông tin",
    "không rõ thông tin",
)
GENERIC_FAIL_MARKERS = (
    "xin lỗi, tôi gặp sự cố",
    "lỗi hệ thống",
    "rag service unavailable",
)


@dataclass
class AnswerCheck:
    turn: int
    ok: bool
    detail: str


@dataclass
class V2CaseResult:
    case_id: str
    priority: str
    status: str  # PASS | FAIL | PARTIAL | SKIP
    memory: v1.CaseResult
    answer_checks: list[AnswerCheck] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_answer(self, turn: int, ok: bool, detail: str) -> None:
        self.answer_checks.append(AnswerCheck(turn, ok, detail))

    def finalize(self) -> None:
        mem_fail = any(not c.ok for c in self.memory.checks)
        ans_fail = any(not c.ok for c in self.answer_checks)
        if self.memory.status == "SKIP":
            self.status = "SKIP"
        elif mem_fail and ans_fail:
            self.status = "FAIL"
        elif mem_fail or ans_fail:
            self.status = "PARTIAL"
        elif self.memory.status == "PARTIAL":
            self.status = "PARTIAL"
        else:
            self.status = "PASS"


CASE_TRANSCRIPTS: dict[str, list[dict[str, Any]]] = {}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _answer_snippet(answer: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", (answer or "").strip())
    return compact[:limit] + ("…" if len(compact) > limit else "")


def _turn_transcript(question: str, result: dict[str, Any]) -> dict[str, Any]:
    state = result.get("session_state") or {}
    rooms = result.get("rooms") or []
    return {
        "question": question,
        "answer": result.get("answer") or "",
        "intent": result.get("intent") or "",
        "rooms_count": len(rooms),
        "room_ids": [r.get("room_id") for r in rooms[:8]],
        "session_state": {
            "constraints": (state.get("constraints") or {}),
            "current_room_id": state.get("current_room_id"),
            "last_result_ids": state.get("last_result_ids") or [],
            "selected_room_ids": state.get("selected_room_ids") or [],
        },
        "abstain": bool(result.get("abstain")),
        "error": result.get("error"),
    }


def _run_turns_real(
    repo: MongoRoomRepository,
    questions: list[str],
    session_id: str,
    *,
    use_qdrant: bool = True,
) -> list[dict[str, Any]]:
    store = InMemorySessionStore()
    history: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    semantic_index = create_semantic_index() if use_qdrant and qdrant_is_available() else None
    transcript: list[dict[str, Any]] = []

    for question in questions:
        result = asyncio.run(
            run_room_assistant(
                question,
                history=history,
                session_id=session_id,
                repository=repo,
                session_store=store,
                semantic_index=semantic_index,
            )
        )
        results.append(result)
        transcript.append(_turn_transcript(question, result))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.get("answer") or ""})

    CASE_TRANSCRIPTS[session_id] = transcript
    return results


v1._run_turns = _run_turns_real


def _failures_generic_answer(result: dict[str, Any]) -> list[str]:
    answer = _norm(result.get("answer") or "")
    failures: list[str] = []
    if len(answer) < 20:
        failures.append("answer too short")
    if any(m in answer for m in GENERIC_FAIL_MARKERS):
        failures.append("generic error answer")
    if result.get("error"):
        failures.append(f"workflow error: {result['error']}")
    return failures


def _room_count(result: dict[str, Any]) -> int:
    rooms = result.get("rooms")
    if rooms is not None:
        return len(rooms)
    if result.get("rooms_count") is not None:
        return int(result["rooms_count"])
    return 0


def _failures_search_answer(result: dict[str, Any], *, expect_rooms: bool = True) -> list[str]:
    failures = _failures_generic_answer(result)
    answer = _norm(result.get("answer") or "")
    room_count = _room_count(result)
    intent = result.get("intent") or ""
    if room_count:
        if any(m in answer for m in NO_RESULT_MARKERS):
            failures.append(f"rooms={room_count} but answer claims zero results")
    elif expect_rooms and intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
        if not any(m in answer for m in NO_RESULT_MARKERS) and not failures:
            failures.append("no rooms but answer does not acknowledge empty search")
    return failures


def _failures_no_info(answer: str, topic_terms: tuple[str, ...]) -> list[str]:
    lower = _norm(answer)
    if not any(t in lower for t in topic_terms):
        return []
    if any(m in lower for m in NO_INFO_MARKERS):
        return [f"answer denies info while discussing {topic_terms[0]}"]
    return []


def _failures_mentions_any(answer: str, terms: tuple[str, ...], label: str) -> list[str]:
    lower = _norm(answer)
    if any(t in lower for t in terms):
        return []
    return [f"answer missing expected {label} ({', '.join(terms[:3])})"]


def _apply_answer_checks(case_id: str, turns: list[dict[str, Any]], checker: Callable[[V2CaseResult, list[dict]], None]) -> list[AnswerCheck]:
    wrapper = V2CaseResult(case_id, "?", "PASS", v1.CaseResult(case_id, "?", "", "PASS"))
    checker(wrapper, turns)
    return wrapper.answer_checks


def _check_ctx01(v2: V2CaseResult, turns: list[dict]) -> None:
    for idx in (0, 1, 4):
        for msg in _failures_search_answer(turns[idx], expect_rooms=idx != 4):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")
    if _room_count(turns[1]) < 2:
        v2.add_answer(2, False, f"T2: only {_room_count(turns[1])} room(s) in results")
    else:
        v2.add_answer(2, True, f"T2: {_room_count(turns[1])} rooms after AC refine")
    for msg in _failures_mentions_any(turns[4]["answer"], ("lựa chọn", "phòng", "gợi ý", "đề xuất"), "recommendation"):
        v2.add_answer(5, False, f"T5: {msg}")


def _check_ctx02(v2: V2CaseResult, turns: list[dict]) -> None:
    for idx in range(5):
        for msg in _failures_search_answer(turns[idx]):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")
    for msg in _failures_no_info(turns[1]["answer"], ("xe", "để xe", "gửi xe", "chỗ để")):
        v2.add_answer(2, False, f"T2 parking refine: {msg}")
    if _room_count(turns[1]) >= 2:
        v2.add_answer(2, True, "T2: multiple rooms after parking refine")
    for msg in _failures_mentions_any(turns[3]["answer"], ("hutech", "đh hutech"), "HUTECH landmark"):
        if _room_count(turns[3]) == 0:
            v2.add_answer(4, False, "T4: no rooms for HUTECH refine")
        else:
            v2.add_answer(4, False, f"T4: {msg}")


def _check_ctx03(v2: V2CaseResult, turns: list[dict]) -> None:
    for idx in range(5):
        for msg in _failures_search_answer(turns[idx]):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")
    a3 = _norm(turns[2]["answer"])
    if "quận 10" in a3 or "quan 10" in a3:
        v2.add_answer(3, False, "T3 answer still references Q10 after pivot")
    if "quận 3" in a3 or "quan 3" in a3:
        v2.add_answer(3, True, "T3 answer references Q3")


def _check_ctx04(v2: V2CaseResult, turns: list[dict]) -> None:
    for idx in range(5):
        for msg in _failures_search_answer(turns[idx]):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")
    a4 = _norm(turns[3]["answer"])
    if any(t in a4 for t in ("ban công", "ban cong", "balcony")):
        v2.add_answer(4, True, "T4 mentions balcony")
    elif _room_count(turns[3]) > 0:
        v2.add_answer(4, False, "T4: rooms returned but answer omits balcony preference")


def _check_ctx05(v2: V2CaseResult, turns: list[dict]) -> None:
    a2 = _norm(turns[1]["answer"])
    if any(t in a2 for t in ("so sánh", "so sanh", "ưu điểm", "nhược điểm", "rẻ hơn", "đắt hơn")):
        v2.add_answer(2, True, "T2 compare phrasing present")
    else:
        v2.add_answer(2, False, "T2: compare answer lacks comparison language")
    for msg in _failures_no_info(turns[2]["answer"], ("nội thất", "noi that", "giường", "tủ")):
        v2.add_answer(3, False, f"T3 furnishings: {msg}")
    for msg in _failures_no_info(turns[3]["answer"], ("mèo", "meo", "pet", "thú cưng")):
        v2.add_answer(4, False, f"T4 pet: {msg}")


def _check_ctx06(v2: V2CaseResult, turns: list[dict]) -> None:
    for idx in (0, 4):
        for msg in _failures_search_answer(turns[idx]):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")
    for msg in _failures_mentions_any(turns[2]["answer"], ("m2", "m²", "diện tích", "dien tich", "met"), "area"):
        v2.add_answer(3, False, f"T3: {msg}")
    for msg in _failures_no_info(turns[3]["answer"], ("xe", "để xe", "gửi xe", "chỗ để")):
        v2.add_answer(4, False, f"T4 parking: {msg}")


def _check_ctx07(v2: V2CaseResult, turns: list[dict]) -> None:
    a3 = _norm(turns[2]["answer"])
    if any(t in a3 for t in ("thủ tục", "thu tuc", "hợp đồng", "hop dong", "giấy tờ", "cọc", "thuê")):
        v2.add_answer(3, True, "T3 FAQ has rental-process substance")
    else:
        v2.add_answer(3, False, "T3 FAQ answer too thin")
    for msg in _failures_no_info(turns[3]["answer"], ("cọc", "deposit", "tiền cọc")):
        v2.add_answer(4, False, f"T4 deposit: {msg}")
    for msg in _failures_mentions_any(turns[4]["answer"], ("tổng", "chi phí", "chi phi", "cọc", "tháng đầu"), "cost breakdown"):
        v2.add_answer(5, False, f"T5 cost: {msg}")


def _check_ctx08(v2: V2CaseResult, turns: list[dict]) -> None:
    for idx in range(8):
        for msg in _failures_search_answer(turns[idx]):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")
    if _room_count(turns[3]) == 0:
        v2.add_answer(4, False, "T4: zero rooms after parking refine (known PDF failure)")
    else:
        v2.add_answer(4, True, f"T4: {_room_count(turns[3])} rooms with parking")
    a7 = _norm(turns[6]["answer"])
    if any(t in a7 for t in ("so sánh", "so sanh", "phòng 1", "phòng 3")):
        v2.add_answer(7, True, "T7 compare answer present")
    else:
        v2.add_answer(7, False, "T7: weak compare answer")


def _check_ctx09(v2: V2CaseResult, turns: list[dict]) -> None:
    # turns is concatenation of session A then B in audit — handled in wrapper
    pass


def _check_ctx10(v2: V2CaseResult, turns: list[dict]) -> None:
    pass


def _check_ctx13(v2: V2CaseResult, turns: list[dict]) -> None:
    if _room_count(turns[0]) == 0:
        v2.add_answer(1, False, "T1: no rooms for Văn Hiên landmark")
    for msg in _failures_no_info(turns[2]["answer"], ("người", "nguoi", "ở", "tối đa")):
        v2.add_answer(3, False, f"T3 occupants: {msg}")
    for msg in _failures_no_info(turns[3]["answer"], ("máy lạnh", "may lanh", "điều hòa")):
        v2.add_answer(4, False, f"T4 AC: {msg}")


def _check_ctx14(v2: V2CaseResult, turns: list[dict]) -> None:
    a2 = _norm(turns[1]["answer"])
    if any(t in a2 for t in ("so sánh", "so sanh")):
        v2.add_answer(2, True, "T2 compare phrasing")
    else:
        v2.add_answer(2, False, "T2: weak compare answer")
    for msg in _failures_search_answer(turns[2]):
        v2.add_answer(3, False, f"T3 budget bump: {msg}")
    for msg in _failures_mentions_any(turns[3]["answer"], ("nên chọn", "đáng chọn", "phù hợp", "gợi ý", "đề xuất"), "recommendation"):
        v2.add_answer(4, False, f"T4: {msg}")


ANSWER_CHECKERS: dict[str, Callable[[V2CaseResult, list[dict]], None]] = {
    "CTX-01": _check_ctx01,
    "CTX-02": _check_ctx02,
    "CTX-03": _check_ctx03,
    "CTX-04": _check_ctx04,
    "CTX-05": _check_ctx05,
    "CTX-06": _check_ctx06,
    "CTX-07": _check_ctx07,
    "CTX-08": _check_ctx08,
    "CTX-13": _check_ctx13,
    "CTX-14": _check_ctx14,
}


def _parse_sse_final(chunks: list[str]) -> dict[str, Any]:
    joined = "".join(chunks)
    final_lines = [
        line[len("data: "):]
        for line in joined.splitlines()
        if line.startswith("data: ") and '"type": "final"' in line
    ]
    if not final_lines:
        raise ValueError("no final SSE event in stream")
    event = json.loads(final_lines[-1])
    return event.get("payload") or event


def _consume_stream(response) -> tuple[list[str], dict[str, Any]]:
    chunks: list[str] = []
    for chunk in response.streaming_content:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
    payload = _parse_sse_final(chunks)
    return chunks, payload


def _rag_payload_to_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": payload.get("answer") or payload.get("reply") or "",
        "intent": payload.get("intent") or "",
        "rooms": payload.get("rooms") or [],
        "session_state": payload.get("session_state") or {},
        "comparison": payload.get("comparison"),
        "abstain": payload.get("abstain"),
        "error": payload.get("error"),
    }


@override_settings(
    RAG_RATE_LIMIT_ENABLED=False,
    ALLOWED_HOSTS=["127.0.0.1", "testserver", "localhost"],
)
def audit_ctx11_stream(repo: MongoRoomRepository) -> V2CaseResult:
    memory = v1.CaseResult("CTX-11", "P0", "Pass", "PASS")
    session_id = f"ctx11-stream-{uuid.uuid4().hex[:10]}"
    client = Client()
    questions = [
        "Tìm phòng ở Gò Vấp dưới 6 triệu",
        "thêm gần công viên Gia Định",
        "có phòng nào phù hợp cho 2 người",
    ]
    history: list[dict[str, str]] = []
    transcript: list[dict[str, Any]] = []
    v2 = V2CaseResult("CTX-11", "P0", "PASS", memory, notes=["endpoint=/api/rag/stream/"])

    for idx, question in enumerate(questions):
        body = {"message": question, "session_id": session_id, "history": history}
        response = client.post(
            "/api/rag/stream/",
            data=json.dumps(body),
            content_type="application/json",
            secure=True,
        )
        if response.status_code != 200:
            v2.add_answer(idx + 1, False, f"HTTP {response.status_code}")
            continue
        try:
            _, payload = _consume_stream(response)
        except Exception as exc:
            v2.add_answer(idx + 1, False, f"stream parse failed: {exc}")
            continue
        result = _rag_payload_to_result(payload)
        transcript.append(_turn_transcript(question, result))
        for msg in _failures_search_answer(result):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.get("answer") or ""})

    if transcript:
        occupants = (transcript[-1]["session_state"].get("constraints") or {}).get("occupants")
        if occupants == 2:
            v2.add_answer(3, True, "stream session occupants=2")
        else:
            v2.add_answer(3, False, f"stream occupants={occupants}")
        districts = ((transcript[-1]["session_state"].get("constraints") or {}).get("location") or {}).get("districts") or []
        if any("go vap" in str(d).lower() or "gò vấp" in str(d).lower() for d in districts):
            v2.add_answer(1, True, "stream kept Go Vap district")
        else:
            v2.add_answer(1, False, f"stream districts={districts}")

    v2.transcript = transcript
    v2.finalize()
    return v2


@override_settings(
    RAG_RATE_LIMIT_ENABLED=False,
    ALLOWED_HOSTS=["127.0.0.1", "testserver", "localhost"],
)
def audit_ctx15_chat_history(repo: MongoRoomRepository) -> V2CaseResult:
    memory = v1.CaseResult("CTX-15", "P0", "", "PASS")
    contact_id = f"ctx15_audit_{uuid.uuid4().hex[:10]}"
    conversation_id: str | None = None
    client = Client()
    questions = [
        "Tìm phòng ở quận 7 dưới 6 triệu",
        "có máy lạnh",
        "gần ĐH TDTU",
    ]
    transcript: list[dict[str, Any]] = []
    v2 = V2CaseResult("CTX-15", "P0", "PASS", memory, notes=["endpoint=/api/chat/ + Mongo chat_history"])

    for idx, question in enumerate(questions):
        body: dict[str, Any] = {
            "message": question,
            "contact_id": contact_id,
            "new_chat": "1" if idx == 0 else "0",
        }
        if conversation_id:
            body["conversation_id"] = conversation_id
        response = client.post(
            "/api/chat/",
            data=json.dumps(body),
            content_type="application/json",
        )
        if response.status_code != 200:
            memory.add(idx + 1, False, f"HTTP {response.status_code}")
            v2.add_answer(idx + 1, False, f"HTTP {response.status_code}")
            continue
        try:
            _, payload = _consume_stream(response)
        except Exception as exc:
            memory.add(idx + 1, False, f"stream parse failed: {exc}")
            v2.add_answer(idx + 1, False, f"stream parse failed: {exc}")
            continue

        conversation_id = payload.get("conversation_id") or conversation_id
        result = {
            "answer": payload.get("reply") or payload.get("answer") or "",
            "intent": payload.get("intent") or "",
            "rooms": payload.get("rooms") or [],
            "session_state": payload.get("session_state") or {},
        }
        transcript.append(_turn_transcript(question, result))
        for msg in _failures_search_answer(result):
            v2.add_answer(idx + 1, False, f"T{idx+1}: {msg}")

    if not conversation_id:
        memory.status = "FAIL"
        memory.add(0, False, "no conversation_id from chat API")
        v2.transcript = transcript
        v2.finalize()
        return v2

    chat_doc = get_collection("chat_history").find_one({"conversation_id": conversation_id})
    if not chat_doc:
        memory.add(0, False, "chat_history document missing")
    else:
        messages = chat_doc.get("messages") or []
        if len(messages) < 6:
            memory.add(0, False, f"expected >=6 messages, got {len(messages)}")
        else:
            memory.add(0, True, f"chat_history has {len(messages)} messages")
        roles = [m.get("role") for m in messages]
        if messages and messages[0].get("role") != "user":
            memory.add(0, False, f"first message role={roles[0]}")
        if chat_doc.get("contact_id") != contact_id:
            memory.add(0, False, f"contact_id mismatch: {chat_doc.get('contact_id')}")
        else:
            memory.add(0, True, "contact_id persisted")

        last_state = (transcript[-1]["session_state"].get("constraints") or {}) if transcript else {}
        amenities = last_state.get("amenities_required") or []
        if "air_conditioner" in amenities:
            memory.add(3, True, "AC constraint accumulated in session")
        else:
            memory.add(3, False, f"AC missing in final constraints: {amenities}")

    v2.transcript = transcript
    v2.finalize()
    return v2


WORKFLOW_AUDITS: list[tuple[str, Callable[[MongoRoomRepository], v1.CaseResult]]] = [
    ("CTX-01", v1.audit_ctx01),
    ("CTX-02", v1.audit_ctx02),
    ("CTX-03", v1.audit_ctx03),
    ("CTX-04", v1.audit_ctx04),
    ("CTX-05", v1.audit_ctx05),
    ("CTX-06", v1.audit_ctx06),
    ("CTX-07", v1.audit_ctx07),
    ("CTX-08", v1.audit_ctx08),
    ("CTX-09", v1.audit_ctx09),
    ("CTX-10", v1.audit_ctx10),
    ("CTX-12", v1.audit_ctx12),
    ("CTX-13", v1.audit_ctx13),
    ("CTX-14", v1.audit_ctx14),
]

SESSION_IDS = {
    "CTX-01": "ctx-01",
    "CTX-02": "ctx-02",
    "CTX-03": "ctx-03",
    "CTX-04": "ctx-04",
    "CTX-05": "ctx-05",
    "CTX-06": "ctx-06",
    "CTX-07": "ctx-07",
    "CTX-08": "ctx-08",
    "CTX-09-a": "ctx-09-a",
    "CTX-09-b": "ctx-09-b",
    "CTX-10-a": "chat1-ctx10",
    "CTX-10-b": "chat2-ctx10",
    "CTX-12": "ctx-12",
    "CTX-13": "ctx-13",
    "CTX-14": "ctx-14",
}


def _run_workflow_case(case_id: str, audit_fn: Callable[[MongoRoomRepository], v1.CaseResult], repo: MongoRoomRepository) -> V2CaseResult:
    memory = audit_fn(repo)
    session_key = SESSION_IDS.get(case_id, case_id.lower())
    transcript = CASE_TRANSCRIPTS.get(session_key, [])
    if case_id == "CTX-09":
        transcript = (CASE_TRANSCRIPTS.get("ctx-09-a") or []) + (CASE_TRANSCRIPTS.get("ctx-09-b") or [])
    if case_id == "CTX-10":
        transcript = (CASE_TRANSCRIPTS.get("chat1-ctx10") or []) + (CASE_TRANSCRIPTS.get("chat2-ctx10") or [])

    v2 = V2CaseResult(case_id, memory.priority, "PASS", memory, transcript=transcript)
    checker = ANSWER_CHECKERS.get(case_id)
    if checker and transcript:
        checker(v2, transcript)
    elif case_id not in {"CTX-09", "CTX-10", "CTX-12"} and checker:
        v2.notes.append("no transcript captured for answer checks")
    v2.finalize()
    return v2


def _normalize_case_id(raw: str) -> str:
    token = raw.strip().upper()
    if token.startswith("CTX-"):
        return token
    if token.startswith("CTX"):
        return "CTX-" + token[3:].lstrip("-")
    return f"CTX-{token}"


def _write_reports(report: list[V2CaseResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "context_memory_audit_v2_report.json"
    md_path = out_dir / "context_memory_audit_v2_transcript.md"

    serializable = []
    for r in report:
        serializable.append({
            "case_id": r.case_id,
            "priority": r.priority,
            "status": r.status,
            "memory_status": r.memory.status,
            "memory_checks": [{"turn": c.turn, "ok": c.ok, "detail": c.detail} for c in r.memory.checks],
            "answer_checks": [{"turn": c.turn, "ok": c.ok, "detail": c.detail} for c in r.answer_checks],
            "transcript": r.transcript,
            "notes": r.notes + r.memory.notes,
        })
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Context Memory Audit v2",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Groq keys: {len(getattr(__import__('config', fromlist=['GROQ_API_KEYS']), 'GROQ_API_KEYS', []))}",
        "",
    ]
    for r in report:
        lines.append(f"## {r.case_id} — {r.status} (memory={r.memory.status})")
        for note in r.notes + r.memory.notes:
            lines.append(f"- note: {note}")
        for turn in r.transcript:
            lines.append(f"### Q: {turn['question']}")
            lines.append(f"- intent: `{turn.get('intent')}` | rooms: {turn.get('rooms_count')}")
            lines.append(f"- answer: {_answer_snippet(turn.get('answer') or '')}")
            constraints = (turn.get("session_state") or {}).get("constraints") or {}
            lines.append(f"- constraints: `{json.dumps(constraints, ensure_ascii=False)[:240]}`")
            lines.append("")
        fails = [c for c in r.memory.checks if not c.ok] + [c for c in r.answer_checks if not c.ok]
        if fails:
            lines.append("**Failures:**")
            for c in fails:
                kind = "memory" if c in r.memory.checks else "answer"
                lines.append(f"- [{kind}] T{c.turn}: {c.detail}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"JSON report: {json_path}")
    print(f"Transcript:  {md_path}")


def _rescore_saved_report(json_path: Path) -> list[V2CaseResult]:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    report: list[V2CaseResult] = []
    for row in rows:
        memory = v1.CaseResult(row["case_id"], row.get("priority", "?"), "", row.get("memory_status", "PASS"))
        for chk in row.get("memory_checks") or []:
            memory.add(chk["turn"], chk["ok"], chk["detail"])
        memory.status = row.get("memory_status", memory.status)
        v2 = V2CaseResult(
            row["case_id"],
            row.get("priority", "?"),
            "PASS",
            memory,
            transcript=row.get("transcript") or [],
            notes=row.get("notes") or [],
        )
        checker = ANSWER_CHECKERS.get(row["case_id"])
        if checker and v2.transcript:
            checker(v2, v2.transcript)
        v2.finalize()
        report.append(v2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Context memory audit v2 (real LLM + answers)")
    parser.add_argument("--case", action="append", help="Run only CTX-01, CTX-11, etc.")
    parser.add_argument("--rescore", action="store_true", help="Re-apply answer checks to saved JSON report")
    args = parser.parse_args()
    selected = {_normalize_case_id(c) for c in (args.case or [])}

    load_dotenv()
    if not mongo_is_configured():
        print("SKIP: MongoDB not configured")
        return 1
    repo = create_mongo_repository()
    if not isinstance(repo, MongoRoomRepository):
        print("SKIP: Mongo repository unavailable")
        return 1

    try:
        from config import GROQ_API_KEYS
    except ImportError:
        GROQ_API_KEYS = []
    if not GROQ_API_KEYS:
        print("WARN: GROQ_API_KEYS empty — routing/answers may use fallbacks")

    out_dir = ROOT / "tmp"
    json_path = out_dir / "context_memory_audit_v2_report.json"

    if args.rescore:
        if not json_path.exists():
            print(f"Missing report: {json_path}")
            return 1
        report = _rescore_saved_report(json_path)
        _write_reports(report, out_dir)
        elapsed = 0
    else:
        report: list[V2CaseResult] = []
        t0 = time.time()

        for case_id, audit_fn in WORKFLOW_AUDITS:
            if selected and case_id not in selected:
                continue
            print(f"Running {case_id} (workflow + answers)...", flush=True)
            try:
                report.append(_run_workflow_case(case_id, audit_fn, repo))
            except Exception as exc:
                cr = V2CaseResult(case_id, "?", "FAIL", v1.CaseResult(case_id, "?", "", "FAIL"))
                cr.notes.append(f"exception: {exc}")
                cr.finalize()
                report.append(cr)

        if not selected or "CTX-11" in selected:
            print("Running CTX-11 (api/rag/stream)...", flush=True)
            try:
                report.append(audit_ctx11_stream(repo))
            except Exception as exc:
                cr = V2CaseResult("CTX-11", "P0", "FAIL", v1.CaseResult("CTX-11", "P0", "", "FAIL"))
                cr.notes.append(f"exception: {exc}")
                cr.finalize()
                report.append(cr)

        if not selected or "CTX-15" in selected:
            print("Running CTX-15 (api/chat + chat_history)...", flush=True)
            try:
                report.append(audit_ctx15_chat_history(repo))
            except Exception as exc:
                cr = V2CaseResult("CTX-15", "P0", "FAIL", v1.CaseResult("CTX-15", "P0", "", "FAIL"))
                cr.notes.append(f"exception: {exc}")
                cr.finalize()
                report.append(cr)

        _write_reports(report, out_dir)
        elapsed = time.time() - t0

    passed = sum(1 for r in report if r.status == "PASS")
    failed = sum(1 for r in report if r.status == "FAIL")
    partial = sum(1 for r in report if r.status == "PARTIAL")
    skipped = sum(1 for r in report if r.status == "SKIP")
    print(f"\n=== V2 SUMMARY ({elapsed:.0f}s): PASS={passed} FAIL={failed} PARTIAL={partial} SKIP={skipped} ===")
    for r in report:
        mark = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "~", "SKIP": "-"}.get(r.status, "?")
        mem_fails = [c for c in r.memory.checks if not c.ok]
        ans_fails = [c for c in r.answer_checks if not c.ok]
        print(f"{mark} {r.case_id} memory={r.memory.status} answers={len(ans_fails)} fail")
        for c in mem_fails + ans_fails:
            kind = "M" if c in r.memory.checks else "A"
            print(f"    [{kind}] T{c.turn}: {c.detail}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
