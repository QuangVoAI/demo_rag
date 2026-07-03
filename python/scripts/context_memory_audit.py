"""Re-run Bo Test Case Manual Test Context Memory (CTX-01..CTX-15)."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from mongo_test_helpers import create_mongo_repository, load_dotenv, mongo_is_configured, qdrant_is_available
from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import MongoRoomRepository, room_matches_constraints
from room_assistant.session_store import InMemorySessionStore, load_session_state
from room_assistant.workflow import run_room_assistant


async def _regex_aligned_llm(question: str, current_state: dict[str, Any] | None) -> dict[str, Any]:
    parsed = parse_intent_and_constraint_patch(question, current_state)
    return {
        "intent": parsed["intent"],
        "confidence": 0.99,
        "operations": parsed["operations"],
        "referenced_room_ids": parsed.get("referenced_room_ids", []),
        "requested_action": parsed.get("requested_action"),
    }


def _run_turns(
    repo: MongoRoomRepository,
    questions: list[str],
    session_id: str,
    *,
    use_qdrant: bool = True,
) -> list[dict[str, Any]]:
    from unittest.mock import patch

    from mongo_test_helpers import create_semantic_index

    store = InMemorySessionStore()
    history: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    semantic_index = create_semantic_index() if use_qdrant and qdrant_is_available() else None

    with patch("room_assistant.intent._llm_classify_intent", _regex_aligned_llm):
        with patch("room_assistant.intent._llm_verify_routing_decision", return_value={}):
            for question in questions:
                result = asyncio.run(run_room_assistant(
                    question,
                    history=history,
                    session_id=session_id,
                    repository=repo,
                    session_store=store,
                    semantic_index=semantic_index,
                ))
                results.append(result)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": result.get("answer") or ""})
    return results


@dataclass
class TurnCheck:
    turn: int
    ok: bool
    detail: str


@dataclass
class CaseResult:
    case_id: str
    priority: str
    old_result: str
    status: str  # PASS | FAIL | PARTIAL | SKIP
    checks: list[TurnCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, turn: int, ok: bool, detail: str) -> None:
        self.checks.append(TurnCheck(turn, ok, detail))


def _constraints(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("constraints") or {}


def _districts(state: dict[str, Any]) -> list[str]:
    return (_constraints(state).get("location") or {}).get("districts") or []


def _landmarks(state: dict[str, Any]) -> list[str]:
    return (_constraints(state).get("location") or {}).get("near_landmarks") or []


def _budget_max(state: dict[str, Any]) -> int | None:
    return (_constraints(state).get("budget") or {}).get("max")


def _amenities(state: dict[str, Any]) -> list[str]:
    return _constraints(state).get("amenities_required") or []


def _vehicles(state: dict[str, Any]) -> list[str]:
    return _constraints(state).get("vehicles") or []


def _categories(state: dict[str, Any]) -> list[str]:
    return _constraints(state).get("categories") or []


def _rooms_ok(rooms: list[dict[str, Any]], constraints: dict[str, Any]) -> tuple[bool, str]:
    if not rooms:
        return False, "no rooms returned"
    for room in rooms:
        if not room_matches_constraints(room, constraints):
            return False, f"room {room.get('room_id')} violates constraints"
    return True, f"{len(rooms)} room(s) match"


def _district_in_rooms(rooms: list[dict[str, Any]], needle: str) -> bool:
    needle = needle.lower()
    for room in rooms:
        district = str(room.get("district") or "").lower()
        if needle in district:
            return True
    return False


def _district_not_in_rooms(rooms: list[dict[str, Any]], needle: str) -> bool:
    needle = needle.lower()
    for room in rooms:
        district = str(room.get("district") or "").lower()
        if needle in district:
            return False
    return True


def audit_ctx01(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-01", "P0", "Fail", "PASS")
    qs = [
        "Tìm phòng ở Gò Vấp dưới 5 triệu",
        "ưu tiên có máy lạnh",
        "gần ĐH Công nghiệp nữa",
        "có phòng nào rộng hơn không",
        "cho tôi 2 lựa chọn phù hợp nhất",
    ]
    results = _run_turns(repo, qs, "ctx-01")
    s1 = results[0]["session_state"]
    if not any("go vap" in d or "gò vấp" in d for d in _districts(s1)):
        r.add(1, False, f"district missing: {_districts(s1)}")
    elif _budget_max(s1) != 5_000_000:
        r.add(1, False, f"budget={_budget_max(s1)}")
    else:
        r.add(1, True, "district+budget ok")
    s2 = results[1]["session_state"]
    if "air_conditioner" not in _amenities(s2):
        r.add(2, False, f"amenities={_amenities(s2)}")
    elif len(results[1].get("rooms") or []) < 2:
        r.add(2, False, f"only {len(results[1].get('rooms') or [])} room(s) at T2")
    else:
        r.add(2, True, "AC added, multiple rooms")
    s3 = results[2]["session_state"]
    if not _landmarks(s3):
        r.add(3, False, "landmark not added")
    elif not _districts(s3):
        r.add(3, False, "lost district at T3")
    else:
        r.add(3, True, f"landmark={_landmarks(s3)}")
    s5 = results[4]["session_state"]
    if not _districts(s5) or _budget_max(s5) != 5_000_000:
        r.add(5, False, "lost district/budget by T5")
    elif not s5.get("last_result_ids"):
        r.add(5, False, "last_result_ids empty at T5")
    else:
        r.add(5, True, f"last_result_ids={len(s5['last_result_ids'])}")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx02(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-02", "P0", "Fail", "PASS")
    qs = [
        "Tìm phòng ở Bình Thạnh dưới 6 triệu",
        "có chỗ để xe máy",
        "không, ngân sách nâng lên 7 triệu",
        "vậy còn phòng nào gần HUTECH",
        "cho loại studio thôi",
    ]
    results = _run_turns(repo, qs, "ctx-02")
    s1 = results[0]["session_state"]
    if _budget_max(s1) != 6_000_000:
        r.add(1, False, f"budget T1={_budget_max(s1)}")
    else:
        r.add(1, True, "T1 ok")
    s2 = results[1]["session_state"]
    parking = "motorbike" in _vehicles(s2)
    if not parking:
        r.add(2, False, f"parking vehicles={_vehicles(s2)}")
    elif results[1].get("intent") not in {"REFINE_SEARCH", "SEARCH_ROOM"}:
        r.add(2, False, f"intent={results[1].get('intent')}")
    elif len(results[1].get("rooms") or []) < 2:
        r.add(2, False, f"only {len(results[1].get('rooms') or [])} room(s) at T2")
    else:
        r.add(2, True, "parking added, multiple rooms")
    s3 = results[2]["session_state"]
    if _budget_max(s3) != 7_000_000:
        r.add(3, False, f"budget not updated: {_budget_max(s3)}")
    elif not any("binh thanh" in d for d in _districts(s3)):
        r.add(3, False, "lost district at T3")
    else:
        r.add(3, True, "budget 7M, district kept")
    s4 = results[3]["session_state"]
    if not _landmarks(s4):
        r.add(4, False, "HUTECH landmark missing")
    else:
        r.add(4, True, f"landmarks={_landmarks(s4)}")
    s5 = results[4]["session_state"]
    if "studio" not in _categories(s5):
        r.add(5, False, f"studio category missing: {_categories(s5)}")
    else:
        r.add(5, True, "studio category set")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx03(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-03", "P0", "Fail", "PASS")
    qs = [
        "Tìm phòng ở quận 10 dưới 5 triệu",
        "thêm máy giặt",
        "không, quận 3 mới đúng",
        "gần công viên Lê Văn Tám",
        "có phòng nào rẻ hơn không",
    ]
    results = _run_turns(repo, qs, "ctx-03")
    s3 = results[2]["session_state"]
    districts = _districts(s3)
    if any("10" in d for d in districts):
        r.add(3, False, f"Q10 still present: {districts}")
    elif not any("3" in d for d in districts):
        r.add(3, False, f"Q3 not set: {districts}")
    else:
        r.add(3, True, f"districts={districts}")
    rooms_t3 = results[2].get("rooms") or []
    if rooms_t3 and not _district_not_in_rooms(rooms_t3, "10"):
        r.add(3, False, "T3 rooms still in Q10")
    s5 = results[4]["session_state"]
    if any("10" in d for d in _districts(s5)):
        r.add(5, False, "Q10 in state at T5")
    elif "washing_machine" not in _amenities(s5):
        r.add(5, False, "lost washing_machine")
    else:
        r.add(5, True, "Q3+washing_machine kept")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx04(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-04", "P0", "Fail", "PASS")
    qs = [
        "Tìm phòng dưới 6 triệu có máy lạnh",
        "thêm gần ĐH Văn Lang",
        "bỏ yêu cầu máy lạnh",
        "ưu tiên có ban công",
        "cho tôi list mới",
    ]
    results = _run_turns(repo, qs, "ctx-04")
    s3 = results[2]["session_state"]
    if "air_conditioner" in _amenities(s3):
        r.add(3, False, "AC not removed")
    elif _budget_max(s3) != 6_000_000:
        r.add(3, False, "budget lost")
    else:
        r.add(3, True, "AC removed, budget kept")
    s4 = results[3]["session_state"]
    balcony = any(a in _amenities(s4) for a in ("balcony", "ban_cong"))
    if not balcony:
        r.add(4, False, f"balcony not in amenities={_amenities(s4)}")
    else:
        r.add(4, True, "balcony added")
    rooms_t4 = results[3].get("rooms") or []
    if rooms_t4:
        for room in rooms_t4:
            text = (room.get("embedding_text") or room.get("title") or "").lower()
            if "ban công: không" in text or "ban cong: khong" in text:
                r.add(4, False, f"room {room.get('room_id')} has no balcony in data")
                break
        else:
            r.add(4, True, "no obvious no-balcony room in top results")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx05(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-05", "P0", "Fail", "PASS")
    qs = [
        "Tìm phòng ở Phú Nhuận dưới 7 triệu",
        "so sánh 2 phòng đầu tiên",
        "phòng thứ 2 có nội thất gì",
        "phòng thứ 1 cho nuôi mèo được không",
        "phòng nào phù hợp 2 người hơn",
    ]
    results = _run_turns(repo, qs, "ctx-05")
    rooms_t1 = results[0].get("rooms") or []
    if len(rooms_t1) < 2:
        r.add(1, False, f"only {len(rooms_t1)} rooms at T1")
    else:
        r.add(1, True, f"{len(rooms_t1)} rooms")
    t2 = results[1]
    if t2.get("intent") != "COMPARE_ROOMS":
        r.add(2, False, f"intent={t2.get('intent')}")
    else:
        rows = (t2.get("comparison") or {}).get("rows") or []
        expected = [room["room_id"] for room in rooms_t1[:2]]
        actual = [row["room_id"] for row in rows[:2]]
        if actual != expected:
            r.add(2, False, f"compare ids {actual} != {expected}")
        else:
            r.add(2, True, "compare first 2 ok")
    t3 = results[2]
    if t3.get("intent") != "ASK_ABOUT_ROOM":
        r.add(3, False, f"T3 intent={t3.get('intent')}")
    else:
        ref = t3.get("session_state", {}).get("current_room_id")
        expected_id = rooms_t1[1]["room_id"] if len(rooms_t1) > 1 else None
        if ref != expected_id:
            r.add(3, False, f"current_room={ref} expected 2nd={expected_id}")
        else:
            r.add(3, True, "ordinal 2 resolved")
    t4 = results[3]
    ref4 = t4.get("session_state", {}).get("current_room_id")
    expected4 = rooms_t1[0]["room_id"] if rooms_t1 else None
    if ref4 != expected4:
        r.add(4, False, f"ordinal 1 room={ref4} expected={expected4}")
    else:
        r.add(4, True, "ordinal 1 resolved")
    answer4 = (t4.get("answer") or "").lower()
    if "không có thông tin" in answer4 and "nuôi" in answer4:
        r.notes.append("T4: contradictory pet info vs compare (check answer quality)")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx06(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-06", "P0", "Fail", "PASS")
    qs = [
        "Tìm phòng gần ĐH Bách Khoa",
        "cho xem chi tiết phòng đầu tiên",
        "phòng này diện tích bao nhiêu",
        "có chỗ để xe không",
        "tìm phòng tương tự",
    ]
    results = _run_turns(repo, qs, "ctx-06")
    if results[0].get("error"):
        r.add(1, False, f"T1 error: {results[0].get('error')}")
        r.status = "FAIL"
        return r
    rooms_t1 = results[0].get("rooms") or []
    if not rooms_t1:
        r.add(1, False, "no rooms T1")
    else:
        r.add(1, True, f"{len(rooms_t1)} rooms, last_result_ids set")
    first_id = rooms_t1[0]["room_id"] if rooms_t1 else None
    for turn_idx, label in ((2, "T2 detail"), (3, "T3 area"), (4, "T4 parking")):
        st = results[turn_idx]["session_state"]
        if st.get("current_room_id") != first_id:
            r.add(turn_idx + 1, False, f"{label} current_room={st.get('current_room_id')}")
        else:
            r.add(turn_idx + 1, True, f"{label} tracks room 1")
    t5 = results[4]
    if t5.get("intent") not in {"FIND_SIMILAR", "SEARCH_ROOM", "REFINE_SEARCH"}:
        r.add(5, False, f"intent={t5.get('intent')}")
    else:
        r.add(5, True, f"find similar intent={t5.get('intent')}")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx07(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-07", "P1", "Pass", "PASS")
    qs = [
        "Tìm phòng ở quận 7 dưới 8 triệu",
        "cho xem phòng đầu tiên",
        "thủ tục thuê như thế nào",
        "quay lại phòng này, tiền cọc bao nhiêu",
        "tính giúp tổng chi phí ban đầu",
    ]
    results = _run_turns(repo, qs, "ctx-07")
    rooms_t1 = results[0].get("rooms") or []
    first_id = rooms_t1[0]["room_id"] if rooms_t1 else None
    s2 = results[1]["session_state"]
    if s2.get("current_room_id") != first_id:
        r.add(2, False, f"T2 current_room={s2.get('current_room_id')}")
    else:
        r.add(2, True, "current_room set")
    t3 = results[2]
    if t3.get("intent") not in {"REQUEST_FAQ", "GENERAL_HELP"}:
        r.add(3, False, f"T3 intent={t3.get('intent')}")
    elif load_session_state("ctx-07", InMemorySessionStore()).get("current_room_id"):
        pass  # store is same instance - check from result state after workflow
    s3 = t3["session_state"]
    if s3.get("current_room_id") != first_id:
        r.add(3, False, f"lost current_room after FAQ: {s3.get('current_room_id')}")
    else:
        r.add(3, True, "FAQ kept current_room")
    t4 = results[3]
    if t4.get("intent") not in {"ASK_ABOUT_ROOM", "CALCULATE_COST"}:
        r.add(4, False, f"T4 intent={t4.get('intent')}")
    elif t4["session_state"].get("current_room_id") != first_id:
        r.add(4, False, "T4 wrong room")
    else:
        r.add(4, True, "returned to correct room")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx08(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-08", "P0", "Fail", "PASS")
    qs = [
        "Tìm phòng ở Tân Bình dưới 5 triệu",
        "phải có máy lạnh",
        "gần Etown",
        "thêm chỗ để xe",
        "có phòng nào lớn hơn 25m2 không",
        "lấy 3 phòng tốt nhất",
        "so sánh phòng 1 và 3",
        "phòng 3 có nuôi mèo được không",
    ]
    results = _run_turns(repo, qs, "ctx-08")
    s4 = results[3]["session_state"]
    parking = "motorbike" in _vehicles(s4)
    if not parking:
        r.add(4, False, f"parking not added: vehicles={_vehicles(s4)}")
    rooms_t4 = results[3].get("rooms") or []
    if not rooms_t4:
        r.add(4, False, "T4 returned zero rooms")
    else:
        r.add(4, True, f"parking added, {len(rooms_t4)} rooms")
    s5 = results[4]["session_state"]
    if not any("tan binh" in d for d in _districts(s5)):
        r.add(5, False, "lost Tan Binh")
    else:
        r.add(5, True, "constraints accumulated to T5")
    t7 = results[6]
    if t7.get("intent") != "COMPARE_ROOMS":
        r.add(7, False, f"compare intent={t7.get('intent')}")
    else:
        r.add(7, True, "compare 1 and 3")
    t8 = results[7]
    last_ids = results[5]["session_state"].get("last_result_ids") or []
    if len(last_ids) >= 3:
        expected3 = last_ids[2]
        if t8["session_state"].get("current_room_id") != expected3:
            r.add(8, False, f"phòng 3={t8['session_state'].get('current_room_id')} expected={expected3}")
        else:
            r.add(8, True, "ordinal 3 resolved")
    else:
        r.add(8, False, f"last_result_ids too short: {len(last_ids)}")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx09(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-09", "P0", "Pass/Partial", "PASS")
    a_results = _run_turns(
        repo,
        [
            "Tìm phòng ở Tân Phú dưới 4 triệu",
            "có máy lạnh",
            "thêm gần Đầm Sen",
        ],
        "ctx-09-a",
    )
    b_results = _run_turns(
        repo,
        [
            "Tìm phòng gần HUTECH",
            "có phòng nào rẻ hơn không",
        ],
        "ctx-09-b",
    )

    s_b1 = b_results[0]["session_state"]
    if any("tan phu" in d for d in _districts(s_b1)):
        r.add(0, False, "Session B inherited Tan Phu from A")
    else:
        r.add(0, True, "Session B isolated")
    s_a3 = a_results[2]["session_state"]
    if _budget_max(s_a3) != 4_000_000:
        r.add(0, False, f"A3 budget={_budget_max(s_a3)}")
    elif "air_conditioner" not in _amenities(s_a3):
        r.add(0, False, f"A3 lost AC: {_amenities(s_a3)}")
        r.status = "PARTIAL"
    else:
        r.add(0, True, "A3 kept Tan Phu+AC+Dam Sen")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL" if r.status != "PARTIAL" else "PARTIAL"
    return r


def audit_ctx10(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-10", "P0", "Fail", "PASS")
    chat1 = _run_turns(
        repo,
        [
            "Tìm phòng quận 1 dưới 9 triệu",
            "có nội thất",
            "thêm ban công",
        ],
        "chat1-ctx10",
    )
    chat2 = _run_turns(
        repo,
        [
            "Tìm phòng gần IUH",
            "có phòng nào rẻ hơn không",
        ],
        "chat2-ctx10",
    )

    s_c2 = chat2[0]["session_state"]
    if any("quan 1" in d or "quận 1" in d for d in _districts(s_c2)):
        r.add(0, False, "Chat2 inherited Q1")
    else:
        r.add(0, True, "Chat2 clean start")
    s_c1 = chat1[2]["session_state"]
    furnished = any(a in _amenities(s_c1) for a in ("furnished", "noi_that", "furniture"))
    if not furnished:
        r.add(0, False, f"furnished not in amenities={_amenities(s_c1)}")
        r.status = "FAIL"
        r.notes.append("nội thất amenity mapping may still be missing")
    elif not any("1" in d for d in _districts(s_c1)):
        r.add(0, False, "Chat1 lost Q1")
        r.status = "FAIL"
    else:
        r.add(0, True, "Chat1 kept Q1+furnished+balcony")
    return r


def audit_ctx11(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-11", "P0", "Pass", "PASS")
    qs = [
        "Tìm phòng ở Gò Vấp dưới 6 triệu",
        "thêm gần công viên Gia Định",
        "có phòng nào phù hợp cho 2 người",
    ]
    results = _run_turns(repo, qs, "ctx-11")
    s1 = results[0]["session_state"]
    s2 = results[1]["session_state"]
    s3 = results[2]["session_state"]
    if not any("go vap" in d for d in _districts(s1)):
        r.add(1, False, "T1 district")
    elif _budget_max(s1) != 6_000_000:
        r.add(1, False, "T1 budget")
    else:
        r.add(1, True, "T1 state saved")
    if not _landmarks(s2) or _budget_max(s2) != 6_000_000:
        r.add(3, False, f"T3 state broken landmarks={_landmarks(s2)} budget={_budget_max(s2)}")
    else:
        r.add(3, True, "T3 accumulated landmark")
    occupants = _constraints(s3).get("occupants")
    if occupants != 2:
        r.add(4, False, f"occupants={occupants}")
    else:
        r.add(4, True, "occupants=2")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx12(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-12", "P1", "Pass", "PASS")
    from room_assistant.session_store import save_session_state
    from room_assistant.schemas import default_session_state

    store = InMemorySessionStore()
    state = default_session_state("ctx-12")
    state["constraints"]["location"]["districts"] = ["binh tan"]
    state["constraints"]["budget"]["max"] = 4_500_000
    state["constraints"]["amenities_required"] = ["air_conditioner"]
    save_session_state(state, store, ttl_seconds=1)
    time.sleep(1.2)
    loaded = load_session_state("ctx-12", store)
    fresh = loaded["constraints"]["location"]["districts"] == [] and loaded["constraints"]["budget"].get("max") is None
    if fresh or loaded.get("state_version", 1) == 1:
        r.add(0, True, "TTL expired -> default state")
    else:
        r.add(0, False, f"state persisted after TTL: {loaded['constraints']}")
        r.status = "FAIL"
    return r


def audit_ctx13(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-13", "P1", "Fail", "PASS")
    qs = [
        "Tìm phòng gần ĐH Văn Hiên",
        "cho xem chi tiết phòng đầu tiên",
        "phòng này ở tối đa mấy người",
        "phòng này có máy lạnh không",
        "tìm phòng tương tự",
    ]
    results = _run_turns(repo, qs, "ctx-13")
    rooms_t1 = results[0].get("rooms") or []
    if not rooms_t1:
        r.add(1, False, "T1 no rooms - landmark resolution failed")
        r.status = "FAIL"
        r.notes.append("ĐH Văn Hiên landmark may not resolve")
        return r
    first_id = rooms_t1[0]["room_id"]
    for turn_idx in (2, 3, 4):
        st = results[turn_idx]["session_state"]
        if st.get("current_room_id") != first_id:
            r.add(turn_idx + 1, False, f"T{turn_idx+1} lost current_room")
        else:
            r.add(turn_idx + 1, True, f"T{turn_idx+1} kept current_room")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx14(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-14", "P1", "", "PASS")
    qs = [
        "Tìm phòng ở quận 5 dưới 7 triệu",
        "so sánh 3 phòng đầu tiên",
        "nếu tăng ngân sách lên 7.5 triệu thì có lựa chọn tốt hơn không",
        "vậy phòng nào đáng chọn nhất",
    ]
    results = _run_turns(repo, qs, "ctx-14")
    t2 = results[1]
    selected = t2["session_state"].get("selected_room_ids") or []
    if len(selected) < 2:
        r.add(2, False, f"selected_room_ids={selected}")
    else:
        r.add(2, True, f"selected {len(selected)} rooms")
    old_ids = t2["session_state"].get("last_result_ids") or []
    t3_rooms = results[2].get("rooms") or []
    t3_ids = [r_["room_id"] for r_ in t3_rooms]
    t4 = results[3]
    if t4.get("intent") not in {"COMPARE_ROOMS", "ASK_ABOUT_ROOM", "GENERAL_HELP", "REFINE_SEARCH"}:
        r.add(4, False, f"T4 intent={t4.get('intent')}")
    else:
        r.add(4, True, "T4 answered recommendation")
    if any(not c.ok for c in r.checks):
        r.status = "FAIL"
    return r


def audit_ctx15(repo: MongoRoomRepository) -> CaseResult:
    r = CaseResult("CTX-15", "P0", "", "SKIP")
    r.notes.append("Requires Django chat_history Mongo integration - not exercised in workflow-only audit")
    return r


def main() -> int:
    load_dotenv()
    if not mongo_is_configured():
        print("SKIP: MongoDB not configured")
        return 1
    repo = create_mongo_repository()
    if not isinstance(repo, MongoRoomRepository):
        print("SKIP: Mongo repository unavailable")
        return 1

    auditors = [
        audit_ctx01, audit_ctx02, audit_ctx03, audit_ctx04, audit_ctx05,
        audit_ctx06, audit_ctx07, audit_ctx08, audit_ctx09, audit_ctx10,
        audit_ctx11, audit_ctx12, audit_ctx13, audit_ctx14, audit_ctx15,
    ]
    report: list[CaseResult] = []
    for fn in auditors:
        case_id = fn.__name__.replace("audit_", "").upper().replace("CTX", "CTX-")
        print(f"Running {case_id}...", flush=True)
        try:
            report.append(fn(repo))
        except Exception as exc:
            cr = CaseResult(case_id, "?", "?", "FAIL")
            cr.notes.append(f"exception: {exc}")
            report.append(cr)

    out_path = Path(__file__).parent.parent.parent / "tmp" / "context_memory_audit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "case_id": r.case_id,
            "priority": r.priority,
            "old_result": r.old_result,
            "status": r.status,
            "checks": [{"turn": c.turn, "ok": c.ok, "detail": c.detail} for c in r.checks],
            "notes": r.notes,
        }
        for r in report
    ]
    out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(1 for r in report if r.status == "PASS")
    failed = sum(1 for r in report if r.status == "FAIL")
    partial = sum(1 for r in report if r.status == "PARTIAL")
    skipped = sum(1 for r in report if r.status == "SKIP")
    print(f"\n=== SUMMARY: PASS={passed} FAIL={failed} PARTIAL={partial} SKIP={skipped} ===")
    for r in report:
        mark = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "~", "SKIP": "-"}.get(r.status, "?")
        fails = [c for c in r.checks if not c.ok]
        print(f"{mark} {r.case_id} [{r.priority}] was={r.old_result} now={r.status}")
        for c in fails:
            print(f"    T{c.turn}: {c.detail}")
        for n in r.notes:
            print(f"    note: {n}")
    print(f"\nFull report: {out_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
