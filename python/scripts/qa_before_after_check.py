"""So sánh hành vi Q&A trước/sau các fix production (Jun–Jul 2026)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import InMemoryRoomRepository, room_matches_constraints
from room_assistant.schemas import default_session_state
from room_assistant.session_store import InMemorySessionStore, apply_operations
from room_assistant.workflow import _compose_answer_template, run_room_assistant


def _rooms() -> list[dict]:
    return [
        {
            "room_id": "studio-1",
            "metadata": {
                "house_name": "Studio Q7",
                "room_code": "S01",
                "price": 4_500_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "available": True,
            "status": "active",
            "title": "Studio Q7",
            "embedding_text": "Studio gọn, có gác, Quận 7",
        },
        {
            "room_id": "tro-1",
            "metadata": {
                "house_name": "Phòng trọ",
                "room_code": "T01",
                "price": 3_200_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "available": True,
            "status": "active",
            "title": "Phòng trọ thường",
            "embedding_text": "Phòng trọ tiện nghi, Quận 7",
        },
        {
            "room_id": "pet-1",
            "metadata": {
                "house_name": "Pet friendly",
                "room_code": "PET1",
                "price": 4_000_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "available": True,
            "status": "active",
            "title": "Phòng cho nuôi mèo",
            "embedding_text": "## Tiện ích\n- Thú cưng: Có\n- Wifi: Có",
        },
        {
            "room_id": "no-pet-1",
            "metadata": {
                "house_name": "No pets",
                "room_code": "NP1",
                "price": 3_800_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "available": True,
            "status": "active",
            "title": "Phòng không thú cưng",
            "embedding_text": "## Tiện ích\n- Thú cưng: Không\n- Wifi: Có",
        },
        {
            "room_id": "p305-room",
            "metadata": {
                "house_name": "C2 HOÀNG QUỐC VIỆT",
                "room_code": "P.305",
                "price": 4_800_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "available": True,
            "status": "active",
            "title": "C2 HOÀNG QUỐC VIỆT - P.305",
            "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7\n## Tiện ích\n- Máy lạnh: Có",
        },
        {
            "room_id": "list-a",
            "metadata": {"house_name": "A", "room_code": "A1", "price": 3_000_000, "status_code": "0", "district_name": "Quận 7"},
            "available": True,
            "status": "active",
            "title": "List A",
            "embedding_text": "Phòng A, Quận 7",
        },
        {
            "room_id": "list-b",
            "metadata": {"house_name": "B", "room_code": "B2", "price": 3_500_000, "status_code": "0", "district_name": "Quận 7"},
            "available": True,
            "status": "active",
            "title": "List B",
            "embedding_text": "Phòng B, Quận 7",
        },
        {
            "room_id": "list-c",
            "metadata": {"house_name": "C", "room_code": "C3", "price": 4_000_000, "status_code": "0", "district_name": "Quận 7"},
            "available": True,
            "status": "active",
            "title": "List C",
            "embedding_text": "Phòng C, Quận 7",
        },
    ]


SCENARIOS = [
    {
        "name": "Studio filter",
        "question": "Tìm studio quận 7",
        "before": "Trả cả phòng trọ lẫn studio vì categories chưa enforce",
        "check": lambda r: all(
            "studio" in (room.get("embedding_text") or "").lower()
            for room in (r.get("rooms") or [])
        ) if r.get("rooms") else False,
    },
    {
        "name": "Nuôi mèo filter",
        "question": "Tìm phòng quận 7 cho nuôi mèo",
        "before": "Có thể gợi ý phòng Thú cưng: Không",
        "check": lambda r: all(
            "thú cưng: không" not in (room.get("embedding_text") or "").lower()
            for room in (r.get("rooms") or [])
        ) if r.get("rooms") else False,
    },
    {
        "name": "Mã phòng P.305",
        "question": "phòng P.305 giá bao nhiêu",
        "before": "GENERAL_HELP hoặc rooms=0 vì lookup theo room_id thay vì room_code",
        "check": lambda r: (
            r.get("intent") == "ASK_ABOUT_ROOM"
            and bool(r.get("rooms"))
            and (r.get("rooms") or [{}])[0].get("room_code", "").replace(".", "") == "P305"
        ),
    },
    {
        "name": "Thú cưng thiếu dữ liệu",
        "question": "Phòng này có cho nuôi mèo không?",
        "before": "Có thể đoán Có/Không dù thiếu dữ liệu",
        "check": lambda r: "chưa có dữ liệu xác minh" in (r.get("answer") or "").lower(),
        "setup_room": "tro-1",
    },
    {
        "name": "Search rõ quận 7",
        "question": "quận 7 dưới 5 triệu",
        "before": "Vẫn đúng nhưng có thể bị LLM lệch intent",
        "check": lambda r: r.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} and bool(r.get("rooms")),
    },
    {
        "name": "Sales handoff khi không có phòng public",
        "question": "Tìm phòng quận 7 dưới 1 triệu",
        "before": "Gợi ý nới ngân sách thay vì chuyển sales",
        "check": lambda r, ctx: (
            "tìm mỏi mắt" in (r.get("answer") or "").lower()
            and ("sales" in (r.get("answer") or "").lower() or "tư vấn" in (r.get("answer") or "").lower())
        ),
        "rooms": [],
    },
    {
        "name": "Detail máy lạnh không bẩn filter",
        "question": "Phòng này có máy lạnh không?",
        "before": "Lưu amenities_required=air_conditioner vào session",
        "check": lambda r, ctx: (
            r.get("intent") == "ASK_ABOUT_ROOM"
            and "air_conditioner" not in (r.get("session_state", {}).get("constraints", {}).get("amenities_required") or [])
        ),
        "setup_results": ["list-a", "list-b", "list-c"],
    },
    {
        "name": "Ordinal vượt list",
        "question": "Giá phòng số 4 bao nhiêu?",
        "before": "Fallback phòng cũ trong list",
        "check": lambda r, ctx: (
            "phòng số 4" in (r.get("answer") or "").lower() or "phòng số **4**" in (r.get("answer") or "").lower()
            or "chỉ có" in (r.get("answer") or "").lower()
        ),
        "setup_results": ["list-a", "list-b", "list-c"],
    },
    {
        "name": "So sánh phòng 1 và 3",
        "question": "So sánh phòng đầu tiên và phòng thứ ba",
        "before": "Chỉ resolve 1 phòng hoặc fallback sai",
        "check": lambda r, ctx: len((r.get("comparison") or {}).get("rows") or []) >= 2,
        "setup_results": ["list-a", "list-b", "list-c"],
    },
    {
        "name": "Search generic không kích price gate",
        "question": "Tìm phòng quận 7",
        "before": "Keyword 'giá' rộng gây abstain/sufficiency sai trên search",
        "check": lambda r: r.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} and bool(r.get("rooms")),
    },
]


async def _run_one(scenario: dict) -> dict:
    rooms = scenario.get("rooms")
    if rooms is None:
        rooms = _rooms()
    repo = InMemoryRoomRepository(rooms)
    store = InMemorySessionStore()
    session_id = f"qa-{scenario['name']}"

    state = default_session_state(session_id)
    if scenario.get("setup_room"):
        state["current_room_id"] = scenario["setup_room"]
        state["last_result_ids"] = [scenario["setup_room"]]
        state["last_intent"] = "ASK_ABOUT_ROOM"
        store.save(session_id, state, ttl_seconds=3600)
    elif scenario.get("setup_results"):
        state["last_result_ids"] = list(scenario["setup_results"])
        state["current_room_id"] = scenario["setup_results"][0]
        state["last_intent"] = "SEARCH_ROOM"
        store.save(session_id, state, ttl_seconds=3600)

    return await run_room_assistant(
        scenario["question"],
        session_id=session_id,
        repository=repo,
        session_store=store,
        semantic_index=None,
    )


def main() -> None:
    print("=" * 72)
    print("KIỂM TRA Q&A SAU FIX PRODUCTION (so với hành vi trước đó)")
    print("=" * 72)

    passed = 0
    for scenario in SCENARIOS:
        result = asyncio.run(_run_one(scenario))
        check = scenario["check"]
        try:
            ok = check(result, scenario)
        except TypeError:
            ok = check(result)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {scenario['name']}")
        print(f"  Q: {scenario['question']}")
        print(f"  Trước: {scenario['before']}")
        print(f"  Sau: intent={result.get('intent')} | rooms={len(result.get('rooms') or [])}")
        if result.get("comparison"):
            print(f"       compare_rows={len((result.get('comparison') or {}).get('rows') or [])}")
        answer = (result.get("answer") or "").replace("\n", " ")[:200]
        print(f"  Answer: {answer}...")

    parsed = parse_intent_and_constraint_patch("phòng P.305 giá bao nhiêu")
    print(f"\n[INTENT] P.305 refs={parsed['referenced_room_ids']} intent={parsed['intent']}")

    constraints = {"categories": ["studio"]}
    studio_ok = room_matches_constraints(_rooms()[0], constraints)
    tro_ok = room_matches_constraints(_rooms()[1], constraints)
    print(f"\n[FILTER] studio room matches studio={studio_ok}, tro room matches studio={tro_ok}")

    pets_c = {"pets_required": ["cat"]}
    print(f"[FILTER] pet room allows pets={room_matches_constraints(_rooms()[2], pets_c)}")
    print(f"[FILTER] no-pet room allows pets={room_matches_constraints(_rooms()[3], pets_c)}")

    grounding = {"rooms": [_rooms()[1]], "constraints": {}}
    answer = _compose_answer_template(
        {"intent": "ASK_ABOUT_ROOM"},
        grounding,
        {},
        question="Phòng này có cho nuôi mèo không?",
    )
    print(f"\n[SUFFICIENCY] pets question without data: {'chưa có dữ liệu xác minh' in answer.lower()}")

    state = default_session_state("intent-detail")
    state["last_result_ids"] = ["list-a"]
    state["current_room_id"] = "list-a"
    detail_parsed = parse_intent_and_constraint_patch("Phòng này có máy lạnh không?", state)
    amenity_ops = [op for op in detail_parsed["operations"] if op.get("path") == "amenities_required"]
    print(f"[INTENT] detail AC ops={amenity_ops} (expect empty)")

    print("\n" + "=" * 72)
    print(f"Kết quả: {passed}/{len(SCENARIOS)} scenario PASS")
    print("=" * 72)
    if passed < len(SCENARIOS):
        sys.exit(1)


if __name__ == "__main__":
    main()
