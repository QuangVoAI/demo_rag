"""So sánh hành vi Q&A trước/sau các fix accuracy (Jun 2026)."""
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
            "available": True,
            "rent_price": 4_500_000,
            "district": "Quận 7",
            "title": "Studio Q7",
            "embedding_text": "Studio gọn, có gác, Quận 7",
        },
        {
            "room_id": "tro-1",
            "available": True,
            "rent_price": 3_200_000,
            "district": "Quận 7",
            "title": "Phòng trọ thường",
            "embedding_text": "Phòng trọ tiện nghi, Quận 7",
        },
        {
            "room_id": "pet-1",
            "available": True,
            "rent_price": 4_000_000,
            "district": "Quận 7",
            "title": "Phòng cho nuôi mèo",
            "embedding_text": "## Tiện ích\n- Thú cưng: Có\n- Wifi: Có",
        },
        {
            "room_id": "no-pet-1",
            "available": True,
            "rent_price": 3_800_000,
            "district": "Quận 7",
            "title": "Phòng không thú cưng",
            "embedding_text": "## Tiện ích\n- Thú cưng: Không\n- Wifi: Có",
        },
        {
            "room_id": "p305-room",
            "available": True,
            "rent_price": 4_800_000,
            "district": "Quận 7",
            "room_code": "P.305",
            "title": "C2 HOÀNG QUỐC VIỆT - P.305",
            "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7",
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
        "before": "GENERAL_HELP — không nhận P.305",
        "check": lambda r: r.get("intent") == "ASK_ABOUT_ROOM" and bool(r.get("rooms")),
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
]


async def _run_one(scenario: dict) -> dict:
    repo = InMemoryRoomRepository(_rooms())
    store = InMemorySessionStore()
    session_id = f"qa-{scenario['name']}"
    if scenario.get("setup_room"):
        state = default_session_state(session_id)
        state["current_room_id"] = scenario["setup_room"]
        state["last_result_ids"] = [scenario["setup_room"]]
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
    print("KIỂM TRA Q&A SAU FIX (so với hành vi trước đó)")
    print("=" * 72)

    passed = 0
    for scenario in SCENARIOS:
        result = asyncio.run(_run_one(scenario))
        ok = scenario["check"](result)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {scenario['name']}")
        print(f"  Q: {scenario['question']}")
        print(f"  Trước: {scenario['before']}")
        print(f"  Sau: intent={result.get('intent')} | rooms={len(result.get('rooms') or [])}")
        answer = (result.get("answer") or "").replace("\n", " ")[:160]
        print(f"  Answer: {answer}...")

    # Intent-only check for P.305 without full run
    parsed = parse_intent_and_constraint_patch("phòng P.305 giá bao nhiêu")
    print(f"\n[INTENT] P.305 refs={parsed['referenced_room_ids']} intent={parsed['intent']}")

    # Filter unit checks
    constraints = {"categories": ["studio"]}
    studio_ok = room_matches_constraints(_rooms()[0], constraints)
    tro_ok = room_matches_constraints(_rooms()[1], constraints)
    print(f"\n[FILTER] studio room matches studio={studio_ok}, tro room matches studio={tro_ok}")

    pets_c = {"pets_required": ["cat"]}
    print(f"[FILTER] pet room allows pets={room_matches_constraints(_rooms()[2], pets_c)}")
    print(f"[FILTER] no-pet room allows pets={room_matches_constraints(_rooms()[3], pets_c)}")

    # Sufficiency template
    grounding = {"rooms": [_rooms()[1]], "constraints": {}}
    answer = _compose_answer_template(
        {"intent": "ASK_ABOUT_ROOM"},
        grounding,
        {},
        question="Phòng này có cho nuôi mèo không?",
    )
    print(f"\n[SUFFICIENCY] pets question without data: {'chưa có dữ liệu xác minh' in answer.lower()}")

    print("\n" + "=" * 72)
    print(f"Kết quả: {passed}/{len(SCENARIOS)} scenario PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
