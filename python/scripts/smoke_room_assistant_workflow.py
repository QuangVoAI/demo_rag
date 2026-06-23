"""Production workflow smoke test for the read-only room assistant.

Runs varied user questions against an in-memory repository while still using
the real orchestration, response writer, and Langfuse decorators.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from room_assistant.repository import InMemoryRoomRepository
from room_assistant.session_store import InMemorySessionStore
from room_assistant.workflow import run_room_assistant


FIXTURES: list[dict[str, Any]] = [
    {
        "room_id": "A101",
        "title": "Studio Bình Thạnh",
        "description": "Phòng sáng, yên tĩnh",
        "available": True,
        "status": "active",
        "district": "Bình Thạnh",
        "rent_price": 4_500_000,
        "deposit": 4_500_000,
        "fees": {"water": 100_000, "parking": 150_000},
        "amenities": ["air_conditioner", "window"],
        "area_m2": 24,
        "source_version": 1,
    },
    {
        "room_id": "B202",
        "title": "Phòng Quận 7",
        "available": True,
        "status": "active",
        "district": "quan 7",
        "rent_price": 6_000_000,
        "amenities": ["balcony"],
        "source_version": 1,
    },
    {
        "room_id": "C303",
        "title": "Phòng hết chỗ",
        "available": False,
        "status": "active",
        "district": "Bình Thạnh",
        "rent_price": 4_000_000,
        "source_version": 1,
    },
    {
        "room_id": "E505",
        "title": "Phòng nhỏ Gò Vấp",
        "available": True,
        "status": "active",
        "district": "Gò Vấp",
        "rent_price": 3_800_000,
        "amenities": ["window"],
        "area_m2": 18,
        "source_version": 1,
    },
]


class RecordingSemanticIndex:
    def search_rooms(self, query_text, candidate_ids, top_k, metadata_filter=None):
        return [{"room_id": item, "score": 1.0} for item in candidate_ids[:top_k]]


CASES: list[dict[str, Any]] = [
    {
        "name": "full_sentence",
        "question": "Mình cần phòng dưới 5 triệu ở quận Bình Thạnh có máy lạnh, ở ngay được không?",
        "expected_intent": "SEARCH_ROOM",
        "expected_rooms": ["A101"],
    },
    {
        "name": "short_no_accent",
        "question": "quan 7 ban cong",
        "expected_intent": "SEARCH_ROOM",
        "expected_rooms": ["B202"],
    },
    {
        "name": "light_typo_no_accent",
        "question": "tim phong binh thanh may lanh duoi 5tr",
        "expected_intent": "SEARCH_ROOM",
        "expected_rooms": ["A101"],
    },
    {
        "name": "many_conditions",
        "question": "Tìm phòng dưới 7 triệu quận 7 có ban công",
        "expected_intent": "SEARCH_ROOM",
        "expected_rooms": ["B202"],
    },
    {
        "name": "negated_feature",
        "question": "tim phong duoi 5tr khong may lanh",
        "expected_intent": "SEARCH_ROOM",
        "expected_rooms": ["E505"],
    },
    {
        "name": "missing_cost_data",
        "question": "Tính tổng chi phí #B202 nếu thuê 6 tháng",
        "expected_intent": "CALCULATE_COST",
        "expected_unknown": "deposit",
    },
    {
        "name": "compare_missing_and_overflow",
        "question": "So sánh #A101 #B202 #Z999 #E505",
        "expected_intent": "COMPARE_ROOMS",
        "expected_missing": "Z999",
        "expected_not_compared": "E505",
    },
    {
        "name": "outside_context",
        "question": "Thời tiết hôm nay ở Sài Gòn sao?",
        "expected_intent": "GENERAL_HELP",
        "expected_rooms": [],
    },
    {
        "name": "read_only_refusal",
        "question": "Đặt lịch xem phòng A101 giúp mình",
        "expected_intent": "REQUEST_ACTION",
        "expected_rooms": [],
    },
]


async def main() -> None:
    repo = InMemoryRoomRepository(FIXTURES)
    store = InMemorySessionStore()
    semantic = RecordingSemanticIndex()
    session_prefix = f"codex-smoke-{int(time.time())}"
    summaries = []

    for idx, case in enumerate(CASES, 1):
        result = await run_room_assistant(
            case["question"],
            session_id=f"{session_prefix}-{idx}-{case['name']}",
            repository=repo,
            session_store=store,
            semantic_index=semantic,
        )
        room_ids = [item.get("room_id") for item in result.get("rooms", [])]
        assert result["intent"] == case["expected_intent"], (case["name"], result["intent"])
        if "expected_rooms" in case:
            assert room_ids == case["expected_rooms"], (case["name"], room_ids)
        if "expected_unknown" in case:
            assert case["expected_unknown"] in result["cost_estimate"]["unknown"], case["name"]
        if "expected_missing" in case:
            assert case["expected_missing"] in result["comparison"]["missing_room_ids"], case["name"]
        if "expected_not_compared" in case:
            assert case["expected_not_compared"] in result["comparison"]["not_compared_room_ids"], case["name"]

        trace = result.get("agent_trace", {})
        assert trace.get("write_tool_calls") == 0, case["name"]
        assert trace.get("read_tool_calls", 0) <= trace.get("max_read_tool_calls_per_turn", 3), case["name"]
        summaries.append({
            "name": case["name"],
            "intent": result["intent"],
            "rooms": room_ids,
            "read_tool_calls": trace.get("read_tool_calls"),
            "latency_ms": result.get("processing_time_ms"),
            "error": trace.get("error_category"),
        })

    try:
        from utils.observability import flush_langfuse
        flush_langfuse()
    except Exception:
        pass

    print(json.dumps({"session_prefix": session_prefix, "cases": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
