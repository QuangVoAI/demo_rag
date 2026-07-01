"""Regression Q&A theo nhận xét production P0/P1 (chạy nhanh, không Qdrant)."""

from __future__ import annotations

import asyncio
import unittest

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import InMemoryRoomRepository, room_matches_constraints
from room_assistant.schemas import default_session_state
from room_assistant.session_store import InMemorySessionStore
from room_assistant.tools import resolve_sensitive_answer_types
from room_assistant.workflow import run_room_assistant


def _fixture_rooms() -> list[dict]:
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
            "embedding_text": "Studio gọn, có gác, Quận 7",
            "available": True,
            "status": "active",
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
            "embedding_text": "Phòng trọ tiện nghi, Quận 7",
            "available": True,
            "status": "active",
        },
        {
            "room_id": "pet-1",
            "metadata": {
                "house_name": "Pet",
                "room_code": "PET1",
                "price": 4_000_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "embedding_text": "## Tiện ích\n- Thú cưng: Có\n- Wifi: Có",
            "available": True,
            "status": "active",
        },
        {
            "room_id": "no-pet-1",
            "metadata": {
                "house_name": "NoPet",
                "room_code": "NP1",
                "price": 3_800_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "embedding_text": "## Tiện ích\n- Thú cưng: Không\n- Wifi: Có",
            "available": True,
            "status": "active",
        },
        {
            "room_id": "mongo-p305",
            "metadata": {
                "house_name": "C2 HQV",
                "room_code": "P.305",
                "price": 4_800_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "embedding_text": "## Tiện ích\n- Máy lạnh: Có",
            "available": True,
            "status": "active",
        },
    ]


class ProductionQaBeforeAfterTests(unittest.TestCase):
    def _run(self, question: str, session_id: str, *, rooms: list[dict] | None = None, state: dict | None = None) -> dict:
        repo = InMemoryRoomRepository(rooms if rooms is not None else _fixture_rooms())
        store = InMemorySessionStore()
        if state:
            store.save(session_id, state, ttl_seconds=3600)
        return asyncio.run(run_room_assistant(
            question,
            session_id=session_id,
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))

    def test_studio_filter_excludes_tro(self):
        result = self._run("Tìm studio quận 7", "qa-studio")
        self.assertTrue(result["rooms"])
        for room in result["rooms"]:
            self.assertTrue(room_matches_constraints(room, {"categories": ["studio"]}))

    def test_pet_filter_excludes_no_pet_room(self):
        result = self._run("Tìm phòng quận 7 cho nuôi mèo", "qa-pet")
        self.assertTrue(result["rooms"])
        for room in result["rooms"]:
            self.assertNotIn("thú cưng: không", (room.get("embedding_text") or "").lower())

    def test_p305_resolves_room_and_price(self):
        result = self._run("phòng P.305 giá bao nhiêu", "qa-p305")
        self.assertEqual(result["intent"], "ASK_ABOUT_ROOM")
        self.assertTrue(result["rooms"])
        self.assertEqual(result["rooms"][0].get("room_code"), "P.305")
        self.assertIn("4.800.000", result["answer"])

    def test_pets_unknown_abstains(self):
        state = default_session_state("qa-pets-unknown")
        state["current_room_id"] = "tro-1"
        state["last_result_ids"] = ["tro-1"]
        result = self._run("Phòng này có cho nuôi mèo không?", "qa-pets-unknown", state=state)
        self.assertIn("chưa có dữ liệu xác minh", result["answer"].lower())

    def test_sales_handoff_when_no_public_match(self):
        result = self._run(
            "Tìm phòng quận 7 dưới 1 triệu",
            "qa-sales",
            rooms=[],
        )
        answer = result["answer"].lower()
        self.assertIn("tìm mỏi mắt", answer)
        self.assertTrue("sales" in answer or "tư vấn" in answer)

    def test_detail_ac_question_does_not_append_amenity_filter(self):
        state = default_session_state("qa-detail-ac")
        state["last_result_ids"] = ["studio-1"]
        state["current_room_id"] = "studio-1"
        state["last_intent"] = "SEARCH_ROOM"
        result = self._run("Phòng này có máy lạnh không?", "qa-detail-ac", state=state)
        amenities = result["session_state"]["constraints"].get("amenities_required") or []
        self.assertNotIn("air_conditioner", amenities)

    def test_ordinal_fourth_asks_clarify_not_fallback(self):
        state = default_session_state("qa-ord-4")
        state["last_result_ids"] = ["studio-1", "tro-1", "pet-1"]
        state["last_intent"] = "SEARCH_ROOM"
        result = self._run("Giá phòng số 4 bao nhiêu?", "qa-ord-4", state=state)
        self.assertIn("chỉ có", result["answer"].lower())
        self.assertNotEqual(result["session_state"].get("current_room_id"), "studio-1")

    def test_compare_first_and_third_returns_two_rows(self):
        state = default_session_state("qa-cmp-13")
        state["last_result_ids"] = ["studio-1", "tro-1", "pet-1"]
        state["last_intent"] = "SEARCH_ROOM"
        result = self._run(
            "So sánh phòng đầu tiên và phòng thứ ba",
            "qa-cmp-13",
            state=state,
        )
        self.assertEqual(result["intent"], "COMPARE_ROOMS")
        rows = (result.get("comparison") or {}).get("rows") or []
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual([row["room_id"] for row in rows[:2]], ["studio-1", "pet-1"])

    def test_generic_search_no_price_sensitive_gate(self):
        types = resolve_sensitive_answer_types(
            "Tìm phòng quận 7",
            intent="SEARCH_ROOM",
            has_room_context=False,
        )
        self.assertNotIn("price_query", types)
        result = self._run("Tìm phòng quận 7", "qa-generic-search")
        self.assertTrue(result["rooms"])

    def test_session_search_then_detail_then_refine_keeps_clean_filters(self):
        store = InMemorySessionStore()
        repo = InMemoryRoomRepository(_fixture_rooms())
        session_id = "qa-session-flow"

        search = asyncio.run(run_room_assistant(
            "Tìm phòng quận 7 dưới 5 triệu",
            session_id=session_id,
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))
        self.assertTrue(search["rooms"])

        detail = asyncio.run(run_room_assistant(
            "Phòng này có wifi không?",
            session_id=session_id,
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))
        self.assertEqual(detail["intent"], "ASK_ABOUT_ROOM")
        self.assertNotIn("wifi", detail["session_state"]["constraints"].get("amenities_required") or [])

        refine = asyncio.run(run_room_assistant(
            "Thêm điều kiện có máy lạnh",
            session_id=session_id,
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))
        self.assertIn("air_conditioner", refine["session_state"]["constraints"].get("amenities_required") or [])


if __name__ == "__main__":
    unittest.main()
