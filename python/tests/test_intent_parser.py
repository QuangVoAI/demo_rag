import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import asyncio

sys.path.append(str(Path(__file__).parent.parent))

from room_assistant.intent import parse_intent_and_constraint_patch, parse_intent_async
from room_assistant.schemas import default_session_state


class IntentParserTests(unittest.TestCase):
    def _values(self, text, path):
        parsed = parse_intent_and_constraint_patch(text)
        return [op["value"] for op in parsed["operations"] if op["path"] == path]

    def test_extract_room_type(self):
        parsed = parse_intent_and_constraint_patch("tim studio co gac")
        ops = parsed["operations"]

        categories = [op["value"] for op in ops if op["path"] == "categories"]
        self.assertIn("studio", categories)

        amenities = [op["value"] for op in ops if op["path"] == "amenities_required"]
        self.assertIn("mezzanine", amenities)

    def test_extract_ev_charging_vs_parking(self):
        self.assertIn(
            "ev_charging",
            self._values("phong co sac xe dien khong", "amenities_required"),
        )
        self.assertIn(
            "electric_bike",
            self._values("cho de xe dien", "vehicles"),
        )
        self.assertIn(
            "ev_charging",
            self._values("phong co nhan xe dien khong", "amenities_required"),
        )

    def test_extract_free_hours(self):
        self.assertIn(
            "free_hours",
            self._values("tim phong gio tu do", "amenities_required"),
        )

    def test_compact_million(self):
        parsed = parse_intent_and_constraint_patch("phong duoi 3tr5 o quan 1")
        budget = next((op["value"] for op in parsed["operations"] if op["path"] == "budget.max"), None)
        self.assertEqual(budget, 3_500_000)

        parsed = parse_intent_and_constraint_patch("phong duoi 1 tr 3 thang")
        budget = next((op["value"] for op in parsed["operations"] if op["path"] == "budget.max"), None)
        self.assertEqual(budget, 1_000_000)

    def test_chdv_2pn(self):
        parsed = parse_intent_and_constraint_patch("muon thue CHDV 2 phong ngu")
        categories = [op["value"] for op in parsed["operations"] if op["path"] == "categories"]
        self.assertIn("chdv", categories)
        self.assertIn("2pn", categories)

    def test_async_llm_summary_does_not_override_detail_question(self):
        state = default_session_state("s-detail")
        state["current_room_id"] = "A101"
        state["last_intent"] = "SEARCH_ROOM"

        async def fake_llm(_question, _state):
            return "SUMMARIZE_ROOM", 0.99

        with patch("room_assistant.intent._llm_classify_intent", fake_llm):
            parsed = asyncio.run(parse_intent_async("Phòng này có tiện ích gì?", state))
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")


if __name__ == "__main__":
    unittest.main()
