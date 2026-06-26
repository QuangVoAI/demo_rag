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

    def test_budget_range(self):
        for question in (
            "Ngân sách 6-8 triệu, 1 người ở",
            "Tôi cần phòng 6 đến 8 triệu",
            "khoảng 6 tới 8tr",
        ):
            with self.subTest(question=question):
                parsed = parse_intent_and_constraint_patch(question)
                self.assertEqual(parsed["intent"], "SEARCH_ROOM")
                self.assertIn({"op": "set", "path": "budget.min", "value": 6_000_000}, parsed["operations"])
                self.assertIn({"op": "set", "path": "budget.max", "value": 8_000_000}, parsed["operations"])

    def test_move_in_this_month(self):
        parsed = parse_intent_and_constraint_patch("Có thể dọn vào trong tháng này")
        self.assertIn({"op": "set", "path": "move_in_date", "value": "this_month"}, parsed["operations"])

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

    def test_sync_intent_routing_matrix(self):
        state = default_session_state("s-routing")
        state["current_room_id"] = "A101"
        state["selected_room_ids"] = ["A101", "B202", "C303"]
        state["last_result_ids"] = ["A101", "B202", "C303"]
        state["last_intent"] = "SEARCH_ROOM"

        cases = [
            ("Xin chào", None, "GENERAL_HELP"),
            ("Tôi muốn tìm phòng ở TP.HCM", None, "SEARCH_ROOM"),
            ("Ngân sách 6-8 triệu, 1 người ở", None, "SEARCH_ROOM"),
            ("Rẻ hơn một chút được không?", state, "REFINE_SEARCH"),
            ("Hãy cho tôi biết giá và tiện ích của phòng đầu tiên", state, "ASK_ABOUT_ROOM"),
            ("Diện tích bao nhiêu?", state, "ASK_ABOUT_ROOM"),
            ("Có nội thất không?", state, "ASK_ABOUT_ROOM"),
            ("Có chỗ để xe không?", state, "ASK_ABOUT_ROOM"),
            ("Chi phí điện nước thế nào?", state, "CALCULATE_COST"),
            ("Hãy so sánh 3 phòng này trong một bảng", state, "COMPARE_ROOMS"),
            ("so sánh 3 phòng đầu tiên nhé", state, "COMPARE_ROOMS"),
            ("Theo bạn phòng nào phù hợp nhất với nhu cầu của tôi?", state, "COMPARE_ROOMS"),
            ("Tôi chọn phòng số 2", state, "ASK_ABOUT_ROOM"),
            ("Giá phòng số 2 bao nhiêu?", state, "ASK_ABOUT_ROOM"),
            ("Diện tích phòng thứ ba thế nào?", state, "ASK_ABOUT_ROOM"),
            ("Tìm phòng tương tự phòng này", state, "FIND_SIMILAR"),
            ("Tóm tắt ưu nhược điểm phòng này", state, "SUMMARIZE_ROOM"),
            ("Thủ tục ký hợp đồng như thế nào?", None, "REQUEST_FAQ"),
            ("Tôi muốn đặt phòng này", state, "REQUEST_ACTION"),
            ("Thanh toán tiền cọc giúp tôi", state, "REQUEST_ACTION"),
            ("Cảm ơn bạn", state, "GENERAL_HELP"),
            ("Ok được rồi", state, "GENERAL_HELP"),
            ("Thời tiết hôm nay sao?", None, "GENERAL_HELP"),
        ]

        for question, current_state, expected in cases:
            with self.subTest(question=question):
                parsed = parse_intent_and_constraint_patch(question, current_state)
                self.assertEqual(parsed["intent"], expected)

    def test_async_intent_keeps_regex_when_llm_is_weaker(self):
        state = default_session_state("s-async-routing")
        state["last_result_ids"] = ["A101", "B202", "C303"]
        state["last_intent"] = "SEARCH_ROOM"

        async def weak_wrong_llm(_question, _state):
            return "GENERAL_HELP", 0.4

        cases = [
            ("Tôi muốn tìm phòng ở TP.HCM", "SEARCH_ROOM"),
            ("Ngân sách 6-8 triệu, 1 người ở", "REFINE_SEARCH"),
            ("Theo bạn phòng nào phù hợp nhất với nhu cầu của tôi?", "COMPARE_ROOMS"),
            ("Tôi muốn đặt phòng này", "REQUEST_ACTION"),
        ]

        with patch("room_assistant.intent._llm_classify_intent", weak_wrong_llm):
            for question, expected in cases:
                with self.subTest(question=question):
                    parsed = asyncio.run(parse_intent_async(question, state))
                    self.assertEqual(parsed["intent"], expected)

    def test_detail_question_can_reference_ordinal_room(self):
        state = default_session_state("s-ordinal")
        state["last_result_ids"] = ["A101", "B202", "C303"]
        state["last_intent"] = "SEARCH_ROOM"

        parsed = parse_intent_and_constraint_patch("Giá phòng số 2 bao nhiêu?", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        self.assertEqual(parsed["current_room_id"], "B202")

        parsed = parse_intent_and_constraint_patch("Diện tích phòng thứ ba thế nào?", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        self.assertEqual(parsed["current_room_id"], "C303")

    def test_async_llm_can_rescue_unclear_general_text(self):
        async def faq_llm(_question, _state):
            return "REQUEST_FAQ", 0.95

        with patch("room_assistant.intent._llm_classify_intent", faq_llm):
            parsed = asyncio.run(parse_intent_async("Quy trình bên mình ra sao?", None))
        self.assertEqual(parsed["intent"], "REQUEST_FAQ")

    def test_async_llm_university_path_is_mapped_to_near_landmark(self):
        async def tdtu_llm(_question, _state):
            return {
                "intent": "SEARCH_ROOM",
                "confidence": 0.95,
                "operations": [{"op": "set", "path": "location.university", "value": "tdtu"}],
                "referenced_room_ids": [],
            }

        with patch("room_assistant.intent._llm_classify_intent", tdtu_llm):
            parsed = asyncio.run(parse_intent_async("có phòng nào gần TDTU k", None))
        self.assertIn(
            {"op": "append", "path": "location.near_landmarks", "value": "tdtu"},
            parsed["operations"],
        )

    def test_async_router_verifier_keeps_regex_hard_slots_on_conflict(self):
        state = default_session_state("s-verifier")

        async def wrong_llm(_question, _state):
            return {
                "intent": "SEARCH_ROOM",
                "confidence": 0.98,
                "operations": [{"op": "append", "path": "location.districts", "value": "quan 7"}],
                "referenced_room_ids": [],
            }

        async def verifier(_question, _state, _regex_candidate, _llm_candidate):
            return {
                "approved_intent": "SEARCH_ROOM",
                "approved_operations": [],
                "approved_room_ids": [],
                "approved_requested_action": None,
                "use_llm_intent": False,
                "use_llm_hard_slots": False,
                "allow_llm_soft_slots": False,
                "hard_conflict": True,
                "reason": "district_conflict_keep_regex",
            }

        with patch("room_assistant.intent._llm_classify_intent", wrong_llm):
            with patch("room_assistant.intent._llm_verify_routing_decision", verifier):
                parsed = asyncio.run(parse_intent_async("tìm cho tôi nhà quận 5", state))
        self.assertIn(
            {"op": "append", "path": "location.districts", "value": "quan 5"},
            parsed["operations"],
        )
        self.assertNotIn(
            {"op": "append", "path": "location.districts", "value": "quan 7"},
            parsed["operations"],
        )

    def test_async_router_verifier_allows_llm_intent_rescue_for_unclear_text(self):
        async def faq_llm(_question, _state):
            return {
                "intent": "REQUEST_FAQ",
                "confidence": 0.97,
                "operations": [],
                "referenced_room_ids": [],
            }

        async def verifier(_question, _state, _regex_candidate, _llm_candidate):
            return {
                "approved_intent": "REQUEST_FAQ",
                "approved_operations": [],
                "approved_room_ids": [],
                "approved_requested_action": None,
                "use_llm_intent": True,
                "use_llm_hard_slots": False,
                "allow_llm_soft_slots": True,
                "hard_conflict": False,
                "reason": "llm_better_for_ambiguous_faq",
            }

        with patch("room_assistant.intent._llm_classify_intent", faq_llm):
            with patch("room_assistant.intent._llm_verify_routing_decision", verifier):
                parsed = asyncio.run(parse_intent_async("quy trình bên mình ra sao?", None))
        self.assertEqual(parsed["intent"], "REQUEST_FAQ")


if __name__ == "__main__":
    unittest.main()
