"""Regression matrix for intent routing edge cases (read-only advisory bot)."""

from __future__ import annotations

import unittest

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.session_store import default_session_state


def _state() -> dict:
    state = default_session_state("matrix")
    state["current_room_id"] = "B202"
    state["last_result_ids"] = ["A101", "B202", "C303"]
    state["last_intent"] = "SEARCH_ROOM"
    return state


class IntentRegressionMatrixTests(unittest.TestCase):
    def test_matrix(self):
        state = _state()
        cases = [
            ("Phòng này có tiện ích gì?", state, "ASK_ABOUT_ROOM", "B202", None),
            ("Phòng đó có wifi không?", state, "ASK_ABOUT_ROOM", "B202", None),
            ("Khu này an ninh không?", state, "ASK_ABOUT_ROOM", "B202", None),
            ("Phòng có yên tĩnh không?", state, "ASK_ABOUT_ROOM", "B202", None),
            ("Tìm phòng tương tự phòng này", state, "FIND_SIMILAR", "B202", None),
            ("Tóm tắt ưu nhược điểm phòng này", state, "SUMMARIZE_ROOM", "B202", None),
            ("Chi phí phòng này thế nào", state, "CALCULATE_COST", "B202", None),
            ("Phòng kia có máy lạnh không?", state, "ASK_ABOUT_ROOM", "C303", None),
            ("Hủy lịch hẹn", state, "REQUEST_ACTION", None, "huy_lich"),
            ("Đổi lịch hẹn xem phòng", state, "REQUEST_ACTION", None, "doi_lich"),
            ("Giảm giá cho em đi", state, "REQUEST_ACTION", None, "negotiate"),
            ("Giảm giá được không?", state, "REQUEST_FAQ", None, None),
            ("Sinh viên nên lưu ý gì khi thuê trọ?", None, "REQUEST_FAQ", None, None),
            ("Làm sao nhận biết tin lừa đảo?", None, "REQUEST_FAQ", None, None),
            ("Có phòng không chung chủ, ban công không?", state, "SEARCH_ROOM", None, None),
            ("Phòng nào yên tĩnh hơn?", state, "REFINE_SEARCH", None, None),
            ("Có thể đặt lịch xem phòng không?", None, "REQUEST_FAQ", None, None),
        ]
        for question, ctx, intent, room_id, action in cases:
            with self.subTest(question=question):
                parsed = parse_intent_and_constraint_patch(question, ctx)
                self.assertEqual(parsed["intent"], intent)
                self.assertEqual(parsed.get("current_room_id"), room_id)
                self.assertEqual(parsed.get("requested_action"), action)
                if intent == "ASK_ABOUT_ROOM" and room_id:
                    preferred = [
                        op for op in parsed["operations"] if op.get("path") == "amenities_preferred"
                    ]
                    self.assertEqual(preferred, [])


if __name__ == "__main__":
    unittest.main()
