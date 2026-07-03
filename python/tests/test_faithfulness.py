"""Tests for post-generation answer faithfulness guards."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.faithfulness import (
    answer_cites_foreign_room_ids,
    answer_denies_inventory_while_rooms_exist,
    repair_answer_against_grounding,
)


class FaithfulnessTests(unittest.TestCase):
    def test_denies_inventory_when_rooms_exist(self):
        room_ids = ["645a2c675a845d1d07828614"]
        bad = "Dạ em tìm mỏi mắt mà chưa thấy phòng nào phù hợp ạ."
        good = "Dạ còn phòng ạ! 1. **P.502** (#645a2c675a845d1d07828614) — 4.200.000 VND/tháng."
        self.assertTrue(answer_denies_inventory_while_rooms_exist(bad, room_ids))
        self.assertFalse(answer_denies_inventory_while_rooms_exist(good, room_ids))

    def test_relaxed_opening_with_room_ids_is_not_denial(self):
        room_ids = ["645a2c675a845d1d07828614"]
        relaxed = (
            "Dạ em chưa thấy căn khớp đúng chỗ để xe nên em xin phép gợi ý mấy căn gần đúng nhất nha. "
            "1. **P.502** (#645a2c675a845d1d07828614) — 4.200.000 VND/tháng."
        )
        self.assertFalse(answer_denies_inventory_while_rooms_exist(relaxed, room_ids))

    def test_foreign_room_id_detection(self):
        allowed = ["645a2c675a845d1d07828614"]
        foreign = "Phòng #64d5c60bcbea327500561f38 có ban công không?"
        self.assertEqual(
            answer_cites_foreign_room_ids(foreign, allowed),
            ["64d5c60bcbea327500561f38"],
        )

    def test_repair_swaps_contradictory_llm_answer(self):
        rooms = [{"room_id": "645a2c675a845d1d07828614", "title": "P.502"}]
        template = "Dạ còn phòng ạ! (#645a2c675a845d1d07828614)"
        llm_bad = "Dạ em không tìm thấy phòng nào phù hợp ạ."
        repaired, meta = repair_answer_against_grounding(
            llm_bad,
            intent="REFINE_SEARCH",
            rooms=rooms,
            template_answer=template,
        )
        self.assertTrue(meta["repaired"])
        self.assertEqual(repaired, template)


if __name__ == "__main__":
    unittest.main()
