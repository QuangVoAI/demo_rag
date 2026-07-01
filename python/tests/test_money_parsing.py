"""Tests for VND money parsing and answer matching."""

from __future__ import annotations

import unittest

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.money import answer_mentions_vnd, extract_colloquial_budget_vnd, money_to_vnd, normalize_money_text


class MoneyParsingTests(unittest.TestCase):
    def test_colloquial_million_units(self):
        cases = {
            "tim phong duoi 5 cu": 5_000_000,
            "ngan sach 5m": 5_000_000,
            "duoi 5000k": 5_000_000,
            "khoang 5 chieu": 5_000_000,
            "5 chai": 5_000_000,
            "3tr5": None,  # handled by compact pattern in intent
        }
        for text, expected in cases.items():
            if expected is None:
                continue
            normalized = normalize_money_text(text)
            self.assertEqual(extract_colloquial_budget_vnd(normalized), expected, msg=text)

    def test_cu_chi_district_not_budget(self):
        normalized = normalize_money_text("tim phong cu chi")
        self.assertIsNone(extract_colloquial_budget_vnd(normalized))

    def test_intent_parses_colloquial_budget_ops(self):
        for question, expected in (
            ("Tìm phòng dưới 5 củ quận 7", 5_000_000),
            ("Phòng dưới 5m Bình Thạnh", 5_000_000),
            ("Ngân sách 5000k", 5_000_000),
            ("Dưới 5 chiệu", 5_000_000),
        ):
            parsed = parse_intent_and_constraint_patch(question)
            budget = next((op["value"] for op in parsed["operations"] if op["path"] == "budget.max"), None)
            self.assertEqual(budget, expected, msg=question)

    def test_money_to_vnd_units(self):
        self.assertEqual(money_to_vnd("5", "cu"), 5_000_000)
        self.assertEqual(money_to_vnd("5", "m"), 5_000_000)
        self.assertEqual(money_to_vnd("5000", "k"), 5_000_000)
        self.assertEqual(money_to_vnd("3.5", "tr"), 3_500_000)


class AnswerPriceMatchTests(unittest.TestCase):
    def test_answer_mentions_canonical_and_colloquial(self):
        amount = 3_900_000
        self.assertTrue(answer_mentions_vnd("Giá 3.900.000 VND/tháng", amount))
        self.assertTrue(answer_mentions_vnd("Giá phòng 3,9 triệu/tháng", amount))
        self.assertTrue(answer_mentions_vnd("khoảng 3.9 trieu", amount))
        self.assertTrue(answer_mentions_vnd("giá 3.9 củ", amount))
        self.assertFalse(answer_mentions_vnd("Giá 4.5 triệu", amount))


class UnifiedMoneyParserTests(unittest.TestCase):
    def test_parse_money_amount_matches_budget_and_fee_paths(self):
        from room_assistant.money import parse_money_amount
        from room_assistant.tools import _money_amount_or_none, _money_value_or_none

        cases = (
            ("5 củ", 5_000_000),
            ("5000k", 5_000_000),
            ("3.900.000", 3_900_000),
            ("3,9 triệu", 3_900_000),
            ("5m", 5_000_000),
        )
        for text, expected in cases:
            self.assertEqual(parse_money_amount(text), expected, msg=text)
            self.assertEqual(_money_value_or_none(text), expected, msg=text)

        self.assertEqual(_money_amount_or_none("miễn phí", "wifi"), 0)
        self.assertEqual(_money_amount_or_none("500", "wifi"), 500_000)
        self.assertEqual(parse_money_amount("500", fee_name="wifi", fee_context=True), 500_000)


if __name__ == "__main__":
    unittest.main()
