"""Reviewer JSON schema compatibility tests."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from agents.reviewer import _parse_result, review_with_retry


class ReviewerSchemaTests(unittest.TestCase):
    def test_legacy_is_approved_schema(self):
        result = _parse_result('{"is_approved": false, "issues": ["hallucination"], "suggestion": "fix"}')
        self.assertFalse(result["is_approved"])
        self.assertIn("hallucination", result["issues"])

    def test_prompt_safe_schema(self):
        raw = (
            '{"safe": false, "violation_type": "hallucination", '
            '"feedback": "Giá không khớp", "corrected_answer": "Giá 4.5 triệu"}'
        )
        result = _parse_result(raw)
        self.assertFalse(result["is_approved"])
        self.assertEqual(result["corrected_answer"], "Giá 4.5 triệu")

    def test_safe_none_violation_approves(self):
        result = _parse_result('{"safe": true, "violation_type": "none", "feedback": ""}')
        self.assertTrue(result["is_approved"])

    def test_review_with_retry_applies_corrected_answer(self):
        async def fake_review(_question, _answer, _room_context=""):
            return {
                "is_approved": False,
                "issues": ["Giá không khớp"],
                "corrected_answer": "Giá 4.5 triệu",
            }

        with patch("agents.reviewer.review", side_effect=fake_review):
            answer, result = asyncio.run(
                review_with_retry("Giá bao nhiêu?", "Giá 9 triệu", room_context="price=4500000")
            )

        self.assertEqual(answer, "Giá 4.5 triệu")
        self.assertTrue(result["is_approved"])
        self.assertTrue(result["used_corrected_answer"])


if __name__ == "__main__":
    unittest.main()
