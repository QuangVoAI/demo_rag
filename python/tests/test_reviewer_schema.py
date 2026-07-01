"""Reviewer JSON schema compatibility tests."""

from __future__ import annotations

import unittest

from agents.reviewer import _parse_result


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


if __name__ == "__main__":
    unittest.main()
