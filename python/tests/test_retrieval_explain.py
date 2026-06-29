"""Tests for retrieval explanation helper."""

from __future__ import annotations

import unittest

from room_assistant.retrieval import build_retrieval_explanation


class RetrievalExplanationTests(unittest.TestCase):
    def test_builds_constraint_and_trace_summary(self):
        lines = build_retrieval_explanation(
            trace={
                "candidate_count": 12,
                "metadata_hit_count": 2,
                "retrieval_confidence": 0.42,
                "retrieval_low_confidence": False,
            },
            constraints={
                "location": {"districts": ["quan 7"], "near_landmarks": ["tdtu"]},
                "budget": {"max": 5_000_000},
            },
            relaxed_fields=["near_landmarks"],
            result_count=3,
        )
        joined = "\n".join(lines)
        self.assertIn("quan 7", joined)
        self.assertIn("tdtu", joined)
        self.assertIn("12 phòng", joined)
        self.assertIn("nới", joined.lower())
        self.assertIn("3 phòng", joined)


if __name__ == "__main__":
    unittest.main()
