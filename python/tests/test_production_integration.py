"""Tests for production verifier/abstain and rerank wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.reviewer import should_abstain
from room_assistant.retrieval import _maybe_rerank_candidates


class ProductionGuardrailTests(unittest.TestCase):
    def test_should_abstain_when_room_context_missing(self) -> None:
        with patch("config.ENABLE_ABSTAIN", True):
            abstain, reason = should_abstain(
                question="Phòng này cọc bao nhiêu?",
                intent="ASK_ABOUT_ROOM",
                grounding={"rooms": []},
                tool_results={},
                answer="Cọc 2 tháng ạ.",
            )
        self.assertTrue(abstain)
        self.assertEqual(reason, "missing_room_context")

    def test_maybe_rerank_skips_when_disabled(self) -> None:
        ranked = [
            {"room_id": "r1", "combined_score": 0.2, "metadata_score": 0.0, "position": 0},
            {"room_id": "r2", "combined_score": 0.1, "metadata_score": 0.0, "position": 1},
        ]

        class Repo:
            def get_by_id(self, room_id: str):
                return {"room_id": room_id, "embedding_text": f"text {room_id}"}

        with patch("config.USE_RERANKER", False):
            result = _maybe_rerank_candidates(
                query_text="tìm phòng quận 7",
                ranked=list(ranked),
                candidates=[{"room_id": "r1"}, {"room_id": "r2"}],
                repository=Repo(),
                metadata_boost=0.5,
            )
        self.assertEqual([item["room_id"] for item in result], ["r1", "r2"])


if __name__ == "__main__":
    unittest.main()
