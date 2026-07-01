"""Tests for retrieval explanation helper."""

from __future__ import annotations

import unittest

from room_assistant.retrieval import build_retrieval_explanation, search_rooms_with_hard_filters
from room_assistant.repository import InMemoryRoomRepository


class _EmptySemanticIndex:
    def search_rooms(self, query_text, candidate_ids, top_k, metadata_filter=None):
        return []


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

    def test_fallback_returns_category_matches_when_semantic_empty(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "studio-q7",
                "metadata": {"district_name": "Quận 7", "price": 4_500_000, "status_code": "0"},
                "embedding_text": "## Tiện ích\n- Studio",
                "available": True,
            },
            {
                "room_id": "tro-q7",
                "metadata": {"district_name": "Quận 7", "price": 3_000_000, "status_code": "0"},
                "embedding_text": "## Tiện ích\n- Phòng trọ",
                "available": True,
            },
        ])
        results = search_rooms_with_hard_filters(
            query_text="tìm studio quận 7",
            constraints={"location": {"districts": ["quan 7"]}, "categories": ["studio"]},
            repository=repo,
            semantic_index=_EmptySemanticIndex(),
            top_k=3,
            candidate_limit=10,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["room_id"], "studio-q7")


if __name__ == "__main__":
    unittest.main()
