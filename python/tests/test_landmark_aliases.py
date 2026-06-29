"""Tests for landmark alias normalization."""

from __future__ import annotations

import unittest

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.landmark_aliases import (
    expand_landmark_tokens,
    landmark_matches_text,
    normalize_landmark,
)
from room_assistant.repository import InMemoryRoomRepository, room_matches_constraints
from room_assistant.schemas import normalize_room


class LandmarkAliasTests(unittest.TestCase):
    def test_tdtu_full_name_variants(self):
        for phrase in (
            "truong dai hoc tdt",
            "truong dai hoc tdtu",
            "dai hoc ton duc thang",
            "ton duc thang",
            "tdt",
            "tdtu",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize_landmark(phrase), "tdtu")

    def test_university_and_mall_aliases(self):
        cases = {
            "dai hoc cong thuong": "huit",
            "truong hutech": "hutech",
            "tttm aeon tan phu": "aeon tan phu",
            "duong nguyen huu tho": "nguyen huu tho",
            "xvnt": "xvnt",
            "xo viet nghe tinh": "xvnt",
            "benh vien cho ray": "cho ray",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_landmark(raw), expected)

    def test_expand_tokens_cover_embedding_text(self):
        tokens = expand_landmark_tokens("truong dai hoc tdt")
        self.assertIn("tdtu", tokens)
        self.assertIn("tdt", tokens)
        self.assertTrue(
            landmark_matches_text("truong dai hoc tdt", "Gan TDTU, di hoc tien")
        )

    def test_intent_parser_normalizes_tdt_full_query(self):
        parsed = parse_intent_and_constraint_patch(
            "mình muốn tìm phòng ở gần trường đại học TDT"
        )
        landmarks = [
            op["value"]
            for op in parsed["operations"]
            if op.get("path") == "location.near_landmarks"
        ]
        self.assertEqual(landmarks, ["tdtu"])

    def test_room_match_via_alias_expansion(self):
        room = normalize_room({
            "room_id": "near-tdtu",
            "metadata": {"price": 3_000_000, "status_code": "0", "district_name": "Quận 7"},
            "embedding_text": "Gan TDTU, di hoc tien",
            "available": True,
            "status": "active",
        })
        self.assertTrue(
            room_matches_constraints(
                room,
                {"location": {"near_landmarks": ["truong dai hoc tdt"]}},
            )
        )

    def test_in_memory_repo_finds_tdtu_rooms(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "tdtu-room",
                "metadata": {
                    "price": 2_300_000,
                    "status_code": "0",
                    "district_name": "Quận 7",
                },
                "embedding_text": "Gan TDTU, di hoc tien",
                "available": True,
                "status": "active",
            },
            {
                "room_id": "other-room",
                "metadata": {
                    "price": 5_000_000,
                    "status_code": "0",
                    "district_name": "Quận Bình Thạnh",
                },
                "embedding_text": "Gan Vincom",
                "available": True,
                "status": "active",
            },
        ])
        hits = repo.search_by_constraints(
            {"location": {"near_landmarks": ["truong dai hoc tdt"]}},
            limit=10,
        )
        ids = [room.get("room_id") for room in hits]
        self.assertIn("tdtu-room", ids)
        self.assertNotIn("other-room", ids)


if __name__ == "__main__":
    unittest.main()
