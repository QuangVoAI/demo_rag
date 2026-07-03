"""Tests for landmark alias normalization."""

from __future__ import annotations

import unittest

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.landmark_aliases import (
    expand_landmark_tokens,
    landmark_matches_text,
    merge_nearby_into_embedding_text,
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

    def test_dhct_resolves_to_ctu_not_huit(self):
        self.assertEqual(normalize_landmark("dhct"), "ctu")
        self.assertEqual(normalize_landmark("dai hoc can tho"), "ctu")
        self.assertEqual(normalize_landmark("DHCT", province_slug="Cần Thơ"), "ctu")

    def test_bare_district_binh_tan_not_aeon(self):
        self.assertIsNone(normalize_landmark("binh tan"))
        self.assertIsNone(normalize_landmark("quan binh tan"))
        self.assertEqual(
            normalize_landmark("aeon mall binh tan"),
            "aeon binh tan",
        )

    def test_airports_are_geo_scoped(self):
        self.assertEqual(normalize_landmark("san bay can tho"), "san bay can tho")
        self.assertEqual(normalize_landmark("san bay tan son nhat"), "tan son nhat")
        self.assertEqual(normalize_landmark("san bay noi bai"), "noi bai")
        self.assertNotEqual(
            normalize_landmark("san bay can tho"),
            normalize_landmark("san bay tan son nhat"),
        )

    def test_ambiguous_bach_khoa_requires_province(self):
        self.assertIsNone(normalize_landmark("bach khoa"))
        self.assertEqual(
            normalize_landmark("bach khoa tp", province_slug="TP.HCM"),
            "hcmut",
        )
        self.assertEqual(
            normalize_landmark("dai hoc bach khoa ha noi"),
            "hust",
        )

    def test_no_false_positive_go_to_huflit(self):
        self.assertIsNone(normalize_landmark("go"))
        tokens = expand_landmark_tokens("go")
        self.assertNotIn("huflit", tokens)

    def test_numeric_one_not_benh_vien_175(self):
        tokens = expand_landmark_tokens("1")
        self.assertNotIn("benh vien 175", tokens)
        self.assertEqual(tokens, [])

    def test_merge_nearby_into_embedding_text(self):
        room = {
            "embedding_text": "## Giá & phí\n- 3tr",
            "tien_ich_xq": "Gần TDTU, Vincom",
        }
        merged = merge_nearby_into_embedding_text(room)
        self.assertIn("## Tiện ích xung quanh", merged)
        self.assertIn("Gần TDTU", merged)
        self.assertIn("## Giá & phí", merged)

    def test_district_blocklist_sample_from_city_txt(self):
        """Một số quận/huyện phổ biến không được nhận nhầm thành landmark."""
        for district in (
            "tan binh",
            "go vap",
            "thu duc",
            "quan 7",
            "nha be",
        ):
            with self.subTest(district=district):
                self.assertIsNone(normalize_landmark(district))


if __name__ == "__main__":
    unittest.main()
