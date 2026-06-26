"""Tests for production source attribution helpers."""

from __future__ import annotations

import unittest

from room_assistant.sources import build_room_source, build_room_sources, parse_listing_identity


class SourceAttributionTests(unittest.TestCase):
    def test_parse_listing_identity_from_room_id(self) -> None:
        room = {
            "room_id": "https://example.com/listing#P101",
            "title": "Phòng 101",
        }
        listing_id, source_url = parse_listing_identity(room)
        self.assertEqual(listing_id, "https://example.com/listing#P101")
        self.assertEqual(source_url, "https://example.com/listing")

    def test_build_room_source_includes_excerpt_and_detail_url(self) -> None:
        room = {
            "room_id": "https://nhatrovn.vn/a#P1",
            "house_id": "house-1",
            "title": "Nhà A - P1",
            "district": "Quận 7",
            "updated_at": "2026-06-01T10:00:00",
            "embedding_text": "## Tiện ích\n- Máy lạnh: Có\n- Wifi: Có",
        }
        source = build_room_source(room, query="máy lạnh")
        self.assertEqual(source["listing_id"], "https://nhatrovn.vn/a#P1")
        self.assertEqual(source["source_url"], "https://nhatrovn.vn/a")
        self.assertEqual(source["detail_url"], "/tim-phong/https://nhatrovn.vn/a#P1/")
        self.assertIn("Máy lạnh", source["excerpt"] or "")

    def test_build_room_sources_deduplicates(self) -> None:
        rooms = [
            {"room_id": "r1", "title": "A"},
            {"room_id": "r1", "title": "A"},
            {"room_id": "r2", "title": "B"},
        ]
        sources = build_room_sources(rooms)
        self.assertEqual(len(sources), 2)


if __name__ == "__main__":
    unittest.main()
