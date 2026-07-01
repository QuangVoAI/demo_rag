"""Smoke tests inventory ~9k: in-memory mock (CI) + Mongo/Qdrant/BGE (integration)."""

from __future__ import annotations

import asyncio
import time
import unittest
from typing import Any

from room_assistant.money import normalize_money_text
from room_assistant.repository import InMemoryRoomRepository, room_matches_constraints
from room_assistant.retrieval import search_rooms_with_hard_filters
from room_assistant.session_store import InMemorySessionStore
from room_assistant.workflow import run_room_assistant

from mongo_test_helpers import production_inventory_ready

INVENTORY_SIZE = 9000


def _synthetic_rooms(count: int = INVENTORY_SIZE) -> list[dict]:
    rooms = []
    districts = ("Quận 1", "Quận 7", "Bình Thạnh", "Tân Bình", "Thủ Đức")
    for idx in range(count):
        district = districts[idx % len(districts)]
        is_studio = idx % 17 == 0
        pets = idx % 23 == 0
        rooms.append({
            "room_id": f"scale-{idx:05d}",
            "metadata": {
                "house_name": f"House {idx}",
                "room_code": f"S{idx % 500:03d}",
                "price": 2_500_000 + (idx % 40) * 100_000,
                "status_code": "0" if idx % 5 != 0 else "1",
                "district_name": district,
            },
            "embedding_text": (
                f"## Thông tin nhà\n- Địa chỉ: {district}\n"
                f"## Tiện ích\n- {'Studio' if is_studio else 'Phòng trọ'}\n"
                f"- Thú cưng: {'Có' if pets else 'Không'}\n"
            ),
            "available": idx % 5 != 0,
            "status": "active",
        })
    return rooms


class KeywordSemanticIndex:
    """Mock semantic rank trên pool candidate_ids (CI, không cần Qdrant/BGE)."""

    def __init__(self, repository: InMemoryRoomRepository) -> None:
        self.repository = repository

    def search_rooms(
        self,
        query_text: str,
        candidate_ids: list[str],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = set(normalize_money_text(query_text).split())
        scored: list[tuple[float, dict[str, Any]]] = []
        for room_id in candidate_ids:
            room = self.repository.get_by_id(room_id)
            if not room:
                continue
            text = normalize_money_text(str(room.get("embedding_text") or ""))
            overlap = sum(1 for token in query_tokens if token in text)
            if overlap <= 0:
                continue
            scored.append((overlap, {**room, "score": overlap / max(len(query_tokens), 1)}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


class InventoryScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rooms = _synthetic_rooms(INVENTORY_SIZE)
        cls.repo = InMemoryRoomRepository(cls.rooms)
        cls.semantic_index = KeywordSemanticIndex(cls.repo)

    def test_constraint_filter_scales_to_9k(self):
        constraints = {
            "location": {"districts": ["quan 7"]},
            "budget": {"max": 4_000_000, "max_operator": "lte"},
            "categories": ["studio"],
        }
        started = time.perf_counter()
        matches = [room for room in self.repo._rooms.values() if room_matches_constraints(room, constraints)]
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 8.0, msg=f"Filter {INVENTORY_SIZE} rooms took {elapsed:.2f}s")
        self.assertTrue(matches)
        for room in matches:
            self.assertIn("studio", (room.get("embedding_text") or "").lower())

    def test_retrieval_with_semantic_index_on_9k(self):
        started = time.perf_counter()
        results = search_rooms_with_hard_filters(
            query_text="tìm studio quận 7",
            constraints={"location": {"districts": ["quan 7"]}, "categories": ["studio"]},
            repository=self.repo,
            semantic_index=self.semantic_index,
            top_k=5,
            candidate_limit=100,
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 10.0, msg=f"Retrieval {INVENTORY_SIZE} took {elapsed:.2f}s")
        self.assertTrue(results)
        self.assertLessEqual(len(results), 5)

    def test_end_to_end_search_on_9k_inventory_uses_template(self):
        result = asyncio.run(run_room_assistant(
            "Tìm studio quận 7 dưới 4 triệu",
            session_id="scale-e2e-9k",
            repository=self.repo,
            session_store=InMemorySessionStore(),
            semantic_index=self.semantic_index,
        ))
        self.assertEqual(result["intent"], "SEARCH_ROOM")
        self.assertTrue(result["rooms"])
        for room in result["rooms"]:
            self.assertLessEqual(room.get("rent_price") or 0, 4_200_000)
        self.assertIn("VND", result["answer"])


@unittest.skipUnless(
    production_inventory_ready(),
    "MongoDB + Qdrant ~9k inventory not ready (set MONGODB_URI, QDRANT_URL)",
)
class ProductionInventoryScaleTests(unittest.TestCase):
    """Integration: Mongo ~9k + Qdrant BGE-M3 semantic rank (prod stack)."""

    @classmethod
    def setUpClass(cls) -> None:
        from mongo_test_helpers import (
            create_mongo_repository,
            create_semantic_index,
            load_dotenv,
            qdrant_points_count,
        )

        load_dotenv()
        cls.repo = create_mongo_repository()
        cls.semantic_index = create_semantic_index()
        cls.qdrant_points = qdrant_points_count()
        if cls.repo is None:
            raise unittest.SkipTest("Mongo repository unavailable")

    def test_qdrant_index_near_production_size(self):
        self.assertGreaterEqual(self.qdrant_points, 8000, msg=f"Qdrant points={self.qdrant_points}")

    def test_bge_qdrant_retrieval_with_candidate_limit_100(self):
        started = time.perf_counter()
        results = search_rooms_with_hard_filters(
            query_text="tìm phòng quận 7 dưới 5 triệu",
            constraints={
                "location": {"districts": ["quan 7"]},
                "budget": {"max": 5_000_000, "max_operator": "lte"},
            },
            repository=self.repo,
            semantic_index=self.semantic_index,
            top_k=5,
            candidate_limit=100,
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 90.0, msg=f"Prod retrieval took {elapsed:.2f}s (incl. BGE cold start)")
        self.assertTrue(results, msg="expected ranked rooms from Qdrant+BGE")
        self.assertLessEqual(len(results), 5)
        for room in results:
            rent = room.get("rent_price")
            if rent is not None:
                self.assertLessEqual(int(rent), 5_200_000)

    def test_room_code_lookup_on_production_mongo(self):
        from room_assistant.repository import normalize_room_code

        code = normalize_room_code("P.305")
        room = self.repo.get_by_room_code(code) if code else None
        if room is None:
            self.skipTest("P.305 not in current Mongo snapshot")
        self.assertTrue(room.get("rent_price"))


if __name__ == "__main__":
    unittest.main()
