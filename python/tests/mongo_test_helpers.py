"""Shared helpers for Mongo/Qdrant integration tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PYTHON_ROOT = Path(__file__).resolve().parents[1]

SALES_CTA_MARKERS: tuple[str, ...] = (
    "xem thực tế",
    "ghé xem",
    "ưng căn",
    "em lọc",
    "ngân sách",
    "khu vực",
    "tư vấn",
    "dạ",
)


def load_dotenv() -> None:
    for env_path in (PYTHON_ROOT.parent / ".env", PYTHON_ROOT / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        break


def mongo_is_configured() -> bool:
    load_dotenv()
    uri = os.getenv("MONGODB_URI", "")
    return bool(uri and "xxx" not in uri and "mongodb" in uri)


def qdrant_is_available() -> bool:
    load_dotenv()
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if not url:
        return False
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, check_compatibility=False, timeout=5)
        collection = os.getenv("QDRANT_ROOMS_COLLECTION", "rooms_v1")
        names = {item.name for item in client.get_collections().collections}
        if collection not in names:
            return False
        return client.get_collection(collection).points_count > 0
    except Exception:
        return False


def integration_stack_ready() -> bool:
    return mongo_is_configured() and qdrant_is_available()


def create_semantic_index():
    from room_assistant.qdrant_index import QdrantRoomSemanticIndex

    return QdrantRoomSemanticIndex()


def assert_answer_invites_customer(test_case, answer: str, *, min_markers: int = 1) -> None:
    lowered = (answer or "").lower()
    hits = sum(1 for marker in SALES_CTA_MARKERS if marker in lowered)
    test_case.assertGreaterEqual(
        hits,
        min_markers,
        msg=f"answer should invite customer action: {answer[:200]}",
    )


def create_mongo_repository():
    load_dotenv()
    from room_assistant.repository import MongoRoomRepository, create_room_repository

    repo = create_room_repository()
    if isinstance(repo, MongoRoomRepository):
        return repo
    uri = os.getenv("MONGODB_URI", "")
    database = os.getenv("MONGODB_DATABASE", os.getenv("MONGODB_DB_NAME", "demo_rag"))
    collection = os.getenv("MONGODB_ROOMS_COLLECTION", "rooms")
    if not uri:
        return None
    return MongoRoomRepository(uri=uri, database=database, collection=collection)


def mongo_room_price(repo: Any, room_id: str) -> int | None:
    room = repo.get_by_id(room_id)
    if not room:
        return None
    return room.get("rent_price")


def assert_rooms_match_mongo(
    test_case,
    repo: Any,
    rooms: list[dict[str, Any]],
    *,
    max_budget: int | None = None,
    district_contains: str | None = None,
) -> None:
    test_case.assertTrue(rooms, "expected at least one room in response")
    for room in rooms:
        room_id = str(room.get("room_id") or "")
        mongo_room = repo.get_by_id(room_id)
        test_case.assertIsNotNone(mongo_room, f"room {room_id} missing in MongoDB")
        test_case.assertEqual(
            room.get("rent_price"),
            mongo_room.get("rent_price"),
            f"rent_price mismatch for {room_id}",
        )
        if max_budget is not None and mongo_room.get("rent_price") is not None:
            test_case.assertLessEqual(
                int(mongo_room["rent_price"]),
                max_budget,
                f"{room_id} exceeds budget cap {max_budget}",
            )
        if district_contains:
            district = str(mongo_room.get("district") or mongo_room.get("metadata", {}).get("district_name") or "")
            test_case.assertIn(
                district_contains.lower(),
                district.lower(),
                f"{room_id} not in expected district ({district_contains})",
            )
