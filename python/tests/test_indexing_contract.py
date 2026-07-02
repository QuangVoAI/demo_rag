"""Contract tests for continuous room indexing (Qdrant CDC pipeline)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).parent.parent))

from room_assistant.indexing import (
    PermanentIndexingError,
    RoomIndexingService,
    build_canonical_embedding_text,
    build_mongo_cdc_event,
    coerce_room_changed_event,
    content_hash,
    validate_room_changed_event,
)
from room_assistant.repository import InMemoryRoomRepository
from room_assistant.schemas import canonical_room_id
from retrieval.qdrant_client import QdrantWrapper

from scripts.index_mongo_to_qdrant import build_payload, room_point_id_for_doc
from scripts.mongo_cdc_publisher import build_event_from_change, resolve_room_id_from_change


class _FakeVectorIndex:
    def __init__(self) -> None:
        self.payloads: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def get_payload(self, room_id: str, chunk_type: str = "room_summary") -> dict | None:
        return self.payloads.get(room_id)

    def upsert_room_chunk(self, room_id: str, chunk_type: str, text: str, embedding: Any, payload: dict) -> str:
        self.payloads[room_id] = {**payload, "text": text}
        return f"point-{room_id}"

    def delete_room(self, room_id: str) -> None:
        self.deleted.append(room_id)
        self.payloads.pop(room_id, None)


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class IndexingContractTests(unittest.TestCase):
    def test_validate_room_changed_event_requires_schema(self):
        with self.assertRaises(PermanentIndexingError):
            validate_room_changed_event({"room_id": "r1"})

    def test_validate_room_changed_event_normalizes(self):
        event = validate_room_changed_event({
            "event_id": "e1",
            "room_id": "scale-00001",
            "operation": "upsert",
            "source_version": 3,
            "occurred_at": "2026-01-01T00:00:00Z",
            "producer": "mongo-cdc",
        })
        self.assertEqual(event["source_version"], 3)

    def test_indexing_service_upserts_room_into_vector_index(self):
        room = {
            "room_id": "scale-00001",
            "source_version": 2,
            "embedding_text": "## Thông tin\n- Quận 7",
            "status": "active",
            "district": "Quận 7",
        }
        repo = InMemoryRoomRepository([room])
        vector = _FakeVectorIndex()
        service = RoomIndexingService(
            repository=repo,
            vector_index=vector,
            embedding_provider=_FakeEmbedder(),
            cache=MagicMock(),
        )
        result = service.process_event({
            "event_id": "e2",
            "room_id": "scale-00001",
            "operation": "upsert",
            "source_version": 2,
            "occurred_at": "2026-01-01T00:00:00Z",
            "producer": "room_index_worker",
        })
        self.assertEqual(result["result"], "upserted")
        self.assertIn("scale-00001", vector.payloads)

    def test_build_canonical_embedding_text_prefers_existing(self):
        room = {"embedding_text": "studio quan 7", "title": "ignored"}
        self.assertEqual(build_canonical_embedding_text(room), "studio quan 7")
        self.assertTrue(content_hash("studio quan 7"))

    def test_canonical_room_id_prefers_room_id_over_mongo_object_id(self):
        doc = {
            "_id": "507f1f77bcf86cd799439011",
            "room_id": "Q7-ROOM-1",
            "metadata": {"status_code": "0", "price": 4_000_000},
        }
        self.assertEqual(canonical_room_id(doc), "Q7-ROOM-1")

    def test_build_payload_keeps_canonical_room_id_and_mongo_id(self):
        doc = {
            "_id": "507f1f77bcf86cd799439011",
            "room_id": "Q7-ROOM-1",
            "metadata": {
                "status_code": "0",
                "price": 4_000_000,
                "district_name": "Quận 7",
                "house_name": "CS1",
                "room_code": "P101",
            },
            "embedding_text": "## Tiện ích\n- Máy lạnh: Có",
        }
        payload = build_payload(doc)
        self.assertEqual(payload["room_id"], "Q7-ROOM-1")
        self.assertEqual(payload["mongo_id"], "507f1f77bcf86cd799439011")
        self.assertNotIn("_id", payload)

    def test_point_id_matches_qdrant_wrapper_and_payload_room_id(self):
        doc = {
            "_id": "507f1f77bcf86cd799439011",
            "room_id": "Q7-ROOM-1",
            "metadata": {"status_code": "0", "price": 4_000_000},
        }
        canonical = canonical_room_id(doc)
        point_id = room_point_id_for_doc(doc)
        self.assertEqual(point_id, QdrantWrapper.room_point_id(canonical))
        self.assertEqual(build_payload(doc)["room_id"], canonical)

    def test_doc_without_room_id_falls_back_to_mongo_id(self):
        doc = {
            "_id": "507f1f77bcf86cd799439011",
            "metadata": {"status_code": "0", "price": 3_000_000},
        }
        self.assertEqual(canonical_room_id(doc), "507f1f77bcf86cd799439011")
        payload = build_payload(doc)
        self.assertEqual(payload["room_id"], "507f1f77bcf86cd799439011")

    def test_coerce_legacy_action_payload_to_full_schema(self):
        coerced = coerce_room_changed_event({
            "action": "upsert",
            "room_id": "Q7-ROOM-1",
        })
        event = validate_room_changed_event(coerced)
        self.assertEqual(event["room_id"], "Q7-ROOM-1")
        self.assertEqual(event["operation"], "upsert")
        self.assertEqual(event["producer"], "mongo-cdc")

    def test_build_mongo_cdc_event_uses_canonical_operation_map(self):
        event = build_mongo_cdc_event(
            room_id="Q7-ROOM-1",
            mongo_operation="update",
            source_version=12,
        )
        self.assertEqual(event["operation"], "upsert")
        self.assertEqual(event["source_version"], 12)
        self.assertIn("event_id", event)

    def test_indexing_service_accepts_legacy_kafka_payload(self):
        room = {
            "room_id": "scale-00001",
            "source_version": 0,
            "embedding_text": "## Thông tin\n- Quận 7",
            "status": "active",
            "district": "Quận 7",
        }
        repo = InMemoryRoomRepository([room])
        vector = _FakeVectorIndex()
        service = RoomIndexingService(
            repository=repo,
            vector_index=vector,
            embedding_provider=_FakeEmbedder(),
            cache=MagicMock(),
        )
        result = service.process_event({
            "action": "upsert",
            "room_id": "scale-00001",
        })
        self.assertEqual(result["result"], "upserted")

    def test_cdc_event_uses_document_room_id_not_object_id(self):
        class _FakeCollection:
            def find_one(self, query):
                return {
                    "_id": "507f1f77bcf86cd799439011",
                    "room_id": "Q7-ROOM-1",
                    "source_version": 4,
                }

        change = {
            "operationType": "update",
            "documentKey": {"_id": "507f1f77bcf86cd799439011"},
            "fullDocument": {
                "_id": "507f1f77bcf86cd799439011",
                "room_id": "Q7-ROOM-1",
                "source_version": 4,
            },
        }
        self.assertEqual(resolve_room_id_from_change(_FakeCollection(), change), "Q7-ROOM-1")
        event = build_event_from_change(_FakeCollection(), change)
        self.assertEqual(event["room_id"], "Q7-ROOM-1")
        self.assertEqual(event["operation"], "upsert")
        self.assertEqual(event["source_version"], 4)


if __name__ == "__main__":
    unittest.main()
