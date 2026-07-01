"""Contract tests for continuous room indexing (Qdrant CDC pipeline)."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from room_assistant.indexing import (
    PermanentIndexingError,
    RoomIndexingService,
    build_canonical_embedding_text,
    content_hash,
    validate_room_changed_event,
)
from room_assistant.repository import InMemoryRoomRepository


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


if __name__ == "__main__":
    unittest.main()
