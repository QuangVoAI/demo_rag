import asyncio
import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from room_assistant.indexing import (
    RoomIndexingService,
    RetryableIndexingError,
    make_dlq_record,
)
from room_assistant.repository import InMemoryRoomRepository
from room_assistant.retrieval import search_rooms_with_hard_filters
from room_assistant.schemas import default_session_state
from room_assistant.session_store import InMemorySessionStore, apply_operations
from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.workflow import run_room_assistant


FIXTURES = [
    {
        "room_id": "A101",
        "title": "Studio Bình Thạnh",
        "description": "Phòng sáng, yên tĩnh",
        "available": True,
        "status": "active",
        "district": "Bình Thạnh",
        "rent_price": 4_500_000,
        "deposit": 4_500_000,
        "fees": {"water": 100_000, "parking": 150_000},
        "amenities": ["air_conditioner", "window"],
        "area_m2": 24,
        "source_version": 1,
    },
    {
        "room_id": "B202",
        "title": "Phòng Quận 7",
        "available": True,
        "status": "active",
        "district": "quan 7",
        "rent_price": 6_000_000,
        "amenities": ["balcony"],
        "source_version": 1,
    },
    {
        "room_id": "C303",
        "title": "Phòng hết chỗ",
        "available": False,
        "status": "active",
        "district": "Bình Thạnh",
        "rent_price": 4_000_000,
        "source_version": 1,
    },
    {
        "room_id": "E505",
        "title": "Phòng nhỏ Gò Vấp",
        "available": True,
        "status": "active",
        "district": "Gò Vấp",
        "rent_price": 3_800_000,
        "amenities": ["window"],
        "area_m2": 18,
        "source_version": 1,
    },
]


class RecordingSemanticIndex:
    def __init__(self):
        self.candidate_ids = []

    def search_rooms(self, query_text, candidate_ids, top_k, metadata_filter=None):
        self.candidate_ids.append(list(candidate_ids))
        return [{"room_id": item, "score": 1.0} for item in reversed(candidate_ids[:top_k])]


class FakeVectorIndex:
    def __init__(self):
        self.payloads = {}
        self.upserts = []
        self.deleted = []

    def get_payload(self, room_id, chunk_type="room_summary"):
        return self.payloads.get((room_id, chunk_type))

    def upsert_room_chunk(self, room_id, chunk_type, text, embedding, payload):
        self.payloads[(room_id, chunk_type)] = dict(payload)
        self.upserts.append((room_id, chunk_type, text, payload))
        return f"room:{room_id}:{chunk_type}"

    def delete_room(self, room_id):
        self.deleted.append(room_id)
        for key in list(self.payloads):
            if key[0] == room_id:
                self.payloads.pop(key)


class CountingEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return [0.1, 0.2, 0.3]


class RoomAssistantWorkflowIndexingTests(unittest.TestCase):
    def test_sparse_search_preserves_house_id(self):
        qdrant_client_source = (
            Path(__file__).resolve().parents[1]
            / "retrieval"
            / "qdrant_client.py"
        ).read_text(encoding="utf-8")
        search_sparse_source = qdrant_client_source.split("def search_sparse", 1)[1].split(
            "def ",
            1,
        )[0]

        self.assertIn('"house_id": r.payload.get("house_id", "")', search_sparse_source)

    def test_search_hard_filters_before_semantic_and_refetches_authoritative(self):
        repo = InMemoryRoomRepository(FIXTURES)
        semantic = RecordingSemanticIndex()
        store = InMemorySessionStore()

        result = asyncio.run(run_room_assistant(
            "Tìm phòng dưới 5 triệu ở quận Bình Thạnh có máy lạnh",
            session_id="s1",
            repository=repo,
            session_store=store,
            semantic_index=semantic,
        ))

        self.assertEqual(result["intent"], "SEARCH_ROOM")
        self.assertEqual([item["room_id"] for item in result["rooms"]], ["A101"])
        self.assertEqual(semantic.candidate_ids, [["A101"]])
        self.assertEqual(result["agent_trace"]["write_tool_calls"], 0)
        self.assertLessEqual(result["agent_trace"]["read_tool_calls"], 3)

    def test_varied_query_formats_preserve_retrieval_correctness(self):
        repo = InMemoryRoomRepository(FIXTURES)
        semantic = RecordingSemanticIndex()
        cases = [
            ("Tìm phòng dưới 5 triệu ở quận Bình Thạnh có máy lạnh", ["A101"]),
            ("quan 7 ban cong", ["B202"]),
            ("tim phong binh thanh may lanh duoi 5tr", ["A101"]),
            ("Tìm phòng dưới 7 triệu quận 7 có ban công", ["B202"]),
            ("tim phong duoi 5tr khong may lanh", ["E505"]),
        ]

        for question, expected_ids in cases:
            state = default_session_state("case")
            parsed = parse_intent_and_constraint_patch(question, state)
            state, _ = apply_operations(state, parsed["operations"])
            results = search_rooms_with_hard_filters(
                query_text=question,
                constraints=state["constraints"],
                repository=repo,
                semantic_index=semantic,
                top_k=5,
                trace={},
            )
            self.assertEqual([room["room_id"] for room in results], expected_ids, question)

    def test_metadata_hit_boosts_room_and_records_trace(self):
        repo = InMemoryRoomRepository(FIXTURES)
        semantic = RecordingSemanticIndex()
        trace = {}

        results = search_rooms_with_hard_filters(
            query_text="Cho mình xem #B202",
            constraints={},
            repository=repo,
            semantic_index=semantic,
            top_k=2,
            trace=trace,
        )

        self.assertEqual(results[0]["room_id"], "B202")
        self.assertGreater(results[0]["metadata_score"], 0)
        self.assertGreater(results[0]["combined_score"], results[0]["rrf_score"])
        self.assertFalse(trace["retrieval_low_confidence"])
        self.assertEqual(trace["retrieval_feedback_retry_count"], 0)
        self.assertEqual(trace["retrieval_attempts"][0]["top_room_ids"][0], "B202")

    def test_request_action_does_not_call_tools(self):
        result = asyncio.run(run_room_assistant(
            "Đặt lịch xem phòng giúp tôi",
            session_id="s2",
            repository=InMemoryRoomRepository(FIXTURES),
            session_store=InMemorySessionStore(),
            semantic_index=RecordingSemanticIndex(),
        ))
        self.assertEqual(result["intent"], "REQUEST_ACTION")
        self.assertEqual(result["agent_trace"]["read_tool_calls"], 0)
        self.assertEqual(result["agent_trace"]["write_tool_calls"], 0)
        self.assertIn("không thể tự thực hiện", result["answer"])

    def test_cost_compare_and_current_room_context(self):
        repo = InMemoryRoomRepository(FIXTURES)
        store = InMemorySessionStore()
        semantic = RecordingSemanticIndex()

        asyncio.run(run_room_assistant(
            "Tìm phòng dưới 5 triệu ở quận Bình Thạnh",
            session_id="s3",
            repository=repo,
            session_store=store,
            semantic_index=semantic,
        ))
        qa = asyncio.run(run_room_assistant(
            "Phòng này có tiện ích gì?",
            session_id="s3",
            repository=repo,
            session_store=store,
            semantic_index=semantic,
        ))
        self.assertEqual(qa["rooms"][0]["room_id"], "A101")
        self.assertIn("Máy lạnh", qa["answer"])
        self.assertIn("Cửa sổ", qa["answer"])

        cost = asyncio.run(run_room_assistant(
            "Tính tổng chi phí #A101",
            session_id="s4",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=semantic,
        ))
        self.assertEqual(cost["cost_estimate"]["total_initial_cost"], 9_250_000)

        cost_six_months = asyncio.run(run_room_assistant(
            "Tính tổng chi phí #A101 nếu thuê 6 tháng",
            session_id="s4b",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=semantic,
        ))
        self.assertEqual(cost_six_months["cost_estimate"]["rental_months"], 6)
        self.assertEqual(cost_six_months["cost_estimate"]["total_period_cost"], 33_000_000)

        compare = asyncio.run(run_room_assistant(
            "So sánh #A101 #B202 #C303 #D404",
            session_id="s5",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=semantic,
        ))
        self.assertEqual(compare["comparison"]["room_ids"], ["A101", "B202", "C303"])
        self.assertIn("D404", compare["comparison"]["not_compared_room_ids"])
        self.assertIn("#D404", compare["answer"])

        compare_missing = asyncio.run(run_room_assistant(
            "So sánh #A101 #Z999",
            session_id="s5b",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=semantic,
        ))
        self.assertIn("Z999", compare_missing["comparison"]["missing_room_ids"])
        self.assertIn("#Z999", compare_missing["answer"])

        missing_data_cost = asyncio.run(run_room_assistant(
            "Tính tổng chi phí #B202 nếu thuê 6 tháng",
            session_id="s4c",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=semantic,
        ))
        self.assertIn("deposit", missing_data_cost["cost_estimate"]["unknown"])

        outside = asyncio.run(run_room_assistant(
            "Thời tiết hôm nay ở Sài Gòn sao?",
            session_id="s-outside",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=semantic,
        ))
        self.assertEqual(outside["intent"], "GENERAL_HELP")
        self.assertEqual(outside["agent_trace"]["read_tool_calls"], 0)
        self.assertIn("Mình có thể giúp tìm phòng", outside["answer"])

    def test_indexing_idempotency_old_event_delete_and_dlq(self):
        repo = InMemoryRoomRepository(FIXTURES)
        vector = FakeVectorIndex()
        embedder = CountingEmbeddingProvider()
        service = RoomIndexingService(repo, vector, embedder, embedding_model="test", embedding_version=1)
        event = {
            "event_id": "e1",
            "room_id": "A101",
            "operation": "upsert",
            "source_version": 1,
            "occurred_at": "2026-06-22T00:00:00Z",
            "producer": "test",
        }

        first = service.process_event(event)
        self.assertEqual(first["result"], "upserted")
        self.assertEqual(embedder.calls, 1)

        duplicate = service.process_event(dict(event, event_id="e2"))
        self.assertEqual(duplicate["result"], "skipped_unchanged")
        self.assertEqual(embedder.calls, 1)

        old = service.process_event(dict(event, event_id="e3", source_version=0))
        self.assertEqual(old["result"], "skipped_old_event")

        old_delete = service.process_event(dict(event, event_id="e3d", operation="delete", source_version=0))
        self.assertEqual(old_delete["result"], "skipped_old_event")
        self.assertEqual(vector.deleted, [])

        deleted = service.process_event(dict(event, event_id="e4", operation="delete"))
        self.assertEqual(deleted["result"], "deleted")
        self.assertEqual(vector.deleted, ["A101"])

        missing_event = dict(event, event_id="e5", room_id="missing")
        with self.assertRaises(RetryableIndexingError) as ctx:
            service.process_with_retry(missing_event, attempts=1)
        dlq = make_dlq_record(missing_event, ctx.exception, attempts=1)
        self.assertEqual(dlq["error_type"], "inconsistent_data")


if __name__ == "__main__":
    unittest.main()
