import asyncio
import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from room_assistant.indexing import (
    ListingIndexingService,
    RetryableIndexingError,
    make_dlq_record,
)
from room_assistant.repository import InMemoryListingRepository
from room_assistant.retrieval import search_listings_with_hard_filters
from room_assistant.session_store import InMemorySessionStore
from room_assistant.workflow import run_room_assistant


FIXTURES = [
    {
        "listing_id": "A101",
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
        "listing_id": "B202",
        "title": "Phòng Quận 7",
        "available": True,
        "status": "active",
        "district": "quan 7",
        "rent_price": 6_000_000,
        "amenities": ["balcony"],
        "source_version": 1,
    },
    {
        "listing_id": "C303",
        "title": "Phòng hết chỗ",
        "available": False,
        "status": "active",
        "district": "Bình Thạnh",
        "rent_price": 4_000_000,
        "source_version": 1,
    },
]


class RecordingSemanticIndex:
    def __init__(self):
        self.candidate_ids = []

    def search_listings(self, query_text, candidate_ids, top_k, metadata_filter=None):
        self.candidate_ids.append(list(candidate_ids))
        return [{"listing_id": item, "score": 1.0} for item in reversed(candidate_ids[:top_k])]


class FakeVectorIndex:
    def __init__(self):
        self.payloads = {}
        self.upserts = []
        self.deleted = []

    def get_payload(self, listing_id, chunk_type="listing_summary"):
        return self.payloads.get((listing_id, chunk_type))

    def upsert_listing_chunk(self, listing_id, chunk_type, text, embedding, payload):
        self.payloads[(listing_id, chunk_type)] = dict(payload)
        self.upserts.append((listing_id, chunk_type, text, payload))
        return f"listing:{listing_id}:{chunk_type}"

    def delete_listing(self, listing_id):
        self.deleted.append(listing_id)
        for key in list(self.payloads):
            if key[0] == listing_id:
                self.payloads.pop(key)


class CountingEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return [0.1, 0.2, 0.3]


class RoomAssistantWorkflowIndexingTests(unittest.TestCase):
    def test_search_hard_filters_before_semantic_and_refetches_authoritative(self):
        repo = InMemoryListingRepository(FIXTURES)
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
        self.assertEqual([item["listing_id"] for item in result["listings"]], ["A101"])
        self.assertEqual(semantic.candidate_ids, [["A101"]])
        self.assertEqual(result["agent_trace"]["write_tool_calls"], 0)
        self.assertLessEqual(result["agent_trace"]["read_tool_calls"], 3)

    def test_metadata_hit_boosts_listing_and_records_trace(self):
        repo = InMemoryListingRepository(FIXTURES)
        semantic = RecordingSemanticIndex()
        trace = {}

        results = search_listings_with_hard_filters(
            query_text="Cho mình xem #B202",
            constraints={},
            repository=repo,
            semantic_index=semantic,
            top_k=2,
            trace=trace,
        )

        self.assertEqual(results[0]["listing_id"], "B202")
        self.assertGreater(results[0]["metadata_score"], 0)
        self.assertGreater(results[0]["combined_score"], results[0]["rrf_score"])
        self.assertFalse(trace["retrieval_low_confidence"])
        self.assertEqual(trace["retrieval_feedback_retry_count"], 0)
        self.assertEqual(trace["retrieval_attempts"][0]["top_listing_ids"][0], "B202")

    def test_request_action_does_not_call_tools(self):
        result = asyncio.run(run_room_assistant(
            "Đặt lịch xem phòng giúp tôi",
            session_id="s2",
            repository=InMemoryListingRepository(FIXTURES),
            session_store=InMemorySessionStore(),
            semantic_index=RecordingSemanticIndex(),
        ))
        self.assertEqual(result["intent"], "REQUEST_ACTION")
        self.assertEqual(result["agent_trace"]["read_tool_calls"], 0)
        self.assertEqual(result["agent_trace"]["write_tool_calls"], 0)
        self.assertIn("không thể tự thực hiện", result["answer"])

    def test_cost_compare_and_current_listing_context(self):
        repo = InMemoryListingRepository(FIXTURES)
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
        self.assertEqual(qa["listings"][0]["listing_id"], "A101")

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
        self.assertEqual(compare["comparison"]["listing_ids"], ["A101", "B202", "C303"])

    def test_indexing_idempotency_old_event_delete_and_dlq(self):
        repo = InMemoryListingRepository(FIXTURES)
        vector = FakeVectorIndex()
        embedder = CountingEmbeddingProvider()
        service = ListingIndexingService(repo, vector, embedder, embedding_model="test", embedding_version=1)
        event = {
            "event_id": "e1",
            "listing_id": "A101",
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

        deleted = service.process_event(dict(event, event_id="e4", operation="delete"))
        self.assertEqual(deleted["result"], "deleted")
        self.assertEqual(vector.deleted, ["A101"])

        missing_event = dict(event, event_id="e5", listing_id="missing")
        with self.assertRaises(RetryableIndexingError) as ctx:
            service.process_with_retry(missing_event, attempts=1)
        dlq = make_dlq_record(missing_event, ctx.exception, attempts=1)
        self.assertEqual(dlq["error_type"], "inconsistent_data")


if __name__ == "__main__":
    unittest.main()
