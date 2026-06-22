import time
import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval.cache import get_cached_answer
from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import InMemoryListingRepository, build_mongo_query, listing_matches_constraints
from room_assistant.schemas import default_session_state
from room_assistant.session_store import InMemorySessionStore, apply_operations, load_session_state
from room_assistant.tools import ReadOnlyToolRegistry, ToolExecutionContext, ToolBudgetExceeded


class RoomAssistantCoreTests(unittest.TestCase):
    def test_parse_refine_budget_and_remove_amenity(self):
        parsed = parse_intent_and_constraint_patch(
            "Tăng ngân sách lên 5 triệu và bỏ yêu cầu máy lạnh."
        )
        self.assertEqual(parsed["intent"], "REFINE_SEARCH")
        self.assertIn({"op": "set", "path": "budget.max", "value": 5_000_000}, parsed["operations"])
        self.assertIn({"op": "remove", "path": "amenities_required", "value": "air_conditioner"}, parsed["operations"])

    def test_budget_parses_grouped_and_compact_values(self):
        parsed = parse_intent_and_constraint_patch("Tìm phòng dưới 1.500.000")
        self.assertIn({"op": "set", "path": "budget.max", "value": 1_500_000}, parsed["operations"])

        parsed = parse_intent_and_constraint_patch("Tìm phòng 3tr5 ở quận 7")
        self.assertIn({"op": "set", "path": "budget.max", "value": 3_500_000}, parsed["operations"])

    def test_patch_keeps_old_constraints_and_remove_trims_value(self):
        state = default_session_state("s1")
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "binh thanh"},
            {"op": "append", "path": "amenities_required", "value": "air_conditioner"},
        ])
        state, applied = apply_operations(state, [
            {"op": "set", "path": "budget.max", "value": 6_000_000},
            {"op": "remove", "path": "amenities_required", "value": "air_conditioner  "},
            {"op": "set", "path": "unknown.field", "value": True},
        ])
        self.assertEqual(applied[0]["path"], "budget.max")
        self.assertEqual(state["constraints"]["location"]["districts"], ["binh thanh"])
        self.assertEqual(state["constraints"]["budget"]["max"], 6_000_000)
        self.assertEqual(state["constraints"]["amenities_required"], [])
        self.assertNotIn("unknown", state["constraints"])

    def test_in_memory_session_ttl(self):
        store = InMemorySessionStore()
        store.save("s1", default_session_state("s1"), ttl_seconds=1)
        self.assertIsNotNone(store.get("s1"))
        expires_at, state = store._data["s1"]
        store._data["s1"] = (time.time() - 1, state)
        self.assertIsNone(store.get("s1"))

    def test_repository_filters_available_and_normalized_district(self):
        listing = {
            "listing_id": "A101",
            "available": True,
            "status": "active",
            "district": "Bình Thạnh",
            "rent_price": 4_500_000,
        }
        constraints = {
            "location": {"districts": ["quan binh thanh"]},
            "budget": {"max": 5_000_000},
        }
        self.assertTrue(listing_matches_constraints(listing, constraints))

        unavailable = dict(listing, available=False)
        self.assertFalse(listing_matches_constraints(unavailable, constraints))

        mongo_query = build_mongo_query(constraints)
        self.assertIn({"available": {"$ne": False}}, mongo_query["$and"])

    def test_registry_is_read_only_and_budgeted(self):
        repo = InMemoryListingRepository([])
        registry = ReadOnlyToolRegistry()
        self.assertEqual(set(registry.names), {
            "search_listings",
            "get_listing_detail",
            "retrieve_listing_context",
            "retrieve_faq",
            "calculate_cost_estimate",
            "compare_listings",
            "find_similar_listings",
        })
        context = ToolExecutionContext(repository=repo)
        for _ in range(3):
            registry.execute("retrieve_faq", {"question": "hello"}, context)
        with self.assertRaises(ToolBudgetExceeded):
            registry.execute("retrieve_faq", {"question": "hello"}, context)
        self.assertEqual(context.write_tool_calls, 0)

    def test_dynamic_listing_answer_cache_disabled_without_safe_context(self):
        self.assertIsNone(get_cached_answer("Phòng này có nuôi mèo không?", context=None, dynamic_listing=True))


if __name__ == "__main__":
    unittest.main()
