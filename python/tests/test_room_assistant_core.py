import time
import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval.cache import get_cached_answer
from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import (
    InMemoryListingRepository,
    MongoListingRepository,
    build_mongo_query,
    listing_matches_constraints,
)
from room_assistant.schemas import default_session_state, normalize_listing
from room_assistant.session_store import InMemorySessionStore, apply_operations, load_session_state
from room_assistant.tools import ReadOnlyToolRegistry, ToolExecutionContext, ToolBudgetExceeded


class FakeMongoCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def skip(self, count):
        self._docs = self._docs[count:]
        return self

    def limit(self, count):
        self._docs = self._docs[:count]
        return self

    def sort(self, field, direction):
        self._docs.sort(key=lambda item: str(item.get(field, "")))
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeMongoCollection:
    def __init__(self, docs):
        self._docs = list(docs)

    def find_one(self, query):
        return next(iter(self.find(query)), None)

    def find(self, query=None, projection=None):
        query = query or {}
        return FakeMongoCursor([doc for doc in self._docs if self._matches(doc, query)])

    def _matches(self, doc, query):
        if "$or" in query:
            return any(self._matches(doc, clause) for clause in query["$or"])
        for field, expected in query.items():
            actual = doc.get(field)
            if isinstance(expected, dict) and "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif isinstance(expected, dict) and "$gt" in expected:
                if actual <= expected["$gt"]:
                    return False
            elif actual != expected:
                return False
        return True


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

    def test_motorbike_parking_is_vehicle_not_missing_amenity(self):
        parsed = parse_intent_and_constraint_patch("À mình muốn có chỗ để xe máy nữa.")
        self.assertIn({"op": "append", "path": "vehicles", "value": "motorbike"}, parsed["operations"])
        self.assertNotIn({"op": "append", "path": "amenities_required", "value": "parking"}, parsed["operations"])

    def test_site_detail_questions_route_to_room_detail(self):
        state = default_session_state("s-detail")
        state["current_listing_id"] = "A101"
        for question in (
            "Phòng này diện tích bao nhiêu?",
            "Phòng này có ban công không?",
            "Còn phòng trống không?",
            "Mã phòng là gì?",
        ):
            parsed = parse_intent_and_constraint_patch(question, state)
            self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        parsed = parse_intent_and_constraint_patch("Giá điện nước phòng này thế nào?", state)
        self.assertEqual(parsed["intent"], "CALCULATE_COST")

    def test_site_search_and_booking_constraints_are_extracted(self):
        state = default_session_state("s-search")
        state["current_listing_id"] = "A101"
        parsed = parse_intent_and_constraint_patch("Tìm phòng phường Bình Hưng Hoà B gần Aeon có tủ lạnh nước nóng ở ngay 2 xe", state)
        self.assertIn({"op": "append", "path": "location.wards", "value": "binh hung hoa b"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "vehicles", "value": "motorbike"}, parsed["operations"])
        self.assertIn({"op": "set", "path": "move_in_date", "value": "immediate"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "amenities_required", "value": "refrigerator"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "amenities_required", "value": "hot_water"}, parsed["operations"])
        self.assertIn(parsed["intent"], {"SEARCH_ROOM", "REFINE_SEARCH"})

        parsed = parse_intent_and_constraint_patch("Có phòng không chung chủ, ban công, cửa sổ không?", state)
        self.assertIn({"op": "append", "path": "amenities_required", "value": "balcony"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "amenities_required", "value": "window"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "amenities_preferred", "value": "no_owner_shared"}, parsed["operations"])
        self.assertIn(parsed["intent"], {"SEARCH_ROOM", "REFINE_SEARCH"})

    def test_site_contact_and_booking_actions_are_read_only_actions(self):
        for question in ("Đặt lịch hẹn xem phòng", "Chat ngay với chủ", "Gọi hỗ trợ giúp mình"):
            parsed = parse_intent_and_constraint_patch(question)
            self.assertEqual(parsed["intent"], "REQUEST_ACTION")

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

    def test_noop_append_does_not_increment_state_version(self):
        state = default_session_state("s-noop")
        state, applied = apply_operations(state, [
            {"op": "append", "path": "amenities_required", "value": "air_conditioner"},
        ])
        version = state["state_version"]
        state, applied = apply_operations(state, [
            {"op": "append", "path": "amenities_required", "value": "air_conditioner"},
        ])
        self.assertEqual(applied, [])
        self.assertEqual(state["state_version"], version)

    def test_normalize_listing_extracts_amenities_one_line_only(self):
        listing = normalize_listing({
            "listing_id": "A101",
            "embedding_text": "Tiện ích: wifi, window\nKhu vực xung quanh: gần trường",
            "available": True,
            "status": "active",
        })
        self.assertEqual(listing["amenities"], ["wifi", "window"])

    def test_normalize_listing_adds_rule_backed_amenities(self):
        listing = normalize_listing({
            "listing_id": "A101",
            "rules": {"window": True, "balcony": True, "toilet": "Riêng", "curfew": "Tự do"},
            "available": True,
            "status": "active",
        })
        self.assertTrue({"window", "balcony", "private_bathroom", "free_hours"}.issubset(set(listing["amenities"])))

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

    def test_repository_filters_motorbike_parking_from_listing_fields(self):
        constraints = {"vehicles": ["motorbike"]}
        with_parking = {
            "listing_id": "P1",
            "available": True,
            "status": "active",
            "shared_parking": True,
        }
        without_parking = {
            "listing_id": "P2",
            "available": True,
            "status": "active",
        }
        self.assertTrue(listing_matches_constraints(with_parking, constraints))
        self.assertFalse(listing_matches_constraints(without_parking, constraints))

    def test_repository_matches_site_amenity_aliases(self):
        listing = {
            "listing_id": "A101",
            "available": True,
            "status": "active",
            "amenities": ["may_lanh", "tu_lanh", "nuoc_nong", "gac"],
        }
        self.assertTrue(listing_matches_constraints(listing, {"amenities_required": ["air_conditioner"]}))
        self.assertTrue(listing_matches_constraints(listing, {"amenities_required": ["refrigerator", "hot_water", "mezzanine"]}))

    def test_mongo_repository_uses_object_id_for_native_listing_documents(self):
        try:
            from bson import ObjectId
        except Exception:
            self.skipTest("bson is not installed")

        object_id = ObjectId("6a38c129041de32cdde5acd3")
        other_id = ObjectId("6a38c129041de32cdde5acd4")
        repo = object.__new__(MongoListingRepository)
        repo._collection = FakeMongoCollection([
            {
                "_id": object_id,
                "room_code": "101",
                "category": "phong_tro",
                "embedding_text": "Địa chỉ: Quận 8, Thành phố Hồ Chí Minh. Giá: từ 4,900,000đ.",
                "price": {"min": 4_900_000, "max": 4_900_000},
                "status": "active",
                "title": "Phòng trọ thường 101 101",
            },
            {
                "_id": other_id,
                "price": {"min": 5_500_000},
                "status": "active",
                "title": "Phòng khác",
            },
        ])

        listing = repo.get_by_id(str(object_id))
        self.assertIsNotNone(listing)
        self.assertEqual(listing["listing_id"], str(object_id))
        self.assertEqual(listing["rent_price"], 4_900_000)

        listings = repo.get_many_by_ids([str(other_id), str(object_id)])
        self.assertEqual([item["listing_id"] for item in listings], [str(other_id), str(object_id)])

        self.assertEqual(
            list(repo.iter_listing_ids(batch_size=2)),
            [[str(object_id), str(other_id)]],
        )
        self.assertEqual(
            list(repo.iter_listing_ids(batch_size=2, resume_after=str(object_id))),
            [[str(other_id)]],
        )

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

    def test_retrieve_faq_covers_deposit_and_fees(self):
        repo = InMemoryListingRepository([])
        registry = ReadOnlyToolRegistry()
        context = ToolExecutionContext(repository=repo)
        result = registry.execute("retrieve_faq", {"question": "Tiền cọc và phí nước thế nào?"}, context)
        self.assertGreaterEqual(len(result), 1)
        self.assertTrue({item["topic"] for item in result} & {"deposit", "fees"})

    def test_calculator_handles_rental_months_deterministically(self):
        repo = InMemoryListingRepository([
            {
                "listing_id": "A101",
                "rent_price": 4_500_000,
                "deposit": 4_500_000,
                "fees": {"water": 100_000, "parking": 150_000},
                "available": True,
                "status": "active",
            }
        ])
        result = ReadOnlyToolRegistry().execute(
            "calculate_cost_estimate",
            {"listing_id": "A101", "rental_months": 6},
            ToolExecutionContext(repository=repo),
        )
        self.assertEqual(result["total_initial_cost"], 9_250_000)
        self.assertEqual(result["total_period_cost"], 33_000_000)
        self.assertEqual(result["recurring_fees_for_period"], 1_500_000)

    def test_calculator_parses_fixed_text_fees_only(self):
        repo = InMemoryListingRepository([
            {
                "listing_id": "A101",
                "rent_price": 4_900_000,
                "fees": {
                    "electricity": "4k/kWh",
                    "water": "30k/m3",
                    "management": "150k/ph",
                    "parking": "100k/xe",
                    "wifi": "Free",
                },
                "available": True,
                "status": "active",
            }
        ])
        result = ReadOnlyToolRegistry().execute(
            "calculate_cost_estimate",
            {
                "listing_id": "A101",
                "rental_months": 6,
                "constraints": {"vehicles": ["motorbike"]},
            },
            ToolExecutionContext(repository=repo),
        )
        self.assertEqual(result["recurring_fees_for_period"], 1_500_000)
        self.assertEqual(result["total_period_cost"], 30_900_000)
        self.assertIn("fees.electricity", result["unknown"])
        self.assertIn("fees.water", result["unknown"])

    def test_dynamic_listing_answer_cache_disabled_without_safe_context(self):
        self.assertIsNone(get_cached_answer("Phòng này có nuôi mèo không?", context=None, dynamic_listing=True))


if __name__ == "__main__":
    unittest.main()
