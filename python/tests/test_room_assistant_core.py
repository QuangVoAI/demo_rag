import asyncio
import time
import unittest
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents import sentiment_analyzer
from retrieval.cache import get_cached_answer
from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import (
    InMemoryRoomRepository,
    MongoRoomRepository,
    build_mongo_query,
    room_matches_constraints,
)
from room_assistant.schemas import default_session_state, normalize_room
from room_assistant.session_store import InMemorySessionStore, apply_operations, load_session_state
from room_assistant.tools import ReadOnlyToolRegistry, ToolExecutionContext, ToolBudgetExceeded
from room_assistant.workflow import _build_llm_context, run_room_assistant


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

    def test_search_phrase_after_phong_is_not_room_id(self):
        parsed = parse_intent_and_constraint_patch(
            "tim phong duoi 5 trieu o quan binh thanh co may lanh"
        )
        self.assertEqual(parsed["intent"], "SEARCH_ROOM")
        self.assertEqual(parsed["referenced_room_ids"], [])

    def test_named_district_search_handles_prefix_and_accents(self):
        parsed = parse_intent_and_constraint_patch("Tìm phòng dưới 5 triệu ở Bình Thạnh")
        self.assertIn({"op": "append", "path": "location.districts", "value": "binh thanh"}, parsed["operations"])

        query = build_mongo_query({
            "location": {"districts": ["quan binh thanh"]},
            "budget": {"max": 5_000_000, "max_operator": "lt"},
        })
        district_patterns = []
        for clause in query["$and"]:
            for option in clause.get("$or", []):
                if "metadata.district_name" in option:
                    district_patterns.append(option["metadata.district_name"]["$regex"])
        self.assertTrue(
            any(re.search(pattern, "Quận Bình Thạnh", re.IGNORECASE) for pattern in district_patterns)
        )

    def test_compact_district_matches_python_post_filter(self):
        room = {
            "room_id": "Q8-1",
            "available": True,
            "district": "Quận 8",
            "rent_price": 4_000_000,
            "amenities": [],
        }
        for district in ("q8", "q.8", "q 8", "quan 8"):
            self.assertTrue(
                room_matches_constraints(room, {"location": {"districts": [district]}}),
                district,
            )

        parsed = parse_intent_and_constraint_patch("Tìm phòng q10 dưới 5 triệu")
        self.assertIn({"op": "append", "path": "location.districts", "value": "quan 10"}, parsed["operations"])

    def test_mixed_amenity_add_and_remove_are_scoped_per_feature(self):
        parsed = parse_intent_and_constraint_patch("Tìm phòng có máy lạnh bỏ tủ lạnh")
        self.assertIn({"op": "append", "path": "amenities_required", "value": "air_conditioner"}, parsed["operations"])
        self.assertIn({"op": "remove", "path": "amenities_required", "value": "refrigerator"}, parsed["operations"])

        parsed = parse_intent_and_constraint_patch("Tìm phòng không cần máy giặt nhưng phải có tủ lạnh")
        self.assertIn({"op": "remove", "path": "amenities_required", "value": "washing_machine"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "amenities_required", "value": "refrigerator"}, parsed["operations"])

    def test_budget_parses_grouped_and_compact_values(self):
        parsed = parse_intent_and_constraint_patch("Tìm phòng dưới 1.500.000")
        self.assertIn({"op": "set", "path": "budget.max", "value": 1_500_000}, parsed["operations"])

        parsed = parse_intent_and_constraint_patch("Tìm phòng 3tr5 ở quận 7")
        self.assertIn({"op": "set", "path": "budget.max", "value": 3_500_000}, parsed["operations"])

    def test_budget_strict_comparison_operators(self):
        parsed = parse_intent_and_constraint_patch("Tìm phòng dưới 5 triệu")
        self.assertIn({"op": "set", "path": "budget.max", "value": 5_000_000}, parsed["operations"])
        self.assertIn({"op": "set", "path": "budget.max_operator", "value": "lt"}, parsed["operations"])

        parsed = parse_intent_and_constraint_patch("Tìm phòng trên 3 triệu")
        self.assertIn({"op": "set", "path": "budget.min", "value": 3_000_000}, parsed["operations"])
        self.assertIn({"op": "set", "path": "budget.min_operator", "value": "gt"}, parsed["operations"])

        parsed = parse_intent_and_constraint_patch("Tìm phòng tối đa 4 triệu")
        self.assertIn({"op": "set", "path": "budget.max_operator", "value": "lte"}, parsed["operations"])

    def test_contextual_booking_selection_and_detail_intents(self):
        state = default_session_state("s")
        state["last_result_ids"] = ["A101", "B202", "C303"]
        state["current_room_id"] = "A101"

        parsed = parse_intent_and_constraint_patch("Tôi muốn đặt phòng này", state)
        self.assertEqual(parsed["intent"], "REQUEST_ACTION")

        parsed = parse_intent_and_constraint_patch("Tôi chọn phòng số 2", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        self.assertEqual(parsed["current_room_id"], "B202")
        self.assertEqual(parsed["referenced_room_ids"], ["B202"])

        parsed = parse_intent_and_constraint_patch("Có chỗ để xe không?", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")

    def test_motorbike_parking_is_vehicle_not_missing_amenity(self):
        parsed = parse_intent_and_constraint_patch("À mình muốn có chỗ để xe máy nữa.")
        self.assertIn({"op": "append", "path": "vehicles", "value": "motorbike"}, parsed["operations"])
        self.assertNotIn({"op": "append", "path": "amenities_required", "value": "parking"}, parsed["operations"])

    def test_site_detail_questions_route_to_room_detail(self):
        state = default_session_state("s-detail")
        state["current_room_id"] = "A101"
        for question in (
            "Phòng này diện tích bao nhiêu?",
            "Phòng này có ban công không?",
            "Phòng này có tiện ích gì?",
            "Còn phòng trống không?",
            "Mã phòng là gì?",
        ):
            parsed = parse_intent_and_constraint_patch(question, state)
            self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        parsed = parse_intent_and_constraint_patch("Giá điện nước phòng này thế nào?", state)
        self.assertEqual(parsed["intent"], "CALCULATE_COST")

    def test_search_context_includes_verified_required_amenities(self):
        context = _build_llm_context(
            {
                "constraints": {"amenities_required": ["air_conditioner"]},
                "rooms": [{
                    "room_id": "A101",
                    "title": "Studio Bình Thạnh",
                    "available": True,
                    "district": "Bình Thạnh",
                    "rent_price": 4_500_000,
                    "amenities": ["air_conditioner", "window"],
                    "area_m2": 24,
                }],
                "unknown": [],
            },
            {},
        )
        self.assertIn("Tiện ích xác minh: Máy lạnh", context)
        self.assertIn("Trạng thái: còn phòng", context)

    def test_neutral_search_text_stays_normal_mood_without_pressure_cue(self):
        class FakeModel:
            def encode(self, text, normalize_embeddings=True, batch_size=None):
                if isinstance(text, list):
                    return np.array([[1.0, 0.0] for _ in text])
                return np.array([1.0, 0.0])

        old_centroids = sentiment_analyzer._centroids
        try:
            sentiment_analyzer._centroids = {
                "frustrated": np.array([1.0, 0.0]),
                "urgent": np.array([0.5, 0.5]),
                "normal": np.array([0.0, 1.0]),
            }
            with patch.object(sentiment_analyzer, "get_embed_model", return_value=FakeModel()):
                mood, _ = sentiment_analyzer.analyze_mood("Tìm phòng dưới 5 triệu ở quận Bình Thạnh")
            self.assertEqual(mood, "normal")
        finally:
            sentiment_analyzer._centroids = old_centroids

    def test_mood_cue_allows_small_filler_words(self):
        self.assertTrue(
            sentiment_analyzer._has_explicit_mood_cue(
                "Tháng này mình phải dọn nhà gấp",
                "urgent",
            )
        )

    def test_runtime_config_validation_reports_invalid_values(self):
        import config

        with patch.object(config, "TOP_K_RETRIEVAL", 0):
            result = config.validate_runtime_config(strict=False)
        self.assertIn("TOP_K_RETRIEVAL must be >= 1", result["errors"])

    def test_input_limit_returns_without_tools_or_retrieval(self):
        with patch("config.MAX_USER_QUESTION_CHARS", 40):
            result = asyncio.run(run_room_assistant(
                "x" * 180,
                session_id="too-long",
                repository=InMemoryRoomRepository([]),
                session_store=InMemorySessionStore(),
                semantic_index=None,
            ))
        self.assertEqual(result["agent_trace"]["error_category"], "input_too_long")
        self.assertEqual(result["agent_trace"]["read_tool_calls"], 0)

    def test_site_search_and_booking_constraints_are_extracted(self):
        state = default_session_state("s-search")
        state["current_room_id"] = "A101"
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

    def test_normalize_room_extracts_amenities_one_line_only(self):
        room = normalize_room({
            "room_id": "A101",
            "embedding_text": "Tiện ích: wifi, window\nKhu vực xung quanh: gần trường",
            "available": True,
            "status": "active",
        })
        self.assertEqual(room["amenities"], [])

    def test_in_memory_session_ttl(self):
        store = InMemorySessionStore()
        store.save("s1", default_session_state("s1"), ttl_seconds=1)
        self.assertIsNotNone(store.get("s1"))
        expires_at, state = store._data["s1"]
        store._data["s1"] = (time.time() - 1, state)
        self.assertIsNone(store.get("s1"))

    def test_repository_filters_available_and_normalized_district(self):
        room = {
            "room_id": "A101",
            "available": True,
            "status": "active",
            "district": "Bình Thạnh",
            "rent_price": 4_500_000,
        }
        constraints = {
            "location": {"districts": ["quan binh thanh"]},
            "budget": {"max": 5_000_000},
        }
        self.assertTrue(room_matches_constraints(room, constraints))

        unavailable = dict(room, available=False)
        self.assertFalse(room_matches_constraints(unavailable, constraints))

        mongo_query = build_mongo_query(constraints)
        self.assertIn({"metadata.status_code": "0"}, mongo_query["$and"])

    def test_mongo_repository_uses_object_id_for_native_room_documents(self):
        try:
            from bson import ObjectId
        except Exception:
            self.skipTest("bson is not installed")

        object_id = ObjectId("6a38c129041de32cdde5acd3")
        other_id = ObjectId("6a38c129041de32cdde5acd4")
        repo = object.__new__(MongoRoomRepository)
        repo._collection = FakeMongoCollection([
            {
                "_id": object_id,
                "room_id": str(object_id),
                "room_code": "101",
                "category": "phong_tro",
                "embedding_text": "Địa chỉ: Quận 8, Thành phố Hồ Chí Minh. Giá: từ 4,900,000đ.",
                "metadata": {
                    "price": 4_900_000,
                    "status_code": "0",
                    "room_code": "101",
                },
                "status": "active",
                "title": "Phòng trọ thường 101 101",
            },
            {
                "_id": other_id,
                "room_id": str(other_id),
                "metadata": {
                    "price": 5_500_000,
                    "status_code": "0",
                },
                "status": "active",
                "title": "Phòng khác",
            },
        ])

        room = repo.get_by_id(str(object_id))
        self.assertIsNotNone(room)
        self.assertEqual(room["room_id"], str(object_id))
        self.assertEqual(room["rent_price"], 4_900_000)

        rooms = repo.get_many_by_ids([str(other_id), str(object_id)])
        self.assertEqual([item["room_id"] for item in rooms], [str(other_id), str(object_id)])

        self.assertEqual(
            list(repo.iter_room_ids(batch_size=2)),
            [[str(object_id), str(other_id)]],
        )
        self.assertEqual(
            list(repo.iter_room_ids(batch_size=2, resume_after=str(object_id))),
            [[str(other_id)]],
        )

    def test_registry_is_read_only_and_budgeted(self):
        repo = InMemoryRoomRepository([])
        registry = ReadOnlyToolRegistry()
        self.assertEqual(set(registry.names), {
            "search_rooms",
            "get_room_detail",
            "retrieve_room_context",
            "retrieve_faq",
            "calculate_cost_estimate",
            "compare_rooms",
            "find_similar_rooms",
        })
        context = ToolExecutionContext(repository=repo)
        for _ in range(3):
            registry.execute("retrieve_faq", {"question": "hello"}, context)
        with self.assertRaises(ToolBudgetExceeded):
            registry.execute("retrieve_faq", {"question": "hello"}, context)
        self.assertEqual(context.write_tool_calls, 0)

    def test_retrieve_faq_covers_deposit_and_fees(self):
        repo = InMemoryRoomRepository([])
        registry = ReadOnlyToolRegistry()
        context = ToolExecutionContext(repository=repo)
        result = registry.execute("retrieve_faq", {"question": "Tiền cọc và phí nước thế nào?"}, context)
        self.assertGreaterEqual(len(result), 1)
        self.assertTrue({item["topic"] for item in result} & {"deposit", "fees"})

    def test_calculator_handles_rental_months_deterministically(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "A101",
                "metadata": {
                    "price": 4_500_000,
                    "status_code": "0",
                },
                "rent_price": 4_500_000,
                "deposit": 4_500_000,
                "fees": {"water": 100_000, "parking": 150_000},
                "available": True,
                "status": "active",
            }
        ])
        # Force overwrite deposit which normally isn't in schemas.py room normalize
        doc = repo.get_by_id("A101")
        doc["deposit"] = 4_500_000
        repo._rooms["A101"] = doc

        result = ReadOnlyToolRegistry().execute(
            "calculate_cost_estimate",
            {"room_id": "A101", "rental_months": 6},
            ToolExecutionContext(repository=repo),
        )
        self.assertEqual(result["total_initial_cost"], 9_250_000)
        self.assertEqual(result["total_period_cost"], 33_000_000)
        self.assertEqual(result["recurring_fees_for_period"], 1_500_000)

    def test_calculator_parses_fixed_text_fees_only(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "A101",
                "metadata": {
                    "price": 4_900_000,
                    "status_code": "0",
                },
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
                "room_id": "A101",
                "rental_months": 6,
                "constraints": {"vehicles": ["motorbike"]},
            },
            ToolExecutionContext(repository=repo),
        )
        self.assertEqual(result["recurring_fees_for_period"], 1_500_000)
        self.assertEqual(result["total_period_cost"], 30_900_000)
        self.assertEqual(result["unknown"], ["deposit"])
        not_calculated = {item["name"]: item["value"] for item in result["not_calculated"]}
        self.assertEqual(not_calculated["fees.electricity"], "4k/kWh")
        self.assertEqual(not_calculated["fees.water"], "30k/m3")

    def test_calculator_handles_text_rent_without_crashing(self):
        result = ReadOnlyToolRegistry().execute(
            "calculate_cost_estimate",
            {
                "room": {
                    "room_id": "A101",
                    "rent_price": "4.5 triệu",
                    "deposit": "4.500.000đ",
                    "fees": {},
                },
                "rental_months": 6,
            },
            ToolExecutionContext(repository=InMemoryRoomRepository([])),
        )
        self.assertEqual(result["total_initial_cost"], 9_000_000)
        self.assertEqual(result["total_period_cost"], 31_500_000)

    def test_compare_missing_room_ids_reports_not_found(self):
        result = asyncio.run(run_room_assistant(
            "So sánh #Z999 #Y888",
            session_id="missing-compare",
            repository=InMemoryRoomRepository([]),
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        self.assertIn("chưa tìm thấy dữ liệu", result["answer"].lower())
        self.assertIn("#Z999", result["answer"])
        self.assertIn("#Y888", result["answer"])

    def test_dynamic_room_answer_cache_disabled_without_safe_context(self):
        self.assertIsNone(get_cached_answer("Phòng này có nuôi mèo không?", context=None, dynamic_room=True))


if __name__ == "__main__":
    unittest.main()
