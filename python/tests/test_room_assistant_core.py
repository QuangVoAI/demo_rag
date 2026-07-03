import asyncio
import json
import time
import unittest
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents import sentiment_analyzer
from retrieval.cache import get_cached_answer
from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import (
    InMemoryRoomRepository,
    MongoRoomRepository,
    _available_status_query,
    build_mongo_query,
    create_room_repository,
    room_matches_constraints,
)
from room_assistant.schemas import default_session_state, normalize_room
from room_assistant.session_store import InMemorySessionStore, apply_operations, load_session_state, update_turn_state
from room_assistant.tools import ReadOnlyToolRegistry, ToolExecutionContext, ToolBudgetExceeded
from room_assistant.workflow import _build_llm_context, _compose_answer_async, _compose_answer_template, run_room_assistant
from room_assistant.prompts import format_request_action_answer


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

    def test_location_replace_clears_previous_location_filters(self):
        state = default_session_state("s-location")
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "binh thanh"},
            {"op": "append", "path": "location.near_landmarks", "value": "dhqg"},
        ])

        parsed = parse_intent_and_constraint_patch("đổi sang quận 7 gần lotte", state)
        self.assertIn({"op": "clear", "path": "location.districts"}, parsed["operations"])
        self.assertIn({"op": "clear", "path": "location.near_landmarks"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "location.districts", "value": "quan 7"}, parsed["operations"])

        next_state, _ = apply_operations(state, parsed["operations"])
        self.assertEqual(next_state["constraints"]["location"]["districts"], ["quan 7"])
        self.assertEqual(next_state["constraints"]["location"]["near_landmarks"], ["lotte"])

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

    def test_price_refinement_cheaper_than_replaces_previous_range_min(self):
        state = default_session_state("s-cheaper")
        state, _ = apply_operations(state, [
            {"op": "set", "path": "budget.min", "value": 10_000_000},
            {"op": "set", "path": "budget.min_operator", "value": "gte"},
            {"op": "set", "path": "budget.max", "value": 20_000_000},
            {"op": "set", "path": "budget.max_operator", "value": "lte"},
            {"op": "append", "path": "location.districts", "value": "quan 1"},
        ])
        state["last_intent"] = "SEARCH_ROOM"

        for question in ("rẻ hơn 10 triệu", "thấp hơn 10 triệu", "dưới 10 triệu", "rẻ hơn 10 củ"):
            with self.subTest(question=question):
                parsed = parse_intent_and_constraint_patch(question, state)
                self.assertIn({"op": "set", "path": "budget.max", "value": 10_000_000}, parsed["operations"])
                self.assertIn({"op": "set", "path": "budget.max_operator", "value": "lt"}, parsed["operations"])
                next_state, _ = apply_operations(state, parsed["operations"])
                budget = next_state["constraints"]["budget"]
                self.assertIsNone(budget["min"])
                self.assertIsNone(budget["min_operator"])
                self.assertEqual(budget["max"], 10_000_000)

    def test_price_refinement_above_conflicting_old_max_clears_max(self):
        state = default_session_state("s-pricier")
        state, _ = apply_operations(state, [
            {"op": "set", "path": "budget.min", "value": 10_000_000},
            {"op": "set", "path": "budget.min_operator", "value": "gte"},
            {"op": "set", "path": "budget.max", "value": 20_000_000},
            {"op": "set", "path": "budget.max_operator", "value": "lte"},
        ])
        parsed = parse_intent_and_constraint_patch("cao hơn 20 triệu", state)
        self.assertIn({"op": "set", "path": "budget.min", "value": 20_000_000}, parsed["operations"])
        self.assertIn({"op": "set", "path": "budget.min_operator", "value": "gt"}, parsed["operations"])
        next_state, _ = apply_operations(state, parsed["operations"])
        budget = next_state["constraints"]["budget"]
        self.assertEqual(budget["min"], 20_000_000)
        self.assertIsNone(budget["max"])
        self.assertIsNone(budget["max_operator"])

    def test_colloquial_higher_than_budget_uses_min_not_max(self):
        parsed = parse_intent_and_constraint_patch("Tìm phòng trên 5 củ")
        self.assertIn({"op": "set", "path": "budget.min", "value": 5_000_000}, parsed["operations"])
        self.assertIn({"op": "set", "path": "budget.min_operator", "value": "gt"}, parsed["operations"])
        self.assertNotIn({"op": "set", "path": "budget.max", "value": 5_000_000}, parsed["operations"])

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
        state["last_result_ids"] = ["A101", "B202"]
        for question in (
            "Phòng này diện tích bao nhiêu?",
            "Phòng này có ban công không?",
            "Phòng này có tiện ích gì?",
            "Còn phòng trống không?",
            "Mã phòng là gì?",
        ):
            parsed = parse_intent_and_constraint_patch(question, state)
            self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
            self.assertEqual(parsed["current_room_id"], "A101", question)
        parsed = parse_intent_and_constraint_patch("Giá điện nước phòng này thế nào?", state)
        self.assertEqual(parsed["intent"], "CALCULATE_COST")
        self.assertEqual(parsed["current_room_id"], "A101")

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
                    return [[1.0, 0.0] for _ in text]
                return [1.0, 0.0]

        old_centroids = sentiment_analyzer._centroids
        try:
            sentiment_analyzer._centroids = {
                "frustrated": [1.0, 0.0],
                "urgent": [0.5, 0.5],
                "normal": [0.0, 1.0],
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

    def test_frustrated_mood_catches_student_cannot_afford_phrase(self):
        self.assertTrue(
            sentiment_analyzer._has_explicit_mood_cue(
                "Giá cao thế, sinh viên sao mà thuê nổi?",
                "frustrated",
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

    def test_deposit_question_does_not_route_to_request_action(self):
        parsed = parse_intent_and_constraint_patch("Tiền đặt cọc bao nhiêu?")
        self.assertEqual(parsed["intent"], "REQUEST_FAQ")
        self.assertIsNone(parsed["requested_action"])

    def test_action_capability_question_routes_to_faq_instead_of_action(self):
        parsed = parse_intent_and_constraint_patch("Có thể đặt phòng qua web không?")
        self.assertEqual(parsed["intent"], "REQUEST_FAQ")
        self.assertIsNone(parsed["requested_action"])

    def test_request_action_templates_are_action_specific(self):
        cases = {
            "huy_lich": "chưa hủy lịch hộ",
            "doi_lich": "chưa đổi lịch hộ",
            "negotiate": "không có quyền thương lượng",
            "dat_lich": "chưa được phép tự đặt lịch hộ",
        }
        for action, snippet in cases.items():
            answer = format_request_action_answer(action)
            self.assertIn(snippet, answer.lower(), action)

        parsed = {"intent": "REQUEST_ACTION", "requested_action": "huy_lich"}
        answer = _compose_answer_template(parsed, {"rooms": [], "constraints": {}}, {})
        self.assertIn("hủy lịch", answer.lower())

    def test_schedule_cancel_and_reschedule_actions(self):
        for question, expected_action in (
            ("Hủy lịch hẹn xem phòng", "huy_lich"),
            ("Đổi lịch hẹn sang chiều mai", "doi_lich"),
        ):
            parsed = parse_intent_and_constraint_patch(question)
            self.assertEqual(parsed["intent"], "REQUEST_ACTION", question)
            self.assertEqual(parsed["requested_action"], expected_action, question)

    def test_rental_advice_questions_route_to_faq(self):
        for question in (
            "Sinh viên nên lưu ý gì khi thuê trọ?",
            "Làm sao nhận biết tin lừa đảo?",
            "Hoàn cọc khi chuyển đi thế nào?",
        ):
            parsed = parse_intent_and_constraint_patch(question)
            self.assertEqual(parsed["intent"], "REQUEST_FAQ", question)

    def test_soft_preference_room_question_does_not_mutate_search(self):
        state = default_session_state("s-soft")
        state["current_room_id"] = "A101"
        state["last_result_ids"] = ["A101", "B202"]
        state["last_intent"] = "SEARCH_ROOM"
        parsed = parse_intent_and_constraint_patch("Khu này an ninh không?", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        self.assertEqual(parsed["current_room_id"], "A101")
        preferred = [op for op in parsed["operations"] if op.get("path") == "amenities_preferred"]
        self.assertEqual(preferred, [])

    def test_compare_by_result_ordinals_collects_multiple_room_ids(self):
        state = default_session_state("s-compare")
        state["last_result_ids"] = ["A101", "B202", "C303"]
        parsed = parse_intent_and_constraint_patch("So sánh phòng số 2 và phòng số 3", state)
        self.assertEqual(parsed["intent"], "COMPARE_ROOMS")
        self.assertEqual(parsed["referenced_room_ids"], ["B202", "C303"])

    def test_room_code_question_resolves_room_detail_without_llm(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "507f1f77bcf86cd799439011",
                "metadata": {
                    "house_name": "CS7",
                    "room_code": "P305",
                    "price": 3_900_000,
                    "status_code": "0",
                    "district_name": "Tân Phú",
                },
                "embedding_text": "## Thông tin cơ bản\n- Diện tích: 20M2\n## Tiện ích\n- Máy lạnh: Có",
                "available": True,
                "status": "active",
            }
        ])
        result = asyncio.run(run_room_assistant(
            "Phòng P305 giá bao nhiêu?",
            session_id="room-code-detail",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        self.assertEqual(result["intent"], "ASK_ABOUT_ROOM")
        self.assertTrue(result["rooms"])
        self.assertEqual(result["rooms"][0]["room_code"], "P305")
        from room_assistant.money import answer_mentions_vnd

        self.assertTrue(
            answer_mentions_vnd(result["answer"], 3_900_000),
            msg=result["answer"],
        )

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
        self.assertIn(_available_status_query(), mongo_query["$and"])

    def test_build_mongo_query_includes_studio_category(self):
        query = build_mongo_query({
            "location": {"districts": ["quan 7"]},
            "categories": ["studio"],
        })
        serialized = json.dumps(query, ensure_ascii=False)
        self.assertIn("metadata.district_name", serialized)
        # Room-type categories rely on post-filter / embedding text, not Mongo `category`.
        self.assertNotIn('"category": "studio"', serialized)
        self.assertNotIn("embedding_text", serialized)
        self.assertGreaterEqual(len(query["$and"]), 2)

    def test_build_mongo_query_uses_structured_amenity_filter(self):
        query = build_mongo_query({
            "amenities_required": ["air_conditioner"],
        })
        serialized = json.dumps(query, ensure_ascii=False)
        self.assertIn('"amenities": "air_conditioner"', serialized)
        self.assertIn("amenities_canonical", serialized)
        self.assertIn("embedding_text", serialized)
        self.assertIn("Máy", serialized)

    def test_current_mongo_shape_amenity_text_is_enough_for_required_filter(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "Q7-NOAC",
                "metadata": {"price": 4_800_000, "status_code": "0", "district_name": "Quận 7"},
                "embedding_text": "## Tiện ích\n- Máy lạnh: Không\n- Wifi: Không có",
            },
            {
                "room_id": "Q7-AC",
                "metadata": {"price": 4_600_000, "status_code": "0", "district_name": "Quận 7"},
                "embedding_text": "## Tiện ích\n- Máy lạnh: Có\n- Wifi: Có",
            },
        ])
        constraints = {
            "location": {"districts": ["quan 7"]},
            "amenities_required": ["air_conditioner"],
        }
        rooms = repo.search_by_constraints(constraints)
        self.assertEqual([room["room_id"] for room in rooms], ["Q7-AC"])

    def test_normalize_room_treats_blank_status_code_as_available(self):
        room = normalize_room({
            "room_id": "Q5-202",
            "metadata": {
                "status_code": "",
                "status_desc": "",
                "price": 4_300_000,
                "district_name": "Quận 5",
            },
            "embedding_text": "",
        })
        self.assertTrue(room["available"])
        self.assertEqual(room["status"], "active")
        self.assertEqual(room["status_desc"], "Còn phòng")

    def test_normalize_room_missing_status_stays_unknown(self):
        room = normalize_room({
            "room_id": "Q5-203",
            "metadata": {"price": 4_300_000, "district_name": "Quận 5"},
            "embedding_text": "",
        })
        self.assertIsNone(room["available"])
        self.assertEqual(room["status_desc"], "Không rõ")
        self.assertEqual(room["status"], "unknown")
        self.assertEqual(room["status_key"], "unknown")

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

    def test_stream_callback_emits_progress_statuses(self):
        events = []

        async def capture(chunk):
            events.append(chunk)

        result = asyncio.run(run_room_assistant(
            "Tìm phòng dưới 5 triệu ở Bình Thạnh",
            session_id="stream-status",
            repository=InMemoryRoomRepository([]),
            session_store=InMemorySessionStore(),
            semantic_index=None,
            stream_callback=capture,
        ))
        self.assertEqual(result["intent"], "SEARCH_ROOM")
        joined = "".join(events)
        self.assertIn("[status:PHÂN TÍCH|gateway]", joined)
        self.assertIn("[status:ĐỊNH HƯỚNG|intent_router]", joined)
        self.assertIn("[status:LÀM RÕ|intent_router]", joined)
        self.assertIn("[status:NGỮ CẢNH|session_store]", joined)
        self.assertIn("[status:TRUY VẤN|retriever]", joined)
        self.assertIn("[status:ĐỐI CHIẾU|grounding_checker]", joined)
        self.assertIn("[status:SOẠN THẢO|response_writer]", joined)

    def test_compose_answer_streams_only_final_answer_after_review(self):
        from unittest.mock import AsyncMock

        events = []
        parsed = {"intent": "SEARCH_ROOM", "operations": []}
        grounding = {
            "rooms": [],
            "constraints": {"budget": {"max": 5_000_000}},
        }
        tool_results = {
            "rooms": [],
            "faq_results": [{"question": "q", "answer": "a"}],
        }

        async def capture(chunk):
            events.append(chunk)

        mock_write = AsyncMock(return_value="DRAFT_SHOULD_NOT_STREAM")
        mock_review = AsyncMock(return_value=(
            "FINAL_REVIEWED_ANSWER",
            {
                "is_approved": True,
                "issues": [],
                "used_corrected_answer": False,
            },
        ))

        with patch("config.ENABLE_REVIEWER", True):
            with patch("agents.response_writer.write_no_result_response", mock_write):
                with patch("agents.reviewer.review_with_retry", mock_review):
                    result = asyncio.run(_compose_answer_async(
                        "Tìm phòng dưới 5 triệu",
                        parsed,
                        grounding,
                        tool_results,
                        history=[],
                        stream_callback=capture,
                    ))

        mock_write.assert_awaited_once()
        self.assertIsNone(mock_write.await_args.kwargs.get("stream_callback"))
        mock_review.assert_awaited_once()
        self.assertEqual(mock_review.await_args.kwargs["answer"], "DRAFT_SHOULD_NOT_STREAM")
        self.assertEqual(result["answer"], "FINAL_REVIEWED_ANSWER")
        self.assertFalse(result["verification"]["corrected_answer_used"])
        joined = "".join(events)
        self.assertNotIn("DRAFT_SHOULD_NOT_STREAM", joined)
        self.assertIn("FINAL_REVIEWED_ANSWER", joined)

    def test_hallucinated_money_falls_back_to_template_answer(self):
        store = InMemorySessionStore()
        state = default_session_state("room-detail-guard")
        state["current_room_id"] = "A101"
        store.save("room-detail-guard", state, ttl_seconds=60)
        repo = InMemoryRoomRepository([
            {
                "room_id": "A101",
                "metadata": {
                    "price": 4_500_000,
                    "status_code": "0",
                    "district_name": "Bình Thạnh",
                },
                "embedding_text": "## Thông tin cơ bản\n- Diện tích: 24M2\n## Tiện ích\n- Máy lạnh: Có",
                "available": True,
                "status": "active",
                "title": "Studio Bình Thạnh",
            }
        ])

        with patch("agents.response_writer.write_response", return_value="Phòng này giá 9 triệu/tháng."):
            result = asyncio.run(run_room_assistant(
                "Phòng này giá bao nhiêu?",
                session_id="room-detail-guard",
                repository=repo,
                session_store=store,
                semantic_index=None,
            ))

        self.assertNotIn("9 triệu", result["answer"])
        self.assertIn("4.500.000", result["answer"])

    def test_cost_template_reads_fixed_items_from_calculator(self):
        parsed = {"intent": "CALCULATE_COST"}
        grounding = {"rooms": [], "constraints": {}}
        tool_results = {
            "cost_estimate": {
                "available": True,
                "rental_months": 6,
                "fixed_items": [
                    {"field": "monthly_rent", "amount": 4_500_000},
                    {"field": "parking", "amount": 150_000},
                ],
                "initial_payment_options": [{"deposit": 4_500_000}],
                "recurring_fees_for_period": 900_000,
                "total_period_cost": 32_400_000,
                "unknown": [],
                "not_calculated": [],
            }
        }
        answer = _compose_answer_template(parsed, grounding, tool_results)
        self.assertIn("Tiền thuê mỗi tháng", answer)
        self.assertIn("Tiền cọc", answer)
        self.assertIn("Tổng tạm tính 6 tháng", answer)

    def test_general_help_price_objection_uses_empathic_script(self):
        parsed = {"intent": "GENERAL_HELP"}
        grounding = {"rooms": [], "constraints": {}}
        answer = _compose_answer_template(
            parsed,
            grounding,
            {},
            question="Giá cao thế, sinh viên sao mà thuê nổi?",
            user_mood="frustrated",
        )
        self.assertIn("Dạ em hiểu", answer)
        self.assertIn("ngân sách", answer)
        self.assertIn("lọc lại", answer)

    def test_general_help_off_topic_request_is_refused_softly(self):
        parsed = {"intent": "GENERAL_HELP"}
        grounding = {"rooms": [], "constraints": {}}
        answer = _compose_answer_template(
            parsed,
            grounding,
            {},
            question="Viết code Python giúp tôi",
        )
        self.assertIn("chỉ hỗ trợ tư vấn phòng trọ", answer)
        self.assertIn("viết code", answer)
        self.assertIn("khu vực", answer)

    def test_frustrated_search_no_result_is_empathetic(self):
        parsed = {"intent": "SEARCH_ROOM"}
        grounding = {"rooms": [], "constraints": {}}
        answer = _compose_answer_template(
            parsed,
            grounding,
            {"rooms": []},
            question="Tìm hoài không thấy phòng nào",
            user_mood="frustrated",
        )
        self.assertIn("em hiểu", answer.lower())
        self.assertIn("mệt", answer.lower())

    def test_compare_template_highlights_price_area_and_feature_tradeoffs(self):
        parsed = {"intent": "COMPARE_ROOMS"}
        grounding = {"rooms": [], "constraints": {}}
        tool_results = {
            "comparison": {
                "rows": [
                    {
                        "room_id": "61ea636e3048d576be90729f",
                        "title": "C2-C3 HOÀNG QUỐC VIỆT - P.206",
                        "rent_price": 4_800_000,
                        "area_m2": 20,
                        "district": "Quận 7",
                        "available": True,
                        "status_desc": "Phòng trống",
                        "amenities": ["Thang máy", "Tủ lạnh", "Giường"],
                        "embedding_text": "## Tiện ích\n- Wifi: Không\n- Gác: Không\n- Ban công: Có\n- Cửa sổ: Có\n- Thang máy: Có\n",
                    },
                    {
                        "room_id": "62cc0e17ff6aae63cefbeda7",
                        "title": "CUBICITY - INDIAN HOUSE Q7 - 107",
                        "rent_price": 4_500_000,
                        "area_m2": 15,
                        "district": "Quận 7",
                        "available": True,
                        "status_desc": "Phòng trống",
                        "amenities": ["Wifi", "Gác", "Tủ lạnh"],
                        "embedding_text": "## Tiện ích\n- Wifi: Có\n- Gác: Có\n- Ban công: Không\n- Cửa sổ: Không\n",
                    },
                ]
            }
        }

        answer = _compose_answer_template(parsed, grounding, tool_results)
        self.assertIn("rẻ hơn", answer)
        self.assertIn("rộng hơn", answer)
        self.assertIn("Ưu điểm C2-C3 HOÀNG QUỐC VIỆT - P.206", answer)
        self.assertIn("Ưu điểm CUBICITY - INDIAN HOUSE Q7 - 107", answer)

    def test_search_template_mentions_matching_landmark_hint(self):
        parsed = {"intent": "SEARCH_ROOM"}
        grounding = {
            "rooms": [
                {
                    "room_id": "A101",
                    "title": "Phong gan truong",
                    "rent_price": 2_500_000,
                    "district": "Quận 7",
                    "embedding_text": "",
                    "tien_ich_xq": "Gan TDTU, di hoc tien",
                }
            ],
            "constraints": {"location": {"near_landmarks": ["tdtu"]}},
        }
        answer = _compose_answer_template(parsed, grounding, {"rooms": grounding["rooms"]})
        self.assertIn("gần TDTU", answer)

    def test_trace_cost_room_id_with_hash_is_resolved(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "61ea636e3048d576be90729b",
                "house_id": "61ea636e3048d576be907293",
                "embedding_text": (
                    "## Giá & phí\n"
                    "- Điện: 4k/kWh\n"
                    "- Nước: 100k/ng\n"
                    "- Quản lý: 250k/ph\n"
                    "- Xe: Không có\n"
                    "- Wifi: Không có\n"
                    "- Máy giặt: Không có\n"
                ),
                "metadata": {
                    "house_name": "C2-C3 HOÀNG QUỐC VIỆT",
                    "room_code": "P.204",
                    "district_name": "Quận 7",
                    "price": 4_800_000,
                    "status_code": "0",
                },
                "available": True,
                "status": "active",
            }
        ])

        with patch("room_assistant.intent._llm_classify_intent", return_value={
            "intent": "CALCULATE_COST",
            "confidence": 1.0,
            "operations": [],
            "referenced_room_ids": ["#61ea636e3048d576be90729b"],
        }):
            result = asyncio.run(run_room_assistant(
                "Tính tổng chi phí cho #61ea636e3048d576be90729b",
                session_id="trace-cost",
                repository=repo,
                session_store=InMemorySessionStore(),
                semantic_index=None,
            ))

        self.assertEqual(result["intent"], "CALCULATE_COST")
        self.assertTrue(result["cost_estimate"]["available"])
        self.assertEqual(result["cost_estimate"]["room_id"], "61ea636e3048d576be90729b")
        self.assertIn("4.800.000", result["answer"])

    def test_refine_budget_adds_relative_to_existing_max(self):
        state = default_session_state("trace-budget")
        state, _ = apply_operations(state, [
            {"op": "set", "path": "budget.max", "value": 5_000_000},
            {"op": "set", "path": "budget.max_operator", "value": "lt"},
        ])

        parsed = parse_intent_and_constraint_patch("Nới ngân sách thêm 1 triệu", state)
        self.assertIn({"op": "set", "path": "budget.max", "value": 6_000_000}, parsed["operations"])
        next_state, _ = apply_operations(state, parsed["operations"])
        self.assertEqual(next_state["constraints"]["budget"]["max"], 6_000_000)

    def test_near_landmark_constraint_matches_embedding_text(self):
        room = normalize_room({
            "room_id": "TDTU-1",
            "metadata": {
                "price": 3_200_000,
                "status_code": "0",
                "district_name": "Quận 7",
            },
            "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7\n- Ghi chú: gần TDTU, tiện đi học",
            "available": True,
            "status": "active",
        })
        self.assertTrue(
            room_matches_constraints(
                room,
                {"location": {"near_landmarks": ["tdtu"]}},
            )
        )

    def test_run_room_assistant_finds_room_near_tdtu_with_chat_suffix_k(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "TDTU-1",
                "metadata": {
                    "house_name": "C2-C3 HOANG QUOC VIET",
                    "room_code": "P206",
                    "price": 4_800_000,
                    "status_code": "0",
                    "district_name": "Quận 7",
                },
                "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7\n- Ghi chú: gần TDTU, tiện đi học",
                "available": True,
                "status": "active",
            },
            {
                "room_id": "Q7-OTHER",
                "metadata": {
                    "house_name": "PHONG KHAC",
                    "room_code": "101",
                    "price": 4_500_000,
                    "status_code": "0",
                    "district_name": "Quận 7",
                },
                "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7\n- Ghi chú: gần chợ, gần siêu thị",
                "available": True,
                "status": "active",
            },
        ])
        state = default_session_state("tdtu-chat-suffix")
        state["last_intent"] = "FIND_SIMILAR"
        store = InMemorySessionStore()
        store.save("tdtu-chat-suffix", state, ttl_seconds=3600)

        async def tdtu_llm(_question, _state):
            return {
                "intent": "SEARCH_ROOM",
                "confidence": 0.95,
                "operations": [{"op": "set", "path": "location.university", "value": "tdtu"}],
                "referenced_room_ids": [],
            }

        with patch("room_assistant.intent._llm_classify_intent", tdtu_llm):
            result = asyncio.run(run_room_assistant(
                "có phòng nào gần TDTU k",
                session_id="tdtu-chat-suffix",
                repository=repo,
                session_store=store,
                semantic_index=None,
            ))

        self.assertEqual(result["intent"], "SEARCH_ROOM")
        self.assertEqual([room["room_id"] for room in result["rooms"]], ["TDTU-1"])

    def test_compare_result_set_respects_requested_first_two_rooms(self):
        repo = InMemoryRoomRepository([
            {"room_id": "R1", "metadata": {"price": 1_000_000, "status_code": "0"}, "available": True, "status": "active", "title": "R1"},
            {"room_id": "R2", "metadata": {"price": 2_000_000, "status_code": "0"}, "available": True, "status": "active", "title": "R2"},
            {"room_id": "R3", "metadata": {"price": 3_000_000, "status_code": "0"}, "available": True, "status": "active", "title": "R3"},
        ])
        state = default_session_state("compare-first-two")
        state["last_intent"] = "SEARCH_ROOM"
        state["last_result_ids"] = ["R1", "R2", "R3"]
        store = InMemorySessionStore()
        store.save("compare-first-two", state, ttl_seconds=3600)

        result = asyncio.run(run_room_assistant(
            "so sánh 2 phòng đầu tiên đi",
            session_id="compare-first-two",
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))

        self.assertEqual(result["intent"], "COMPARE_ROOMS")
        self.assertEqual([room["room_id"] for room in result["rooms"]], ["R1", "R2"])

    def test_new_district_replaces_previous_search_district(self):
        state = default_session_state("trace-district")
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "quan 7"},
        ])

        parsed = parse_intent_and_constraint_patch("tìm cho tôi nhà quận 5", state)
        self.assertIn({"op": "clear", "path": "location.districts"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "location.districts", "value": "quan 5"}, parsed["operations"])

    def test_search_below_budget_does_not_select_previous_lower_room(self):
        state = default_session_state("search-below-budget")
        state["last_intent"] = "SEARCH_ROOM"
        state["current_room_id"] = "BINH_THANH_2"
        state["last_result_ids"] = ["BINH_THANH_1", "BINH_THANH_2"]
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "binh thanh"},
            {"op": "set", "path": "budget.max", "value": 5_000_000},
        ])

        parsed = parse_intent_and_constraint_patch("Tìm phòng dưới 5 triệu ở quận 7", state)

        self.assertIn(parsed["intent"], {"SEARCH_ROOM", "REFINE_SEARCH"})
        self.assertIsNone(parsed["current_room_id"])
        self.assertEqual(parsed["referenced_room_ids"], [])
        self.assertIn({"op": "clear", "path": "location.districts"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "location.districts", "value": "quan 7"}, parsed["operations"])

    def test_repeated_same_district_clears_stale_ward_and_landmark(self):
        state = default_session_state("s-stale-loc")
        state["last_intent"] = "SEARCH_ROOM"
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "quan 7"},
            {"op": "append", "path": "location.wards", "value": "tan hung"},
            {"op": "append", "path": "location.near_landmarks", "value": "nguyen huu tho"},
        ])

        parsed = parse_intent_and_constraint_patch("Tìm phòng ở quận 7", state)
        self.assertIn({"op": "clear", "path": "location.wards"}, parsed["operations"])
        self.assertIn({"op": "clear", "path": "location.near_landmarks"}, parsed["operations"])

        next_state, _ = apply_operations(state, parsed["operations"])
        location = next_state["constraints"]["location"]
        self.assertEqual(location["districts"], ["quan 7"])
        self.assertEqual(location["wards"], [])
        self.assertEqual(location["near_landmarks"], [])

    def test_ward_query_is_accent_flexible(self):
        query = build_mongo_query({"location": {"wards": ["tan hung"]}})
        ward_patterns = []
        for clause in query["$and"]:
            for option in clause.get("$or", []):
                if "metadata.ward_name" in option:
                    ward_patterns.append(option["metadata.ward_name"]["$regex"])
        self.assertTrue(
            any(re.search(pattern, "Tân Hưng", re.IGNORECASE) for pattern in ward_patterns)
        )

    def test_search_relaxes_unmatchable_landmark_and_returns_area_rooms(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "Q7-1",
                "metadata": {"house_name": "HQV", "room_code": "204", "price": 4_800_000, "status_code": "0", "district_name": "Quận 7"},
                "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7",
                "available": True,
                "status": "active",
            },
            {
                "room_id": "Q7-2",
                "metadata": {"house_name": "CUBI", "room_code": "107", "price": 4_500_000, "status_code": "0", "district_name": "Quận 7"},
                "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7",
                "available": True,
                "status": "active",
            },
        ])
        state = default_session_state("relax-landmark")
        state["last_intent"] = "SEARCH_ROOM"
        state, _ = apply_operations(state, [{"op": "append", "path": "location.districts", "value": "quan 7"}])
        store = InMemorySessionStore()
        store.save("relax-landmark", state, ttl_seconds=3600)

        result = asyncio.run(run_room_assistant(
            "Căn nào mà gần TDTU á",
            session_id="relax-landmark",
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))
        self.assertTrue(result["rooms"])
        self.assertTrue(all(room["district"] == "Quận 7" for room in result["rooms"]))
        self.assertIn("vị trí gần mốc", result["answer"])

    def test_repeated_district_run_recovers_after_stale_filters(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "Q7-1",
                "metadata": {"house_name": "HQV", "room_code": "204", "price": 4_800_000, "status_code": "0", "district_name": "Quận 7"},
                "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 7",
                "available": True,
                "status": "active",
            },
        ])
        state = default_session_state("stale-recover")
        state["last_intent"] = "SEARCH_ROOM"
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "quan 7"},
            {"op": "append", "path": "location.wards", "value": "tan hung"},
            {"op": "append", "path": "location.near_landmarks", "value": "nguyen huu tho"},
        ])
        store = InMemorySessionStore()
        store.save("stale-recover", state, ttl_seconds=3600)

        result = asyncio.run(run_room_assistant(
            "Tìm phòng ở quận 7",
            session_id="stale-recover",
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))
        self.assertTrue(result["rooms"])
        self.assertEqual(result["rooms"][0]["room_id"], "Q7-1")
        location = result["session_state"]["constraints"]["location"]
        self.assertEqual(location["near_landmarks"], [])
        self.assertEqual(location["wards"], [])

    def test_repository_enforces_studio_category(self):
        studio_room = {
            "room_id": "s1",
            "available": True,
            "category": "studio",
            "category_key": "studio",
            "rent_price": 4_000_000,
        }
        normal_room = {
            "room_id": "n1",
            "available": True,
            "category": "phong_tro",
            "category_key": "phong_tro",
            "rent_price": 3_000_000,
        }
        constraints = {"categories": ["studio"]}
        self.assertTrue(room_matches_constraints(studio_room, constraints))
        self.assertFalse(room_matches_constraints(normal_room, constraints))

    def test_repository_enforces_studio_category_from_embedding_fallback(self):
        studio_room = {
            "room_id": "s2",
            "available": True,
            "embedding_text": "Studio gọn, giá tốt",
            "rent_price": 4_000_000,
        }
        constraints = {"categories": ["studio"]}
        self.assertTrue(room_matches_constraints(studio_room, constraints))

    def test_repository_enforces_pets_required(self):
        pet_room = {
            "room_id": "p1",
            "available": True,
            "pet_policy": "allowed",
            "rent_price": 4_000_000,
        }
        no_pet_room = {
            "room_id": "np1",
            "available": True,
            "pet_policy": "denied",
            "rent_price": 3_500_000,
        }
        constraints = {"pets_required": ["cat"]}
        self.assertTrue(room_matches_constraints(pet_room, constraints))
        self.assertFalse(room_matches_constraints(no_pet_room, constraints))

    def test_repository_pets_required_allows_unknown_policy(self):
        unknown_pet_room = {
            "room_id": "unk1",
            "available": True,
            "rent_price": 3_000_000,
            "embedding_text": "## Tiện ích\n- Wifi: Có",
        }
        constraints = {"pets_required": ["cat"]}
        self.assertTrue(room_matches_constraints(unknown_pet_room, constraints))

    def test_ask_room_pets_without_verified_data_is_insufficient(self):
        parsed = {"intent": "ASK_ABOUT_ROOM"}
        grounding = {
            "rooms": [{
                "room_id": "r1",
                "title": "Phòng A",
                "rent_price": 4_000_000,
                "district": "Quận 7",
                "embedding_text": "## Tiện ích\n- Wifi: Có",
            }],
            "constraints": {},
        }
        answer = _compose_answer_template(
            parsed,
            grounding,
            {},
            question="Phòng này có cho nuôi mèo không?",
            user_mood="normal",
        )
        self.assertIn("chưa có dữ liệu xác minh", answer.lower())
        self.assertIn("thú cưng", answer.lower())

    def test_narrow_keywords_do_not_flag_generic_search(self):
        from room_assistant.tools import resolve_sensitive_answer_types

        types = resolve_sensitive_answer_types(
            "Tìm phòng quận 7",
            intent="SEARCH_ROOM",
            has_room_context=False,
        )
        self.assertNotIn("price_query", types)
        self.assertNotIn("utilities_query", types)

    def test_pets_slot_plus_question_triggers_pets_sufficiency(self):
        from room_assistant.tools import SufficiencyStatus, evaluate_room_data_sufficiency

        room = {
            "room_id": "r1",
            "embedding_text": "## Tiện ích\n- Wifi: Có",
        }
        status, missing = evaluate_room_data_sufficiency(
            "Phòng này nuôi mèo được không?",
            room,
            intent="ASK_ABOUT_ROOM",
            constraints={"pets_required": ["cat"]},
        )
        self.assertEqual(status, SufficiencyStatus.INSUFFICIENT)
        self.assertIn("pets_policy", missing)

    def test_dotted_room_code_p305_resolves_by_room_code_norm(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "mongo-p305",
                "metadata": {
                    "house_name": "CS7",
                    "room_code": "P.305",
                    "price": 3_900_000,
                    "status_code": "0",
                    "district_name": "Tân Phú",
                },
                "embedding_text": "## Tiện ích\n- Máy lạnh: Có",
                "available": True,
                "status": "active",
            }
        ])
        result = asyncio.run(run_room_assistant(
            "Phòng P.305 giá bao nhiêu?",
            session_id="room-code-dotted",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        self.assertEqual(result["intent"], "ASK_ABOUT_ROOM")
        self.assertTrue(result["rooms"])
        self.assertEqual(result["rooms"][0]["room_code"], "P.305")

    def test_detail_amenity_question_does_not_set_search_filter(self):
        state = default_session_state("detail-ac")
        state["last_result_ids"] = ["A101"]
        state["current_room_id"] = "A101"
        state["last_intent"] = "SEARCH_ROOM"
        parsed = parse_intent_and_constraint_patch("Phòng này có máy lạnh không?", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        amenities = [op["value"] for op in parsed["operations"] if op.get("path") == "amenities_required"]
        self.assertNotIn("air_conditioner", amenities)

    def test_ordinal_out_of_range_does_not_fallback_room(self):
        state = default_session_state("ordinal-miss")
        state["last_result_ids"] = ["A101", "B202", "C303"]
        state["last_intent"] = "SEARCH_ROOM"
        parsed = parse_intent_and_constraint_patch("Giá phòng số 4 bao nhiêu?", state)
        self.assertTrue(parsed.get("ordinal_out_of_range"))
        self.assertIsNone(parsed.get("current_room_id"))

    def test_compare_first_and_third_room_by_ordinal(self):
        state = default_session_state("compare-1-3")
        state["last_result_ids"] = ["A101", "B202", "C303"]
        parsed = parse_intent_and_constraint_patch("So sánh phòng đầu tiên và phòng thứ ba", state)
        self.assertEqual(parsed["intent"], "COMPARE_ROOMS")
        self.assertEqual(parsed["referenced_room_ids"], ["A101", "C303"])

    def test_search_no_result_routes_to_sales_handoff(self):
        repo = InMemoryRoomRepository([])
        result = asyncio.run(run_room_assistant(
            "Tìm phòng quận 7 dưới 1 triệu",
            session_id="sales-handoff",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        self.assertIn("sales", result["answer"].lower())

    def test_retrieval_candidate_limit_defaults_to_100(self):
        from room_assistant.retrieval import _retrieval_config
        self.assertEqual(_retrieval_config()["candidate_limit"], 100)

    def test_province_and_ward_post_filter_blocks_wrong_metadata(self):
        room = {
            "room_id": "W1",
            "available": True,
            "province": "Hồ Chí Minh",
            "district": "Quận 7",
            "ward": "Tân Phú Trung",
            "rent_price": 4_000_000,
            "amenities": [],
        }
        self.assertTrue(room_matches_constraints(room, {
            "location": {"province": "Hồ Chí Minh", "wards": ["tan phu trung"]},
        }))
        self.assertFalse(room_matches_constraints(room, {
            "location": {"province": "Đồng Nai", "wards": ["tan phu trung"]},
        }))
        self.assertFalse(room_matches_constraints(room, {
            "location": {"wards": ["phuoc long"]},
        }))

    def test_metadata_candidates_cannot_bypass_province_constraint(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "HN-1",
                "available": True,
                "province": "Hà Nội",
                "district": "Cầu Giấy",
                "rent_price": 5_000_000,
                "embedding_text": "studio quan 7",
                "metadata": {"status_code": "0"},
            },
            {
                "room_id": "HCM-1",
                "available": True,
                "province": "Hồ Chí Minh",
                "district": "Quận 7",
                "rent_price": 4_500_000,
                "embedding_text": "studio quan 7",
                "metadata": {"status_code": "0"},
            },
        ])
        from room_assistant.retrieval import search_rooms_with_hard_filters
        rooms = search_rooms_with_hard_filters(
            "studio quan 7",
            {"location": {"province": "Hồ Chí Minh"}},
            repository=repo,
            semantic_index=None,
            top_k=5,
        )
        self.assertEqual([room["room_id"] for room in rooms], ["HCM-1"])

    def test_composite_category_chdv_and_2pn_matches_room_with_both_signals(self):
        room = {
            "room_id": "CHDV-2PN",
            "available": True,
            "category": "chdv",
            "rent_price": 8_000_000,
            "embedding_text": "CHDV 2 phòng ngủ ban công",
            "amenities": [],
        }
        self.assertTrue(room_matches_constraints(room, {"categories": ["chdv", "2pn"]}))

    def test_composite_category_query_does_not_and_impossible_scalar_filters(self):
        query = build_mongo_query({"categories": ["chdv", "2pn"]})
        category_clauses = [
            clause for clause in query["$and"]
            if any("category" in option for option in clause.get("$or", []))
        ]
        self.assertEqual(category_clauses, [])

    def test_detail_landmark_question_does_not_set_search_filters(self):
        state = default_session_state("detail-landmark")
        state["last_result_ids"] = ["A101"]
        state["current_room_id"] = "A101"
        state["last_intent"] = "SEARCH_ROOM"
        parsed = parse_intent_and_constraint_patch("Phòng này gần TDTU không?", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        paths = {op.get("path") for op in parsed["operations"]}
        self.assertFalse(paths & {"location.near_landmarks", "location.districts", "budget.max"})

    def test_detail_budget_question_does_not_set_search_filters(self):
        state = default_session_state("detail-budget")
        state["last_result_ids"] = ["A101"]
        state["current_room_id"] = "A101"
        parsed = parse_intent_and_constraint_patch("phòng này dưới 5 triệu không?", state)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        paths = {op.get("path") for op in parsed["operations"]}
        self.assertNotIn("budget.max", paths)

    def test_warm_session_balcony_search_routes_to_refine_search(self):
        state = default_session_state("balcony-search")
        state["last_intent"] = "SEARCH_ROOM"
        state["last_result_ids"] = ["A101", "B202"]
        parsed = parse_intent_and_constraint_patch("Có căn nào có ban công không?", state)
        self.assertIn(parsed["intent"], {"SEARCH_ROOM", "REFINE_SEARCH"})
        self.assertIn("balcony", [op["value"] for op in parsed["operations"] if op.get("path") == "amenities_required"])

    def test_compare_room_1_and_4_unresolved_with_three_results(self):
        state = default_session_state("compare-1-4")
        state["last_result_ids"] = ["R1", "R2", "R3"]
        parsed = parse_intent_and_constraint_patch("So sánh phòng số 1 và số 4", state)
        self.assertEqual(parsed["intent"], "COMPARE_ROOMS")
        self.assertTrue(parsed.get("compare_unresolved"))

    def test_refine_single_result_does_not_replace_last_result_ids(self):
        from room_assistant.session_store import update_turn_state
        state = default_session_state("refine-one")
        state["last_result_ids"] = ["OLD1", "OLD2", "OLD3"]
        next_state = update_turn_state(
            state,
            intent="REFINE_SEARCH",
            current_room_id="NEW1",
            referenced_room_ids=[],
            result_ids=["NEW1"],
        )
        self.assertEqual(next_state["last_result_ids"], ["OLD1", "OLD2", "OLD3"])

    def test_compare_turn_sets_selected_room_ids(self):
        from room_assistant.session_store import update_turn_state
        state = default_session_state("compare-selected")
        state["last_result_ids"] = ["R1", "R2", "R3", "R4"]
        next_state = update_turn_state(
            state,
            intent="COMPARE_ROOMS",
            current_room_id=None,
            referenced_room_ids=["R1", "R3"],
            result_ids=["R1", "R3"],
        )
        self.assertEqual(next_state["selected_room_ids"], ["R1", "R3"])
        self.assertEqual(next_state["last_result_ids"], ["R1", "R2", "R3", "R4"])

    def test_take_top_n_rooms_is_refine_not_single_ordinal(self):
        state = default_session_state("top-n")
        state["last_intent"] = "REFINE_SEARCH"
        state["last_result_ids"] = ["A101", "B202", "C303", "D404"]
        parsed = parse_intent_and_constraint_patch("lấy 3 phòng tốt nhất", state)
        self.assertEqual(parsed["intent"], "REFINE_SEARCH")
        self.assertIsNone(parsed.get("current_room_id"))

    def test_area_size_question_does_not_set_rent_budget(self):
        state = default_session_state("area-q")
        state["last_intent"] = "REFINE_SEARCH"
        state["last_result_ids"] = ["A101", "B202"]
        parsed = parse_intent_and_constraint_patch("có phòng nào lớn hơn 25m2 không", state)
        self.assertNotIn(
            {"op": "set", "path": "budget.min", "value": 25_000_000},
            parsed["operations"],
        )

    def test_find_similar_without_current_room_asks_for_source(self):
        result = asyncio.run(run_room_assistant(
            "Tìm phòng tương tự",
            session_id="find-similar-missing",
            repository=InMemoryRoomRepository([]),
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        self.assertEqual(result["intent"], "FIND_SIMILAR")
        self.assertIn("mã phòng", result["answer"].lower())

    def test_location_pivot_does_not_set_bright_preference(self):
        state = default_session_state("pivot-bright")
        state, _ = apply_operations(state, [{"op": "append", "path": "location.districts", "value": "binh thanh"}])
        parsed = parse_intent_and_constraint_patch("đổi sang quận 7", state)
        preferred = [op["value"] for op in parsed["operations"] if op.get("path") == "amenities_preferred"]
        self.assertNotIn("bright", preferred)

    def test_preferred_amenities_boost_ranks_matching_room_higher(self):
        from room_assistant.retrieval import _preferred_amenities_boost
        with_balcony = {
            "amenities_canonical": ["balcony"],
            "amenities": [],
            "embedding_text": "",
        }
        without = {"amenities_canonical": [], "amenities": [], "embedding_text": ""}
        self.assertGreater(
            _preferred_amenities_boost(with_balcony, ["balcony"]),
            _preferred_amenities_boost(without, ["balcony"]),
        )

    def test_stale_category_cleared_on_full_search_without_room_type(self):
        state = default_session_state("sticky-category")
        state, _ = apply_operations(state, [{"op": "append", "path": "categories", "value": "phong_tro"}])

        parsed = parse_intent_and_constraint_patch("Tìm phòng dưới 5tr ở bình thạnh", state)
        self.assertIn({"op": "clear", "path": "categories"}, parsed["operations"])

        next_state, _ = apply_operations(state, parsed["operations"])
        self.assertEqual(next_state["constraints"].get("categories"), [])

    def test_district_pivot_clears_sticky_category(self):
        state = default_session_state("pivot-category")
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "binh thanh"},
            {"op": "append", "path": "categories", "value": "phong_tro"},
        ])

        parsed = parse_intent_and_constraint_patch("tim phong o Binh Tan duoi 1 trieu", state)
        self.assertIn({"op": "clear", "path": "categories"}, parsed["operations"])

    def test_phong_tro_matches_room_without_category_metadata(self):
        room = {
            "room_id": "BT-1",
            "category": None,
            "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận Bình Thạnh",
            "rent_price": 4_800_000,
            "available": True,
        }
        self.assertTrue(room_matches_constraints(room, {"categories": ["phong_tro"]}))

        studio_room = dict(room, embedding_text="Studio cao cap Quan 1")
        self.assertFalse(room_matches_constraints(studio_room, {"categories": ["phong_tro"]}))

    def test_location_pivot_after_landmark_search_clears_tdtu(self):
        state = default_session_state("pivot-tdtu")
        state["last_intent"] = "SEARCH_ROOM"
        state["last_result_ids"] = ["room-a", "room-b", "room-c"]
        state, _ = apply_operations(state, [
            {"op": "set", "path": "budget.max", "value": 5_000_000},
            {"op": "append", "path": "location.near_landmarks", "value": "tdtu"},
        ])

        parsed = parse_intent_and_constraint_patch("đổi sang quận 2", state)
        self.assertIn(parsed["intent"], {"SEARCH_ROOM", "REFINE_SEARCH"})
        self.assertIn({"op": "clear", "path": "location.near_landmarks"}, parsed["operations"])
        self.assertIn({"op": "append", "path": "location.districts", "value": "quan 2"}, parsed["operations"])

        next_state, _ = apply_operations(state, parsed["operations"])
        location = next_state["constraints"]["location"]
        self.assertEqual(location["districts"], ["quan 2"])
        self.assertEqual(location["near_landmarks"], [])

    def test_relative_budget_without_state_skips_absolute_parse(self):
        parsed = parse_intent_and_constraint_patch("Nới ngân sách thêm 1 triệu", None)
        budget_ops = [op for op in parsed["operations"] if str(op.get("path", "")).startswith("budget.")]
        self.assertEqual(budget_ops, [])

    def test_conversation_flow_binh_thanh_returns_rooms(self):
        try:
            repo = create_room_repository()
        except Exception:
            self.skipTest("Mongo repository unavailable")
        if not isinstance(repo, MongoRoomRepository):
            self.skipTest("MONGODB_URI not configured; Mongo integration not exercised")

        from room_assistant.workflow import _execute_workflow
        from room_assistant.session_store import load_session_state, save_session_state

        store = InMemorySessionStore()
        session_id = "conv-binh-thanh"
        state = load_session_state(session_id, store)
        queries = [
            "tim phong tro Binh Thanh duoi 5 trieu",
            "Nới ngân sách thêm 1 triệu",
            "Tim phong duoi 5tr o binh thanh",
        ]
        for question in queries:
            parsed = parse_intent_and_constraint_patch(question, state)
            merged, _ = apply_operations(state, parsed["operations"])
            ctx = ToolExecutionContext(repository=repo)
            results = _execute_workflow(question, parsed, merged, ctx)
            state = update_turn_state(
                merged,
                parsed["intent"],
                None,
                [],
                [item.get("room_id") for item in (results.get("rooms") or []) if item.get("room_id")],
            )
            save_session_state(state, store, 3600)

        self.assertGreater(len(results.get("rooms") or []), 0)


def _acceptance_rooms() -> list[dict]:
    """Fixture matching production Mongo shape: Room A/B (Bình Thạnh), Room C (Quận 11)."""
    return [
        {
            "room_id": "ROOM_A",
            "metadata": {"price": 4_500_000, "status_code": "0", "district_name": "Quận Bình Thạnh"},
            "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận Bình Thạnh",
            "available": True,
            "status": "active",
        },
        {
            "room_id": "ROOM_B",
            "metadata": {"price": 5_500_000, "status_code": "0", "district_name": "Quận Bình Thạnh"},
            "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận Bình Thạnh",
            "available": True,
            "status": "active",
        },
        {
            "room_id": "ROOM_C",
            "metadata": {"price": 1_500_000, "status_code": "0", "district_name": "Quận 11"},
            "embedding_text": "## Thông tin nhà\n- Địa chỉ: Quận 11",
            "available": True,
            "status": "active",
        },
    ]


class _ZeroResultSemanticIndex:
    """Semantic index that ranks nothing (Qdrant filter/index mismatch)."""

    def __init__(self):
        self.calls = []

    def search_rooms(self, query_text, candidate_ids, top_k, metadata_filter=None):
        self.calls.append({"metadata_filter": metadata_filter, "candidate_ids": list(candidate_ids)})
        return []


class _BrokenSemanticIndex:
    """Semantic index that raises (Qdrant unavailable)."""

    def search_rooms(self, query_text, candidate_ids, top_k, metadata_filter=None):
        raise ConnectionError("qdrant down")


class FreshSearchAcceptanceTests(unittest.TestCase):
    """Section-12 acceptance matrix: fresh search on a deterministic fake repo."""

    def _run_turn(self, question, state, repo, semantic_index=None):
        parsed = parse_intent_and_constraint_patch(question, state)
        merged, _ = apply_operations(state, parsed["operations"])
        from room_assistant.workflow import _execute_workflow
        ctx = ToolExecutionContext(repository=repo, semantic_index=semantic_index)
        results = _execute_workflow(question, parsed, merged, ctx)
        next_state = update_turn_state(
            merged,
            parsed["intent"],
            None,
            [],
            [item.get("room_id") for item in (results.get("rooms") or []) if item.get("room_id")],
        )
        return results, next_state, ctx

    def test_binh_thanh_under_5m_returns_room_a(self):
        repo = InMemoryRoomRepository(_acceptance_rooms())
        results, _, _ = self._run_turn(
            "Tìm phòng Bình Thạnh dưới 5 triệu", default_session_state("fs-1"), repo,
        )
        self.assertEqual(
            [room["room_id"] for room in results["rooms"]], ["ROOM_A"],
        )

    def test_binh_thanh_under_5m_without_accents_returns_room_a(self):
        repo = InMemoryRoomRepository(_acceptance_rooms())
        results, _, _ = self._run_turn(
            "Binh Thanh duoi 5 trieu", default_session_state("fs-2"), repo,
        )
        self.assertEqual(
            [room["room_id"] for room in results["rooms"]], ["ROOM_A"],
        )

    def test_phong_tro_binh_thanh_under_5m_returns_room_a(self):
        """The original production failure: 'phong tro' category must not drop rooms without category metadata."""
        repo = InMemoryRoomRepository(_acceptance_rooms())
        results, _, _ = self._run_turn(
            "tim phong tro Binh Thanh duoi 5 trieu", default_session_state("fs-3"), repo,
        )
        self.assertEqual(
            [room["room_id"] for room in results["rooms"]], ["ROOM_A"],
        )

    def test_binh_thanh_under_6m_returns_rooms_a_and_b(self):
        repo = InMemoryRoomRepository(_acceptance_rooms())
        results, _, _ = self._run_turn(
            "Tìm phòng Bình Thạnh dưới 6 triệu", default_session_state("fs-4"), repo,
        )
        self.assertEqual(
            sorted(room["room_id"] for room in results["rooms"]), ["ROOM_A", "ROOM_B"],
        )

    def test_quan_11_under_1m_returns_zero_with_reason_code(self):
        repo = InMemoryRoomRepository(_acceptance_rooms())
        results, _, ctx = self._run_turn(
            "Tìm phòng ở Quận 11 dưới 1 triệu", default_session_state("fs-5"), repo,
        )
        self.assertEqual(results["rooms"], [])
        self.assertEqual(
            ctx.retrieval_trace.get("empty_result_reason"), "NO_HARD_FILTER_CANDIDATES",
        )


class MultiTurnRefinementAcceptanceTests(unittest.TestCase):
    """Section-12 critical acceptance: relative budget relax keeps location and reruns retrieval."""

    def _run_turn(self, question, state, repo):
        parsed = parse_intent_and_constraint_patch(question, state)
        merged, _ = apply_operations(state, parsed["operations"])
        from room_assistant.workflow import _execute_workflow
        ctx = ToolExecutionContext(repository=repo)
        results = _execute_workflow(question, parsed, merged, ctx)
        next_state = update_turn_state(
            merged,
            parsed["intent"],
            None,
            [],
            [item.get("room_id") for item in (results.get("rooms") or []) if item.get("room_id")],
        )
        return results, next_state

    def test_binh_thanh_relax_budget_makes_room_b_eligible(self):
        repo = InMemoryRoomRepository(_acceptance_rooms())
        state = default_session_state("mt-bt")

        results, state = self._run_turn("Tìm phòng Bình Thạnh dưới 5 triệu", state, repo)
        self.assertEqual([room["room_id"] for room in results["rooms"]], ["ROOM_A"])

        results, state = self._run_turn("Nới ngân sách thêm 1 triệu", state, repo)
        constraints = state["constraints"]
        self.assertEqual(constraints["budget"]["max"], 6_000_000)
        self.assertEqual(constraints["location"]["districts"], ["binh thanh"])
        self.assertIn("ROOM_B", [room["room_id"] for room in results["rooms"]])

    def test_quan_11_zero_then_relax_budget_returns_room_c(self):
        repo = InMemoryRoomRepository(_acceptance_rooms())
        state = default_session_state("mt-q11")
        version_start = state["state_version"]

        results, state = self._run_turn("Tìm phòng ở Quận 11 dưới 1 triệu", state, repo)
        self.assertEqual(results["rooms"], [])
        self.assertEqual(state["constraints"]["budget"]["max"], 1_000_000)
        version_after_t1 = state["state_version"]
        self.assertGreater(version_after_t1, version_start)

        results, state = self._run_turn("Nới ngân sách thêm 1 triệu", state, repo)
        constraints = state["constraints"]
        self.assertEqual(constraints["budget"]["max"], 2_000_000)
        self.assertEqual(constraints["location"]["districts"], ["quan 11"])
        self.assertGreater(state["state_version"], version_after_t1)
        self.assertEqual([room["room_id"] for room in results["rooms"]], ["ROOM_C"])


class SemanticRankingFallbackTests(unittest.TestCase):
    """Section-12 pipeline tests: Qdrant failures must not become false empty results."""

    def test_semantic_zero_results_falls_back_to_mongo_ranked(self):
        from room_assistant.retrieval import search_rooms_with_hard_filters

        repo = InMemoryRoomRepository(_acceptance_rooms())
        index = _ZeroResultSemanticIndex()
        trace = {}
        rooms = search_rooms_with_hard_filters(
            "Tìm phòng Bình Thạnh dưới 6 triệu",
            {"location": {"districts": ["binh thanh"]}, "budget": {"max": 6_000_000}},
            repository=repo,
            semantic_index=index,
            top_k=5,
            trace=trace,
        )
        self.assertEqual(
            sorted(room["room_id"] for room in rooms), ["ROOM_A", "ROOM_B"],
        )
        self.assertIsNone(trace["empty_result_reason"])
        self.assertEqual(trace["retrieval_attempts"][0]["semantic_result_count"], 0)
        # Hard filters live in candidate_ids; no district metadata filter may reach Qdrant
        # (normalized constraint values never match display-name payloads).
        for call in index.calls:
            self.assertIsNone(call["metadata_filter"])
            self.assertTrue(call["candidate_ids"])

    def test_semantic_index_exception_falls_back_instead_of_raising(self):
        from room_assistant.retrieval import search_rooms_with_hard_filters

        repo = InMemoryRoomRepository(_acceptance_rooms())
        trace = {}
        rooms = search_rooms_with_hard_filters(
            "Tìm phòng Bình Thạnh dưới 5 triệu",
            {"location": {"districts": ["binh thanh"]}, "budget": {"max": 5_000_000}},
            repository=repo,
            semantic_index=_BrokenSemanticIndex(),
            top_k=5,
            trace=trace,
        )
        self.assertEqual([room["room_id"] for room in rooms], ["ROOM_A"])
        self.assertTrue(trace["semantic_error"])
        self.assertIsNone(trace["empty_result_reason"])

    def test_no_candidates_reports_reason_not_qdrant(self):
        from room_assistant.retrieval import search_rooms_with_hard_filters

        repo = InMemoryRoomRepository([])
        trace = {}
        rooms = search_rooms_with_hard_filters(
            "Tìm phòng Bình Thạnh dưới 5 triệu",
            {"location": {"districts": ["binh thanh"]}, "budget": {"max": 5_000_000}},
            repository=repo,
            semantic_index=_BrokenSemanticIndex(),
            top_k=5,
            trace=trace,
        )
        self.assertEqual(rooms, [])
        self.assertEqual(trace["empty_result_reason"], "NO_HARD_FILTER_CANDIDATES")


if __name__ == "__main__":
    unittest.main()
