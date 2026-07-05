"""Behavior scenarios validated against MongoDB + Qdrant with sales-oriented dialogue."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import MongoRoomRepository, room_matches_constraints
from room_assistant.schemas import default_session_state
from room_assistant.session_store import InMemorySessionStore
from room_assistant.workflow import run_room_assistant

from mongo_test_helpers import (
    assert_answer_invites_customer,
    assert_rooms_match_mongo,
    create_mongo_repository,
    create_semantic_index,
    integration_stack_ready,
    load_dotenv,
    mongo_is_configured,
    mongo_room_price,
    qdrant_is_available,
)

YUHOME_ROOM_ID = "62963aae137e2a3d7e03c9d0"
PANDJ_ROOM_ID = "6399d72f07985f204285aed9"


async def _regex_aligned_llm(question: str, current_state: dict[str, Any] | None) -> dict[str, Any]:
    parsed = parse_intent_and_constraint_patch(question, current_state)
    return {
        "intent": parsed["intent"],
        "confidence": 0.99,
        "operations": parsed["operations"],
        "referenced_room_ids": parsed.get("referenced_room_ids", []),
        "requested_action": parsed.get("requested_action"),
    }


def _run_session(
    repo: MongoRoomRepository,
    questions: list[str],
    session_id: str = "mongo-behavior",
    *,
    use_qdrant: bool | None = None,
) -> list[dict[str, Any]]:
    if use_qdrant is None:
        use_qdrant = qdrant_is_available()

    store = InMemorySessionStore()
    history: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    semantic_index = create_semantic_index() if use_qdrant else None

    with patch("room_assistant.intent._llm_classify_intent", _regex_aligned_llm):
        with patch("room_assistant.intent._llm_verify_routing_decision", return_value={}):
            with patch("agents.sentiment_analyzer.analyze_mood", return_value=("normal", 0.0)):
                for question in questions:
                    result = asyncio.run(run_room_assistant(
                        question,
                        history=history,
                        session_id=session_id,
                        repository=repo,
                        session_store=store,
                        semantic_index=semantic_index,
                    ))
                    results.append(result)
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": result.get("answer") or ""})
    return results


@unittest.skipUnless(mongo_is_configured(), "MONGODB_URI not configured")
class MongoBehaviorScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()
        cls.repo = create_mongo_repository()
        if not isinstance(cls.repo, MongoRoomRepository):
            raise unittest.SkipTest("Mongo room repository unavailable")

    def test_off_topic_redirects_back_to_room_consulting(self) -> None:
        cases = [
            ("Em ơi giải hộ chị bài toán đi", "GENERAL_HELP"),
            ("Cho chị hỏi ngoài lề: viết giúp chị email xin việc với", "GENERAL_HELP"),
            ("Thời tiết Sài Gòn hôm nay mưa không em?", "GENERAL_HELP"),
        ]
        for question, expected_intent in cases:
            with self.subTest(question=question):
                parsed = parse_intent_and_constraint_patch(question)
                self.assertEqual(parsed["intent"], expected_intent)
                result = _run_session(self.repo, [question], session_id=f"off-topic-{expected_intent}")[-1]
                answer = (result.get("answer") or "").lower()
                self.assertEqual(result["intent"], expected_intent)
                self.assertFalse(result.get("rooms"))
                self.assertTrue(
                    "phòng trọ" in answer or "tìm phòng" in answer or "khu vực" in answer,
                    msg=answer[:160],
                )
                assert_answer_invites_customer(self, result.get("answer") or "")

    def test_hesitant_customer_can_pivot_district_and_budget(self) -> None:
        questions = [
            "Em gợi ý giúp chị mấy căn view đẹp ở quận 2 nha",
            "Ủa khoan, chị đổi sang quận 7 cho tiện đi làm",
            "Có căn nào dưới 4 triệu mà sạch sẽ, ở ngay được không em?",
        ]
        results = _run_session(self.repo, questions, session_id="mind-change-sales")

        self.assertIn(results[0]["intent"], {"SEARCH_ROOM", "REFINE_SEARCH"})
        self.assertIn(results[1]["intent"], {"SEARCH_ROOM", "REFINE_SEARCH"})
        self.assertEqual(results[2]["intent"], "REFINE_SEARCH")

        final_state = results[2]["session_state"]
        self.assertIn("quan 7", final_state["constraints"]["location"]["districts"])
        self.assertLessEqual(final_state["constraints"]["budget"]["max"], 4_000_000)

        rooms = results[2].get("rooms") or []
        if rooms:
            assert_rooms_match_mongo(
                self,
                self.repo,
                rooms,
                max_budget=4_000_000,
                district_contains="7",
            )
            assert_answer_invites_customer(self, results[2].get("answer") or "")

    def test_bargain_tone_keeps_trust_and_invites_alternatives(self) -> None:
        bargain_cases = [
            (
                "Phòng đẹp mà giá hơi cao, em thương chị giảm giá tí được không?",
                "REQUEST_FAQ",
                ("không có dữ liệu", "không tự chốt"),
            ),
            (
                "Đắt quá em ơi, bớt giá chút cho chị đi",
                "REQUEST_ACTION",
                ("chưa", "thao tác"),
            ),
            (
                "Giá cao thế, sinh viên như chị sao thuê nổi trời",
                "GENERAL_HELP",
                ("hiểu", "ngân sách"),
            ),
        ]
        for question, expected_intent, markers in bargain_cases:
            with self.subTest(question=question):
                parsed = parse_intent_and_constraint_patch(question)
                self.assertEqual(parsed["intent"], expected_intent)
                result = _run_session(self.repo, [question], session_id=f"bargain-{expected_intent}")[-1]
                answer = (result.get("answer") or "").lower()
                self.assertTrue(any(marker in answer for marker in markers), msg=answer[:200])
                self.assertNotRegex(answer, r"giảm\s+\d+\s*triệu|giảm\s+\d+%")
                assert_answer_invites_customer(self, result.get("answer") or "")

    def test_operational_requests_declined_with_viewing_invite(self) -> None:
        cases = [
            ("Chị muốn thanh toán cọc luôn, em hỗ trợ giúp chị nha", "payment"),
            ("Đặt lịch xem phòng chiều nay giúp chị với", "dat_lich"),
            ("Giữ phòng này giúp chị, chị rất ưng rồi", "hold_room"),
        ]
        state = default_session_state("ops-sales")
        state["current_room_id"] = YUHOME_ROOM_ID
        state["last_result_ids"] = [YUHOME_ROOM_ID, PANDJ_ROOM_ID]

        for question, action in cases:
            with self.subTest(question=question):
                parsed = parse_intent_and_constraint_patch(question, state)
                self.assertEqual(parsed["intent"], "REQUEST_ACTION")
                self.assertEqual(parsed.get("requested_action"), action)
                result = _run_session(self.repo, [question], session_id=f"ops-{action}")[-1]
                answer = (result.get("answer") or "").lower()
                self.assertIn("chưa", answer)
                self.assertTrue("thao tác" in answer or "hỗ trợ" in answer or "xem" in answer)

    def test_favorite_room_detail_pitch_matches_mongo(self) -> None:
        expected_price = mongo_room_price(self.repo, YUHOME_ROOM_ID)
        self.assertIsNotNone(expected_price, "fixture room must exist in MongoDB")

        question = f"Phòng #{YUHOME_ROOM_ID} nhìn ưng quá — em mô tả chi tiết giúp chị với ạ"
        parsed = parse_intent_and_constraint_patch(question)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")
        self.assertEqual(parsed["current_room_id"], YUHOME_ROOM_ID)

        result = _run_session(self.repo, [question], session_id="ask-detail-sales")[-1]
        self.assertEqual(result["intent"], "ASK_ABOUT_ROOM")
        rooms = result.get("rooms") or []
        self.assertEqual(len(rooms), 1)
        assert_rooms_match_mongo(self, self.repo, rooms)
        self.assertEqual(rooms[0]["rent_price"], expected_price)

        answer = result.get("answer") or ""
        self.assertIn(str(expected_price // 1_000_000), answer.replace(".", ""))
        assert_answer_invites_customer(self, answer)

    def test_room_pet_question_stays_personal_not_generic_faq(self) -> None:
        question = f"Chị nuôi mèo á, phòng #{YUHOME_ROOM_ID} cho nuôi được không em?"
        parsed = parse_intent_and_constraint_patch(question)
        self.assertEqual(parsed["intent"], "ASK_ABOUT_ROOM")

        result = _run_session(self.repo, [question], session_id="pet-room-sales")[-1]
        self.assertEqual(result["intent"], "ASK_ABOUT_ROOM")
        rooms = result.get("rooms") or []
        self.assertEqual([room["room_id"] for room in rooms], [YUHOME_ROOM_ID])
        assert_rooms_match_mongo(self, self.repo, rooms)

    def test_hard_pet_denied_current_room_suggests_mongo_backed_alternative(self) -> None:
        denied, alternatives = self._find_pet_denied_room_with_alternatives()
        if not denied or not alternatives:
            self.skipTest("No Mongo pet-denied room with same-district pet-friendly alternative found")

        question = (
            f"Em có nuôi mèo, phòng #{denied['room_id']} được không? "
            "Nếu không thì gợi ý căn cùng khu tầm giá gần đó giúp em."
        )
        result = _run_session(self.repo, [question], session_id="hard-pet-denied-mongo", use_qdrant=False)[-1]
        answer = result.get("answer") or ""

        self.assertEqual(result["intent"], "ASK_ABOUT_ROOM")
        self.assertEqual([room["room_id"] for room in (result.get("rooms") or [])], [denied["room_id"]])
        self.assertIn("Thú cưng", answer)
        self.assertIn("Không", answer)
        self.assertNotIn("chưa có dữ liệu xác minh đủ", answer.lower())
        alt_ids = [room["room_id"] for room in alternatives[:3]]
        self.assertTrue(any(room_id in answer for room_id in alt_ids), msg=answer[:300])
        for alt in alternatives[:3]:
            if alt["room_id"] in answer:
                expected_price = f"{int(alt['rent_price']):,}".replace(",", ".")
                self.assertIn(expected_price, answer)

    def _find_pet_denied_room_with_alternatives(self) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        available_rooms = self.repo.search_by_constraints({}, limit=1000)
        for room in available_rooms:
            text = (room.get("embedding_text") or "").lower()
            rent = room.get("rent_price")
            district = room.get("district")
            if "thú cưng: không" not in text or not isinstance(rent, int) or not district:
                continue
            constraints = {
                "location": {"districts": [district]},
                "budget": {"max": rent + 1_000_000, "max_operator": "lte"},
                "pets_required": ["cat"],
            }
            alternatives = [
                item for item in self.repo.search_by_constraints(constraints, limit=4)
                if item.get("room_id") != room.get("room_id")
            ]
            if alternatives:
                return room, alternatives
        return None, []

    def test_warm_session_recovers_after_off_topic_detour(self) -> None:
        questions = [
            "Chị mới chuyển vào Sài Gòn, em tìm giúp chị phòng quận 7 dưới 4 triệu nha",
            "Khoan, em viết code Python giúp chị trước đã",
            f"Thôi thôi, chị thích căn #{YUHOME_ROOM_ID} — em kể thêm chi tiết đi",
        ]
        results = _run_session(self.repo, questions, session_id="mixed-session-sales")

        search_rooms = results[0].get("rooms") or []
        if search_rooms:
            assert_rooms_match_mongo(self, self.repo, search_rooms, max_budget=4_000_000)
            assert_answer_invites_customer(self, results[0].get("answer") or "")

        self.assertEqual(results[1]["intent"], "GENERAL_HELP")
        self.assertFalse(results[1].get("rooms"))

        self.assertEqual(results[2]["intent"], "ASK_ABOUT_ROOM")
        detail_room = (results[2].get("rooms") or [None])[0]
        self.assertIsNotNone(detail_room)
        self.assertEqual(detail_room["room_id"], YUHOME_ROOM_ID)
        assert_rooms_match_mongo(self, self.repo, [detail_room])
        assert_answer_invites_customer(self, results[2].get("answer") or "")

    def test_compare_top_picks_uses_mongo_backed_rows(self) -> None:
        questions = [
            "Em lọc giúp chị mấy căn quận 7 dưới 4 triệu, ưu tiên sạch đẹp nha",
            "So sánh giúp chị 2 căn đầu tiên để chị chọn cho nhanh",
        ]
        results = _run_session(self.repo, questions, session_id="compare-two-sales")
        search_ids = [room["room_id"] for room in (results[0].get("rooms") or [])[:2]]
        self.assertGreaterEqual(len(search_ids), 2)

        compare = results[1]
        self.assertEqual(compare["intent"], "COMPARE_ROOMS")
        rows = (compare.get("comparison") or {}).get("rows") or []
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual([row["room_id"] for row in rows[:2]], search_ids)

        for row in rows[:2]:
            self.assertEqual(row.get("rent_price"), mongo_room_price(self.repo, row["room_id"]))
        assert_answer_invites_customer(self, compare.get("answer") or "")

    def test_mongo_fixture_rooms_match_search_constraints(self) -> None:
        constraints = {
            "location": {"districts": ["quan 7"]},
            "budget": {"max": 4_000_000, "max_operator": "lt", "type": "rent_only"},
        }
        rooms = self.repo.search_by_constraints(constraints, limit=20)
        self.assertTrue(rooms)
        for room in rooms:
            self.assertTrue(room_matches_constraints(room, constraints), room.get("room_id"))
            assert_rooms_match_mongo(self, self.repo, [room], max_budget=4_000_000, district_contains="7")


@unittest.skipUnless(integration_stack_ready(), "MongoDB + Qdrant stack not ready")
class MongoQdrantIntegrationScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()
        cls.repo = create_mongo_repository()
        if not isinstance(cls.repo, MongoRoomRepository):
            raise unittest.SkipTest("Mongo room repository unavailable")

    def test_full_stack_search_near_tdtu_with_qdrant_ranking(self) -> None:
        questions = [
            "Em ơi, chị là sinh viên TDTU, tìm giúp chị căn quận 7 dưới 4 triệu, gần trường cho tiện đi học nha",
        ]
        results = _run_session(self.repo, questions, session_id="qdrant-tdtu", use_qdrant=True)
        result = results[0]
        rooms = result.get("rooms") or []
        self.assertTrue(rooms, msg=result.get("answer"))
        assert_rooms_match_mongo(self, self.repo, rooms, max_budget=4_200_000, district_contains="7")
        assert_answer_invites_customer(self, result.get("answer") or "", min_markers=2)

        room_ids = {room["room_id"] for room in rooms}
        if YUHOME_ROOM_ID in room_ids:
            self.assertEqual(mongo_room_price(self.repo, YUHOME_ROOM_ID), 2_300_000)


if __name__ == "__main__":
    unittest.main()
