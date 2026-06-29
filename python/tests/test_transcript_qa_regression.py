"""Regression tests for production transcript Q&A scenarios (Jun 2026)."""

from __future__ import annotations

import asyncio
import unittest

from room_assistant.intent import parse_intent_and_constraint_patch
from room_assistant.repository import InMemoryRoomRepository
from room_assistant.schemas import default_session_state
from room_assistant.session_store import InMemorySessionStore, apply_operations
from room_assistant.workflow import run_room_assistant


def _q7_rooms() -> list[dict]:
    def room(room_id: str, price: int, ward: str = "Phú Thuận", street: str = "", extras: str = "") -> dict:
        return {
            "room_id": room_id,
            "metadata": {
                "house_name": room_id,
                "room_code": room_id,
                "price": price,
                "status_code": "0",
                "district_name": "Quận 7",
                "ward_name": ward,
            },
            "embedding_text": f"## Thông tin nhà\n- Địa chỉ: {street or ward}, Quận 7\n{extras}",
            "available": True,
            "status": "active",
            "title": room_id,
        }

    return [
        room("61ea636e3048d576be90729b", 4_800_000, street="C2 Hoàng Quốc Việt, Phú Thuận"),
        room("62963aae137e2a3d7e03c9d0", 2_300_000, ward="Phú Mỹ", extras="Gan TDTU, di hoc tien"),
        room("6399d72f07985f204285aed9", 3_400_000, ward="Bình Thuận"),
        room("nguyen-huu-tho-q7", 4_600_000, ward="Tân Hưng", street="Nguyễn Hữu Thọ, Tân Hưng"),
    ]


NO_RESULT_MARKERS = (
    "tìm mỏi mắt",
    "chưa thấy phòng nào khớp 100%",
    "không tìm thấy phòng phù hợp",
)


class TranscriptQaRegressionTests(unittest.TestCase):
    def _run_session(self, questions: list[str], session_id: str = "transcript-qa") -> list[dict]:
        repo = InMemoryRoomRepository(_q7_rooms())
        store = InMemorySessionStore()
        history: list[dict] = []
        results: list[dict] = []
        for question in questions:
            result = asyncio.run(run_room_assistant(
                question,
                history=history,
                session_id=session_id,
                repository=repo,
                session_store=store,
                semantic_index=None,
            ))
            results.append(result)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.get("answer") or ""})
        return results

    def test_gan_do_does_not_create_garbage_landmark(self):
        parsed = parse_intent_and_constraint_patch("Có gợi ý phòng nào gần đó k?")
        landmark_ops = [
            op for op in parsed["operations"]
            if op.get("path") == "location.near_landmarks"
        ]
        values = [op.get("value") for op in landmark_ops if op.get("op") == "append"]
        self.assertFalse(any(str(v).lower() in {"do", "đó"} for v in values))

    def test_repeated_quan_7_search_returns_rooms_after_stale_session(self):
        session_id = "transcript-stale"
        store = InMemorySessionStore()
        state = default_session_state(session_id)
        state["last_intent"] = "SEARCH_ROOM"
        state, _ = apply_operations(state, [
            {"op": "append", "path": "location.districts", "value": "quan 7"},
            {"op": "append", "path": "location.wards", "value": "tan hung"},
            {"op": "append", "path": "location.near_landmarks", "value": "nguyen huu tho"},
        ])
        store.save(session_id, state, ttl_seconds=3600)

        result = asyncio.run(run_room_assistant(
            "Tìm phòng ở quận 7",
            session_id=session_id,
            repository=InMemoryRoomRepository(_q7_rooms()),
            session_store=store,
            semantic_index=None,
        ))
        self.assertTrue(result["rooms"], msg=result.get("answer"))
        location = result["session_state"]["constraints"]["location"]
        self.assertEqual(location["wards"], [])
        self.assertEqual(location["near_landmarks"], [])

    def test_full_transcript_critical_turns(self):
        questions = [
            "Chị mới về Sài Gòn, em tìm giúp chị phòng quận 7 gần đường Nguyễn Hữu Thọ nha",
            "Trong list vừa rồi có căn nào gần đó không em?",
            "Ủa vậy chị muốn xem thêm phòng ở phường Tân Hưng",
            "Em lọc lại giúp chị phòng quận 7 nha",
            "Ngân sách chị khoảng 4 triệu thôi",
            "Có căn nào quận 7 dưới 5 triệu mà sạch sẽ không?",
            "Hơi căng quá, chị muốn dưới 4 triệu thôi",
            "Căn nào gần TDTU cho tiện đi học vậy em?",
        ]
        results = self._run_session(questions, session_id="transcript-full")

        for idx in (0, 3, 5, 6, 7):
            answer = (results[idx].get("answer") or "").lower()
            rooms = results[idx].get("rooms") or []
            self.assertTrue(
                rooms,
                msg=f"Lượt {idx + 1} '{questions[idx]}' không trả phòng: {answer[:120]}",
            )
            self.assertFalse(
                any(marker in answer for marker in NO_RESULT_MARKERS),
                msg=f"Lượt {idx + 1} vẫn trả template 0 kết quả",
            )

        turn4_location = results[3]["session_state"]["constraints"]["location"]
        self.assertEqual(turn4_location["near_landmarks"], [])
        self.assertEqual(turn4_location["wards"], [])

        turn8_prices = [room.get("rent_price") for room in results[6]["rooms"] if room.get("rent_price")]
        if turn8_prices:
            self.assertTrue(all(price <= 4_200_000 for price in turn8_prices))

        turn9_ids = [room.get("room_id") for room in results[7]["rooms"]]
        self.assertIn("62963aae137e2a3d7e03c9d0", turn9_ids)

    def test_landmark_tdt_full_name_finds_tdtu_room(self):
        result = asyncio.run(run_room_assistant(
            "mình muốn tìm phòng ở gần trường đại học TDT",
            session_id="landmark-tdt-full",
            repository=InMemoryRoomRepository(_q7_rooms()),
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        rooms = result.get("rooms") or []
        answer = (result.get("answer") or "").lower()
        self.assertTrue(rooms, msg=answer[:200])
        self.assertFalse(any(room.get("relaxed_search") for room in rooms))
        self.assertFalse(any(marker in answer for marker in NO_RESULT_MARKERS))
        room_ids = [room.get("room_id") for room in rooms]
        self.assertIn("62963aae137e2a3d7e03c9d0", room_ids)
        landmarks = result["session_state"]["constraints"]["location"].get("near_landmarks") or []
        self.assertIn("tdtu", landmarks)

    def test_follow_up_deictic_after_search(self):
        questions = [
            "Em lọc giúp chị vài căn quận 7 dưới 5 triệu nha",
            "Căn ở trên có ban công không em?",
        ]
        results = self._run_session(questions, session_id="deictic-follow-up")
        self.assertTrue(results[0].get("rooms"))
        second = results[1]
        self.assertEqual(second.get("intent"), "ASK_ABOUT_ROOM")
        first_room_id = results[0]["rooms"][0].get("room_id")
        self.assertEqual(second["session_state"].get("current_room_id"), first_room_id)
        self.assertTrue(second.get("retrieval_explanation"))

    def test_off_topic_and_empathy_short_paths(self):
        off_topic = asyncio.run(run_room_assistant(
            "Viết code Python giúp tôi",
            session_id="off-topic",
            repository=InMemoryRoomRepository([]),
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        self.assertIn("phòng trọ", off_topic["answer"].lower())

        empathy = asyncio.run(run_room_assistant(
            "Giá cao quá em ơi, sinh viên như chị sao thuê nổi",
            session_id="empathy",
            repository=InMemoryRoomRepository(_q7_rooms()),
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        answer = empathy["answer"].lower()
        self.assertTrue("hiểu" in answer or "ngân sách" in answer)

    def test_budget_cap_not_relaxed_to_expensive_rooms(self):
        repo = InMemoryRoomRepository([
            {
                "room_id": "q1-expensive",
                "metadata": {
                    "house_name": "Q1-VIP",
                    "price": 8_800_000,
                    "status_code": "0",
                    "district_name": "Quận 1",
                },
                "embedding_text": "Quận 1 trung tâm",
                "available": True,
                "status": "active",
            },
            {
                "room_id": "q1-ok",
                "metadata": {
                    "house_name": "Q1-Budget",
                    "price": 5_500_000,
                    "status_code": "0",
                    "district_name": "Quận 1",
                },
                "embedding_text": "Quận 1 giá mềm",
                "available": True,
                "status": "active",
            },
        ])
        result = asyncio.run(run_room_assistant(
            "Em tìm giúp chị phòng quận 1 dưới 6 triệu, ưu tiên giá mềm nha",
            session_id="budget-cap",
            repository=repo,
            session_store=InMemorySessionStore(),
            semantic_index=None,
        ))
        prices = [room.get("rent_price") for room in (result.get("rooms") or []) if room.get("rent_price")]
        if prices:
            self.assertTrue(all(price <= 6_000_000 for price in prices))
        else:
            answer = (result.get("answer") or "").lower()
            self.assertTrue("6" in answer or "ngân sách" in answer or "triệu" in answer)

    def test_compare_keeps_list_after_single_room_refine(self):
        repo = InMemoryRoomRepository(_q7_rooms())
        store = InMemorySessionStore()
        session_id = "compare-preserve"
        first = asyncio.run(run_room_assistant(
            "Em lọc giúp chị vài căn quận 7 dưới 5 triệu nha",
            session_id=session_id,
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))
        self.assertGreaterEqual(len(first.get("rooms") or []), 2)

        asyncio.run(run_room_assistant(
            "Chị thu hẹp còn dưới 3 triệu thôi em",
            session_id=session_id,
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))

        compare = asyncio.run(run_room_assistant(
            "So sánh giúp chị 2 căn đầu tiên để chị chọn nhanh",
            session_id=session_id,
            repository=repo,
            session_store=store,
            semantic_index=None,
        ))
        rows = (compare.get("comparison") or {}).get("rows") or []
        self.assertGreaterEqual(len(rows), 2, msg=compare.get("answer"))

    def test_pet_and_deposit_route_to_staff_faq(self):
        for question, marker in (
            ("Nuôi mèo được không?", "chủ nhà"),
            ("Tiền cọc bao nhiêu?", "1 tháng"),
        ):
            parsed = parse_intent_and_constraint_patch(question)
            self.assertEqual(parsed["intent"], "REQUEST_FAQ", msg=question)
            result = asyncio.run(run_room_assistant(
                question,
                session_id=f"faq-{marker}",
                repository=InMemoryRoomRepository(_q7_rooms()),
                session_store=InMemorySessionStore(),
                semantic_index=None,
            ))
            answer = (result.get("answer") or "").lower()
            self.assertIn(marker, answer, msg=f"{question} -> {answer[:160]}")


if __name__ == "__main__":
    unittest.main()
