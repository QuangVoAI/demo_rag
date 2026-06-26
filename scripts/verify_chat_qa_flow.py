"""One-off multi-turn Q&A verification for chat + Mongo sync (in-memory Mongo)."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

from django.test import Client  # noqa: E402

from apps.rooms.tests import ChatApiTests  # noqa: E402


class QaScenarioRunner(ChatApiTests):
    """Reuse fake Mongo helpers from ChatApiTests."""

    def setUp(self):
        super().setUp()
        self.client = Client()

    async def _qa_stream(self, question, history=None, session_id="", stream_callback=None):
        answers = {
            "Tìm phòng quận 7 dưới 5 triệu": "Dạ em tìm thấy vài phòng ở Quận 7 phù hợp ngân sách ạ.",
            "Tìm phòng ở quận 7": "Dạ vẫn còn phòng Quận 7 phù hợp, em gửi anh/chị danh sách bên dưới ạ.",
            "Căn nào gần TDTU": "Dạ em gợi ý các phòng gần khu vực TDTU ở Quận 7 ạ.",
            "Giá cao thế, sinh viên sao thuê nổi": "Dạ em hiểu lo ngại về giá, mình thử lọc ngân sách thấp hơn nhé.",
            "Viết code Python giúp tôi": "Em chỉ hỗ trợ tư vấn phòng trọ, không viết code ạ.",
            "Khách cần studio quận 7": "Dạ em đã ghi nhận nhu cầu studio Quận 7.",
            "Mở đoạn chat đầu tiên": "Dạ em sẵn sàng hỗ trợ đoạn chat mới ạ.",
        }
        if stream_callback is not None:
            await stream_callback("[status:planning] Đang xử lý.\n")
        reply = answers.get(question, f"Đã nhận: {question}")
        return {
            "answer": reply,
            "rooms": [{"room_id": "Q7-1", "title": "Studio Q7", "rent_price": 4_800_000, "district": "Quận 7"}]
            if "quận 7" in question.lower() or "tdtu" in question.lower()
            else [],
            "suggested_questions": ["Rẻ hơn", "Gần trung tâm hơn"],
            "session_state": {"constraints": {}},
            "sources": [],
        }

    def _post_chat(self, body: dict) -> dict:
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            with patch("apps.rooms.views.run_streaming", new=self._qa_stream):
                response = self.client.post(
                    "/api/chat/",
                    data=json.dumps(body),
                    content_type="application/json",
                )
        self.assertEqual(response.status_code, 200)
        chunks = []
        for chunk in response.streaming_content:
            chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        joined = "".join(chunks)
        final_line = [line for line in joined.splitlines() if line.startswith("data: ") and '"type": "final"' in line][-1]
        return json.loads(final_line[len("data: "):])["payload"]

    def _get_chat(self, params: dict) -> dict:
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            response = self.client.get("/api/chat/", params)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        return payload

    def run_scenarios(self) -> list[str]:
        logs: list[str] = []

        # Scenario A — customer fixed thread + multi-turn Q&A
        init = self._get_chat({
            "contact_phone": "0900111222",
            "contact_name": "Khách QA",
            "demo_role": "customer",
        })
        conv_a = init["conversation_id"]
        logs.append(f"A0 init: contact={init['contact_id']} conv={conv_a} msgs={len(init['messages'])}")

        qa_pairs = [
            ("Tìm phòng quận 7 dưới 5 triệu", "Quận 7"),
            ("Tìm phòng ở quận 7", "Quận 7"),
            ("Căn nào gần TDTU", "TDTU"),
            ("Giá cao thế, sinh viên sao thuê nổi", "hiểu"),
            ("Viết code Python giúp tôi", "phòng trọ"),
        ]
        for idx, (question, expect_snippet) in enumerate(qa_pairs, start=1):
            payload = self._post_chat({
                "message": question,
                "contact_phone": "0900111222",
                "contact_name": "Khách QA",
                "demo_role": "customer",
                "conversation_id": conv_a,
            })
            self.assertEqual(payload["conversation_id"], conv_a)
            self.assertIn(expect_snippet.lower(), payload["reply"].lower())
            logs.append(f"A{idx} Q: {question[:40]}… → OK ({expect_snippet})")

        reload_a = self._get_chat({
            "contact_phone": "0900111222",
            "contact_name": "Khách QA",
            "demo_role": "customer",
        })
        self.assertEqual(reload_a["conversation_id"], conv_a)
        self.assertGreaterEqual(len(reload_a["messages"]), len(qa_pairs) * 2)
        logs.append(f"A reload: {len(reload_a['messages'])} messages persisted on single thread")

        # Scenario B — staff multi-thread
        staff_init = self._get_chat({
            "contact_phone": "0900333444",
            "contact_name": "NV QA",
            "demo_role": "landlord",
            "new_chat": "1",
        })
        conv_b1 = staff_init["conversation_id"]
        p1 = self._post_chat({
            "message": "Mở đoạn chat đầu tiên",
            "contact_phone": "0900333444",
            "contact_name": "NV QA",
            "demo_role": "landlord",
            "conversation_id": conv_b1,
            "new_chat": "1",
        })
        self.assertEqual(p1["conversation_id"], conv_b1)

        time.sleep(1.1)
        staff_new = self._get_chat({
            "contact_phone": "0900333444",
            "contact_name": "NV QA",
            "demo_role": "landlord",
            "new_chat": "1",
        })
        conv_b2 = staff_new["conversation_id"]
        self.assertNotEqual(conv_b1, conv_b2)
        p2 = self._post_chat({
            "message": "Khách cần studio quận 7",
            "contact_phone": "0900333444",
            "contact_name": "NV QA",
            "demo_role": "landlord",
            "conversation_id": conv_b2,
            "new_chat": "1",
        })
        threads = self._get_chat({
            "contact_phone": "0900333444",
            "contact_name": "NV QA",
            "demo_role": "landlord",
        })["threads"]
        self.assertGreaterEqual(len(threads), 2)
        logs.append(f"B staff: 2 threads ({conv_b1}, {conv_b2}), sidebar={len(threads)} items")
        logs.append(f"B2 reply OK: {'studio' in p2['reply'].lower() or 'ghi nhận' in p2['reply'].lower()}")

        return logs


if __name__ == "__main__":
    runner = QaScenarioRunner()
    runner.setUp()
    try:
        results = runner.run_scenarios()
    except Exception as exc:
        print("FAILED:", exc)
        raise
    else:
        print("=== Chat Q&A multi-turn verification ===")
        for line in results:
            print(line)
        print("ALL SCENARIOS PASSED")
