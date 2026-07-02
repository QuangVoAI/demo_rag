from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from unittest.mock import patch

from bson import ObjectId
from django.test import Client, TestCase, override_settings

from apps.rooms import views


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, spec, direction=None):
        if isinstance(spec, list):
            fields = spec
        else:
            fields = [(spec, direction)]

        for field, sort_dir in reversed(fields):
            reverse = sort_dir == -1
            self._docs.sort(key=lambda doc: doc.get(field) or datetime.min, reverse=reverse)
        return self

    def limit(self, count):
        self._docs = self._docs[:count]
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    def __init__(self, docs=None):
        self.docs = [deepcopy(doc) for doc in (docs or [])]
        self.indexes = []

    def _get_nested(self, doc, key):
        current = doc
        for part in str(key).split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    def _match_condition(self, actual, expected):
        if isinstance(expected, dict):
            if "$regex" in expected:
                import re

                pattern = expected["$regex"]
                flags = re.IGNORECASE if "i" in str(expected.get("$options", "")) else 0
                return isinstance(actual, str) and re.search(pattern, actual, flags) is not None
            if "$gte" in expected and not (actual is not None and actual >= expected["$gte"]):
                return False
            if "$lte" in expected and not (actual is not None and actual <= expected["$lte"]):
                return False
            return True
        return actual == expected

    def _matches(self, doc, query):
        query = query or {}
        for key, value in query.items():
            if key == "$and":
                if not all(self._matches(doc, clause) for clause in value):
                    return False
                continue
            if key == "$or":
                if not any(self._matches(doc, clause) for clause in value):
                    return False
                continue
            if not self._match_condition(self._get_nested(doc, key), value):
                return False
        return True

    def find_one(self, query=None, sort=None):
        docs = [doc for doc in self.docs if self._matches(doc, query or {})]
        if sort:
            docs = list(FakeCursor(docs).sort(sort))
        return deepcopy(docs[0]) if docs else None

    def find(self, query=None):
        return FakeCursor([deepcopy(doc) for doc in self.docs if self._matches(doc, query or {})])

    def distinct(self, key):
        values = []
        seen = set()
        for doc in self.docs:
            value = self._get_nested(doc, key)
            marker = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            if marker in seen or value is None:
                continue
            seen.add(marker)
            values.append(deepcopy(value))
        return values

    def update_one(self, query, update, upsert=False):
        target = None
        for doc in self.docs:
            if self._matches(doc, query):
                target = doc
                break

        if target is None:
            if not upsert:
                return
            target = deepcopy(query)
            self.docs.append(target)
            for key, value in update.get("$setOnInsert", {}).items():
                target[key] = deepcopy(value)

        for key, value in update.get("$set", {}).items():
            target[key] = deepcopy(value)

        for key, value in update.get("$push", {}).items():
            target.setdefault(key, [])
            target[key].append(deepcopy(value))

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, deepcopy(kwargs)))
        return kwargs.get("name", "mock_index")

    def list_indexes(self):
        indexes = [{"key": {"_id": 1}, "name": "_id_"}]
        for keys, kwargs in self.indexes:
            if isinstance(keys, list):
                key_doc = {key: direction for key, direction in keys}
            elif isinstance(keys, str):
                key_doc = {keys: 1}
            else:
                key_doc = deepcopy(keys)
            index_doc = {"key": key_doc, "name": kwargs.get("name", "mock_index")}
            index_doc.update(kwargs)
            indexes.append(index_doc)
        return indexes


class RecordingCollection(FakeCollection):
    def __init__(self, docs=None):
        super().__init__(docs=docs)
        self.last_find_query = None

    def find(self, query=None):
        self.last_find_query = deepcopy(query or {})
        return super().find(query)


class NestedCollection(FakeCollection):
    pass


@override_settings(ALLOWED_HOSTS=["127.0.0.1", "testserver", "localhost"])
class ChatApiTests(TestCase):
    def setUp(self):
        if hasattr(views._ensure_contacts_phone_unique_index, "_ready"):
            delattr(views._ensure_contacts_phone_unique_index, "_ready")
        self.client = Client(HTTP_HOST="127.0.0.1")
        self.client.defaults["HTTP_HOST"] = "127.0.0.1"
        now = datetime(2026, 6, 25, 12, 0, 0)
        self.collections = {
            "contacts": FakeCollection([
                {"contact_id": "c_usr_999", "name": "John Customer", "phone_number": "0987654322", "role": "user"},
                {"contact_id": "c_staff_001", "name": "Jane Staff", "phone_number": "0987654321", "role": "staff"},
            ]),
            "chat_history": FakeCollection([
                {
                    "contact_id": "c_usr_999",
                    "conversation_id": "conv_c_usr_999",
                    "title": "Phòng quận 10",
                    "sender_role": "user",
                    "created_at": now - timedelta(minutes=10),
                    "updated_at": now - timedelta(minutes=9),
                    "messages": [
                        {"role": "user", "content": "Phòng Quận 10 dưới 6 triệu", "created_at": now - timedelta(minutes=10)},
                        {"role": "assistant", "content": "Có 2 phòng phù hợp.", "created_at": now - timedelta(minutes=9)},
                    ],
                },
                {
                    "contact_id": "c_staff_001",
                    "conversation_id": "conv_staff_001",
                    "title": "Khách studio quận 7",
                    "sender_role": "nhan_vien",
                    "created_at": now - timedelta(minutes=30),
                    "updated_at": now - timedelta(minutes=20),
                    "messages": [
                        {"role": "staff", "content": "Khách hỏi studio quận 7", "created_at": now - timedelta(minutes=30)},
                    ],
                },
                {
                    "contact_id": "c_staff_001",
                    "conversation_id": "conv_staff_002",
                    "title": "Khách cần ban công",
                    "sender_role": "nhan_vien",
                    "created_at": now - timedelta(minutes=8),
                    "updated_at": now - timedelta(minutes=5),
                    "messages": [
                        {"role": "staff", "content": "Khách cần ban công", "created_at": now - timedelta(minutes=8)},
                        {"role": "assistant", "content": "Đã ghi nhận.", "created_at": now - timedelta(minutes=5)},
                    ],
                },
            ]),
            "rooms": FakeCollection([]),
        }

    def _get_collection(self, name):
        return self.collections.setdefault(name, FakeCollection([]))

    async def _stream_response(self, question, history=None, session_id="", stream_callback=None):
        if stream_callback is not None:
            await stream_callback("[status:planning] Mình đang xác định nhu cầu.\n")
            await stream_callback("Đây là câu trả lời nháp.")
        return {
            "answer": "Đây là câu trả lời cuối.",
            "rooms": [],
            "suggested_questions": ["Còn phòng nào rẻ hơn không?"],
            "session_state": {"constraints": {}},
        }

    def test_get_initializes_user_from_phone_and_loads_history(self):
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            response = self.client.get("/api/chat/", {
                "contact_phone": "0987654322",
                "contact_name": "John Customer",
                "demo_role": "customer",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["contact_id"], "c_usr_999")
        self.assertEqual(payload["conversation_id"], "conv_c_usr_999")
        self.assertEqual(payload["role"], "user")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["threads"][0]["conversation_id"], "conv_c_usr_999")

    def test_get_returns_employee_threads_and_requested_conversation(self):
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            response = self.client.get("/api/chat/", {
                "contact_phone": "0987654321",
                "contact_name": "Jane Staff",
                "demo_role": "landlord",
                "conversation_id": "conv_staff_001",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["contact_id"], "c_staff_001")
        self.assertEqual(payload["role"], "staff")
        self.assertEqual(payload["conversation_id"], "conv_staff_001")
        self.assertEqual(len(payload["threads"]), 2)
        self.assertTrue(any(thread["active"] for thread in payload["threads"]))
        self.assertEqual(payload["messages"][0]["content"], "Khách hỏi studio quận 7")

    def test_get_ignores_staff_session_when_new_user_identity_is_provided(self):
        session = self.client.session
        session["contact_id"] = "c_staff_001"
        session["conversation_id"] = "conv_staff_002"
        session.save()

        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            response = self.client.get("/api/chat/", {
                "contact_phone": "0987654322",
                "contact_name": "John Customer",
                "demo_role": "customer",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["contact_id"], "c_usr_999")
        self.assertEqual(payload["role"], "user")
        self.assertEqual(payload["conversation_id"], "conv_c_usr_999")

    def test_post_streams_final_payload_and_persists_new_staff_conversation(self):
        body = {
            "message": "Tìm phòng Bình Thạnh dưới 5 triệu",
            "contact_phone": "0987654321",
            "contact_name": "Jane Staff",
            "demo_role": "landlord",
            "new_chat": "1",
        }

        with (
            patch("apps.rooms.views.get_collection", side_effect=self._get_collection),
            patch("apps.rooms.views.run_streaming", new=self._stream_response),
        ):
            response = self.client.post(
                "/api/chat/",
                data=json.dumps(body),
                content_type="application/json",
            )
            chunks = []
            for chunk in response.streaming_content:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        self.assertEqual(response.status_code, 200)
        joined = "".join(chunks)
        self.assertIn('"type": "status"', joined)
        self.assertIn('"type": "token"', joined)
        self.assertIn('"type": "final"', joined)

        final_line = [line for line in joined.splitlines() if line.startswith("data: ") and '"type": "final"' in line][-1]
        payload = json.loads(final_line[len("data: "):])["payload"]
        self.assertEqual(payload["role"], "staff")
        self.assertTrue(payload["conversation_id"].startswith("conv_c_staff_001_"))
        self.assertEqual(payload["reply"], "Đây là câu trả lời cuối.")
        self.assertEqual(payload["follow_ups"], ["Còn phòng nào rẻ hơn không?"])

        chat_doc = self._get_collection("chat_history").find_one({"conversation_id": payload["conversation_id"]})
        self.assertIsNotNone(chat_doc)
        self.assertEqual(chat_doc["contact_id"], "c_staff_001")
        self.assertEqual(chat_doc["messages"][0]["role"], "staff")
        self.assertEqual(chat_doc["messages"][-1]["role"], "assistant")

    def test_post_new_chat_ignores_existing_staff_session_conversation(self):
        session = self.client.session
        session["contact_id"] = "c_staff_001"
        session["conversation_id"] = "conv_staff_002"
        session.save()

        body = {
            "message": "Tạo đoạn chat staff mới",
            "contact_phone": "0987654321",
            "contact_name": "Jane Staff",
            "demo_role": "landlord",
            "new_chat": "1",
        }

        with (
            patch("apps.rooms.views.get_collection", side_effect=self._get_collection),
            patch("apps.rooms.views.run_streaming", new=self._stream_response),
        ):
            response = self.client.post(
                "/api/chat/",
                data=json.dumps(body),
                content_type="application/json",
            )
            chunks = []
            for chunk in response.streaming_content:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        joined = "".join(chunks)
        final_line = [line for line in joined.splitlines() if line.startswith("data: ") and '"type": "final"' in line][-1]
        payload = json.loads(final_line[len("data: "):])["payload"]
        self.assertNotEqual(payload["conversation_id"], "conv_staff_002")
        self.assertTrue(payload["conversation_id"].startswith("conv_c_staff_001_"))

    def test_post_existing_staff_conversation_persists_staff_and_assistant_messages(self):
        body = {
            "message": "Khách muốn phòng có ban công",
            "contact_phone": "0987654321",
            "contact_name": "Jane Staff",
            "demo_role": "landlord",
            "conversation_id": "conv_staff_002",
        }

        with (
            patch("apps.rooms.views.get_collection", side_effect=self._get_collection),
            patch("apps.rooms.views.run_streaming", new=self._stream_response),
        ):
            response = self.client.post(
                "/api/chat/",
                data=json.dumps(body),
                content_type="application/json",
            )
            chunks = []
            for chunk in response.streaming_content:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        joined = "".join(chunks)
        final_line = [line for line in joined.splitlines() if line.startswith("data: ") and '"type": "final"' in line][-1]
        payload = json.loads(final_line[len("data: "):])["payload"]
        self.assertEqual(payload["conversation_id"], "conv_staff_002")

        chat_doc = self._get_collection("chat_history").find_one({"conversation_id": "conv_staff_002"})
        self.assertEqual(chat_doc["messages"][-2]["role"], "staff")
        self.assertEqual(chat_doc["messages"][-2]["content"], "Khách muốn phòng có ban công")
        self.assertEqual(chat_doc["messages"][-1]["role"], "assistant")
        self.assertEqual(chat_doc["messages"][-1]["content"], "Đây là câu trả lời cuối.")

    def test_post_staff_without_thread_auto_creates_conversation(self):
        body = {
            "message": "Mở đoạn chat đầu tiên",
            "contact_phone": "1234",
            "contact_name": "test2",
            "demo_role": "landlord",
        }

        with (
            patch("apps.rooms.views.get_collection", side_effect=self._get_collection),
            patch("apps.rooms.views.run_streaming", new=self._stream_response),
        ):
            response = self.client.post(
                "/api/chat/",
                data=json.dumps(body),
                content_type="application/json",
            )
            chunks = []
            for chunk in response.streaming_content:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        joined = "".join(chunks)
        final_line = [line for line in joined.splitlines() if line.startswith("data: ") and '"type": "final"' in line][-1]
        payload = json.loads(final_line[len("data: "):])["payload"]
        self.assertTrue(payload["conversation_id"].startswith("conv_c_staff_1234_"))

        chat_doc = self._get_collection("chat_history").find_one({"conversation_id": payload["conversation_id"]})
        self.assertIsNotNone(chat_doc)
        self.assertEqual(chat_doc["contact_id"], "c_staff_1234")
        self.assertEqual(chat_doc["messages"][0]["role"], "staff")
        self.assertEqual(chat_doc["messages"][0]["content"], "Mở đoạn chat đầu tiên")
        self.assertEqual(chat_doc["messages"][-1]["role"], "assistant")

    def test_phone_number_is_global_unique_key_across_roles(self):
        contacts = self._get_collection("contacts")
        contacts.docs.append({
            "contact_id": "c_usr_shared",
            "name": "Shared User",
            "phone_number": "0909000111",
            "role": "user",
        })

        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            response = self.client.get("/api/chat/", {
                "contact_phone": "0909000111",
                "contact_name": "Shared Staff Attempt",
                "demo_role": "landlord",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["contact_id"], "c_usr_shared")
        self.assertEqual(payload["role"], "user")

        same_phone_docs = [doc for doc in contacts.docs if doc.get("phone_number") == "0909000111"]
        self.assertEqual(len(same_phone_docs), 1)

    def test_load_filtered_rooms_escapes_search_query_regex(self):
        rooms = RecordingCollection([])

        with patch("apps.rooms.views.get_collection", return_value=rooms):
            views._load_filtered_rooms(search_query="(a+)+")

        self.assertEqual(
            rooms.last_find_query["$and"][1]["$or"][0]["metadata.house_name"]["$regex"],
            r"\(a\+\)\+",
        )

    def test_load_filtered_rooms_is_not_capped_at_sixty(self):
        room_docs = [
            {
                "room_id": f"R{i}",
                "metadata": {
                    "status_code": "0",
                    "price": i,
                    "house_name": f"House {i}",
                    "room_code": f"P{i}",
                    "district_name": "Quận 7",
                    "province_name": "Thành phố Hồ Chí Minh",
                },
                "embedding_text": "",
            }
            for i in range(61)
        ]
        rooms = NestedCollection(room_docs)

        with patch("apps.rooms.views.get_collection", return_value=rooms):
            payload = views._load_filtered_rooms()

        self.assertEqual(len(payload), 61)

    def test_load_filtered_rooms_accepts_blank_status_code_as_available(self):
        rooms = NestedCollection([
            {
                "room_id": "Q5-202",
                "metadata": {
                    "status_code": "",
                    "price": 4_300_000,
                    "house_name": "1362 VÕ VĂN KIỆT",
                    "room_code": "202",
                    "district_name": "Quận 5",
                    "province_name": "Thành phố Hồ Chí Minh",
                },
                "embedding_text": "",
            }
        ])

        with patch("apps.rooms.views.get_collection", return_value=rooms):
            payload = views._load_filtered_rooms(district_slug="quan-5", price_range="0-5")

        self.assertEqual([room["room_id"] for room in payload], ["Q5-202"])

    def test_serialize_rag_response_fills_missing_image_from_room_doc(self):
        rooms_collection = FakeCollection([
            {
                "room_id": "A101",
                "house_id": "665f00000000000000000001",
                "metadata": {
                    "status_code": "0",
                    "price": 4_500_000,
                    "house_name": "Studio Bình Thạnh",
                    "room_code": "P101",
                },
                "media": {
                    "cover_image": "https://cdn.example.com/cover-a101.jpg",
                },
                "embedding_text": "",
            }
        ])

        def fake_get_collection(name):
            if name == "rooms":
                return rooms_collection
            return FakeCollection([])

        with patch("apps.rooms.views.get_collection", side_effect=fake_get_collection):
            payload = views._serialize_rag_response({
                "answer": "OK",
                "rooms": [{"room_id": "A101", "title": "Studio Bình Thạnh", "image": ""}],
            })

        self.assertEqual(payload["rooms"][0]["id"], "A101")
        self.assertEqual(payload["rooms"][0]["image"], "https://cdn.example.com/cover-a101.jpg")

    def test_serialize_rag_response_serializes_object_id_fields(self):
        oid_room = ObjectId("665f00000000000000000001")
        oid_source = ObjectId("665f00000000000000000002")
        oid_trace = ObjectId("665f00000000000000000003")
        payload = views._serialize_rag_response({
            "answer": "OK",
            "rooms": [{
                "room_id": "A101",
                "title": "Studio",
                "property_id": oid_room,
            }],
            "sources": [{"type": "room", "room_id": "A101", "property_id": oid_source}],
            "retrieval_attempts": [{"top_room_ids": ["A101"], "trace_id": oid_trace}],
        })
        self.assertEqual(payload["rooms"][0]["property_id"], "665f00000000000000000001")
        self.assertEqual(payload["sources"][0]["property_id"], "665f00000000000000000002")
        self.assertEqual(payload["retrieval_attempts"][0]["trace_id"], "665f00000000000000000003")

    def test_customer_get_after_post_returns_persisted_messages(self):
        body = {
            "message": "Tìm phòng quận 7",
            "contact_phone": "0900111333",
            "contact_name": "Persist QA",
            "demo_role": "customer",
        }

        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            with patch("apps.rooms.views.run_streaming", new=self._stream_response):
                post = self.client.post(
                    "/api/chat/",
                    data=json.dumps(body),
                    content_type="application/json",
                )
                chunks = []
                for chunk in post.streaming_content:
                    chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
                final_line = [
                    line for line in "".join(chunks).splitlines()
                    if line.startswith("data: ") and '"type": "final"' in line
                ][-1]
                post_payload = json.loads(final_line[len("data: "):])["payload"]
                conv_id = post_payload["conversation_id"]

                reload = self.client.get("/api/chat/", {
                    "contact_phone": "0900111333",
                    "contact_name": "Persist QA",
                    "demo_role": "customer",
                })

        reload_payload = reload.json()
        self.assertEqual(reload_payload["conversation_id"], conv_id)
        self.assertGreaterEqual(len(reload_payload["messages"]), 2)
        roles = [msg["role"] for msg in reload_payload["messages"]]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)


@override_settings(ALLOWED_HOSTS=["127.0.0.1", "testserver", "localhost"], RAG_API_KEY="")
class ChatTranscriptApiTests(ChatApiTests):
    """Multi-turn chat API cases mirroring production transcript."""

    async def _transcript_stream(self, question, history=None, session_id="", stream_callback=None):
        q = (question or "").lower()
        if "viết code" in q:
            return {
                "answer": "Em chỉ hỗ trợ tư vấn phòng trọ, không viết code ạ.",
                "rooms": [],
                "suggested_questions": [],
                "session_state": {"constraints": {}},
                "intent": "GENERAL_HELP",
            }
        if "quận 7" in q or "tdtu" in q:
            return {
                "answer": "Dạ còn phòng Quận 7 ạ.",
                "rooms": [{"room_id": "Q7-ROOM", "title": "Studio Q7", "rent_price": 4_500_000, "district": "Quận 7"}],
                "suggested_questions": ["Rẻ hơn"],
                "session_state": {"constraints": {"location": {"districts": ["quan 7"]}}},
                "intent": "SEARCH_ROOM",
            }
        return {
            "answer": "Dạ em tìm mỏi mắt mà chưa thấy phòng nào khớp 100% điều kiện của mình ạ.",
            "rooms": [],
            "suggested_questions": [],
            "session_state": {"constraints": {}},
            "intent": "SEARCH_ROOM",
        }

    def _post_chat(self, body: dict) -> dict:
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            with patch("apps.rooms.views.run_streaming", new=self._transcript_stream):
                response = self.client.post(
                    "/api/chat/",
                    data=json.dumps(body),
                    content_type="application/json",
                )
        joined = "".join(
            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            for chunk in response.streaming_content
        )
        final_line = [
            line for line in joined.splitlines()
            if line.startswith("data: ") and '"type": "final"' in line
        ][-1]
        return json.loads(final_line[len("data: "):])["payload"]

    def test_transcript_multi_turn_persists_on_single_customer_thread(self):
        init = self.client.get("/api/chat/", {
            "contact_phone": "0900222444",
            "contact_name": "Transcript QA",
            "demo_role": "customer",
        }).json()
        conv_id = init["conversation_id"]

        questions = [
            "Tìm phòng ở quận 7",
            "Căn nào gần TDTU á",
            "Viết code Python giúp tôi",
        ]
        for question in questions:
            payload = self._post_chat({
                "message": question,
                "contact_phone": "0900222444",
                "contact_name": "Transcript QA",
                "demo_role": "customer",
                "conversation_id": conv_id,
            })
            self.assertEqual(payload["conversation_id"], conv_id)

        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            reload = self.client.get("/api/chat/", {
                "contact_phone": "0900222444",
                "contact_name": "Transcript QA",
                "demo_role": "customer",
            }).json()

        self.assertEqual(reload["conversation_id"], conv_id)
        self.assertGreaterEqual(len(reload["messages"]), len(questions) * 2)

    def test_staff_can_load_previous_thread_by_conversation_id(self):
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            payload = self.client.get("/api/chat/", {
                "contact_phone": "0987654321",
                "contact_name": "Jane Staff",
                "demo_role": "landlord",
                "conversation_id": "conv_staff_001",
            }).json()

        self.assertEqual(payload["conversation_id"], "conv_staff_001")
        self.assertEqual(payload["messages"][0]["content"], "Khách hỏi studio quận 7")
        self.assertGreaterEqual(len(payload["threads"]), 2)
        active_threads = [t for t in payload["threads"] if t.get("active")]
        self.assertEqual(len(active_threads), 1)
        self.assertEqual(active_threads[0]["conversation_id"], "conv_staff_001")


class RagApiQueryTests(TestCase):
    def setUp(self):
        self.client = Client(HTTP_HOST="127.0.0.1")

    async def _response(self, question, history=None, session_id="", stream_callback=None):
        return {
            "session_id": session_id or "rag-session-1",
            "answer": "Phòng phù hợp là A101.",
            "intent": "SEARCH_ROOM",
            "session_state": {"constraints": {"budget": {"max": 5000000}}},
            "rooms": [{"room_id": "A101", "title": "Studio Bình Thạnh"}],
            "cost_estimate": None,
            "comparison": None,
            "suggested_questions": ["Còn phòng nào rẻ hơn không?"],
            "sources": [{"type": "room", "room_id": "A101"}],
            "retrieval_confidence": 0.82,
            "retrieval_low_confidence": False,
            "retrieval_feedback_retry_count": 0,
            "retrieval_attempts": [{"top_room_ids": ["A101"]}],
            "processing_time_ms": 123,
        }

    async def _streaming_response(self, question, history=None, session_id="", stream_callback=None):
        if stream_callback is not None:
            await stream_callback("[status:planning|intent_router] Dang phan loai intent.\n")
            await stream_callback("[status:retrieving|retriever] Dang truy van du lieu phong.\n")
        return await self._response(question, history=history, session_id=session_id, stream_callback=stream_callback)

    def test_rag_query_returns_json_payload(self):
        body = {
            "message": "Tìm phòng dưới 5 triệu",
            "session_id": "web-123",
            "history": [{"role": "user", "content": "Xin chào"}],
        }

        with patch("apps.rooms.views.run_streaming", new=self._response):
            response = self.client.post(
                "/api/rag/query/",
                data=json.dumps(body),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["session_id"], "web-123")
        self.assertEqual(payload["reply"], "Phòng phù hợp là A101.")
        self.assertEqual(payload["rooms"][0]["room_id"], "A101")
        self.assertEqual(payload["retrieval_confidence"], 0.82)
        self.assertEqual(payload["follow_ups"], ["Còn phòng nào rẻ hơn không?"])
        self.assertTrue(payload["request_id"])
        self.assertEqual(payload["correlation_id"], response["X-Correlation-ID"])
        self.assertTrue(response["X-Request-ID"])

    @override_settings(RAG_API_KEY="secret-key")
    def test_rag_query_requires_api_key_when_configured(self):
        body = {"message": "Tìm phòng"}

        with patch("apps.rooms.views.run_streaming", new=self._response):
            unauthorized = self.client.post(
                "/api/rag/query/",
                data=json.dumps(body),
                content_type="application/json",
            )
            authorized = self.client.post(
                "/api/rag/query/",
                data=json.dumps(body),
                content_type="application/json",
                HTTP_X_API_KEY="secret-key",
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.json()["error"]["code"], "unauthorized")
        self.assertFalse(unauthorized.json()["error"]["retryable"])
        self.assertEqual(authorized.status_code, 200)

    def test_health_endpoint_returns_ok(self):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(response["X-Request-ID"])

    def test_demo_template_uses_chat_api_for_production_ui(self):
        response = self.client.get("/", secure=True)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8", errors="ignore")
        self.assertIn("/api/chat/", content)
        self.assertIn("chat-thread-panel", content)
        self.assertIn("bootstrapChatSession", content)
        self.assertNotIn("/api/rag/stream/", content)

    @override_settings(RAG_RATE_LIMIT_MAX_REQUESTS=1, RAG_RATE_LIMIT_WINDOW_SECONDS=60)
    def test_rag_query_rate_limit_returns_cooldown_payload(self):
        body = {"message": "TÃ¬m phÃ²ng", "session_id": "same-session"}

        with patch("apps.rooms.views.run_streaming", new=self._response):
            first = self.client.post(
                "/api/rag/query/",
                data=json.dumps(body),
                content_type="application/json",
                REMOTE_ADDR="10.0.0.1",
            )
            second = self.client.post(
                "/api/rag/query/",
                data=json.dumps(body),
                content_type="application/json",
                REMOTE_ADDR="10.0.0.1",
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(second.json()["success"])
        self.assertTrue(second.json()["rate_limited"])
        self.assertEqual(second.json()["message"], "Too many requests. Please retry later.")
        self.assertGreaterEqual(second.json()["retry_after_seconds"], 1)
        self.assertEqual(second.json()["error"]["code"], "rate_limited")
        self.assertTrue(second.json()["error"]["retryable"])
        self.assertTrue(second["Retry-After"])

    @override_settings(
        RAG_API_KEY="secret-key",
        RAG_RATE_LIMIT_MAX_REQUESTS=1,
        RAG_RATE_LIMIT_WINDOW_SECONDS=60,
    )
    def test_rag_query_rate_limit_is_per_conversation_with_api_key(self):
        with patch("apps.rooms.views.run_streaming", new=self._response):
            first_conv = self.client.post(
                "/api/rag/query/",
                data=json.dumps({"message": "Tim phong", "conversation_id": "conv_user_a"}),
                content_type="application/json",
                HTTP_X_API_KEY="secret-key",
            )
            second_conv = self.client.post(
                "/api/rag/query/",
                data=json.dumps({"message": "Tim phong", "conversation_id": "conv_user_b"}),
                content_type="application/json",
                HTTP_X_API_KEY="secret-key",
            )
            repeat_first_conv = self.client.post(
                "/api/rag/query/",
                data=json.dumps({"message": "Tim phong tiep", "conversation_id": "conv_user_a"}),
                content_type="application/json",
                HTTP_X_API_KEY="secret-key",
            )

        self.assertTrue(first_conv.json()["success"])
        self.assertTrue(second_conv.json()["success"])
        self.assertFalse(repeat_first_conv.json()["success"])
        self.assertTrue(repeat_first_conv.json()["rate_limited"])

    def test_rag_stream_emits_status_and_final_events(self):
        body = {"message": "Tim phong duoi 5 trieu", "session_id": "stream-123"}

        with patch("apps.rooms.views.run_streaming", new=self._streaming_response):
            response = self.client.post(
                "/api/rag/stream/",
                data=json.dumps(body),
                content_type="application/json",
                secure=True,
            )
            chunks = []
            for chunk in response.streaming_content:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        joined = "".join(chunks)
        self.assertIn('"type": "status"', joined)
        self.assertIn('"agent": "intent_router"', joined)
        self.assertIn('"agent": "retriever"', joined)
        self.assertIn('"type": "final"', joined)

    async def _streaming_response_with_object_id(self, question, history=None, session_id="", stream_callback=None):
        return {
            "session_id": session_id or "rag-session-oid",
            "answer": "OK",
            "intent": "SEARCH_ROOM",
            "session_state": {"constraints": {}},
            "rooms": [{"room_id": "A101", "title": "Studio", "property_id": ObjectId("665f00000000000000000001")}],
            "cost_estimate": None,
            "comparison": None,
            "suggested_questions": ["next"],
            "sources": [{"type": "room", "room_id": "A101", "property_id": ObjectId("665f00000000000000000002")}],
            "retrieval_confidence": 0.82,
            "retrieval_low_confidence": False,
            "retrieval_feedback_retry_count": 0,
            "retrieval_attempts": [{"top_room_ids": ["A101"], "trace_id": ObjectId("665f00000000000000000003")}],
            "processing_time_ms": 123,
        }

    def test_rag_stream_serializes_object_id_in_final_payload(self):
        body = {"message": "Tim phong", "session_id": "stream-oid"}

        with patch("apps.rooms.views.run_streaming", new=self._streaming_response_with_object_id):
            response = self.client.post(
                "/api/rag/stream/",
                data=json.dumps(body),
                content_type="application/json",
                secure=True,
            )
            chunks = []
            for chunk in response.streaming_content:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        joined = "".join(chunks)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "final"', joined)
        self.assertIn('"665f00000000000000000001"', joined)
        self.assertIn('"665f00000000000000000002"', joined)
        self.assertIn('"665f00000000000000000003"', joined)


@override_settings(ALLOWED_HOSTS=["127.0.0.1", "testserver", "localhost"])
class ChatApiContractTests(ChatApiTests):
    async def _full_rag_response(self, question, history=None, session_id="", stream_callback=None):
        return {
            "answer": "Dạ còn phòng phù hợp ạ.",
            "intent": "SEARCH_ROOM",
            "session_state": {"constraints": {"budget": {"max": 5_000_000}}},
            "rooms": [{"room_id": "A101", "title": "Studio Bình Thạnh", "rent_price": 4_500_000}],
            "cost_estimate": {"total_initial": 9_000_000},
            "comparison": None,
            "suggested_questions": ["Còn phòng rẻ hơn không?"],
            "sources": [{"type": "room", "room_id": "A101"}],
            "verification": {"approved": True, "corrected_answer_used": False},
            "retrieval_confidence": 0.81,
            "retrieval_low_confidence": False,
            "retrieval_feedback_retry_count": 0,
            "retrieval_attempts": [{"top_room_ids": ["A101"]}],
            "processing_time_ms": 150,
        }

    def test_chat_final_payload_matches_rag_contract_fields(self):
        body = {
            "message": "Tìm phòng dưới 5 triệu",
            "contact_phone": "0900555666",
            "contact_name": "Contract QA",
            "demo_role": "customer",
        }
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            with patch("apps.rooms.views.run_streaming", new=self._full_rag_response):
                post = self.client.post(
                    "/api/chat/",
                    data=json.dumps(body),
                    content_type="application/json",
                )
                chunks = []
                for chunk in post.streaming_content:
                    chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
                final_line = [
                    line for line in "".join(chunks).splitlines()
                    if line.startswith("data: ") and '"type": "final"' in line
                ][-1]
                payload = json.loads(final_line[len("data: "):])["payload"]

        for key in (
            "session_state",
            "verification",
            "cost_estimate",
            "comparison",
            "retrieval_confidence",
            "retrieval_attempts",
            "sources",
            "follow_ups",
            "intent",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["cost_estimate"]["total_initial"], 9_000_000)

    def test_chat_reload_restores_assistant_metadata(self):
        body = {
            "message": "Tìm phòng quận 7",
            "contact_phone": "0900777888",
            "contact_name": "Reload Meta",
            "demo_role": "customer",
        }
        with patch("apps.rooms.views.get_collection", side_effect=self._get_collection):
            with patch("apps.rooms.views.run_streaming", new=self._full_rag_response):
                post = self.client.post(
                    "/api/chat/",
                    data=json.dumps(body),
                    content_type="application/json",
                )
                chunks = []
                for chunk in post.streaming_content:
                    chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
                final_line = [
                    line for line in "".join(chunks).splitlines()
                    if line.startswith("data: ") and '"type": "final"' in line
                ][-1]
                conv_id = json.loads(final_line[len("data: "):])["payload"]["conversation_id"]

                reload = self.client.get("/api/chat/", {
                    "contact_phone": "0900777888",
                    "contact_name": "Reload Meta",
                    "demo_role": "customer",
                    "conversation_id": conv_id,
                }).json()

        assistant_msgs = [msg for msg in reload["messages"] if msg.get("role") == "assistant"]
        self.assertTrue(assistant_msgs)
        last = assistant_msgs[-1]
        self.assertEqual(last.get("intent"), "SEARCH_ROOM")
        self.assertTrue(last.get("rooms"))
        self.assertTrue(last.get("follow_ups"))
        self.assertIn("sources", last)

    def test_health_deep_endpoint_reports_dependency_checks(self):
        with patch("apps.rooms.views._get_rooms_collection", return_value=FakeCollection([])):
            response = self.client.get("/api/health/deep/")
        payload = response.json()
        self.assertIn("checks", payload)
        self.assertIn("mongodb", payload["checks"])
