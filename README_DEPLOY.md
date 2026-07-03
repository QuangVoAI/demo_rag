# README Deploy

## Mục tiêu

Expose RAG backend cho `nhatrovn.vn` gọi qua REST:
- `POST /api/rag/query/`
- `GET /api/health/`

First-party demo/UI nội bộ:
- `POST /api/rag/stream/`

## Runtime tối thiểu

- Python `3.11`
- Django app từ repo này
- MongoDB cho dữ liệu phòng
- Redis khuyến nghị cho production rate limiting

Optional:
- Qdrant
- Kafka workers
- Langfuse

## Biến môi trường chính

```env
DJANGO_DEBUG=false
DJANGO_SECRET_KEY=replace-with-a-long-random-secret
DJANGO_ALLOWED_HOSTS=rag.your-domain.com

DJANGO_SECURE_SSL_REDIRECT=true
DJANGO_SESSION_COOKIE_SECURE=true
DJANGO_CSRF_COOKIE_SECURE=true
DJANGO_SECURE_HSTS_SECONDS=3600
DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS=true
DJANGO_SECURE_HSTS_PRELOAD=true

MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=demo_rag
MONGODB_ROOMS_COLLECTION=rooms
MONGODB_SERVER_SELECTION_TIMEOUT_MS=3000
MONGODB_CONNECT_TIMEOUT_MS=3000

RAG_API_KEY=replace-with-shared-secret-between-web-and-rag
RAG_RATE_LIMIT_CACHE_ALIAS=default
RAG_RATE_LIMIT_ENABLED=true
RAG_RATE_LIMIT_WINDOW_SECONDS=60
RAG_RATE_LIMIT_MAX_REQUESTS=30

DJANGO_CACHE_REDIS_URL=redis://127.0.0.1:6379/1
DJANGO_CACHE_KEY_PREFIX=nhatrovn
DJANGO_CACHE_DEFAULT_TIMEOUT=300

# Room assistant / RAG pipeline (python/)
RETRIEVAL_CANDIDATE_LIMIT=100
TOP_K_RETRIEVAL=6
QDRANT_URL=http://localhost:6333
QDRANT_ROOMS_COLLECTION=rooms_v1
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=demo_rag
MONGODB_ROOMS_COLLECTION=rooms
GROQ_API_KEY=
ENABLE_REVIEWER=false
```

Ghi chú:
- Nếu `DJANGO_CACHE_REDIS_URL` rỗng, app fallback sang `LocMemCache`
- Production nhiều instance nên bật Redis để rate limit đồng bộ giữa các app instance

Xem thêm:
- [.env.example](/C:/Users/Admin/Desktop/work/nhatrovn/.env.example)

## Endpoint tích hợp

### 1. Health check

`GET /api/health/`

Expected:
- HTTP `200`
- JSON `{"success": true, "status": "ok", ...}`

### 2. RAG query

`POST /api/rag/query/`

Headers:

```http
Content-Type: application/json
X-API-Key: <RAG_API_KEY>
X-Request-ID: <optional-client-request-id>
```

Request body:

```json
{
  "message": "Tìm phòng dưới 5 triệu ở Bình Thạnh",
  "session_id": "web-user-123",
  "history": [
    { "role": "user", "content": "Mình cần gần trung tâm" }
  ]
}
```

## Response / headers

Success:
- HTTP `200`
- JSON chứa:
  - `success`
  - `request_id`
  - `correlation_id`
  - `reply`
  - `answer`
  - `intent`
  - `rooms`
  - `follow_ups`
  - `suggested_questions`

Response headers:
- `X-Request-ID`
- `X-Correlation-ID`
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

Rate limited:
- HTTP `200`
- JSON `rate_limited=true`
- `retry_after_seconds`
- `Retry-After` header

## JSON error contract

Mọi JSON error response đều có envelope:

```json
{
  "success": false,
  "message": "Human readable error message",
  "request_id": "req-123",
  "correlation_id": "req-123",
  "error": {
    "code": "validation_error",
    "message": "Human readable error message",
    "retryable": false,
    "http_status": 400
  }
}
```

`error.code` hiện hỗ trợ:
- `validation_error`
- `unauthorized`
- `rate_limited`
- `service_unavailable`

## Rate limiting

Rate limit áp ở lớp HTTP ngoài cùng của `POST /api/rag/query/` và `POST /api/rag/stream/`.

Nó giới hạn:
- số lượt gọi **theo từng cuộc hội thoại** khi có `session_id` hoặc `conversation_id`
- mỗi conversation có bucket riêng (khuyến nghị: dùng `conversation_id` từ Mongo `chat_history`)
- nếu thiếu `session_id`/`conversation_id`: fallback theo `API key`, hoặc theo `IP`

Nó không giới hạn:
- các bước nội bộ trong cùng một request
- planner / retrieval / reviewer / multi-agent bên trong pipeline RAG

Nói ngắn gọn:
- đây là limit usage của caller / end-user traffic
- không phải limit “số agent con” bên trong 1 lượt xử lý

## Reverse proxy

Khuyến nghị:
- terminate HTTPS ở Nginx / load balancer
- forward:
  - `X-Forwarded-For`
  - `X-Forwarded-Proto`
  - `X-Request-ID` nếu proxy có sinh request id

## Checklist tích hợp cho web team `nhatrovn.vn`

1. Chỉ gọi `POST /api/rag/query/` từ backend/server, không gọi trực tiếp từ browser public.
2. Lưu `RAG_API_KEY` trong secret manager hoặc biến môi trường deploy, không hardcode vào frontend.
3. Gửi `session_id` ổn định theo từng cuộc hội thoại người dùng.
4. Chỉ gửi `history` ngắn, tối đa vài lượt gần nhất.
5. Bật HTTPS end-to-end ở domain public.
6. Log lại `request_id` và `correlation_id` của mỗi response để trace lỗi.
7. Khi nhận `rate_limited=true`, đọc `retry_after_seconds` để khóa nút gửi ở UI hoặc defer retry ở backend.
8. Render `rooms`, `follow_ups`, `intent` theo hướng defensive, không assume field nào luôn luôn có.
9. Nếu `error.retryable=true`, backend web có thể retry có kiểm soát; nếu `false`, trả lỗi chuẩn cho người dùng.
10. Trước cutover thật, chạy smoke test `GET /api/health/` và 1 request thật tới `POST /api/rag/query/`.

## Room assistant (RAG) — hành vi production

### Inventory ~9k

- MongoDB: nguồn authoritative (`rent_price`, `room_code`, trạng thái còn phòng).
- Qdrant `rooms_v1` (~9k points): semantic index (BGE-M3), cập nhật qua Kafka `room_index_worker` (CDC).
- Bot **chỉ gợi ý phòng public**; không bịa từ kho nội bộ sales.

### Biến môi trường retrieval

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `RETRIEVAL_CANDIDATE_LIMIT` | `100` | Số phòng sau Mongo hard filter đưa vào Qdrant rank. Tăng (150–200) nếu recall thấp ở quận dày. |
| `TOP_K_RETRIEVAL` | `6` | Số phòng trả về sau rank |
| `QDRANT_ROOMS_COLLECTION` | `rooms_v1` | Collection Qdrant cho phòng |

### Lookup `room_code`

Khách có thể hỏi `P.305` / `P305` — backend chuẩn hóa `room_code_norm` và tra Mongo trước `room_id`.

### Sales handoff (không có phòng public)

Khi `rooms=[]` sau search/refine khớp điều kiện nhưng inventory public trống, bot trả template thấu cảm + đề xuất **sales tư vấn thêm** (không gợi ý nới ngân sách). Web nên hiển thị như luồng chuyển nhân viên, không phải lỗi hệ thống.

### Giá hiển thị

Câu hỏi giá (`giá bao nhiêu`, `giá phòng P305`) dùng template `X.XXX.XXX VND` từ Mongo. Parser ngân sách chấp nhận: `5 triệu`, `5 củ`, `5m`, `5000k`, `5 chiệu`.

### Test integration stack

```bash
cd python
python -m pytest tests/test_inventory_scale.py::ProductionInventoryScaleTests -q
```

Cần `MONGODB_URI` + Qdrant ≥8k points. CI chạy nhánh in-memory `InventoryScaleTests` (không cần Qdrant).

## Production gates (bắt buộc trước cutover)

Chạy tuần tự và **không cutover** nếu gate bắt buộc fail:

| Gate | Lệnh / endpoint | Pass criteria |
|---|---|---|
| Django system check | `python manage.py check` | exit 0 |
| Django API tests | `python manage.py test apps.rooms` | all pass |
| Python compile | `python -m compileall -q python apps config` | exit 0 |
| Intent/parser | `python -m pytest python/tests/test_intent_parser.py -q` | all pass |
| Indexing contract | `python -m pytest python/tests/test_indexing_contract.py -q` | all pass |
| Core assistant | `python -m pytest python/tests/test_room_assistant_core.py -q` | all pass (hoặc isolate test treo) |
| Shallow health | `GET /api/health/` | HTTP 200, `status=ok` |
| Deep health | `GET /api/health/deep/` | Mongo `ok`; Qdrant `ok` hoặc `skipped` nếu chưa bật semantic |
| Qdrant inventory | `rooms_v1` point count ≈ Mongo available rooms | lệch >1% cần reindex |
| Kafka CDC (nếu bật) | worker consume + upsert 1 event test | `result=upserted` |
| RAG smoke | `POST /api/rag/query/` 1 câu search thật | `success=true`, có `intent` |
| Intent regression | `python -m pytest python/tests/test_intent_regression_matrix.py python/tests/test_landmark_aliases.py -q` | routing + landmark |

Ghi chú:
- `GET /api/health/` giữ nhanh (không gọi Mongo/Qdrant).
- `GET /api/health/deep/` dùng trước deploy/cutover; trả `503` khi Mongo down.
- Staging không có Mongo/Qdrant live: integration tests được skip nhưng phải ghi rõ trong release note.

## Checklist DevOps

1. `manage.py check --deploy`
2. `manage.py test apps.rooms`
3. xác nhận `DJANGO_SECRET_KEY`, `RAG_API_KEY`, `MONGODB_URI` có đủ
4. nếu scale nhiều instance, bật `DJANGO_CACHE_REDIS_URL`
5. xác nhận proxy forward đúng `X-Forwarded-For` và HTTPS
6. xác nhận log thu được `X-Request-ID` / `X-Correlation-ID`

## Tài liệu liên quan

- [RAG_API_CONTRACT.md](/C:/Users/Admin/Desktop/work/nhatrovn/RAG_API_CONTRACT.md)
- [RAG_API_EXAMPLES.md](/C:/Users/Admin/Desktop/work/nhatrovn/RAG_API_EXAMPLES.md)
