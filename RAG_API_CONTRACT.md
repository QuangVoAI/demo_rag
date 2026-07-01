# RAG API Contract

Base paths:
- `POST /api/rag/query/`
- `POST /api/rag/stream/` for first-party demo/UI SSE
- `GET /api/health/`

## Purpose

- `POST /api/rag/query/`: synchronous JSON REST endpoint for `nhatrovn.vn` backend-to-backend integration
- `POST /api/rag/stream/`: same-origin SSE endpoint for the demo chat UI

## Authentication

`POST /api/rag/query/`
- Optional
- If `RAG_API_KEY` is configured, callers must send one of:
  - `X-API-Key: <key>`
  - `Authorization: Bearer <key>`

`POST /api/rag/stream/`
- Same-origin browser endpoint
- Protected by CSRF
- Not intended for third-party/public server-to-server callers

## Request

```json
{
  "message": "Tìm phòng dưới 5 triệu ở Bình Thạnh",
  "session_id": "web-user-123",
  "history": [
    { "role": "user", "content": "Mình cần phòng gần trung tâm" },
    { "role": "assistant", "content": "Bạn muốn ngân sách khoảng bao nhiêu?" }
  ]
}
```

Request fields:
- `message` or `question`: required string
- `session_id`: optional string
- `history`: optional array of `{ "role": string, "content": string }`

## Success response

HTTP:
- `200 OK`

Body:
```json
{
  "success": true,
  "session_id": "web-user-123",
  "reply": "Mình thấy một số phòng phù hợp...",
  "answer": "Mình thấy một số phòng phù hợp...",
  "intent": "SEARCH_ROOM",
  "session_state": {},
  "rooms": [],
  "cost_estimate": null,
  "comparison": null,
  "follow_ups": [],
  "suggested_questions": [],
  "sources": [],
  "retrieval_confidence": 0.82,
  "retrieval_low_confidence": false,
  "retrieval_feedback_retry_count": 0,
  "retrieval_attempts": [],
  "processing_time_ms": 123,
  "request_id": "req-123",
  "correlation_id": "req-123"
}
```

## Standard error contract

Every JSON error response uses this envelope:

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

Possible `error.code` values:
- `validation_error`
- `unauthorized`
- `rate_limited`
- `service_unavailable`

## Rate limit behavior

When limited:
- HTTP `200`
- `Retry-After` header is set
- `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` headers are set

Body:
```json
{
  "success": false,
  "message": "Too many requests. Please retry later.",
  "request_id": "req-123",
  "correlation_id": "req-123",
  "rate_limited": true,
  "retry_after_seconds": 60,
  "error": {
    "code": "rate_limited",
    "message": "Too many requests. Please retry later.",
    "retryable": true,
    "http_status": 200,
    "rate_limited": true,
    "retry_after_seconds": 60
  }
}
```

This is intentionally a cooldown payload for client UX, not a hard `429` response.

## SSE event contract for `/api/rag/stream/`

Event types:
- `status`
- `token`
- `final`
- `error`

Example status event:
```json
{
  "type": "status",
  "stage": "retrieving",
  "agent": "retriever",
  "message": "Mình đang tìm dữ liệu phòng phù hợp và đối chiếu thông tin."
}
```

Example final event:
```json
{
  "type": "final",
  "payload": {
    "success": true,
    "reply": "..."
  }
}
```

## Notes

- `reply` and `answer` contain the same assistant text for client compatibility.
- `session_id` should be stable per end-user conversation if the caller wants multi-turn memory.
- `conversation_id` is accepted as an alias for `session_id` in the request body.
- Rate limiting is applied **per conversation** when `session_id` or `conversation_id` is sent (recommended: Mongo `chat_history.conversation_id`). Without it, limits fall back to API key or client IP.
- `history` is sanitized and bounded server-side; callers should still keep it concise.
- `POST /api/rag/stream/` is for first-party UI experience only. Web team should integrate `POST /api/rag/query/`.

## Assistant behavior (inventory & accuracy)

The bot only recommends rooms visible in **public web/app inventory** (Mongo authoritative prices + Qdrant semantic index). It does **not** invent listings from internal sales-only stock.

### Hybrid retrieval defaults

| Setting | Default | Notes |
|---|---|---|
| `RETRIEVAL_CANDIDATE_LIMIT` | `100` | Max rooms after Mongo hard filter before Qdrant BGE-M3 rank (~9k index). Override via env if recall drops in dense districts. |
| `TOP_K_RETRIEVAL` | `6` | Final rooms returned to client after rank |

Pipeline: **Mongo hard filter → semantic rank on `candidate_ids` → authoritative `rent_price` from Mongo**.

### Room reference resolution

Users may cite rooms by:

- Mongo `room_id` (24-char hex)
- Public **`room_code`** (e.g. `P.305`, `P305`) — normalized via `room_code_norm` before lookup
- Ordinal in last result list (“phòng số 2”, “phòng đầu tiên”)

`ASK_ABOUT_ROOM` / price questions use verified Mongo fields; displayed rent uses `X.XXX.XXX VND` format.

### No public match → sales handoff

When search/refine finds **zero** public rooms (or budget+district miss), `intent` stays `SEARCH_ROOM` / `REFINE_SEARCH`, `rooms` is `[]`, and `reply` uses an empathetic **sales handoff** template (not “nới ngân sách”):

- Acknowledges exhaustive public search (“em tìm mỏi mắt…”)
- Offers internal sales follow-up for off-inventory options
- Mood variants: `normal`, `urgent`, `frustrated` (inferred server-side)

Clients should treat `rooms=[]` + sales wording as **handoff**, not hard error.

### Streaming status events

`/api/rag/stream/` emits progress before tokens, e.g.:

- `[status:Phân tích|Hệ thống] Đang phân tích yêu cầu...`
- `[status:Truy vấn|Cơ sở dữ liệu] Đang tìm kiếm các phòng phù hợp...`
- `[status:Tổng hợp|Trợ lý AI] Đang tổng hợp câu trả lời...`

Template answers (search hits, sales handoff, ordinal errors, verified price) may stream without LLM.
