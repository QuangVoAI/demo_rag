# RAG API Examples

## Health Check (shallow)

Fast liveness probe — does not call Mongo/Qdrant/LLM.

```bash
curl https://your-rag-host/api/health/
```

Example response:

```json
{
  "success": true,
  "status": "ok",
  "service": "nhatrovn-rag",
  "api": {
    "rag_query_path": "/api/rag/query/",
    "rag_stream_path": "/api/rag/stream/",
    "health_deep_path": "/api/health/deep/"
  }
}
```

## Deep Health Check (optional)

Use before cutover or in staging monitors. May return `503` when Mongo is unreachable.

```bash
curl https://your-rag-host/api/health/deep/
```

Example response:

```json
{
  "success": true,
  "status": "ok",
  "service": "nhatrovn-rag",
  "checks": {
    "mongodb": {"status": "ok"},
    "qdrant": {"status": "ok"},
    "llm": {"status": "ok"}
  }
}
```

## Success query

```bash
curl -X POST https://your-rag-host/api/rag/query/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_RAG_API_KEY" \
  -d '{
    "message": "Tìm phòng dưới 5 triệu ở Bình Thạnh",
    "session_id": "web-user-123",
    "history": [
      { "role": "user", "content": "Mình cần gần trung tâm" }
    ]
  }'
```

Example response (trimmed):

```json
{
  "success": true,
  "session_id": "web-user-123",
  "reply": "Dạ em tìm được vài phòng phù hợp ạ...",
  "intent": "SEARCH_ROOM",
  "session_state": {"constraints": {"budget": {"max": 5000000}}},
  "rooms": [{"room_id": "A101", "title": "Studio Bình Thạnh"}],
  "sources": [{"type": "room", "room_id": "A101"}],
  "verification": {"approved": true},
  "retrieval_confidence": 0.82,
  "retrieval_attempts": [
    {
      "query": "Tìm phòng dưới 5 triệu ở Bình Thạnh",
      "result_count": 5,
      "confidence": 0.26,
      "top_room_ids": ["6741770ef0dcd06be1634f17"],
      "semantic_result_count": 6,
      "semantic_error": false,
      "fallback_used": false
    }
  ],
  "retrieval_explanation": [
    "Lọc quận: binh thanh",
    "Ngân sách tối đa: 5,000,000 VND",
    "Sau lọc cứng MongoDB: 100 phòng ứng viên",
    "Kết quả trả về: 5 phòng"
  ],
  "empty_result_reason": null,
  "processing_time_ms": 1234
}
```

## Multi-turn budget refinement

Turn 1: `"Tìm phòng Bình Thạnh dưới 5 triệu"` → `session_state.constraints.budget.max = 5000000`, `location.districts = ["binh thanh"]`.

Turn 2: `"Nới ngân sách thêm 1 triệu"` → `intent: REFINE_SEARCH`, `budget.max = 6000000` (cộng từ state, không parse “1 triệu” thành max tuyệt đối), quận giữ nguyên, retrieval chạy lại.

## No-result sales handoff

When inventory miss is confirmed for district/budget constraints:

```json
{
  "success": true,
  "intent": "SEARCH_ROOM",
  "rooms": [],
  "empty_result_reason": "NO_HARD_FILTER_CANDIDATES",
  "retrieval_explanation": [
    "Lọc quận: quan 7",
    "Ngân sách tối đa: 100,000 VND",
    "Sau lọc cứng MongoDB: 0 phòng ứng viên",
    "Kết quả trả về: 0 phòng"
  ],
  "reply": "Dạ em tìm mỏi mắt mà chưa thấy phòng nào khớp 100% điều kiện của mình ạ...",
  "suggested_questions": ["Nới ngân sách lên 6 triệu", "Xem phòng quận lân cận"]
}
```

`empty_result_reason` helps distinguish true inventory miss from retrieval/index failures. User-facing `reply` remains the sales handoff template.

## Request action refusal

```json
{
  "success": true,
  "intent": "REQUEST_ACTION",
  "rooms": [],
  "reply": "Dạ tính năng thao tác tự động em chưa được học ạ..."
}
```

## Validation error

```json
{
  "success": false,
  "message": "Vui lòng nhập câu hỏi.",
  "error": {
    "code": "validation_error",
    "message": "Vui lòng nhập câu hỏi.",
    "retryable": false,
    "http_status": 400
  }
}
```

## Rate limit

```json
{
  "success": false,
  "message": "Too many requests. Please retry later.",
  "rate_limited": true,
  "retry_after_seconds": 42,
  "error": {
    "code": "rate_limited",
    "message": "Too many requests. Please retry later.",
    "retryable": true,
    "http_status": 200
  }
}
```

## Source fields

Typical `sources` item:

```json
{
  "type": "room",
  "room_id": "A101",
  "title": "Studio Bình Thạnh",
  "property_id": "665f00000000000000000001"
}
```

## Frontend fetch

```js
const response = await fetch("/api/rag/query/", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": window.RAG_API_KEY,
  },
  body: JSON.stringify({
    message: userMessage,
    session_id: conversationId,
    history: history.slice(-10),
  }),
});

const payload = await response.json();
if (!response.ok || !payload.success) {
  throw new Error(payload.message || "RAG request failed");
}
```

## `/api/chat/` contract note

`POST /api/chat/` streams SSE and its `final` payload now mirrors the RAG contract fields (`session_state`, `verification`, `cost_estimate`, `comparison`, retrieval metrics). Reload via `GET /api/chat/` restores assistant message metadata (`rooms`, `follow_ups`, `sources`, `intent`) from persisted chat history.

`POST /api/chat/` does **not** enforce `RAG_API_KEY` or the RAG rate-limit bucket by default — first-party demo UI only. External integrators should use `/api/rag/query/` or `/api/rag/stream/`.
