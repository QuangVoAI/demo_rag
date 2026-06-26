# RAG API Examples

## Health Check

```bash
curl https://your-rag-host/api/health/
```

## cURL

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

## Backend axios

```js
import axios from "axios";

const { data } = await axios.post(
  "https://your-rag-host/api/rag/query/",
  {
    message: userMessage,
    session_id: conversationId,
    history: history.slice(-10),
  },
  {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.RAG_API_KEY,
    },
    timeout: 65000,
  }
);
```
