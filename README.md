# Nhatrovn Assistant

Read-only AI assistant for customers looking for rooms on Nhatrovn.

The assistant can search rooms, analyze fit, compare rooms, estimate costs, summarize room details, and suggest next questions. It must not perform business actions such as booking, messaging landlords, saving favorites, holding rooms, payments, negotiation, profile updates, or room mutations.

## Architecture

```text
Frontend
-> Rust Actix WebSocket Gateway
-> Redpanda/Kafka query.request
-> Python Query Worker
-> Read-only Room Assistant Workflow
-> MongoDB RoomRepository + Qdrant room vectors
-> Redpanda/Kafka query.response
-> Rust Gateway
-> Frontend final response
```

Continuous indexing is separate from query serving:

```text
crawler/backfill
-> room.changed
-> python/kafka_workers/room_index_worker.py
-> MongoDB fetch latest room
-> canonical embedding text + content hash
-> Qdrant named dense+sparse upsert/delete
-> room.index.dlq on permanent/retry-exhausted failures
```

## Production Flow

```text
START
-> normalize_input
-> parse_intent_async
-> analyze_mood
-> load_session_state
-> merge_and_validate_state
-> route_workflow
-> execute_read_only_tools
-> grounding_check
-> compose_answer_with_review
-> persist_state_and_trace
-> END
```

## Read-Only Tool Registry

Only these tools are registered:

```text
search_rooms
get_room_detail
retrieve_room_context
retrieve_faq
calculate_cost_estimate
compare_rooms
find_similar_rooms
```

Invariants:

```text
write_tool_calls_per_turn = 0
max_read_tool_calls_per_turn <= 3
```

Requests for booking, messaging, saving, holding, payment, negotiation, or room edits are classified as `REQUEST_ACTION` and return UI guidance only.

## Client Contract

The assistant response exposes room identity only:

```json
{
  "rooms": [
    {
      "room_id": "A101",
      "house_id": "H001"
    }
  ],
  "sources": [
    {
      "type": "room",
      "room_id": "A101",
      "house_id": "H001"
    }
  ]
}
```

Full room documents stay inside the Python workflow for grounding and cost calculation. They are not sent to the frontend.

## Session State

Redis key:

```text
ai:session:{session_id}
```

Minimum state:

```json
{
  "session_id": "...",
  "constraints": {
    "location": {
      "province": null,
      "districts": [],
      "wards": [],
      "near_landmarks": [],
      "max_distance_km": null
    },
    "budget": {
      "min": null,
      "max": null,
      "type": "rent_only"
    },
    "occupants": null,
    "vehicles": [],
    "pets_required": [],
    "amenities_required": [],
    "amenities_preferred": [],
    "excluded_features": [],
    "move_in_date": null
  },
  "current_room_id": null,
  "selected_room_ids": [],
  "last_result_ids": [],
  "last_intent": null,
  "conversation_summary": "",
  "state_version": 1,
  "updated_at": "..."
}
```

The Rust gateway reads recent messages from server-side SQLite and sends only a capped recent history. Frontend history is fallback compatibility, not the source of truth.

## Event Schema

Topic: `room.changed`

Kafka key: `room_id`

```json
{
  "event_id": "uuid",
  "room_id": "string",
  "operation": "upsert",
  "source_version": 12,
  "occurred_at": "ISO-8601",
  "producer": "crawler"
}
```

Supported operations:

```text
upsert
delete
publish
unpublish
```

DLQ topic: `room.index.dlq`

```json
{
  "original_event": {},
  "error_type": "...",
  "error_message": "...",
  "attempts": 3,
  "failed_at": "..."
}
```

## Qdrant Room Schema

Collection default: `rooms_v1`

Named vectors:

```text
dense
sparse
```

Point IDs are deterministic:

```text
room:{room_id}:{chunk_type}
```

Payload:

```json
{
  "room_id": "...",
  "house_id": "...",
  "chunk_type": "room_summary",
  "source_version": 12,
  "content_hash": "...",
  "embedding_model": "BAAI/bge-m3",
  "embedding_version": 1,
  "status": "active",
  "district": "...",
  "indexed_at": "..."
}
```

Qdrant is used for semantic retrieval and reranking only. Final answers fetch current authoritative room data from `RoomRepository`.

## Local Setup

Create a local env file from the example:

```bash
cp .env.example .env
```

Start infrastructure:

```bash
docker compose up -d qdrant redpanda redpanda-console mongo
```

Install Python dependencies:

```bash
cd python
pip install -r requirements.txt
```

Create Kafka topics:

```bash
cd python
python kafka_workers/kafka_config.py
```

Run query worker:

```bash
cd python
python kafka_workers/query_worker.py
```

Run room index worker:

```bash
cd python
python kafka_workers/room_index_worker.py
```

Optional one-shot Mongo to Qdrant backfill:

```bash
cd python
python scripts/index_mongo_to_qdrant.py
```

Run Rust gateway:

```bash
cd rust_backend
cargo run
```

Frontend:

```text
http://localhost:8083
```

## Crawler Integration

For each changed room, the crawler should publish one compact `room.changed` event with Kafka key equal to `room_id`. Do not put the full room document in the event. The index worker fetches the latest document from MongoDB by `room_id`, compares `source_version`, skips unchanged semantic content by `content_hash`, and upserts/deletes Qdrant points idempotently.

## Tests

Run Python unit tests:

```bash
python -m unittest discover -s python/tests
```

Run Rust checks:

```bash
cd rust_backend
cargo test
```

Some integration checks require local MongoDB, Qdrant, Redis or model downloads. Unit tests use fake repositories/vector indexes and do not require crawler data.
