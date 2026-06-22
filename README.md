# Nhatrovn Assistant

Read-only AI assistant for customers looking for rooms on Nhatrovn.

The assistant can search, analyze, compare, estimate costs, summarize listings, explain fit, and suggest next questions. It must not perform business actions such as booking, messaging landlords, saving favorites, holding rooms, payments, negotiation, profile updates, or listing mutations.

## Architecture

```text
Frontend
-> Rust Actix WebSocket Gateway
-> Redpanda/Kafka query.request
-> Python Query Worker
-> Read-only Room Assistant Workflow
-> MongoDB ListingRepository + Qdrant listing vectors
-> Redpanda/Kafka query.response
-> Rust Gateway
-> Frontend final response
```

Continuous indexing is separate from query serving:

```text
crawler/backfill
-> listing.changed
-> python/kafka_workers/listing_index_worker.py
-> MongoDB fetch latest listing
-> canonical embedding text + content hash
-> Qdrant upsert/delete
-> listing.index.dlq on permanent/retry-exhausted failures
```

## Production Flow

```text
START
-> normalize_input
-> parse_intent_and_constraint_patch
-> load_session_state
-> merge_and_validate_state
-> route_workflow
-> execute_read_only_tools
-> grounding_check
-> compose_response
-> persist_state_and_trace
-> END
```

The legacy support graph is no longer imported by the production entry point `python/agents/graph.py`.

## Read-Only Tool Registry

Only these tools are registered:

```text
search_listings
get_listing_detail
retrieve_listing_context
retrieve_faq
calculate_cost_estimate
compare_listings
find_similar_listings
```

Invariants:

```text
write_tool_calls_per_turn = 0
max_read_tool_calls_per_turn <= 3
```

Requests for booking, messaging, saving, holding, payment, negotiation, or listing edits are classified as `REQUEST_ACTION` and return UI guidance only.

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
  "current_listing_id": null,
  "selected_listing_ids": [],
  "last_result_ids": [],
  "last_intent": null,
  "conversation_summary": "",
  "state_version": 1,
  "updated_at": "..."
}
```

The Rust gateway reads recent messages from server-side SQLite and sends only a capped recent history. Frontend history is fallback compatibility, not the source of truth.

## Event Schema

Topic: `listing.changed`

Kafka key: `listing_id`

```json
{
  "event_id": "uuid",
  "listing_id": "string",
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

DLQ topic: `listing.index.dlq`

```json
{
  "original_event": {},
  "error_type": "...",
  "error_message": "...",
  "attempts": 3,
  "failed_at": "..."
}
```

## Qdrant Listing Schema

Point IDs are deterministic:

```text
listing:{listing_id}:{chunk_type}
```

Payload:

```json
{
  "listing_id": "...",
  "chunk_type": "listing_summary",
  "source_version": 12,
  "content_hash": "...",
  "embedding_model": "BAAI/bge-m3",
  "embedding_version": 1,
  "status": "active",
  "district": "...",
  "indexed_at": "..."
}
```

Qdrant is used for semantic retrieval and reranking only. Final answers always fetch current authoritative listing data from `ListingRepository`.

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

Run listing index worker:

```bash
cd python
python kafka_workers/listing_index_worker.py
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

## Backfill

Dry run:

```bash
python scripts/backfill_listings.py --dry-run --batch-size 100
```

Publish `listing.changed` events:

```bash
python scripts/backfill_listings.py --mode events --resume --batch-size 100
```

Directly call the same indexing service used by the worker:

```bash
python scripts/backfill_listings.py --mode direct --resume --batch-size 100
```

Backfill uses checkpoint `data/state/listing_backfill_checkpoint.json` by default and does not recreate Qdrant collections.

## Crawler Integration

For each changed listing, the crawler should publish one compact `listing.changed` event with Kafka key equal to `listing_id`. Do not put the full listing document in the event. The index worker fetches the latest document from MongoDB by `listing_id`, compares `source_version`, skips unchanged semantic content by `content_hash`, and upserts/deletes Qdrant points idempotently.

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
