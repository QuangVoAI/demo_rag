"""Backfill listing indexing events or direct indexing service calls.

Usage:
  python scripts/backfill_listings.py --dry-run
  python scripts/backfill_listings.py --mode events --batch-size 100
  python scripts/backfill_listings.py --mode direct --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "python"))

from kafka_workers.kafka_config import BROKERS, TOPIC_LISTING_CHANGED, serialize
from room_assistant.indexing import BgeEmbeddingProvider, ListingIndexingService
from room_assistant.qdrant_index import QdrantListingVectorIndex
from room_assistant.repository import create_listing_repository


def load_checkpoint(path: Path) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("last_listing_id")


def save_checkpoint(path: Path, listing_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_listing_id": listing_id, "updated_at": now_iso()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_event(listing_id: str, source_version: int) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "listing_id": listing_id,
        "operation": "upsert",
        "source_version": source_version,
        "occurred_at": now_iso(),
        "producer": "backfill",
    }


def create_producer():
    from confluent_kafka import Producer

    return Producer({"bootstrap.servers": BROKERS, "acks": "all"})


def run(args: argparse.Namespace) -> int:
    repository = create_listing_repository()
    checkpoint = Path(args.checkpoint)
    resume_after = load_checkpoint(checkpoint) if args.resume else None

    producer = None
    service = None
    if args.mode == "events" and not args.dry_run:
        producer = create_producer()
    if args.mode == "direct" and not args.dry_run:
        service = ListingIndexingService(
            repository=repository,
            vector_index=QdrantListingVectorIndex(),
            embedding_provider=BgeEmbeddingProvider(),
        )

    processed = 0
    for batch in repository.iter_listing_ids(batch_size=args.batch_size, resume_after=resume_after):
        for listing_id in batch:
            version = repository.get_current_version(listing_id) or 0
            event = build_event(listing_id, version)
            if args.dry_run:
                print(json.dumps({"dry_run": True, "event": event}, ensure_ascii=False))
            elif args.mode == "events":
                producer.produce(
                    TOPIC_LISTING_CHANGED,
                    key=listing_id.encode("utf-8"),
                    value=serialize(event),
                )
                producer.poll(0)
            else:
                result = service.process_with_retry(event, attempts=args.attempts)
                print(json.dumps({"direct_index": result}, ensure_ascii=False))

            processed += 1
            if not args.dry_run:
                save_checkpoint(checkpoint, listing_id)

        print(json.dumps({"progress": processed, "last_listing_id": batch[-1]}, ensure_ascii=False))

    if producer:
        producer.flush()
    print(json.dumps({"done": True, "processed": processed}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["events", "direct"], default="events")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--checkpoint", default="data/state/listing_backfill_checkpoint.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

