"""
MongoDB Change Stream (CDC) Publisher cho Nhatrovn.
Theo dõi các thay đổi trực tiếp trên MongoDB (insert, update, delete)
và đẩy sự kiện đầy đủ schema vào topic Kafka `room.changed`.
"""
import sys
import json
from pathlib import Path
from pymongo import MongoClient
from confluent_kafka import Producer

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_ROOMS_COLLECTION,
    KAFKA_BROKERS,
    ROOM_CHANGED_TOPIC,
)
from room_assistant.indexing import build_mongo_cdc_event, source_version_from_change
from room_assistant.schemas import canonical_room_id
from utils.console import console


def delivery_report(err, msg):
    if err is not None:
        console.print(f"[red]❌ Gửi tin nhắn Kafka thất bại: {err}[/]")
    else:
        console.print(f"[green]📨 Đã gửi sự kiện lên Kafka: {msg.topic()} [{msg.partition()}][/]")


def resolve_room_id_from_change(collection, change: dict) -> str:
    """Resolve canonical room_id for indexing (not raw Mongo _id when room_id exists)."""
    doc_key = change.get("documentKey") or {}
    raw_id = doc_key.get("_id")
    if change.get("operationType") == "delete":
        return str(raw_id)

    full_doc = change.get("fullDocument")
    if not full_doc and raw_id is not None:
        full_doc = collection.find_one({"_id": raw_id})
    if full_doc:
        return canonical_room_id(full_doc)
    return str(raw_id)


def build_event_from_change(collection, change: dict) -> dict:
    op_type = str(change.get("operationType") or "update")
    room_id = resolve_room_id_from_change(collection, change)
    full_doc = change.get("fullDocument")
    if not full_doc and change.get("operationType") != "delete":
        raw_id = (change.get("documentKey") or {}).get("_id")
        if raw_id is not None:
            full_doc = collection.find_one({"_id": raw_id})
    return build_mongo_cdc_event(
        room_id=room_id,
        mongo_operation=op_type,
        source_version=source_version_from_change(change, full_doc),
    )


def main():
    if not MONGODB_URI or "cluster0.xxx" in MONGODB_URI:
        console.print("[bold red]LỖI: Chưa cấu hình MONGODB_URI trong .env![/]")
        sys.exit(1)

    console.print(f"[bold green]1. Kết nối MongoDB: {MONGODB_URI.split('@')[-1]}...[/]")
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DATABASE]
    collection = db[MONGODB_ROOMS_COLLECTION]

    console.print(f"[bold green]2. Kết nối Kafka Broker: {KAFKA_BROKERS}...[/]")
    producer = Producer({"bootstrap.servers": KAFKA_BROKERS})

    console.print(f"\n[bold blue]👀 Đang theo dõi collection '{MONGODB_ROOMS_COLLECTION}' để phát hiện thay đổi...[/]")
    console.print("Nhấn Ctrl+C để dừng.\n")

    pipeline = [
        {
            "$match": {
                "operationType": {"$in": ["insert", "update", "replace", "delete"]},
            }
        }
    ]

    try:
        with collection.watch(pipeline, full_document="updateLookup") as stream:
            for change in stream:
                op_type = change["operationType"]
                payload = build_event_from_change(collection, change)

                console.print(
                    f"🔔 Phát hiện thay đổi DB: [bold yellow]{op_type}[/] "
                    f"room_id=[cyan]{payload['room_id']}[/] op={payload['operation']} -> Kafka..."
                )

                producer.produce(
                    ROOM_CHANGED_TOPIC,
                    key=str(payload["room_id"]).encode("utf-8"),
                    value=json.dumps(payload).encode("utf-8"),
                    callback=delivery_report,
                )
                producer.poll(0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Đang tắt MongoDB CDC Publisher...[/]")
    except Exception as e:
        console.print(f"[bold red]Lỗi Change Stream: {e}[/]")
        console.print("[yellow]Lưu ý: MongoDB Change Stream yêu cầu MongoDB Replica Set (Atlas mặc định hỗ trợ).[/]")
    finally:
        producer.flush()
        console.print("[bold green]Đã dừng CDC Publisher.[/]")


if __name__ == "__main__":
    main()
