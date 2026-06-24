"""
MongoDB Change Stream (CDC) Publisher cho Nhatrovn.
Theo dõi các thay đổi trực tiếp trên MongoDB (insert, update, delete) 
và đẩy sự kiện vào topic Kafka `room.changed` để Qdrant tự động cập nhật.
"""
import sys
import json
from pathlib import Path
from pymongo import MongoClient
from confluent_kafka import Producer

# Đưa thư mục python/ vào sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_ROOMS_COLLECTION,
    KAFKA_BROKERS,
    ROOM_CHANGED_TOPIC
)
from utils.console import console

def delivery_report(err, msg):
    if err is not None:
        console.print(f"[red]❌ Gửi tin nhắn Kafka thất bại: {err}[/]")
    else:
        console.print(f"[green]📨 Đã gửi sự kiện lên Kafka: {msg.topic()} [{msg.partition()}][/]")

def main():
    if not MONGODB_URI or "cluster0.xxx" in MONGODB_URI:
        console.print("[bold red]LỖI: Chưa cấu hình MONGODB_URI trong .env![/]")
        sys.exit(1)

    console.print(f"[bold green]1. Kết nối MongoDB: {MONGODB_URI.split('@')[-1]}...[/]")
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DATABASE]
    collection = db[MONGODB_ROOMS_COLLECTION]
    
    console.print(f"[bold green]2. Kết nối Kafka Broker: {KAFKA_BROKERS}...[/]")
    producer = Producer({'bootstrap.servers': KAFKA_BROKERS})
    
    console.print(f"\n[bold blue]👀 Đang theo dõi collection '{MONGODB_ROOMS_COLLECTION}' để phát hiện thay đổi...[/]")
    console.print("Nhấn Ctrl+C để dừng.\n")
    
    pipeline = [
        {
            "$match": {
                "operationType": {"$in": ["insert", "update", "replace", "delete"]}
            }
        }
    ]
    
    try:
        with collection.watch(pipeline) as stream:
            for change in stream:
                op_type = change["operationType"]
                doc_id = str(change["documentKey"]["_id"])
                
                action = "upsert"
                if op_type == "delete":
                    action = "delete"
                
                payload = {
                    "action": action,
                    "room_id": doc_id
                }
                
                console.print(f"🔔 Phát hiện thay đổi DB: [bold yellow]{op_type}[/] trên room_id: [cyan]{doc_id}[/] -> Đang gửi Kafka...")
                
                producer.produce(
                    ROOM_CHANGED_TOPIC,
                    value=json.dumps(payload).encode('utf-8'),
                    callback=delivery_report
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
