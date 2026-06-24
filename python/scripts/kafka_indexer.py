"""
Kafka (Redpanda) Consumer cho Nhatrovn.
Lắng nghe sự kiện thay đổi dữ liệu từ topic `room.changed` 
để cập nhật Qdrant Vector Database theo thời gian thực (Real-time).
"""
import sys
import json
import signal
from pathlib import Path

# Đưa thư mục python/ vào sys.path
sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Consumer, KafkaError, KafkaException
from pymongo import MongoClient
from bson import ObjectId

from config import (
    KAFKA_BROKERS,
    ROOM_CHANGED_TOPIC,
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_ROOMS_COLLECTION,
    QDRANT_ROOMS_COLLECTION
)
from retrieval.qdrant_client import QdrantWrapper
from scripts.index_mongo_to_qdrant import extract_text_for_embedding, build_payload
from utils.console import console

running = True

def signal_handler(sig, frame):
    global running
    console.print("\n[bold yellow]Đang tắt Kafka Consumer an toàn...[/]")
    running = False

signal.signal(signal.SIGINT, signal_handler)


def _room_lookup_query(room_id: str) -> dict:
    try:
        return {"$or": [{"room_id": room_id}, {"_id": ObjectId(room_id)}]}
    except Exception:
        return {"room_id": room_id}

def main():
    if not MONGODB_URI or "cluster0.xxx" in MONGODB_URI:
        console.print("[bold red]LỖI: Chưa cấu hình MONGODB_URI trong .env![/]")
        sys.exit(1)

    # 1. Kết nối MongoDB
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DATABASE]
    collection = db[MONGODB_ROOMS_COLLECTION]
    
    # 2. Khởi tạo Qdrant & Embedding Model
    qdrant = QdrantWrapper(collection_name=QDRANT_ROOMS_COLLECTION)
    qdrant.create_collection(recreate=False)
    console.print("[bold green]Đang nạp mô hình AI Embedding (BAAI/bge-m3)...[/]")
    from agents.model_registry import get_embed_model
    embed_model = get_embed_model()

    # 3. Khởi tạo Kafka Consumer
    conf = {
        'bootstrap.servers': KAFKA_BROKERS,
        'group.id': 'nhatrovn_qdrant_indexer_group',
        'auto.offset.reset': 'earliest' # Đọc từ đầu nếu là lần đầu tiên connect
    }
    consumer = Consumer(conf)
    consumer.subscribe([ROOM_CHANGED_TOPIC])

    console.print(f"[bold green]🚀 Kafka Indexer đang lắng nghe topic: {ROOM_CHANGED_TOPIC}[/]")
    console.print(f"Broker: {KAFKA_BROKERS}")
    console.print("Nhấn Ctrl+C để thoát.\n")

    try:
        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            
            if msg.error():
                err_code = msg.error().code()
                if err_code == KafkaError._PARTITION_EOF:
                    continue
                elif err_code == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    console.print(f"[yellow]Chờ topic {ROOM_CHANGED_TOPIC} được tạo...[/]", end="\r")
                    continue
                else:
                    raise KafkaException(msg.error())

            # Decode message
            try:
                data = json.loads(msg.value().decode('utf-8'))
                action = data.get("operation") or data.get("action") or "upsert"
                room_id_str = data.get("room_id")
                
                if not room_id_str:
                    console.print("[yellow]Bỏ qua tin nhắn không có room_id[/]")
                    continue

                if action in {"delete", "unpublish"}:
                    qdrant.delete_room_points(room_id_str)
                    console.print(f"🗑️ Đã xóa room [red]{room_id_str}[/] khỏi Qdrant.")
                
                elif action in {"upsert", "publish"}:
                    # Lấy data mới nhất từ Mongo
                    doc = collection.find_one(_room_lookup_query(room_id_str))
                    if not doc:
                        console.print(f"[yellow]Không tìm thấy room {room_id_str} trong MongoDB.[/]")
                        continue
                    
                    text = extract_text_for_embedding(doc)
                    if not text.strip():
                        console.print(f"[yellow]Room {room_id_str} không có text để nhúng.[/]")
                        continue
                    
                    # Tính vector & Push lên Qdrant
                    embedding = embed_model.encode(text, normalize_embeddings=True)
                    payload = build_payload(doc)

                    qdrant.upsert_room_chunk(
                        room_id=str(payload.get("room_id") or room_id_str),
                        chunk_type="room_summary",
                        text=text,
                        embedding=embedding,
                        payload=payload,
                    )
                    console.print(f"✅ Đã upsert room [cyan]{room_id_str}[/] vào Qdrant thành công!")

            except json.JSONDecodeError:
                console.print("[red]Lỗi: Message không phải định dạng JSON hợp lệ.[/]")
            except Exception as e:
                console.print(f"[red]Lỗi xử lý message: {e}[/]")

    finally:
        consumer.close()
        console.print("[bold green]Đã đóng Kafka Consumer.[/]")

if __name__ == "__main__":
    main()
