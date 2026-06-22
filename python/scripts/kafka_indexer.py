"""
Kafka (Redpanda) Consumer cho Nhatrovn.
Lắng nghe sự kiện thay đổi dữ liệu từ topic `listing.changed` 
để cập nhật Qdrant Vector Database theo thời gian thực (Real-time).
"""
import sys
import json
import uuid
import signal
from pathlib import Path

# Đưa thư mục python/ vào sys.path
sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Consumer, KafkaError, KafkaException
from pymongo import MongoClient
from bson import ObjectId
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from config import (
    KAFKA_BROKERS,
    LISTING_CHANGED_TOPIC,
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_LISTINGS_COLLECTION,
    QDRANT_URL,
    QDRANT_COLLECTION
)
from agents.model_registry import get_embed_model
from scripts.index_mongo_to_qdrant import extract_text_for_embedding, build_payload
from utils.console import console

running = True

def signal_handler(sig, frame):
    global running
    console.print("\n[bold yellow]Đang tắt Kafka Consumer an toàn...[/]")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def main():
    if not MONGODB_URI or "cluster0.xxx" in MONGODB_URI:
        console.print("[bold red]LỖI: Chưa cấu hình MONGODB_URI trong .env![/]")
        sys.exit(1)

    # 1. Kết nối MongoDB
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DATABASE]
    collection = db[MONGODB_LISTINGS_COLLECTION]
    
    # 2. Khởi tạo Qdrant & Embedding Model
    qclient = QdrantClient(url=QDRANT_URL)
    console.print("[bold green]Đang nạp mô hình AI Embedding (BAAI/bge-m3)...[/]")
    embed_model = get_embed_model()

    # 3. Khởi tạo Kafka Consumer
    conf = {
        'bootstrap.servers': KAFKA_BROKERS,
        'group.id': 'nhatrovn_qdrant_indexer_group',
        'auto.offset.reset': 'earliest' # Đọc từ đầu nếu là lần đầu tiên connect
    }
    consumer = Consumer(conf)
    consumer.subscribe([LISTING_CHANGED_TOPIC])

    console.print(f"[bold green]🚀 Kafka Indexer đang lắng nghe topic: {LISTING_CHANGED_TOPIC}[/]")
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
                    console.print(f"[yellow]Chờ topic {LISTING_CHANGED_TOPIC} được tạo...[/]", end="\r")
                    continue
                else:
                    raise KafkaException(msg.error())

            # Decode message
            try:
                data = json.loads(msg.value().decode('utf-8'))
                action = data.get("action", "upsert")
                listing_id_str = data.get("listing_id")
                
                if not listing_id_str:
                    console.print("[yellow]Bỏ qua tin nhắn không có listing_id[/]")
                    continue

                point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, listing_id_str))

                if action == "delete":
                    qclient.delete(collection_name=QDRANT_COLLECTION, points_selector=[point_id])
                    console.print(f"🗑️ Đã xóa listing [red]{listing_id_str}[/] khỏi Qdrant.")
                
                elif action == "upsert":
                    # Lấy data mới nhất từ Mongo
                    doc = collection.find_one({"_id": ObjectId(listing_id_str)})
                    if not doc:
                        console.print(f"[yellow]Không tìm thấy listing {listing_id_str} trong MongoDB.[/]")
                        continue
                    
                    text = extract_text_for_embedding(doc)
                    if not text.strip():
                        console.print(f"[yellow]Listing {listing_id_str} không có text để nhúng.[/]")
                        continue
                    
                    # Tính vector & Push lên Qdrant
                    vector = embed_model.encode(text, normalize_embeddings=True).tolist()
                    payload = build_payload(doc)

                    qclient.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=[PointStruct(id=point_id, vector=vector, payload=payload)]
                    )
                    console.print(f"✅ Đã upsert listing [cyan]{listing_id_str}[/] vào Qdrant thành công!")

            except json.JSONDecodeError:
                console.print("[red]Lỗi: Message không phải định dạng JSON hợp lệ.[/]")
            except Exception as e:
                console.print(f"[red]Lỗi xử lý message: {e}[/]")

    finally:
        consumer.close()
        console.print("[bold green]Đã đóng Kafka Consumer.[/]")

if __name__ == "__main__":
    main()
