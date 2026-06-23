import sys
import json
from pathlib import Path

# Thêm root dự án vào sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Nhập KAFKA_BROKERS từ file python/config.py thực thụ
try:
    from config import KAFKA_BROKERS
except ImportError:
    KAFKA_BROKERS = "localhost:9092"

# ─── Kafka Broker ────────────────────────────────────────────
BROKERS = KAFKA_BROKERS

# ─── Consumer Groups ────────────────────────────────────────
GROUP_INGESTION = "python-ingestion-worker"
GROUP_INDEXER = "python-llamaindex-indexer"
GROUP_ROOM_INDEXER = "qdrant-room-indexer-v1"
GROUP_QUERY = "python-query-processor"

# ─── Topics ──────────────────────────────────────────────────
TOPIC_PAPER_UPLOADED = "paper.uploaded"
TOPIC_DOCUMENT_CHUNKS = "document.chunks"
TOPIC_DOCUMENT_INDEXED = "document.indexed"
TOPIC_QUERY_REQUEST = "query.request"
TOPIC_QUERY_RESPONSE = "query.response"
TOPIC_ROOM_CHANGED = "room.changed"
TOPIC_ROOM_INDEX_DLQ = "room.index.dlq"

ALL_TOPICS = [
    TOPIC_PAPER_UPLOADED,
    TOPIC_DOCUMENT_CHUNKS,
    TOPIC_DOCUMENT_INDEXED,
    TOPIC_QUERY_REQUEST,
    TOPIC_QUERY_RESPONSE,
    TOPIC_ROOM_CHANGED,
    TOPIC_ROOM_INDEX_DLQ,
]

# ─── Serialization Helpers ───────────────────────────────────

def serialize(data: dict) -> bytes:
    """Serialize dict → JSON bytes cho Kafka message."""
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def deserialize(raw: bytes) -> dict:
    """Deserialize Kafka message bytes → dict."""
    return json.loads(raw.decode("utf-8"))


def ensure_topics_exist():
    """Tạo tất cả Kafka topics nếu chưa tồn tại."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": BROKERS})

    try:
        # Kiểm tra topics hiện tại
        existing = set(admin.list_topics(timeout=10).topics.keys())

        new_topics = []
        for topic in ALL_TOPICS:
            if topic not in existing:
                new_topics.append(
                    NewTopic(
                        topic,
                        num_partitions=3,
                        replication_factor=1,
                    )
                )

        if new_topics:
            futures = admin.create_topics(new_topics)
            for topic, future in futures.items():
                try:
                    future.result()
                    print(f"✅ Created topic: {topic}")
                except Exception as e:
                    print(f"⚠️ Topic {topic}: {e}")
        else:
            print("✅ All topics already exist")
    except Exception as e:
        print(f"❌ Error connecting to Kafka: {e}")
        print("Giả định bạn chưa bật Docker. Hãy chạy 'docker-compose up -d' trước.")

if __name__ == "__main__":
    print(f"📡 Kafka brokers: {BROKERS}")
    ensure_topics_exist()
