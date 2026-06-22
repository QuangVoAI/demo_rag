"""
Script test Kafka Producer: Gửi thử sự kiện 'upsert' vào Redpanda
để kích hoạt Kafka Indexer (Consumer).
"""
import sys
import json
from pathlib import Path

# Đưa thư mục python/ vào sys.path
sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Producer
from config import KAFKA_BROKERS, LISTING_CHANGED_TOPIC
from utils.console import console

def delivery_report(err, msg):
    """ Callback khi Kafka gửi tin nhắn xong. """
    if err is not None:
        console.print(f"[red]Gửi thất bại: {err}[/]")
    else:
        console.print(f"[green]Đã gửi sự kiện tới {msg.topic()} [{msg.partition()}][/]")

def main():
    if len(sys.argv) < 2:
        console.print("[yellow]Hướng dẫn sử dụng:[/]")
        console.print("  python test_kafka_producer.py <MONOGO_OBJECT_ID> [action]")
        console.print("\n[yellow]Ví dụ:[/]")
        console.print("  python test_kafka_producer.py 6a38c129041de32cdde5acd3 upsert")
        console.print("  python test_kafka_producer.py 6a38c129041de32cdde5acd3 delete")
        sys.exit(1)

    listing_id = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "upsert"

    producer = Producer({'bootstrap.servers': KAFKA_BROKERS})

    # Data mô phỏng Crawler gửi
    event_data = {
        "action": action,
        "listing_id": listing_id
    }

    console.print(f"Đang gửi sự kiện [bold cyan]{action}[/] cho ID: [bold yellow]{listing_id}[/] tới Kafka...")
    
    producer.produce(
        LISTING_CHANGED_TOPIC,
        json.dumps(event_data).encode('utf-8'),
        callback=delivery_report
    )
    
    # Chờ đẩy dữ liệu qua network
    producer.flush()

if __name__ == "__main__":
    main()
