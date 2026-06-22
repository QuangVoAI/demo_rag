"""
Đồng bộ dữ liệu từ MongoDB Atlas sang Qdrant Vector Database.
Đọc danh sách 'listings' từ Mongo, tính toán embedding vector bằng bge-m3, 
và đẩy (upsert) toàn bộ vào Qdrant.
"""
import sys
import uuid
from pathlib import Path
from typing import Any

# Đưa thư mục python/ vào sys.path để import các module của project
sys.path.append(str(Path(__file__).parent.parent))

from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_LISTINGS_COLLECTION,
    QDRANT_URL,
    QDRANT_COLLECTION,
    EMBEDDING_DIM
)
from agents.model_registry import get_embed_model
from utils.console import console

def extract_text_for_embedding(doc: dict[str, Any]) -> str:
    """Trích xuất text để tạo vector từ document Mongo."""
    # Ưu tiên trường embedding_text có sẵn
    if doc.get("embedding_text"):
        return str(doc["embedding_text"])
    
    parts = []
    if "category" in doc:
        parts.append(f"Loại: {doc['category']}")
    if "description" in doc:
        parts.append(str(doc["description"]))
    if "summary" in doc:
        parts.append(str(doc["summary"]))
    
    return " ".join(parts)

def build_payload(doc: dict[str, Any]) -> dict[str, Any]:
    """Chuyển đổi dữ liệu Mongo thành Payload hợp lệ cho Qdrant."""
    payload = dict(doc)
    
    # Qdrant không nhận ObjectId của Mongo, phải ép kiểu về string
    if "_id" in payload:
        payload["listing_id"] = str(payload["_id"])
        del payload["_id"]
    if "property_id" in payload:
        payload["property_id"] = str(payload["property_id"])
    if "crawl_job_id" in payload:
        payload["crawl_job_id"] = str(payload["crawl_job_id"])
    
    # Ép kiểu các _id trong object lồng nhau (nếu có)
    for key, value in payload.items():
        if hasattr(value, "__class__") and value.__class__.__name__ == "ObjectId":
            payload[key] = str(value)
            
    return payload

def main():
    if not MONGODB_URI or "cluster0.xxx" in MONGODB_URI:
        console.print("[bold red]LỖI: Chưa cấu hình MONGODB_URI trong file .env![/]")
        console.print("Vui lòng mở file [bold].env[/] và cập nhật link kết nối MongoDB Atlas của bạn.")
        sys.exit(1)

    console.print(f"[bold green]1. Kết nối MongoDB: {MONGODB_URI.split('@')[-1]}...[/]")
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DATABASE]
    collection = db[MONGODB_LISTINGS_COLLECTION]
    
    total_docs = collection.count_documents({})
    console.print(f"   Đã tìm thấy [bold blue]{total_docs}[/] documents phòng trọ.")
    
    if total_docs == 0:
        console.print("   Không có data để index. Dừng chương trình.")
        sys.exit(0)

    # 1. Load embedding model
    console.print("[bold green]2. Đang nạp mô hình AI Embedding (BAAI/bge-m3)...[/]")
    embed_model = get_embed_model()
    
    # 2. Connect to Qdrant
    console.print(f"[bold green]3. Kết nối Qdrant tại {QDRANT_URL}...[/]")
    qclient = QdrantClient(url=QDRANT_URL)
    
    # Kiểm tra/tạo collection Qdrant
    collections_response = qclient.get_collections()
    collection_names = [c.name for c in collections_response.collections]
    
    if QDRANT_COLLECTION not in collection_names:
        console.print(f"   Tạo mới collection [bold blue]{QDRANT_COLLECTION}[/] (DIM={EMBEDDING_DIM})...")
        qclient.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
    else:
        console.print(f"   Collection [bold blue]{QDRANT_COLLECTION}[/] đã tồn tại.")

    # 3. Chạy từng batch
    BATCH_SIZE = 50
    cursor = collection.find({})
    
    batch_points = []
    processed_count = 0
    
    console.print("[bold green]4. Bắt đầu tính toán Vector & đẩy vào Qdrant...[/]")
    
    for doc in cursor:
        text = extract_text_for_embedding(doc)
        if not text.strip():
            continue
            
        # Tạo vector
        vector = embed_model.encode(text, normalize_embeddings=True).tolist()
        payload = build_payload(doc)
        
        # UUID5 hash từ Mongo ObjectId để Qdrant Point ID luôn cố định
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(doc["_id"])))
        
        batch_points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )
        
        # Khi batch đủ lớn thì upsert
        if len(batch_points) >= BATCH_SIZE:
            qclient.upsert(
                collection_name=QDRANT_COLLECTION,
                points=batch_points
            )
            processed_count += len(batch_points)
            console.print(f"   → Đã index [cyan]{processed_count}/{total_docs}[/] listings...")
            batch_points = []
            
    # Xử lý nốt batch cuối
    if batch_points:
        qclient.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch_points
        )
        processed_count += len(batch_points)
        console.print(f"   → Đã index [cyan]{processed_count}/{total_docs}[/] listings...")

    console.print("\n[bold green]🎉 HOÀN TẤT ĐỒNG BỘ DỮ LIỆU TỪ MONGO SANG QDRANT![/]")

if __name__ == "__main__":
    main()
