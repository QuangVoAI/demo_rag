"""
Đồng bộ dữ liệu từ MongoDB Atlas sang Qdrant Vector Database.
Đọc danh sách 'rooms' từ Mongo, tính toán embedding vector bằng bge-m3, 
và đẩy (upsert) toàn bộ vào Qdrant.
"""
import sys
from pathlib import Path
from typing import Any

# Đưa thư mục python/ vào sys.path để import các module của project
sys.path.append(str(Path(__file__).parent.parent))

from pymongo import MongoClient
from qdrant_client.http.models import PointStruct, PointIdsList

from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_ROOMS_COLLECTION,
    QDRANT_URL,
    QDRANT_ROOMS_COLLECTION,
)
from agents.model_registry import get_embed_model
from utils.console import console
from room_assistant.schemas import canonical_room_id, normalize_room
from retrieval.qdrant_client import QdrantWrapper


def extract_text_for_embedding(doc: dict[str, Any]) -> str:
    """Trích xuất text để tạo vector từ document Mongo."""
    if doc.get("embedding_text"):
        return str(doc["embedding_text"])
    
    parts = []
    metadata = doc.get("metadata") or {}
    if "house_name" in metadata:
        parts.append(f"Nhà: {metadata['house_name']}")
    if "district_name" in metadata:
        parts.append(f"Quận/Huyện: {metadata['district_name']}")
    if doc.get("house_remark"):
        parts.append(str(doc["house_remark"]))
    if doc.get("tien_ich_xq"):
        parts.append(str(doc["tien_ich_xq"]))
    
    return " ".join(parts)


def build_payload(doc: dict[str, Any]) -> dict[str, Any]:
    """Chuyển đổi dữ liệu Mongo thành Payload hợp lệ cho Qdrant."""
    normalized = normalize_room(doc)
    if not normalized:
        normalized = dict(doc)

    room_id = canonical_room_id(doc)
    payload = dict(normalized)

    raw_id = doc.get("_id")
    if raw_id is not None:
        payload["mongo_id"] = str(raw_id)
    payload.pop("_id", None)
    payload["room_id"] = room_id

    # Ép kiểu ObjectId và Datetime trong dict
    import datetime
    for key, value in list(payload.items()):
        if hasattr(value, "__class__") and value.__class__.__name__ == "ObjectId":
            payload[key] = str(value)
        elif isinstance(value, datetime.datetime):
            payload[key] = value.isoformat()

    payload["chunk_type"] = "room_summary"
    return payload


def room_point_id_for_doc(doc: dict[str, Any], chunk_type: str = "room_summary") -> str:
    return QdrantWrapper.room_point_id(canonical_room_id(doc), chunk_type=chunk_type)


def _format_datetime(val: Any) -> str | None:
    if val is None:
        return None
    import datetime
    if isinstance(val, datetime.datetime):
        return val.isoformat()
    if isinstance(val, dict) and "$date" in val:
        d = val["$date"]
        if isinstance(d, (int, float)):
            return datetime.datetime.fromtimestamp(d / 1000.0, tz=datetime.timezone.utc).isoformat()
        return str(d)
    return str(val)


def main():
    if not MONGODB_URI or "cluster0.xxx" in MONGODB_URI:
        console.print("[bold red]LỖI: Chưa cấu hình MONGODB_URI trong file .env![/]")
        console.print("Vui lòng mở file [bold].env[/] và cập nhật link kết nối MongoDB Atlas của bạn.")
        sys.exit(1)

    console.print(f"[bold green]1. Kết nối MongoDB: {MONGODB_URI.split('@')[-1]}...[/]")
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DATABASE]
    collection = db[MONGODB_ROOMS_COLLECTION]
    
    total_docs = collection.count_documents({})
    console.print(f"   Đã tìm thấy [bold blue]{total_docs}[/] documents phòng trọ trong collection '{MONGODB_ROOMS_COLLECTION}'.")
    
    if total_docs == 0:
        console.print("   Không có data để index. Dừng chương trình.")
        sys.exit(0)

    # 1. Load embedding model
    console.print("[bold green]2. Đang nạp mô hình AI Embedding (BAAI/bge-m3)...[/]")
    embed_model = get_embed_model()
    
    # 2. Connect to Qdrant
    console.print(f"[bold green]3. Kết nối Qdrant tại {QDRANT_URL}...[/]")
    from qdrant_client.http.models import SparseVector
    
    force_recreate = "--force" in sys.argv
    if force_recreate:
        console.print("[bold yellow]⚠️ Phát hiện tham số --force. Sẽ xóa collection cũ và tạo lại để index toàn bộ từ đầu![/]")
        
    wrapper = QdrantWrapper(url=QDRANT_URL, collection_name=QDRANT_ROOMS_COLLECTION)
    wrapper.create_collection(recreate=force_recreate)
    qclient = wrapper.client

    # Lấy metadata hiện tại trong Qdrant để hỗ trợ checkpoint/resume
    existing_meta = {}
    try:
        console.print("[dim]  Đang đồng bộ danh sách phòng đã có trong Qdrant để làm checkpoint...[/]")
        next_page_offset = None
        while True:
            records, next_page_offset = qclient.scroll(
                collection_name=QDRANT_ROOMS_COLLECTION,
                limit=1000,
                offset=next_page_offset,
                with_payload=["updated_at", "room_id"],
                with_vectors=False
            )
            for r in records:
                if r.payload:
                    existing_meta[str(r.id)] = r.payload.get("updated_at")
            if next_page_offset is None:
                break
        console.print(f"   Tìm thấy [bold blue]{len(existing_meta)}[/] phòng đã được index trước đó.")
    except Exception as e:
        console.print(f"[yellow]  Không thể đọc checkpoint từ Qdrant: {e}. Tiến hành chạy mới hoàn toàn.[/]")

    # 3. Chạy từng batch
    BATCH_SIZE = 128
    cursor = collection.find({})
    
    current_batch_docs = []
    processed_count = 0
    skipped_count = 0
    new_or_updated_count = 0
    mongo_point_ids = set()
    
    console.print("[bold green]4. Bắt đầu tính toán Vector & đẩy vào Qdrant...[/]")
    
    def process_and_upsert_batch(batch_docs):
        if not batch_docs:
            return
        
        # Trích xuất text cho từng doc
        batch_data = []
        for doc in batch_docs:
            text = extract_text_for_embedding(doc)
            if text.strip():
                batch_data.append((doc, text))
                
        if not batch_data:
            return
            
        # Batch encode dense vectors
        texts = [item[1] for item in batch_data]
        dense_vectors = embed_model.encode(
            texts, 
            batch_size=len(texts), 
            show_progress_bar=False, 
            normalize_embeddings=True
        ).tolist()
        
        batch_points = []
        for idx, (doc, text) in enumerate(batch_data):
            vector_dense = dense_vectors[idx]
            
            # Tạo vector sparse
            sparse_indices, sparse_values = wrapper._text_to_sparse(text)
            payload = build_payload(doc)
            
            point_id = room_point_id_for_doc(doc)
            
            batch_points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": vector_dense,
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        )
                    },
                    payload=payload
                )
            )
            
        # Đẩy lên Qdrant
        qclient.upsert(
            collection_name=QDRANT_ROOMS_COLLECTION,
            points=batch_points
        )

    for doc in cursor:
        point_id = room_point_id_for_doc(doc)
        mongo_point_ids.add(point_id)
        
        # Kiểm tra updated_at để quyết định có skip hay không
        mongo_updated_at_str = _format_datetime(doc.get("updated_at"))
        qdrant_updated_at_str = _format_datetime(existing_meta.get(point_id)) if point_id in existing_meta else None
        
        if point_id in existing_meta and mongo_updated_at_str == qdrant_updated_at_str:
            skipped_count += 1
            processed_count += 1
            if processed_count % 500 == 0 or processed_count == total_docs:
                console.print(f"   → Đã quét [cyan]{processed_count}/{total_docs}[] phòng (Bỏ qua [green]{skipped_count}[/], Cập nhật/Mới [blue]{new_or_updated_count}[/])...")
            continue
            
        current_batch_docs.append(doc)
        new_or_updated_count += 1
        
        if len(current_batch_docs) >= BATCH_SIZE:
            process_and_upsert_batch(current_batch_docs)
            processed_count += len(current_batch_docs)
            console.print(f"   → Đã index [cyan]{processed_count}/{total_docs}[] phòng (Bỏ qua [green]{skipped_count}[/], Cập nhật/Mới [blue]{new_or_updated_count}[/])...")
            current_batch_docs = []
            
    # Xử lý nốt batch cuối
    if current_batch_docs:
        process_and_upsert_batch(current_batch_docs)
        processed_count += len(current_batch_docs)
        console.print(f"   → Đã index [cyan]{processed_count}/{total_docs}[] phòng (Bỏ qua [green]{skipped_count}[/], Cập nhật/Mới [blue]{new_or_updated_count}[/])...")

    # Dọn dẹp các phòng đã bị xóa khỏi MongoDB
    deleted_ids = set(existing_meta.keys()) - mongo_point_ids
    if deleted_ids:
        console.print(f"\n[bold yellow]5. Phát hiện {len(deleted_ids)} phòng đã bị xóa khỏi MongoDB. Đang dọn dẹp khỏi Qdrant...[/]")
        deleted_ids_list = list(deleted_ids)
        for i in range(0, len(deleted_ids_list), 250):
            batch_to_delete = deleted_ids_list[i:i+250]
            qclient.delete(
                collection_name=QDRANT_ROOMS_COLLECTION,
                points_selector=PointIdsList(points=batch_to_delete)
            )
        console.print(f"   ✅ Đã dọn dẹp thành công [red]{len(deleted_ids)}[/] phòng thừa khỏi Qdrant.")
    else:
        console.print("\n[bold green]5. Không phát hiện phòng nào bị xóa khỏi MongoDB.[/]")

    console.print("\n[bold green]🎉 HOÀN TẤT ĐỒNG BỘ DỮ LIỆU TỪ MONGO SANG QDRANT![/]")


if __name__ == "__main__":
    main()
