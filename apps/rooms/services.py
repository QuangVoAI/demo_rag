import os
from urllib.parse import parse_qsl, urlsplit

import pymongo
from django.conf import settings
from bson import ObjectId

_mongo_client = None


def _should_enable_mongo_tls_ca(uri: str) -> bool:
    normalized = str(uri or "").strip().lower()
    if normalized.startswith("mongodb+srv://"):
        return True
    try:
        query = dict(parse_qsl(urlsplit(normalized).query, keep_blank_values=True))
    except Exception:
        query = {}
    return query.get("tls") == "true" or query.get("ssl") == "true"


def _build_mongo_client_options(uri: str) -> dict:
    options = {
        "serverSelectionTimeoutMS": getattr(settings, "MONGODB_SERVER_SELECTION_TIMEOUT_MS", 3000),
        "connectTimeoutMS": getattr(settings, "MONGODB_CONNECT_TIMEOUT_MS", 3000),
    }
    tls_ca_file = os.getenv("MONGODB_TLS_CA_FILE", "").strip()
    if tls_ca_file:
        options["tlsCAFile"] = tls_ca_file
        return options
    if _should_enable_mongo_tls_ca(uri):
        try:
            import certifi

            options["tlsCAFile"] = certifi.where()
        except Exception:
            pass
    return options

def get_mongodb_client():
    global _mongo_client
    if _mongo_client is None:
        uri = getattr(settings, 'MONGODB_URI', 'mongodb://localhost:27017/')
        _mongo_client = pymongo.MongoClient(uri, **_build_mongo_client_options(uri))
    return _mongo_client

def get_mongodb_db():
    client = get_mongodb_client()
    db_name = getattr(settings, 'MONGODB_DB_NAME', 'demo_rag')
    return client[db_name]

# Helper collection accessors
def get_collection(name):
    db = get_mongodb_db()
    return db[name]

def get_properties_collection():
    return get_collection("properties")

def get_rooms_collection():
    return get_collection("rooms")

def get_users_collection():
    return get_collection("users")

def get_sessions_collection():
    return get_collection("sessions")

def get_bookings_collection():
    return get_collection("bookings")

def get_payments_collection():
    return get_collection("payments")

def get_vouchers_collection():
    return get_collection("vouchers")

def get_user_voucher_logs_collection():
    return get_collection("user_voucher_logs")

def get_reviews_collection():
    return get_collection("reviews")

def get_comments_collection():
    return get_collection("comments")

def get_support_tickets_collection():
    return get_collection("support_tickets")

def get_consignments_collection():
    return get_collection("consignments")

def get_jobs_collection():
    return get_collection("jobs")

def get_job_applications_collection():
    return get_collection("job_applications")

def get_faqs_collection():
    return get_collection("faqs")

def get_chat_logs_collection():
    return get_collection("chat_logs")

def get_media_assets_collection():
    return get_collection("media_assets")

def get_crawl_jobs_collection():
    return get_collection("crawl_jobs")

def upsert_listing(listing_data, job_id=None):
    from apps.crawler.normalizers import build_embedding_text, build_search_text
    
    properties_col = get_properties_collection()
    rooms_col = get_rooms_collection()
    
    source_url = listing_data.get('source', {}).get('url')
    if not source_url:
        raise ValueError("listing_data must have 'source.url'")
        
    # 1. Extract property-level fields
    property_fields = {
        "source": listing_data.get("source"),
        "address": listing_data.get("address"),
        "amenities": listing_data.get("amenities", []),
        "fees": listing_data.get("fees", {}),
        "rules": listing_data.get("rules", {}),
        "media": listing_data.get("media", {}),
        "nearby_places": listing_data.get("nearby_places", []),
        "property_info": {
            "total_room_count": listing_data.get("property_info", {}).get("total_room_count"),
            "floor_position": listing_data.get("property_info", {}).get("floor_position"),
        }
    }
    if job_id:
        property_fields["crawl_job_id"] = ObjectId(job_id) if isinstance(job_id, str) else job_id
        
    # Landlord assignment
    landlord_id = listing_data.get("landlord_id")
    if landlord_id:
        property_fields["landlord_id"] = landlord_id
        
    # Upsert property
    prop_query = {"source.url": source_url}
    properties_col.update_one(prop_query, {"$set": property_fields}, upsert=True)
    property_doc = properties_col.find_one(prop_query)
    property_id = property_doc["_id"]
    
    # 2. Extract available rooms
    available_rooms = listing_data.get("available_rooms", [])
    if not available_rooms:
        # Create a single default room unit
        room_code = listing_data.get("source", {}).get("listing_code") or "P.Chung"
        available_rooms = [{
            "room_code": room_code,
            "price": listing_data.get("price", {}).get("min") or 0
        }]
        
    # 3. Insert rooms referencing the property_id
    results = []
    for room in available_rooms:
        room_code = room.get("room_code")
        price_val = room.get("price") or 0
        room_id = f"{source_url}#{room_code}"
        
        room_title = f"{listing_data.get('title', 'Phòng')} {room_code}"
        room_price_data = {
            "min": price_val,
            "max": price_val,
            "currency": "VND",
            "period": "month",
            "display_text": f"{price_val:,}".replace(",", ".") if price_val else "Liên hệ"
        }
        
        room_listing = {
            "room_id": room_id,
            "house_id": str(property_id),
            "property_id": property_id,
            "room_code": room_code,
            "category": listing_data.get("category", "phong_tro"),
            "metadata": {
                "status_code": "0",
                "price": price_val,
                "house_name": listing_data.get("title", "Phòng"),
                "room_code": room_code,
                "ward_name": listing_data.get("address", {}).get("ward"),
                "district_name": listing_data.get("address", {}).get("district"),
                "province_name": listing_data.get("address", {}).get("city"),
            },
            "title": room_title,
            "description": listing_data.get("description", ""),
            "summary": f"{room_title} tại {listing_data.get('address', {}).get('district', '')}, giá {room_price_data['display_text']} VND.",
            "price": room_price_data,
            "property_info": {
                "area_m2": listing_data.get("property_info", {}).get("area_m2"),
                "floor_position": listing_data.get("property_info", {}).get("floor_position"),
                "available_room_count": len(available_rooms)
            },
            "audience": listing_data.get("audience", {}),
            "tags": listing_data.get("tags", []),
            "status": "active",
            "source": {
                "site": "nhatrovn",
                "url": source_url,
                "listing_code": room_code,
                "crawl_time": listing_data.get("source", {}).get("crawl_time")
            }
        }
        if job_id:
            room_listing["crawl_job_id"] = ObjectId(job_id) if isinstance(job_id, str) else job_id
            
        # Re-build RAG fields using the full property attributes combined with listing specifics
        # We temporarily mock properties structure for RAG functions
        mock_full_listing = {**room_listing, **property_fields}
        room_listing["embedding_text"] = build_embedding_text(mock_full_listing)
        room_listing["search_text"] = build_search_text(mock_full_listing)
        
        # Upsert room based on its source URL and room code.
        res = rooms_col.update_one(
            {"room_id": room_id},
            {"$set": room_listing},
            upsert=True
        )
        results.append(res)
        
    return results

