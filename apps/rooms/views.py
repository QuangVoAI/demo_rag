from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any

from bson import ObjectId
from django.core.cache import cache, caches
from django.http import Http404, HttpResponseNotAllowed, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.conf import settings

# Add python path for RAG room assistant imports
python_path = os.path.join(settings.BASE_DIR, 'python')
if python_path not in sys.path:
    sys.path.insert(0, python_path)

from agents.graph import run_streaming
from asgiref.sync import async_to_sync

from .services import get_bookings_collection, get_collection


logger = logging.getLogger(__name__)


FALLBACK_IMAGES: tuple[str, ...] = (
    "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=1200&q=80",
    "https://images.unsplash.com/photo-1484154218962-a197022b5858?auto=format&fit=crop&w=1200&q=80",
    "https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=1200&q=80",
)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_rooms_collection():
    return get_collection("rooms")


def _available_status_query() -> dict[str, Any]:
    return {
        "$or": [
            {"metadata.status_code": "0"},
            {"metadata.status_code": ""},
        ]
    }


def _extract_primary_image(doc: dict[str, Any] | None) -> str | None:
    if not isinstance(doc, dict):
        return None

    candidates: list[Any] = [
        doc.get("image"),
        doc.get("thumbnail"),
        doc.get("featured_image_url"),
        doc.get("cover_image"),
    ]
    media = doc.get("media") if isinstance(doc.get("media"), dict) else {}
    candidates.extend([
        media.get("cover_image"),
        (media.get("images") or [None])[0] if isinstance(media.get("images"), list) else None,
    ])
    candidates.extend([
        (doc.get("images") or [None])[0] if isinstance(doc.get("images"), list) else None,
        (doc.get("image_urls") or [None])[0] if isinstance(doc.get("image_urls"), list) else None,
    ])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _lookup_property_image(doc: dict[str, Any] | None) -> str | None:
    if not isinstance(doc, dict):
        return None

    property_ref = doc.get("property_id") or doc.get("house_id")
    if not property_ref:
        return None

    try:
        properties_col = get_collection("properties")
        query: dict[str, Any] = {"_id": property_ref}
        if isinstance(property_ref, str) and len(property_ref) == 24:
            try:
                query = {"_id": ObjectId(property_ref)}
            except Exception:
                query = {"_id": property_ref}
        property_doc = properties_col.find_one(query)
        return _extract_primary_image(property_doc)
    except Exception:
        return None


def _resolve_room_image(doc: dict[str, Any] | None, room_id: str) -> str:
    primary_image = _extract_primary_image(doc) or _lookup_property_image(doc)
    if primary_image:
        return primary_image
    img_idx = int(hashlib.md5(room_id.encode("utf-8")).hexdigest(), 16) % len(FALLBACK_IMAGES)
    return FALLBACK_IMAGES[img_idx]


def _load_rooms() -> list[dict[str, Any]]:
    try:
        docs = list(
            _get_rooms_collection()
            .find(_available_status_query())
            .sort("metadata.price", 1)
            .limit(24)
        )
    except Exception:
        docs = []

    if not docs:
        return []

    rooms = [_normalize_room_from_rooms_collection(doc) for doc in docs]
    rooms.sort(key=lambda item: item["price_value"] or 0)
    return rooms


def home(request):
    rooms = _load_rooms()
    context = {
        "featured_rooms": rooms[:6],
        "categories": [
            "Phòng trọ",
            "Căn hộ",
            "Nhà phố",
            "Mặt bằng",
            "Studio",
            "CHDV 1PN",
        ],
        "cities": [
            "TP. Hồ Chí Minh",
            "Hà Nội",
            "Đà Nẵng",
            "Bình Dương",
            "Cần Thơ",
        ],
    }
    return render(request, "rooms/home.html", context)


def _slugify_simple(text: str) -> str:
    import unicodedata
    import re
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text).strip('-')
    return text


def _get_district_slug_map() -> dict[str, str]:
    try:
        districts = _get_rooms_collection().distinct("metadata.district_name")
        return {_slugify_simple(d): d for d in districts if d}
    except Exception:
        return {}


def _get_city_slug_map() -> dict[str, str]:
    try:
        cities = _get_rooms_collection().distinct("metadata.province_name")
        return {_slugify_simple(c): c for c in cities if c}
    except Exception:
        return {}


def _get_category_slug_map() -> dict[str, str]:
    db_categories = [
        'can_ho', 'chdv_1pn', 'chdv_2pn', 'chdv_3pn', 'duplex', 
        'giuong_nam', 'giuong_nu', 'mat_bang', 'nha_pho', 'phong_tro', 
        'sleepbox_nam', 'sleepbox_nu', 'studio'
    ]
    slug_map = {}
    for cat in db_categories:
        slug_map[_slugify_simple(cat)] = cat
        display = cat.replace("_", " ")
        slug_map[_slugify_simple(display)] = cat
        
    slug_map["chdv-1-phong-ngu"] = "chdv_1pn"
    slug_map["chdv-1pn"] = "chdv_1pn"
    slug_map["giuong-nam"] = "giuong_nam"
    slug_map["giuong-nu"] = "giuong_nu"
    slug_map["sleepbox-nam"] = "sleepbox_nam"
    slug_map["sleepbox-nu"] = "sleepbox_nu"
    return slug_map


def _load_filtered_rooms(
    city_slug: str | None = None,
    district_slug: str | None = None,
    category_slug: str | None = None,
    price_range: str | None = None,
    amenity: str | None = None,
    quick_filter: str | None = None,
    search_query: str | None = None
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"$and": [_available_status_query()]}
    
    if category_slug:
        cat_map = _get_category_slug_map()
        db_cat = cat_map.get(category_slug)
        if db_cat:
            query["$and"].append({"embedding_text": {"$regex": db_cat.replace("_", " "), "$options": "i"}})
            
    if price_range:
        parts = price_range.split("-")
        if len(parts) == 2:
            try:
                min_price = float(parts[0]) * 1_000_000
                max_price = float(parts[1]) * 1_000_000
                query["$and"].append({"metadata.price": {"$gte": min_price, "$lte": max_price}})
            except ValueError:
                pass

    if search_query:
        import re
        escaped_query = re.escape(search_query)
        query["$and"].append({
            "$or": [
                {"metadata.house_name": {"$regex": escaped_query, "$options": "i"}},
                {"metadata.room_code": {"$regex": escaped_query, "$options": "i"}},
                {"embedding_text": {"$regex": escaped_query, "$options": "i"}},
            ]
        })
    
    if city_slug:
        city_map = _get_city_slug_map()
        db_city = city_map.get(city_slug)
        if db_city:
            query["$and"].append({"metadata.province_name": db_city})
            
    if district_slug:
        dist_map = _get_district_slug_map()
        db_dist = dist_map.get(district_slug)
        if db_dist:
            query["$and"].append({"metadata.district_name": db_dist})
            
    filter_features = []
    if amenity:
        filter_features.append(amenity)
    if quick_filter:
        qf_map = {
            "co-may-lanh": "may_lanh",
            "nuoi-thu-cung": "thu_cung",
            "co-gac": "gac",
            "co-ban-cong": "ban_cong",
            "co-cua-so": "cua_so",
            "gio-tu-do": "gio_tu_do"
        }
        feat = qf_map.get(quick_filter)
        if feat:
            filter_features.append(feat)
            
    feature_conditions: list[dict[str, Any]] = []
    for feat in filter_features:
        if feat == "may_lanh":
            feature_conditions.append({"embedding_text": {"$regex": "Máy lạnh\\s*:\\s*Có|điều hòa", "$options": "i"}})
        elif feat == "gac":
            feature_conditions.append({"embedding_text": {"$regex": "Gác\\s*:\\s*Có", "$options": "i"}})
        elif feat == "ban_cong":
            feature_conditions.append({"embedding_text": {"$regex": "Ban công\\s*:\\s*Có", "$options": "i"}})
        elif feat == "cua_so":
            feature_conditions.append({"embedding_text": {"$regex": "Cửa sổ\\s*:\\s*Có", "$options": "i"}})
        elif feat == "thu_cung":
            feature_conditions.append({"embedding_text": {"$regex": "Thú cưng\\s*:\\s*(Có|Cho|Được)", "$options": "i"}})
        elif feat == "gio_tu_do":
            feature_conditions.append({"embedding_text": {"$regex": "Giờ giấc\\s*:\\s*Tự do", "$options": "i"}})
            
    if feature_conditions:
        query.setdefault("$and", []).extend(feature_conditions)
    
    try:
        docs = list(
            _get_rooms_collection()
            .find(query)
            .sort("metadata.price", 1)
        )
    except Exception:
        docs = []

    rooms = [_normalize_room_from_rooms_collection(doc) for doc in docs]
    rooms.sort(key=lambda item: item["price_value"] or 0)
    return rooms


def _extract_fee_rows(text: str) -> list[dict[str, str]]:
    import re

    fee_patterns = {
        "Điện": r"(?:Điện|Giá điện)\s*:\s*([^\n\r;]+)",
        "Nước": r"(?:Nước|Giá nước)\s*:\s*([^\n\r;]+)",
        "Xe": r"(?:Xe|Gửi xe|Phí xe)\s*:\s*([^\n\r;]+)",
        "Wifi": r"(?:Wifi|Internet)\s*:\s*([^\n\r;]+)",
        "Quản lý": r"(?:Quản lý|Phí quản lý)\s*:\s*([^\n\r;]+)",
    }
    fees: list[dict[str, str]] = []
    for label, pattern in fee_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fees.append({"label": label, "value": match.group(1).strip(" -")})
    return fees


def room_list(request):
    city_slug = request.GET.get("city")
    district_slug = request.GET.get("district")
    category_slug = request.GET.get("category")
    price_range = request.GET.get("price_range")
    amenity = request.GET.get("amenity")
    quick_filter = request.GET.get("quick_filter")
    search_query = request.GET.get("q")
    
    rooms = _load_filtered_rooms(
        city_slug=city_slug,
        district_slug=district_slug,
        category_slug=category_slug,
        price_range=price_range,
        amenity=amenity,
        quick_filter=quick_filter,
        search_query=search_query
    )
    
    import json
    
    city_district_map = {}
    try:
        rooms_col = _get_rooms_collection()
        cities = [c for c in rooms_col.distinct("metadata.province_name") if c]
        if not cities:
            cities = ["TP. Hồ Chí Minh", "TP. Hà Nội", "TP. Đà Nẵng", "Tỉnh Bình Dương", "TP. Cần Thơ"]
            
        if city_slug:
            city_map = _get_city_slug_map()
            db_city = city_map.get(city_slug)
            if db_city:
                districts = [d for d in rooms_col.distinct("metadata.district_name", {"metadata.province_name": db_city}) if d]
            else:
                districts = [d for d in rooms_col.distinct("metadata.district_name") if d]
        else:
            districts = [d for d in rooms_col.distinct("metadata.district_name") if d]
            
        if not districts:
            districts = ["Quận 1", "Quận 7", "Quận 10", "Quận Bình Tân", "Thành phố Thủ Đức"]
        districts.sort()

        # Build dynamic Javascript map grouping districts by city
        pipeline = [
            {"$group": {
                "_id": "$metadata.province_name",
                "districts": {"$addToSet": "$metadata.district_name"}
            }}
        ]
        groups = list(rooms_col.aggregate(pipeline))
        for g in groups:
            city_name = g.get("_id")
            if not city_name:
                continue
            c_slug = _slugify_simple(city_name)
            d_list = []
            for d in g.get("districts", []):
                if d:
                    d_list.append({
                        "name": d,
                        "slug": _slugify_simple(d)
                    })
            d_list.sort(key=lambda x: x["name"])
            city_district_map[c_slug] = d_list
    except Exception:
        cities = ["TP. Hồ Chí Minh", "TP. Hà Nội", "TP. Đà Nẵng", "Tỉnh Bình Dương", "TP. Cần Thơ"]
        districts = ["Quận 1", "Quận 7", "Quận 10", "Quận Bình Tân", "Thành phố Thủ Đức"]
        city_district_map = {}

    context = {
        "rooms": rooms,
        "result_count": len(rooms),
        "categories": [
            "Phòng trọ",
            "Căn hộ",
            "Nhà phố",
            "Mặt bằng",
            "Giường Nam",
            "Giường Nữ",
            "Studio",
            "CHDV 1 Phòng ngủ",
        ],
        "cities": cities,
        "districts": districts,
        "filter_chips": [
            "Có máy lạnh",
            "Nuôi thú cưng",
            "Có gác",
            "Có ban công",
            "Có cửa sổ",
            "Giờ tự do",
        ],
        "current_city": city_slug,
        "current_district": district_slug,
        "current_category": category_slug,
        "current_price_range": price_range,
        "current_amenity": amenity,
        "current_quick_filter": quick_filter,
        "current_search_query": search_query,
        "city_district_json": json.dumps(city_district_map, ensure_ascii=False)
    }
    return render(request, "rooms/room_list.html", context)


def _normalize_room_from_rooms_collection(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = doc.get("metadata", {}) or {}
    room_id = doc.get("room_id") or str(doc.get("_id"))
    house_id = doc.get("house_id") or ""
    
    price_val = metadata.get("price") or 0
    if price_val:
        price_text = f"{price_val:,.0f}đ / tháng".replace(",", ".")
    else:
        price_text = "Liên hệ"
        
    embedding_text = doc.get("embedding_text") or ""
    amenities = []
    import re
    feature_matches = re.findall(r"-\s*([^:]+)\s*:\s*(?:Có|Riêng|Tự do|True|Yes|Free|Có sẵn)", embedding_text, re.IGNORECASE)
    if feature_matches:
        amenities = [item.strip().title() for item in feature_matches]
    
    if not amenities:
        tien_ich_xq = doc.get("tien_ich_xq") or ""
        if tien_ich_xq:
            if isinstance(tien_ich_xq, list):
                amenities = [str(item).strip().title() for item in tien_ich_xq]
            else:
                amenities = [item.strip().title() for item in str(tien_ich_xq).split(",") if item.strip()]

    resolved_image = _resolve_room_image(doc, room_id)
    
    area_match = re.search(r"(?:Diện tích|Diện tích sử dụng)\s*:\s*(\d+(?:\.\d+)?)\s*(?:m2|m²)", embedding_text, re.IGNORECASE)
    area_text = f"{area_match.group(1)}m2" if area_match else ""

    title = f"{metadata.get('house_name') or 'Nhà trọ'} - Phòng {metadata.get('room_code') or ''}".strip(" -")
    if not title:
        title = "Phòng trọ tiện nghi"

    ward = metadata.get("ward_name") or ""
    district = metadata.get("district_name") or ""
    city = metadata.get("province_name") or ""
    
    address_parts = [ward, district, city]
    full_address = ", ".join(p for p in address_parts if p).strip(", ")
    if not full_address:
        full_address = ""

    rules = []
    if "tự do" in embedding_text.lower():
        rules.append({"label": "Giờ giấc", "value": "Tự do"})
    if "thú cưng" in embedding_text.lower() or "chó" in embedding_text.lower() or "mèo" in embedding_text.lower():
        rules.append({"label": "Thú cưng", "value": "Cho phép"})

    return {
        "id": room_id,
        "room_id": room_id,
        "house_id": house_id,
        "property_id": house_id,
        "title": title,
        "category": doc.get("category") or "",
        "district": district,
        "city": city,
        "ward": ward,
        "address": full_address,
        "price_text": price_text,
        "price_value": price_val,
        "area_text": area_text,
        "floor_position": "",
        "available_room_count": None,
        "total_room_count": None,
        "status_text": metadata.get("status_desc") or "",
        "amenities": amenities[:8],
        "description": doc.get("house_remark") or doc.get("embedding_text") or "",
        "tags": [metadata.get("status_desc")] if metadata.get("status_desc") else [],
        "hero_badge": "",
        "image": resolved_image,
        "available_rooms": [
            {
                "room_code": metadata.get("room_code", "-"),
                "price_text": price_text,
            }
        ],
        "fees": _extract_fee_rows(embedding_text),
        "rules": rules,
    }


def room_detail(request, room_id: str):
    room = None
    try:
        rooms_col = _get_rooms_collection()
        if len(room_id) == 24:
            room_doc = rooms_col.find_one({"_id": ObjectId(room_id)})
        else:
            room_doc = None
        if not room_doc:
            room_doc = rooms_col.find_one({"room_id": room_id})
        if room_doc:
            room = _normalize_room_from_rooms_collection(room_doc)
    except Exception:
        room = None

    if room is None:
        raise Http404("Không tìm thấy phòng")

    rooms = _load_rooms()
    context = {
        "room": room,
        "related_rooms": [item for item in rooms if item["id"] != room["id"]][:3],
    }
    return render(request, "rooms/room_detail.html", context)


@require_POST
@csrf_protect
def book_viewing(request):
    try:
        col = get_bookings_collection()
        house_id = request.POST.get("house_id")
        room_id = request.POST.get("room_id")
        name = request.POST.get("name")
        phone = request.POST.get("phone")
        number_people = request.POST.get("number_people")
        number_vehicles = request.POST.get("number_vehicles")
        have_pet = request.POST.get("have_pet")
        datetime_str = request.POST.get("datetime")
        estimated_time = request.POST.get("estimated_time")
        remark = request.POST.get("remark")
        
        room_object_id = ObjectId(room_id) if room_id and len(room_id) == 24 else None
        booking_doc = {
            "tenant_name": name,
            "phone": phone,
            "property_id": ObjectId(house_id) if house_id and len(house_id) == 24 else None,
            "room_id": room_id,
            "room_object_id": room_object_id,
            "number_people": int(number_people) if number_people else 1,
            "number_vehicles": int(number_vehicles) if number_vehicles else 0,
            "have_pet": True if have_pet == "Y" else False,
            "viewing_time": datetime_str,
            "estimated_move_in": estimated_time,
            "remark": remark,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
        col.insert_one(booking_doc)
        return JsonResponse({"success": True, "message": "Đặt lịch xem phòng thành công!"})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Có lỗi xảy ra: {str(e)}"})


def _normalize_chat_role(raw_role: str | None) -> str:
    value = str(raw_role or "").strip().lower()
    if value in {"staff", "employee", "landlord", "nhan_vien", "nhanvien"}:
        return "staff"
    return "user"


def _sender_role_for_chat(role: str) -> str:
    return "user" if role == "user" else "nhan_vien"


def _normalize_phone_number(phone: str | None) -> str:
    return "".join(ch for ch in str(phone or "") if ch.isdigit())


def _extract_request_payload(request) -> tuple[dict[str, Any], str]:
    try:
        data = json.loads(request.body)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    if not data:
        data = dict(request.POST.items())
    message = str(data.get("message") or data.get("question") or "").strip()
    return data, message


def _sanitize_history(history: Any, limit: int = 10) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    sanitized: list[dict[str, str]] = []
    for item in history[-limit:]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        sanitized.append({
            "role": str(item.get("role") or "user"),
            "content": content,
        })
    return sanitized


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return value


def _serialize_rag_response(response_dict: dict[str, Any], session_id: str = "") -> dict[str, Any]:
    rooms = response_dict.get("rooms", [])
    for room in rooms:
        room_id = str(room.get("room_id") or room.get("id") or "").strip()
        if "rent_price" in room and "price_text" not in room:
            try:
                room["price_text"] = f"{int(room['rent_price']):,} VND"
            except (ValueError, TypeError):
                room["price_text"] = str(room.get("rent_price", ""))
        if "area_m2" in room and "area_text" not in room:
            room["area_text"] = f"{room['area_m2']} m²" if room["area_m2"] else ""
        if "status_desc" in room and "status_text" not in room:
            room["status_text"] = room["status_desc"]
        if "province_name" in room and "city" not in room:
            room["city"] = room["province_name"]
        if "category" not in room:
            room["category"] = "Phòng trọ"
        if "id" not in room and room_id:
            room["id"] = room_id
        if not room.get("image"):
            room_doc = None
            if room_id:
                try:
                    rooms_col = _get_rooms_collection()
                    room_doc = rooms_col.find_one({"room_id": room_id})
                    if not room_doc and len(room_id) == 24:
                        room_doc = rooms_col.find_one({"_id": ObjectId(room_id)})
                except Exception:
                    room_doc = None
            room["image"] = _resolve_room_image(room_doc or room, room_id or str(room.get("title") or "room"))
            
    payload = {
        "success": True,
        "session_id": response_dict.get("session_id") or session_id,
        "reply": response_dict.get("answer") or "",
        "answer": response_dict.get("answer") or "",
        "intent": response_dict.get("intent"),
        "session_state": response_dict.get("session_state", {}),
        "rooms": rooms,
        "cost_estimate": response_dict.get("cost_estimate"),
        "comparison": response_dict.get("comparison"),
        "follow_ups": response_dict.get("suggested_questions", []),
        "suggested_questions": response_dict.get("suggested_questions", []),
        "sources": response_dict.get("sources", []),
        "retrieval_confidence": response_dict.get("retrieval_confidence"),
        "retrieval_low_confidence": response_dict.get("retrieval_low_confidence"),
        "retrieval_feedback_retry_count": response_dict.get("retrieval_feedback_retry_count", 0),
        "retrieval_attempts": response_dict.get("retrieval_attempts", []),
        "processing_time_ms": response_dict.get("processing_time_ms", 0),
    }
    return _json_safe_value(payload)


def _rag_error_payload(
    request,
    *,
    code: str,
    message: str,
    retryable: bool,
    http_status: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "message": message,
        "request_id": getattr(request, "request_id", None),
        "correlation_id": getattr(request, "correlation_id", None),
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "http_status": http_status,
        },
    }
    if extra:
        payload.update(extra)
        payload["error"].update(extra)
    return _json_safe_value(payload)


def _json_rag_error(
    request,
    *,
    code: str,
    message: str,
    retryable: bool,
    http_status: int,
    extra: dict[str, Any] | None = None,
) -> JsonResponse:
    return JsonResponse(
        _rag_error_payload(
            request,
            code=code,
            message=message,
            retryable=retryable,
            http_status=http_status,
            extra=extra,
        ),
        status=http_status,
    )


def _is_rag_api_authorized(request) -> bool:
    required_key = str(getattr(settings, "RAG_API_KEY", "") or "").strip()
    if not required_key:
        return True
    header_key = str(request.headers.get("X-API-Key", "") or "").strip()
    auth_header = str(request.headers.get("Authorization", "") or "").strip()
    bearer_key = ""
    if auth_header.lower().startswith("bearer "):
        bearer_key = auth_header[7:].strip()
    return header_key == required_key or bearer_key == required_key


def _request_log_context(request, **extra: Any) -> dict[str, Any]:
    payload = {
        "request_id": getattr(request, "request_id", None),
        "correlation_id": getattr(request, "correlation_id", None),
        "path": getattr(request, "path", ""),
    }
    payload.update(extra)
    return payload


def _get_rag_rate_limit_cache():
    alias = str(getattr(settings, "RAG_RATE_LIMIT_CACHE_ALIAS", "default") or "default").strip() or "default"
    try:
        return caches[alias]
    except Exception:
        return cache


def _attach_rate_limit_headers(response, rate_state: dict[str, int]) -> None:
    response["X-RateLimit-Limit"] = str(rate_state["limit"])
    response["X-RateLimit-Remaining"] = str(rate_state["remaining"])
    response["X-RateLimit-Reset"] = str(rate_state["retry_after"])


def _rag_rate_limit_identity(request, session_id: str) -> str:
    header_key = str(request.headers.get("X-API-Key", "") or "").strip()
    auth_header = str(request.headers.get("Authorization", "") or "").strip()
    if auth_header.lower().startswith("bearer "):
        header_key = auth_header[7:].strip() or header_key
    if header_key:
        return "api_key:" + hashlib.sha256(header_key.encode("utf-8")).hexdigest()[:16]
    forwarded_for = str(request.headers.get("X-Forwarded-For", "") or "").split(",")[0].strip()
    remote_addr = forwarded_for or str(request.META.get("REMOTE_ADDR", "") or "").strip() or "unknown"
    if session_id:
        return f"ip_session:{remote_addr}:{session_id[:64]}"
    return f"ip:{remote_addr}"


def _check_rag_rate_limit(request, session_id: str) -> tuple[bool, dict[str, int]]:
    if not bool(getattr(settings, "RAG_RATE_LIMIT_ENABLED", True)):
        return True, {"limit": 0, "remaining": 0, "retry_after": 0, "window_seconds": 0}

    limit = max(1, int(getattr(settings, "RAG_RATE_LIMIT_MAX_REQUESTS", 30)))
    window_seconds = max(1, int(getattr(settings, "RAG_RATE_LIMIT_WINDOW_SECONDS", 60)))
    identity = _rag_rate_limit_identity(request, session_id)
    cache_key = "rag_rate_limit:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
    rate_limit_cache = _get_rag_rate_limit_cache()
    now = int(time.time())
    record = rate_limit_cache.get(cache_key) or {"count": 0, "window_started": now}
    elapsed = max(0, now - int(record.get("window_started", now)))
    if elapsed >= window_seconds:
        record = {"count": 0, "window_started": now}
        elapsed = 0

    if int(record.get("count", 0)) >= limit:
        retry_after = max(1, window_seconds - elapsed)
        return False, {
            "limit": limit,
            "remaining": 0,
            "retry_after": retry_after,
            "window_seconds": window_seconds,
        }

    record["count"] = int(record.get("count", 0)) + 1
    rate_limit_cache.set(cache_key, record, timeout=window_seconds)
    remaining = max(0, limit - record["count"])
    return True, {
        "limit": limit,
        "remaining": remaining,
        "retry_after": max(0, window_seconds - elapsed),
        "window_seconds": window_seconds,
    }


def _build_contact_id(role: str, normalized_phone: str, name: str) -> str:
    prefix = "c_staff_" if role == "staff" else "c_user_"
    if normalized_phone:
        suffix = normalized_phone[-10:]
    else:
        suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{suffix}"


def _serialize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        created_at = msg.get("created_at")
        serialized.append({
            "role": msg.get("role"),
            "content": msg.get("content"),
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
        })
    return serialized


def _list_chat_threads(contact_id: str, role: str, active_conversation_id: str | None = None) -> list[dict[str, Any]]:
    chat_history_col = get_collection("chat_history")
    try:
        docs = list(
            chat_history_col.find({"contact_id": contact_id})
            .sort([("updated_at", -1), ("created_at", -1)])
            .limit(25)
        )
    except Exception:
        docs = []

    threads: list[dict[str, Any]] = []
    for doc in docs:
        messages = doc.get("messages", [])
        last_message = messages[-1] if messages else {}
        last_timestamp = doc.get("updated_at") or last_message.get("created_at") or doc.get("created_at")
        participants = {str(msg.get("role") or "") for msg in messages}
        threads.append({
            "conversation_id": doc.get("conversation_id"),
            "title": doc.get("title") or "Đoạn chat chưa đặt tên",
            "last_message": last_message.get("content", ""),
            "last_sender_role": last_message.get("role", ""),
            "message_count": len(messages),
            "updated_at": last_timestamp.isoformat() if hasattr(last_timestamp, "isoformat") else last_timestamp,
            "has_staff_messages": "staff" in participants or "nhan_vien" in participants,
            "active": doc.get("conversation_id") == active_conversation_id,
        })

    if role == "user" and not threads and contact_id:
        conversation_id = f"conv_{contact_id}"
        threads.append({
            "conversation_id": conversation_id,
            "title": "Hỗ trợ tìm phòng",
            "last_message": "",
            "message_count": 0,
            "updated_at": None,
            "active": conversation_id == active_conversation_id,
        })

    return threads


def _load_chat_messages(conversation_id: str | None) -> list[dict[str, Any]]:
    if not conversation_id:
        return []
    try:
        chat_doc = get_collection("chat_history").find_one({"conversation_id": conversation_id})
    except Exception:
        chat_doc = None
    return list(chat_doc.get("messages", [])) if chat_doc else []


def _ensure_contacts_phone_unique_index() -> None:
    if getattr(_ensure_contacts_phone_unique_index, "_ready", False):
        return

    contacts_col = get_collection("contacts")
    try:
        for index in contacts_col.list_indexes():
            key = index.get("key", {})
            if key == {"phone_number": 1} and index.get("unique"):
                _ensure_contacts_phone_unique_index._ready = True
                return

        contacts_col.create_index(
            [("phone_number", 1)],
            name="uniq_phone_number",
            unique=True,
            sparse=True,
        )
        _ensure_contacts_phone_unique_index._ready = True
    except Exception:
        logger.exception("Failed to ensure unique phone_number index for contacts")


def _resolve_contact_identity(
    *,
    contact_id: str | None = None,
    contact_phone: str | None = None,
    contact_name: str | None = None,
    demo_role: str | None = None,
) -> dict[str, Any]:
    contacts_col = get_collection("contacts")
    _ensure_contacts_phone_unique_index()
    normalized_phone = _normalize_phone_number(contact_phone)
    normalized_role = _normalize_chat_role(demo_role)
    cleaned_name = str(contact_name or "").strip()
    contact_doc = None

    if normalized_phone:
        try:
            contact_doc = contacts_col.find_one({"phone_number": normalized_phone})
        except Exception:
            contact_doc = None

    if contact_doc is None and contact_id:
        try:
            contact_doc = contacts_col.find_one({"contact_id": contact_id})
        except Exception:
            contact_doc = None

    if contact_doc:
        resolved_contact_id = contact_doc.get("contact_id") or contact_id
        resolved_role = _normalize_chat_role(contact_doc.get("role"))
        update_fields: dict[str, Any] = {"updated_at": datetime.utcnow()}
        if cleaned_name:
            update_fields["name"] = cleaned_name
        if normalized_phone:
            update_fields["phone_number"] = normalized_phone
        if resolved_role != _normalize_chat_role(contact_doc.get("role")):
            update_fields["role"] = resolved_role
        try:
            contacts_col.update_one({"contact_id": resolved_contact_id}, {"$set": update_fields}, upsert=True)
        except Exception:
            pass
        return {
            "contact_id": resolved_contact_id,
            "contact_name": cleaned_name or contact_doc.get("name", ""),
            "contact_phone": normalized_phone or contact_doc.get("phone_number", ""),
            "role": resolved_role,
        }

    resolved_contact_id = contact_id or _build_contact_id(normalized_role, normalized_phone, cleaned_name or "guest")
    set_fields: dict[str, Any] = {
        "name": cleaned_name,
        "role": normalized_role,
        "updated_at": datetime.utcnow(),
    }
    if normalized_phone:
        set_fields["phone_number"] = normalized_phone
    try:
        contacts_col.update_one(
            {"contact_id": resolved_contact_id},
            {
                "$set": set_fields,
                "$setOnInsert": {
                    "created_at": datetime.utcnow(),
                },
            },
            upsert=True,
        )
    except Exception:
        pass

    return {
        "contact_id": resolved_contact_id,
        "contact_name": cleaned_name,
        "contact_phone": normalized_phone,
        "role": normalized_role,
    }


def _resolve_conversation_id(
    *,
    contact_id: str,
    role: str,
    requested_conversation_id: str | None,
    session_conversation_id: str | None,
    create_new: bool = False,
) -> str | None:
    if role == "user":
        return f"conv_{contact_id}"

    chat_history_col = get_collection("chat_history")
    if create_new:
        return f"conv_{contact_id}_{int(datetime.utcnow().timestamp())}"

    candidates = [requested_conversation_id, session_conversation_id]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            chat_doc = chat_history_col.find_one({"conversation_id": candidate})
        except Exception:
            chat_doc = None
        if chat_doc and chat_doc.get("contact_id") == contact_id:
            return candidate

    try:
        latest_doc = chat_history_col.find_one(
            {"contact_id": contact_id},
            sort=[("updated_at", -1), ("created_at", -1)],
        )
    except Exception:
        latest_doc = None
    return latest_doc.get("conversation_id") if latest_doc else None


def _build_rooms_data(response_dict: dict[str, Any]) -> list[dict[str, Any]]:
    rooms_data = []
    for room in response_dict.get("rooms", []):
        room_id = room.get("room_id")
        if not room_id:
            continue

        try:
            rooms_col = _get_rooms_collection()
            room_doc = None
            from bson.errors import InvalidId
            try:
                room_doc = rooms_col.find_one({"_id": ObjectId(str(room_id))})
            except (InvalidId, TypeError, ValueError):
                pass
            
            if not room_doc:
                room_doc = rooms_col.find_one({"room_id": room_id})
                
            normalized = _normalize_room_from_rooms_collection(room_doc) if room_doc else None
        except Exception:
            normalized = None

        if not normalized:
            # Fallback to room data directly from the assistant's results
            rent_price = room.get("rent_price") or 0
            normalized = {
                "id": room_id,
                "room_id": room_id,
                "house_id": room.get("house_id", ""),
                "title": room.get("title", "Phòng trọ"),
                "category": room.get("category", "Phòng Trọ"),
                "price_text": f"{rent_price:,.0f}đ / tháng".replace(",", ".") if rent_price else "Liên hệ",
                "address": room.get("address", ""),
                "image": FALLBACK_IMAGES[int(hashlib.md5(str(room_id).encode("utf-8")).hexdigest(), 16) % len(FALLBACK_IMAGES)] if room_id else FALLBACK_IMAGES[0],
                "area_text": f"{room.get('area_m2')}m2" if room.get("area_m2") else "",
                "district": room.get("district", ""),
                "city": room.get("province", ""),
                "available_room_count": 1,
                "status_text": room.get("status_desc", "Còn phòng"),
                "amenities": room.get("amenities", [])[:4],
            }

        if not normalized:
            continue

        rooms_data.append({
            "id": normalized["id"],
            "room_id": normalized.get("room_id", normalized["id"]),
            "house_id": normalized.get("house_id", ""),
            "title": normalized["title"],
            "category": normalized["category"],
            "price_text": normalized["price_text"],
            "address": normalized["address"],
            "image": normalized["image"],
            "area_text": normalized["area_text"],
            "district": normalized.get("district", ""),
            "city": normalized.get("city", ""),
            "available_room_count": normalized.get("available_room_count"),
            "status_text": normalized.get("status_text", ""),
            "amenities": normalized.get("amenities", [])[:4],
        })
    return rooms_data


def _build_ui_filters(response_dict: dict[str, Any]) -> dict[str, Any]:
    state = response_dict.get("session_state", {}) or {}
    constraints = state.get("constraints", {}) or {}
    budget = constraints.get("budget", {}) or {}
    location = constraints.get("location", {}) or {}
    return {
        "category": None,
        "district": location.get("districts", [""])[0] if location.get("districts") else "",
        "price_max": budget.get("max"),
        "amenities": constraints.get("amenities_required", []) + constraints.get("amenities_preferred", []),
    }


def _build_chat_payload(
    *,
    response_dict: dict[str, Any] | None,
    contact_id: str,
    role: str,
    conversation_id: str | None,
) -> dict[str, Any]:
    response_dict = response_dict or {}
    reply = response_dict.get("answer") or "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi."
    return {
        "success": True,
        "contact_id": contact_id,
        "conversation_id": conversation_id,
        "role": role,
        "reply": reply,
        "rooms": _build_rooms_data(response_dict),
        "follow_ups": response_dict.get("suggested_questions") or [
            "Tìm phòng dưới 5 triệu ở Bình Thạnh",
            "Có gác lửng",
            "Gần trung tâm",
            "Cho nuôi thú cưng",
        ],
        "filters": _build_ui_filters(response_dict),
        "threads": _list_chat_threads(contact_id, role, active_conversation_id=conversation_id),
    }


def _stream_chunk_to_event(chunk: str) -> dict[str, Any] | None:
    if not chunk:
        return None
    if chunk.startswith("[status:"):
        prefix, _, remainder = chunk.partition("]")
        status_spec = prefix[len("[status:"):].strip()
        stage, _, agent = status_spec.partition("|")
        return {
            "type": "status",
            "stage": stage.strip() or "working",
            "agent": agent.strip() or None,
            "message": remainder.strip() or "Đang xử lý...",
        }
    return {"type": "token", "content": chunk}


def _sse_event(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(_json_safe_value(event), ensure_ascii=False)}\n\n"


def api_chat(request):
    if request.method not in ["GET", "POST"]:
        return HttpResponseNotAllowed(["GET", "POST"])

    if request.method == "GET":
        explicit_contact_id = request.GET.get("contact_id")
        explicit_contact_phone = request.GET.get("contact_phone")
        explicit_contact_name = request.GET.get("contact_name")
        explicit_demo_role = request.GET.get("demo_role")

        requested_contact_id = explicit_contact_id
        if not requested_contact_id and not (explicit_contact_phone or explicit_contact_name):
            requested_contact_id = request.session.get("contact_id")
        if not requested_contact_id and not explicit_contact_phone:
            requested_contact_id = "c_user_001"
        identity = _resolve_contact_identity(
            contact_id=requested_contact_id,
            contact_phone=explicit_contact_phone,
            contact_name=explicit_contact_name,
            demo_role=explicit_demo_role,
        )
        contact_id = identity["contact_id"]
        role = identity["role"]
        conversation_id = _resolve_conversation_id(
            contact_id=contact_id,
            role=role,
            requested_conversation_id=request.GET.get("conversation_id"),
            session_conversation_id=request.session.get("conversation_id"),
            create_new=request.GET.get("new_chat") == "1",
        )
        request.session["contact_id"] = contact_id
        if conversation_id:
            request.session["conversation_id"] = conversation_id
        else:
            request.session.pop("conversation_id", None)
        return JsonResponse({
            "success": True,
            "contact_id": contact_id,
            "conversation_id": conversation_id,
            "role": role,
            "contact_name": identity.get("contact_name", ""),
            "contact_phone": identity.get("contact_phone", ""),
            "threads": _list_chat_threads(contact_id, role, active_conversation_id=conversation_id),
            "messages": _serialize_chat_messages(_load_chat_messages(conversation_id)),
        })

    data, message = _extract_request_payload(request)

    if not message:
        return JsonResponse({"success": False, "message": "Vui lòng nhập tin nhắn."})

    explicit_contact_id = data.get("contact_id") or request.POST.get("contact_id")
    explicit_contact_phone = data.get("contact_phone") or request.POST.get("contact_phone")
    explicit_contact_name = data.get("contact_name") or request.POST.get("contact_name")
    explicit_demo_role = data.get("demo_role") or request.POST.get("demo_role")

    requested_contact_id = explicit_contact_id
    if not requested_contact_id and not (explicit_contact_phone or explicit_contact_name):
        requested_contact_id = request.session.get("contact_id")

    identity = _resolve_contact_identity(
        contact_id=requested_contact_id,
        contact_phone=explicit_contact_phone,
        contact_name=explicit_contact_name,
        demo_role=explicit_demo_role,
    )
    contact_id = identity["contact_id"]
    role = identity["role"]
    sender_role = _sender_role_for_chat(role)
    conversation_id = _resolve_conversation_id(
        contact_id=contact_id,
        role=role,
        requested_conversation_id=data.get("conversation_id") or request.POST.get("conversation_id"),
        session_conversation_id=request.session.get("conversation_id"),
        create_new=(data.get("new_chat") == "1") or (request.POST.get("new_chat") == "1"),
    )
    if role == "staff" and not conversation_id:
        conversation_id = f"conv_{contact_id}_{int(datetime.utcnow().timestamp())}"

    request.session["contact_id"] = contact_id
    if conversation_id:
        request.session["conversation_id"] = conversation_id

    history = []
    chat_history_col = get_collection("chat_history")
    try:
        chat_doc = chat_history_col.find_one({"conversation_id": conversation_id})
        if chat_doc:
            msgs = chat_doc.get("messages", [])[-10:]
            for msg in msgs:
                history.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
    except Exception:
        history = request.session.get("chat_history", [])

    title = message[:30] + "..." if len(message) > 30 else message
    try:
        chat_doc = chat_history_col.find_one({"conversation_id": conversation_id})
        if chat_doc and chat_doc.get("title"):
            title = chat_doc["title"]
    except Exception:
        pass

    now = datetime.utcnow()
    try:
        chat_history_col.update_one(
            {"conversation_id": conversation_id},
            {
                "$setOnInsert": {
                    "title": title,
                    "created_at": now,
                },
                "$set": {
                    "contact_id": contact_id,
                    "sender_role": sender_role,
                    "updated_at": now,
                },
                "$push": {
                    "messages": {
                        "role": "user" if role == "user" else "staff",
                        "content": message,
                        "created_at": now,
                    }
                }
            },
            upsert=True
        )
    except Exception:
        logger.exception(
            "Failed to persist inbound chat message",
            extra={
                "conversation_id": conversation_id,
                "contact_id": contact_id,
                "role": role,
            },
        )

    event_queue: queue.Queue[str | None] = queue.Queue()

    def enqueue_event(event: dict[str, Any]) -> None:
        event_queue.put(_sse_event(event))

    async def stream_callback(chunk: str) -> None:
        event = _stream_chunk_to_event(chunk)
        if event:
            enqueue_event(event)

    def worker() -> None:
        try:
            response_dict = async_to_sync(run_streaming)(
                question=message,
                history=history,
                session_id=conversation_id,
                stream_callback=stream_callback,
            )
            reply = response_dict.get("answer") or "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi."
            finished_at = datetime.utcnow()
            try:
                chat_history_col.update_one(
                    {"conversation_id": conversation_id},
                    {
                        "$set": {
                            "updated_at": finished_at,
                        },
                        "$push": {
                            "messages": {
                                "role": "assistant",
                                "content": reply,
                                "created_at": finished_at,
                            }
                        },
                    },
                )
            except Exception:
                logger.exception(
                    "Failed to persist assistant chat message",
                    extra={
                        "conversation_id": conversation_id,
                        "contact_id": contact_id,
                    },
                )

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            payload = _build_chat_payload(
                response_dict=response_dict,
                contact_id=contact_id,
                role=role,
                conversation_id=conversation_id,
            )
            enqueue_event({"type": "final", "payload": payload})
        except Exception as exc:
            enqueue_event({
                "type": "error",
                "message": f"Lỗi hệ thống trợ lý ảo: {str(exc)}",
            })
        finally:
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            chunk = event_queue.get()
            if chunk is None:
                break
            yield chunk

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@require_POST
@csrf_protect
def api_rag_stream(request):
    data, message = _extract_request_payload(request)
    if not message:
        return _json_rag_error(
            request,
            code="validation_error",
            message="Vui lòng nhập câu hỏi.",
            retryable=False,
            http_status=400,
        )

    session_id = str(data.get("session_id") or data.get("conversation_id") or "").strip()
    history = _sanitize_history(data.get("history"))
    allowed, rate_state = _check_rag_rate_limit(request, session_id=session_id)
    if not allowed:
        logger.warning(
            "rag_stream_rate_limited",
            extra=_request_log_context(request, session_id=session_id, rate_limit=rate_state),
        )
        response = _json_rag_error(
            request,
            code="rate_limited",
            message="Too many requests. Please retry later.",
            retryable=True,
            http_status=200,
            extra={
                "rate_limited": True,
                "retry_after_seconds": rate_state["retry_after"],
            },
        )
        response["Retry-After"] = str(rate_state["retry_after"])
        _attach_rate_limit_headers(response, rate_state)
        return response

    event_queue: queue.Queue[str | None] = queue.Queue()

    def enqueue_event(event: dict[str, Any]) -> None:
        event_queue.put(_sse_event(event))

    async def stream_callback(chunk: str) -> None:
        event = _stream_chunk_to_event(chunk)
        if event:
            enqueue_event(event)

    def worker() -> None:
        try:
            response_dict = async_to_sync(run_streaming)(
                question=message,
                history=history,
                session_id=session_id,
                stream_callback=stream_callback,
            )
            payload = _serialize_rag_response(response_dict, session_id=session_id)
            payload["request_id"] = getattr(request, "request_id", None)
            payload["correlation_id"] = getattr(request, "correlation_id", None)
            enqueue_event({"type": "final", "payload": payload})
        except Exception:
            logger.exception("rag_stream_failed", extra=_request_log_context(request, session_id=session_id))
            enqueue_event(
                {
                    "type": "error",
                    "payload": _rag_error_payload(
                        request,
                        code="service_unavailable",
                        message="RAG service unavailable.",
                        retryable=True,
                        http_status=503,
                    ),
                }
            )
        finally:
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            chunk = event_queue.get()
            if chunk is None:
                break
            yield chunk

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    _attach_rate_limit_headers(response, rate_state)
    return response


@require_POST
@csrf_exempt
def api_rag_query(request):
    if not _is_rag_api_authorized(request):
        logger.warning("rag_query_unauthorized", extra=_request_log_context(request))
        return _json_rag_error(
            request,
            code="unauthorized",
            message="Unauthorized.",
            retryable=False,
            http_status=401,
        )

    data, message = _extract_request_payload(request)
    if not message:
        return _json_rag_error(
            request,
            code="validation_error",
            message="Vui lòng nhập câu hỏi.",
            retryable=False,
            http_status=400,
        )

    session_id = str(data.get("session_id") or data.get("conversation_id") or "").strip()
    history = _sanitize_history(data.get("history"))
    allowed, rate_state = _check_rag_rate_limit(request, session_id=session_id)
    if not allowed:
        logger.warning(
            "rag_query_rate_limited",
            extra=_request_log_context(request, session_id=session_id, rate_limit=rate_state),
        )
        response = _json_rag_error(
            request,
            code="rate_limited",
            message="Too many requests. Please retry later.",
            retryable=True,
            http_status=200,
            extra={
                "rate_limited": True,
                "retry_after_seconds": rate_state["retry_after"],
            },
        )
        response["Retry-After"] = str(rate_state["retry_after"])
        _attach_rate_limit_headers(response, rate_state)
        return response

    try:
        response_dict = async_to_sync(run_streaming)(
            question=message,
            history=history,
            session_id=session_id,
        )
    except Exception:
        logger.exception("rag_query_failed", extra=_request_log_context(request, session_id=session_id))
        response = _json_rag_error(
            request,
            code="service_unavailable",
            message="RAG service unavailable.",
            retryable=True,
            http_status=503,
        )
        _attach_rate_limit_headers(response, rate_state)
        return response

    payload = _serialize_rag_response(response_dict, session_id=session_id)
    payload["request_id"] = getattr(request, "request_id", None)
    payload["correlation_id"] = getattr(request, "correlation_id", None)
    logger.info(
        "rag_query_ok",
        extra=_request_log_context(
            request,
            session_id=payload.get("session_id"),
            intent=payload.get("intent"),
            processing_time_ms=payload.get("processing_time_ms"),
        ),
    )
    response = JsonResponse(payload)
    _attach_rate_limit_headers(response, rate_state)
    return response


def api_health(request):
    return JsonResponse({
        "success": True,
        "status": "ok",
        "service": "nhatrovn-rag",
        "api": {
            "rag_query_path": "/api/rag/query/",
            "rag_stream_path": "/api/rag/stream/",
        },
        "security": {
            "https_redirect": bool(getattr(settings, "SECURE_SSL_REDIRECT", False)),
            "rag_api_key_required": bool(getattr(settings, "RAG_API_KEY", "")),
        },
        "cache": {
            "rate_limit_cache_alias": str(getattr(settings, "RAG_RATE_LIMIT_CACHE_ALIAS", "default")),
            "default_backend": settings.CACHES["default"]["BACKEND"],
        },
    })
