from __future__ import annotations

from typing import Any
from datetime import datetime
import os
import sys

from bson import ObjectId
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from django.conf import settings

# Add python path for RAG room assistant imports
python_path = os.path.join(settings.BASE_DIR, 'python')
if python_path not in sys.path:
    sys.path.insert(0, python_path)

from agents.graph import run_streaming
from asgiref.sync import async_to_sync

# Force the workflow to re-initialize its MongoDB repository singleton using
# the environment variables that Django loaded (avoids Django's 'config' package
# shadowing the python/config.py module during the first import).
try:
    import room_assistant.workflow as _wf
    _wf._room_repository = None  # will be lazily re-created with correct env vars
except Exception:
    pass

from .services import get_bookings_collection, get_collection


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


def _load_rooms() -> list[dict[str, Any]]:
    try:
        docs = list(
            _get_rooms_collection()
            .find({"metadata.status_code": "0"})
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
    query: dict[str, Any] = {"metadata.status_code": "0"}
    
    if category_slug:
        cat_map = _get_category_slug_map()
        db_cat = cat_map.get(category_slug)
        if db_cat:
            query["embedding_text"] = {"$regex": db_cat.replace("_", " "), "$options": "i"}
            
    if price_range:
        parts = price_range.split("-")
        if len(parts) == 2:
            try:
                min_price = float(parts[0]) * 1_000_000
                max_price = float(parts[1]) * 1_000_000
                query["metadata.price"] = {"$gte": min_price, "$lte": max_price}
            except ValueError:
                pass

    if search_query:
        query["$or"] = [
            {"metadata.house_name": {"$regex": search_query, "$options": "i"}},
            {"metadata.room_code": {"$regex": search_query, "$options": "i"}},
            {"embedding_text": {"$regex": search_query, "$options": "i"}},
        ]
    
    if city_slug:
        city_map = _get_city_slug_map()
        db_city = city_map.get(city_slug)
        if db_city:
            query["metadata.province_name"] = db_city
            
    if district_slug:
        dist_map = _get_district_slug_map()
        db_dist = dist_map.get(district_slug)
        if db_dist:
            query["metadata.district_name"] = db_dist
            
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
            .limit(60)
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

    import hashlib
    img_idx = int(hashlib.md5(room_id.encode("utf-8")).hexdigest(), 16) % len(FALLBACK_IMAGES)
    fallback_image = FALLBACK_IMAGES[img_idx]
    
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
        "image": fallback_image,
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


@require_POST
def api_chat(request):
    import json
    try:
        data = json.loads(request.body)
        message = data.get("message", "")
    except Exception:
        message = request.POST.get("message", "")

    if not message:
        return JsonResponse({"success": False, "message": "Vui lòng nhập tin nhắn."})

    # Retrieve history and session ID from Django session
    session_id = request.session.session_key
    if not session_id:
        request.session.save()
        session_id = request.session.session_key

    history = request.session.get("chat_history", [])
    
    # Run the RAG workflow synchronously using async_to_sync
    try:
        response_dict = async_to_sync(run_streaming)(
            question=message,
            history=history,
            session_id=session_id
        )
    except Exception as exc:
        return JsonResponse({
            "success": False,
            "message": f"Lỗi hệ thống trợ lý ảo: {str(exc)}"
        })

    reply = response_dict.get("answer") or "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi."
    
    # Update history in session
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    request.session["chat_history"] = history[-8:]
    
    # Process rooms returned by RAG
    rooms_data = []
    for r in response_dict.get("rooms", []):
        room_id = r.get("room_id")
        if not room_id:
            continue

        try:
            rooms_col = _get_rooms_collection()
            if len(room_id) == 24:
                room_doc = rooms_col.find_one({"_id": ObjectId(room_id)})
            else:
                room_doc = None
            if not room_doc:
                room_doc = rooms_col.find_one({"room_id": room_id})
            norm = _normalize_room_from_rooms_collection(room_doc) if room_doc else None
        except Exception:
            norm = None

        if norm:
            rooms_data.append({
                "id": norm["id"],
                "room_id": norm.get("room_id", norm["id"]),
                "house_id": norm.get("house_id", ""),
                "title": norm["title"],
                "category": norm["category"],
                "price_text": norm["price_text"],
                "address": norm["address"],
                "image": norm["image"],
                "area_text": norm["area_text"],
                "district": norm.get("district", ""),
                "city": norm.get("city", ""),
                "available_room_count": norm.get("available_room_count"),
                "status_text": norm.get("status_text", ""),
                "amenities": norm.get("amenities", [])[:4],
            })
            
    follow_up_chips = response_dict.get("suggested_questions") or [
        "Tìm phòng dưới 5 triệu ở Bình Thạnh",
        "Có gác lửng",
        "Gần trung tâm",
        "Cho nuôi thú cưng"
    ]
    
    # Construct filters from constraints
    state = response_dict.get("session_state", {}) or {}
    constraints = state.get("constraints", {}) or {}
    budget = constraints.get("budget", {}) or {}
    location = constraints.get("location", {}) or {}
    
    ui_filters = {
        "category": None,
        "district": location.get("districts", [""])[0] if location.get("districts") else "",
        "price_max": budget.get("max"),
        "amenities": constraints.get("amenities_required", []) + constraints.get("amenities_preferred", []),
    }
        
    return JsonResponse({
        "success": True,
        "reply": reply,
        "rooms": rooms_data,
        "follow_ups": follow_up_chips,
        "filters": ui_filters
    })
