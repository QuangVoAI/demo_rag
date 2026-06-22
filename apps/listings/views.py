from __future__ import annotations

from typing import Any
from datetime import datetime

from bson import ObjectId
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect

from .services import get_listings_collection, get_bookings_collection


FALLBACK_LISTINGS: list[dict[str, Any]] = [
    {
        "id": "sample-q10-102",
        "title": "Phòng thường 102",
        "category": "Phòng trọ",
        "district": "Quận 10",
        "city": "TP. Hồ Chí Minh",
        "ward": "Phường 04",
        "address": "400/xx Ngô Gia Tự, Phường 04, Quận 10, TP. Hồ Chí Minh",
        "price_text": "4.500.000đ / tháng",
        "price_value": 4500000,
        "area_text": "20m2",
        "floor_position": "Lầu 1",
        "available_room_count": 3,
        "total_room_count": 10,
        "amenities": ["Wifi", "Máy lạnh", "Thang máy", "Tủ lạnh", "Gác"],
        "description": "Gần UEH, thuận tiện đi Quận 1, Quận 3 và khu trung tâm.",
        "tags": ["Đã xác thực", "Hot", "Còn 3 phòng trống"],
        "hero_badge": "Nổi bật",
        "image": "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=1200&q=80",
        "available_rooms": [
            {"room_code": "102", "price_text": "4.500.000đ"},
            {"room_code": "103", "price_text": "4.500.000đ"},
            {"room_code": "P.001", "price_text": "4.500.000đ"},
        ],
        "fees": [
            {"label": "Điện", "value": "4k/kWh"},
            {"label": "Nước", "value": "100k/ng"},
            {"label": "Quản lý", "value": "150k/ph"},
            {"label": "Xe", "value": "Free"},
        ],
        "rules": [
            {"label": "Toilet", "value": "Riêng"},
            {"label": "Giờ giấc", "value": "Tự do"},
            {"label": "Thú cưng", "value": "Không"},
            {"label": "Ban công", "value": "Không"},
        ],
    },
    {
        "id": "sample-td-8",
        "title": "Studio có ban công",
        "category": "Studio",
        "district": "Thành phố Thủ Đức",
        "city": "TP. Hồ Chí Minh",
        "ward": "Phường Hiệp Phú",
        "address": "8/xx Trần Thị Điệu, Phường Hiệp Phú, TP. Thủ Đức, TP. Hồ Chí Minh",
        "price_text": "3.400.000đ / tháng",
        "price_value": 3400000,
        "area_text": "24m2",
        "floor_position": "Lầu 2",
        "available_room_count": 3,
        "total_room_count": 6,
        "amenities": ["Ban công", "Cửa sổ", "Wifi", "Máy lạnh"],
        "description": "Phù hợp sinh viên, gần tuyến Metro và khu công nghệ cao.",
        "tags": ["Mới", "Đã xác thực"],
        "hero_badge": "Sinh viên",
        "image": "https://images.unsplash.com/photo-1484154218962-a197022b5858?auto=format&fit=crop&w=1200&q=80",
        "available_rooms": [{"room_code": "A203", "price_text": "3.400.000đ"}],
        "fees": [
            {"label": "Điện", "value": "3.8k/kWh"},
            {"label": "Nước", "value": "90k/ng"},
            {"label": "Wifi", "value": "Free"},
        ],
        "rules": [
            {"label": "Giờ giấc", "value": "Tự do"},
            {"label": "Xe điện", "value": "Có"},
            {"label": "Thú cưng", "value": "Cân nhắc"},
        ],
    },
    {
        "id": "sample-bt-958",
        "title": "Căn hộ mini full nội thất",
        "category": "Căn hộ",
        "district": "Quận Bình Tân",
        "city": "TP. Hồ Chí Minh",
        "ward": "Phường An Lạc",
        "address": "958/xx An Dương Vương, Phường An Lạc, Quận Bình Tân, TP. Hồ Chí Minh",
        "price_text": "4.500.000đ / tháng",
        "price_value": 4500000,
        "area_text": "30m2",
        "floor_position": "Lầu 3",
        "available_room_count": 2,
        "total_room_count": 3,
        "amenities": ["Nội thất", "Máy lạnh", "Kệ bếp", "Tủ lạnh"],
        "description": "Phù hợp người đi làm, gần bến xe Miền Tây.",
        "tags": ["Sắp hết", "Đã xác thực"],
        "hero_badge": "Full nội thất",
        "image": "https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=1200&q=80",
        "available_rooms": [{"room_code": "B301", "price_text": "4.500.000đ"}],
        "fees": [
            {"label": "Điện", "value": "4k/kWh"},
            {"label": "Nước", "value": "100k/ng"},
        ],
        "rules": [
            {"label": "Toilet", "value": "Riêng"},
            {"label": "Giờ giấc", "value": "Tự do"},
        ],
    },
]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_listing(doc: dict[str, Any]) -> dict[str, Any]:
    address = doc.get("address", {}) or {}
    price = doc.get("price", {}) or {}
    property_info = doc.get("property_info", {}) or {}
    media = doc.get("media", {}) or {}
    source = doc.get("source", {}) or {}

    price_min = _safe_int(price.get("min") or price.get("amount"))
    price_max = _safe_int(price.get("max") or price.get("amount"))
    if price_min and price_max and price_min != price_max:
        price_text = f"{price_min:,.0f}đ - {price_max:,.0f}đ / tháng".replace(",", ".")
    elif price_min:
        price_text = f"{price_min:,.0f}đ / tháng".replace(",", ".")
    else:
        price_text = price.get("display_text") or "Liên hệ"

    amenities = doc.get("amenities") or []
    fees = doc.get("fees") or {}
    rules = doc.get("rules") or {}
    available_rooms = doc.get("available_rooms") or []

    return {
        "id": str(doc.get("_id") or source.get("url") or doc.get("title")),
        "property_id": str(doc.get("property_id") or ""),
        "title": doc.get("title") or "Listing",
        "category": str(doc.get("category") or "Nhà trọ").replace("_", " ").title(),
        "district": address.get("district") or "Chưa rõ quận/huyện",
        "city": address.get("city") or "Chưa rõ thành phố",
        "ward": address.get("ward") or "",
        "address": address.get("full") or "Chưa có địa chỉ",
        "price_text": price_text,
        "price_value": price_min or 0,
        "area_text": f"{property_info.get('area_m2')}m2" if property_info.get("area_m2") else "Đang cập nhật",
        "floor_position": property_info.get("floor_position") or "Đang cập nhật",
        "available_room_count": property_info.get("available_room_count"),
        "total_room_count": property_info.get("total_room_count"),
        "amenities": [str(item).replace("_", " ").title() for item in amenities][:8],
        "description": doc.get("description") or doc.get("summary") or "Chưa có mô tả",
        "tags": [str(tag).replace("_", " ").title() for tag in (doc.get("tags") or [])][:3],
        "hero_badge": "Live data",
        "image": media.get("cover_image") or (media.get("images") or [None])[0] or FALLBACK_LISTINGS[0]["image"],
        "available_rooms": [
            {
                "room_code": room.get("room_code", "-"),
                "price_text": f"{_safe_int(room.get('price') or 0):,.0f}đ".replace(",", ".")
                if _safe_int(room.get("price"))
                else "Liên hệ",
            }
            for room in available_rooms[:5]
        ],
        "fees": [{"label": k.replace("_", " ").title(), "value": v} for k, v in fees.items()],
        "rules": [{"label": k.replace("_", " ").title(), "value": v} for k, v in rules.items()],
    }


def _load_listings() -> list[dict[str, Any]]:
    try:
        col = get_listings_collection()
        pipeline = [
            {
                "$lookup": {
                    "from": "properties",
                    "localField": "property_id",
                    "foreignField": "_id",
                    "as": "property"
                }
            },
            {
                "$unwind": {
                    "path": "$property",
                    "preserveNullAndEmptyArrays": True
                }
            },
            {"$limit": 24}
        ]
        docs = list(col.aggregate(pipeline))
    except Exception:
        docs = []

    if not docs:
        return FALLBACK_LISTINGS

    merged_docs = []
    for doc in docs:
        prop = doc.get("property") or {}
        merged = {**prop, **doc}
        for field in ["address", "fees", "rules", "media", "amenities", "nearby_places"]:
            if field not in doc or not doc[field]:
                merged[field] = prop.get(field)
        merged_docs.append(merged)

    listings = [_normalize_listing(doc) for doc in merged_docs]
    listings.sort(key=lambda item: item["price_value"] or 0)
    return listings


def home(request):
    listings = _load_listings()
    context = {
        "featured_listings": listings[:6],
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
    return render(request, "listings/home.html", context)


def _slugify_simple(text: str) -> str:
    import unicodedata
    import re
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text).strip('-')
    return text


def _get_district_slug_map() -> dict[str, str]:
    from .services import get_properties_collection
    try:
        col = get_properties_collection()
        districts = col.distinct("address.district")
        return {_slugify_simple(d): d for d in districts if d}
    except Exception:
        return {}


def _get_city_slug_map() -> dict[str, str]:
    from .services import get_properties_collection
    try:
        col = get_properties_collection()
        cities = col.distinct("address.city")
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


def _load_filtered_listings(
    city_slug: str | None = None,
    district_slug: str | None = None,
    category_slug: str | None = None,
    price_range: str | None = None,
    amenity: str | None = None,
    quick_filter: str | None = None,
    search_query: str | None = None
) -> list[dict[str, Any]]:
    col = get_listings_collection()
    
    match_listing = {}
    
    if category_slug:
        cat_map = _get_category_slug_map()
        db_cat = cat_map.get(category_slug)
        if db_cat:
            match_listing["category"] = db_cat
            
    if price_range:
        parts = price_range.split("-")
        if len(parts) == 2:
            try:
                min_price = float(parts[0]) * 1_000_000
                max_price = float(parts[1]) * 1_000_000
                match_listing["price.min"] = {"$gte": min_price, "$lte": max_price}
            except ValueError:
                pass

    if search_query:
        match_listing["$or"] = [
            {"title": {"$regex": search_query, "$options": "i"}},
            {"description": {"$regex": search_query, "$options": "i"}},
            {"search_text": {"$regex": search_query, "$options": "i"}},
        ]
        
    pipeline = [
        {"$match": match_listing},
        {
            "$lookup": {
                "from": "properties",
                "localField": "property_id",
                "foreignField": "_id",
                "as": "property"
            }
        },
        {
            "$unwind": {
                "path": "$property",
                "preserveNullAndEmptyArrays": True
            }
        }
    ]
    
    match_property = {}
    
    if city_slug:
        city_map = _get_city_slug_map()
        db_city = city_map.get(city_slug)
        if db_city:
            match_property["property.address.city"] = db_city
            
    if district_slug:
        dist_map = _get_district_slug_map()
        db_dist = dist_map.get(district_slug)
        if db_dist:
            match_property["property.address.district"] = db_dist
            
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
            
    feature_conditions = []
    for feat in filter_features:
        if feat == "may_lanh":
            feature_conditions.append({
                "$or": [
                    {"property.amenities": "may_lanh"},
                    {"property.amenities": {"$regex": "may_lanh|máy lạnh|điều hòa", "$options": "i"}},
                    {"amenities": {"$regex": "may_lanh|máy lạnh|điều hòa", "$options": "i"}},
                ]
            })
        elif feat == "gac":
            feature_conditions.append({
                "$or": [
                    {"property.amenities": "gac"},
                    {"property.amenities": {"$regex": "gac|gác", "$options": "i"}},
                    {"amenities": {"$regex": "gac|gác", "$options": "i"}},
                ]
            })
        elif feat == "ban_cong":
            feature_conditions.append({
                "$or": [
                    {"property.rules.balcony": True},
                    {"property.rules.balcony": "True"},
                    {"property.amenities": "ban_cong"},
                    {"property.amenities": {"$regex": "ban_cong|ban công", "$options": "i"}},
                    {"amenities": {"$regex": "ban_cong|ban công", "$options": "i"}},
                ]
            })
        elif feat == "cua_so":
            feature_conditions.append({
                "$or": [
                    {"property.rules.window": True},
                    {"property.rules.window": "True"},
                    {"property.amenities": "cua_so"},
                    {"property.amenities": {"$regex": "cua_so|cửa sổ", "$options": "i"}},
                    {"amenities": {"$regex": "cua_so|cửa sổ", "$options": "i"}},
                ]
            })
        elif feat == "thu_cung":
            feature_conditions.append({
                "$or": [
                    {"property.rules.pet_allowed": True},
                    {"property.rules.pet_allowed": "True"},
                    {"property.rules.thu_cung": {"$regex": "Có|Cho|Được", "$options": "i"}},
                    {"property.rules.have_pet": {"$regex": "Có|Cho|Được|Y", "$options": "i"}},
                ]
            })
        elif feat == "gio_tu_do":
            feature_conditions.append({
                "$or": [
                    {"property.rules.curfew": "Tự do"},
                    {"property.rules.curfew": {"$regex": "tự do|free|24", "$options": "i"}},
                    {"property.rules.gio_giac": {"$regex": "Tự do|Free|24", "$options": "i"}},
                ]
            })
            
    if feature_conditions:
        match_property["$and"] = feature_conditions

    if match_property:
        pipeline.append({"$match": match_property})
        
    pipeline.append({"$limit": 60})
    
    try:
        docs = list(col.aggregate(pipeline))
    except Exception:
        docs = []
        
    merged_docs = []
    for doc in docs:
        prop = doc.get("property") or {}
        merged = {**prop, **doc}
        for field in ["address", "fees", "rules", "media", "amenities", "nearby_places"]:
            if field not in doc or not doc[field]:
                merged[field] = prop.get(field)
        merged_docs.append(merged)
        
    listings = [_normalize_listing(doc) for doc in merged_docs]
    listings.sort(key=lambda item: item["price_value"] or 0)
    return listings


def listing_list(request):
    city_slug = request.GET.get("city")
    district_slug = request.GET.get("district")
    category_slug = request.GET.get("category")
    price_range = request.GET.get("price_range")
    amenity = request.GET.get("amenity")
    quick_filter = request.GET.get("quick_filter")
    search_query = request.GET.get("q")
    
    listings = _load_filtered_listings(
        city_slug=city_slug,
        district_slug=district_slug,
        category_slug=category_slug,
        price_range=price_range,
        amenity=amenity,
        quick_filter=quick_filter,
        search_query=search_query
    )
    
    from .services import get_properties_collection
    import json
    
    city_district_map = {}
    try:
        prop_col = get_properties_collection()
        cities = [c for c in prop_col.distinct("address.city") if c]
        if not cities:
            cities = ["TP. Hồ Chí Minh", "TP. Hà Nội", "TP. Đà Nẵng", "Tỉnh Bình Dương", "TP. Cần Thơ"]
            
        if city_slug:
            city_map = _get_city_slug_map()
            db_city = city_map.get(city_slug)
            if db_city:
                districts = [d for d in prop_col.distinct("address.district", {"address.city": db_city}) if d]
            else:
                districts = [d for d in prop_col.distinct("address.district") if d]
        else:
            districts = [d for d in prop_col.distinct("address.district") if d]
            
        if not districts:
            districts = ["Quận 1", "Quận 7", "Quận 10", "Quận Bình Tân", "Thành phố Thủ Đức"]
        districts.sort()

        # Build dynamic Javascript map grouping districts by city
        pipeline = [
            {"$group": {
                "_id": "$address.city",
                "districts": {"$addToSet": "$address.district"}
            }}
        ]
        groups = list(prop_col.aggregate(pipeline))
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
        "listings": listings,
        "result_count": len(listings),
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
    return render(request, "listings/listing_list.html", context)


def listing_detail(request, listing_id: str):
    listings = _load_listings()
    listing = next((item for item in listings if item["id"] == listing_id), None)

    if listing is None:
        try:
            col = get_listings_collection()
            pipeline = [
                {"$match": {"_id": ObjectId(listing_id)}},
                {
                    "$lookup": {
                        "from": "properties",
                        "localField": "property_id",
                        "foreignField": "_id",
                        "as": "property"
                    }
                },
                {
                    "$unwind": {
                        "path": "$property",
                        "preserveNullAndEmptyArrays": True
                    }
                }
            ]
            results = list(col.aggregate(pipeline))
            doc = results[0] if results else None
        except Exception:
            doc = None
        if doc:
            prop = doc.get("property") or {}
            merged = {**prop, **doc}
            for field in ["address", "fees", "rules", "media", "amenities", "nearby_places"]:
                if field not in doc or not doc[field]:
                    merged[field] = prop.get(field)
            listing = _normalize_listing(merged)

    if listing is None:
        raise Http404("Không tìm thấy listing")

    context = {
        "listing": listing,
        "related_listings": [item for item in listings if item["id"] != listing["id"]][:3],
    }
    return render(request, "listings/listing_detail.html", context)


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
        
        # Save to database
        booking_doc = {
            "tenant_name": name,
            "phone": phone,
            "property_id": ObjectId(house_id) if house_id and len(house_id) == 24 else None,
            "listing_id": ObjectId(room_id) if room_id and len(room_id) == 24 else None,
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


def _parse_chat_message(message: str) -> dict[str, Any]:
    import re
    message_lower = message.lower()
    filters = {}
    
    # 1. Parse Category
    categories_map = {
        "phòng trọ": "phong_tro",
        "phòng thường": "phong_tro",
        "nhà trọ": "phong_tro",
        "căn hộ": "can_ho",
        "chdv": "chdv_1pn",
        "căn hộ dịch vụ": "chdv_1pn",
        "studio": "studio",
        "sleepbox": "sleepbox_nam",
        "giường": "giuong_nam",
        "ktx": "sleepbox_nam",
        "mặt bằng": "mat_bang",
        "nhà phố": "nha_pho",
        "nhà nguyên căn": "nha_pho",
        "duplex": "duplex"
    }
    for keyword, cat_val in categories_map.items():
        if keyword in message_lower:
            filters["category"] = cat_val
            break
            
    # 2. Parse District
    districts = [
        "quận 10", "q10", "q.10", "quận 11", "q11", "q.11", "quận 12", "q12", "q.12",
        "quận 1", "q1", "q.1", "quận 3", "q3", "q.3", "quận 4", "q4", "q.4",
        "quận 5", "q5", "q.5", "quận 6", "q6", "q.6", "quận 7", "q7", "q.7",
        "quận 8", "q8", "q.8", "thủ đức", "thu duc", "gò vấp", "go vap", 
        "bình thạnh", "binh thanh", "tân bình", "tan binh", "tân phú", "tan phu",
        "phú nhuận", "phu nhuan", "bình tân", "binh tan", "hóc môn", "hoc mon",
        "nhà bè", "nha be", "dĩ an", "di an", "thuận an", "thuan an"
    ]
    
    district_normalized = {
        "quận 10": "Quận 10", "q10": "Quận 10", "q.10": "Quận 10",
        "quận 11": "Quận 11", "q11": "Quận 11", "q.11": "Quận 11",
        "quận 12": "Quận 12", "q12": "Quận 12", "q.12": "Quận 12",
        "quận 1": "Quận 1", "q1": "Quận 1", "q.1": "Quận 1",
        "quận 3": "Quận 3", "q3": "Quận 3", "q.3": "Quận 3",
        "quận 4": "Quận 4", "q4": "Quận 4", "q.4": "Quận 4",
        "quận 5": "Quận 5", "q5": "Quận 5", "q.5": "Quận 5",
        "quận 6": "Quận 6", "q6": "Quận 6", "q.6": "Quận 6",
        "quận 7": "Quận 7", "q7": "Quận 7", "q.7": "Quận 7",
        "quận 8": "Quận 8", "q8": "Quận 8", "q.8": "Quận 8",
        "thủ đức": "Thành phố Thủ Đức", "thu duc": "Thành phố Thủ Đức",
        "gò vấp": "Quận Gò Vấp", "go vap": "Quận Gò Vấp",
        "bình thạnh": "Quận Bình Thạnh", "binh thanh": "Quận Bình Thạnh",
        "tân bình": "Quận Tân Bình", "tan binh": "Quận Tân Bình",
        "tân phú": "Quận Tân Phú", "tan phu": "Quận Tân Phú",
        "phú nhuận": "Quận Phú Nhuận", "phu nhuan": "Quận Phú Nhuận",
        "bình tân": "Quận Bình Tân", "binh tan": "Quận Bình Tân",
        "hóc môn": "Huyện Hóc Môn", "hoc mon": "Huyện Hóc Môn",
        "nhà bè": "Huyện Nhà Bè", "nha be": "Huyện Nhà Bè",
        "dĩ an": "Thành phố Dĩ An", "di an": "Thành phố Dĩ An",
        "thuận an": "Thành phố Thuận An", "thuan an": "Thành phố Thuận An"
    }
    
    for d in districts:
        if d in message_lower:
            filters["district"] = district_normalized[d]
            break
            
    # 3. Parse Price limit (e.g., "dưới 5 triệu", "dưới 4.5tr", "tầm 3 triệu")
    price_match = re.search(r'(?:dưới|tầm|khoảng|max|bằng|giá)\s+(\d+(?:[.,]\d+)?)\s*(?:triệu|tr)', message_lower)
    if price_match:
        val_str = price_match.group(1).replace(",", ".")
        try:
            val_float = float(val_str)
            filters["price_max"] = int(val_float * 1_000_000)
        except ValueError:
            pass
            
    # 4. Parse Amenities
    amenity_keywords = {
        "máy lạnh": "may_lanh",
        "điều hòa": "may_lanh",
        "tủ lạnh": "tu_lanh",
        "máy giặt": "may_giat",
        "gác": "gac",
        "ban công": "ban_cong",
        "thú cưng": "thu_cung",
        "chó": "thu_cung",
        "mèo": "thu_cung",
    }
    matched_amenities = []
    for kw, val in amenity_keywords.items():
        if kw in message_lower:
            matched_amenities.append(val)
    if matched_amenities:
        filters["amenities"] = matched_amenities
        
    return filters


def _search_listings(filters: dict[str, Any]) -> list[dict[str, Any]]:
    col = get_listings_collection()
    
    # Base match for listings
    match_listing = {}
    if "category" in filters:
        match_listing["category"] = filters["category"]
    
    if "price_max" in filters:
        match_listing["price.min"] = {"$lte": filters["price_max"]}
        
    pipeline = [
        {"$match": match_listing},
        {
            "$lookup": {
                "from": "properties",
                "localField": "property_id",
                "foreignField": "_id",
                "as": "property"
            }
        },
        {
            "$unwind": {
                "path": "$property",
                "preserveNullAndEmptyArrays": True
            }
        }
    ]
    
    # Filter by property level fields (district & amenities)
    match_property = {}
    if "district" in filters:
        match_property["property.address.district"] = filters["district"]
        
    if "amenities" in filters:
        feature_conditions = []
        for feat in filters["amenities"]:
            if feat == "may_lanh":
                feature_conditions.append({
                    "$or": [
                        {"property.amenities": "may_lanh"},
                        {"property.amenities": {"$regex": "may_lanh|máy lạnh|điều hòa", "$options": "i"}},
                        {"amenities": {"$regex": "may_lanh|máy lạnh|điều hòa", "$options": "i"}},
                    ]
                })
            elif feat == "gac":
                feature_conditions.append({
                    "$or": [
                        {"property.amenities": "gac"},
                        {"property.amenities": {"$regex": "gac|gác", "$options": "i"}},
                        {"amenities": {"$regex": "gac|gác", "$options": "i"}},
                    ]
                })
            elif feat == "ban_cong":
                feature_conditions.append({
                    "$or": [
                        {"property.rules.balcony": True},
                        {"property.rules.balcony": "True"},
                        {"property.amenities": "ban_cong"},
                        {"property.amenities": {"$regex": "ban_cong|ban công", "$options": "i"}},
                        {"amenities": {"$regex": "ban_cong|ban công", "$options": "i"}},
                    ]
                })
            elif feat == "cua_so":
                feature_conditions.append({
                    "$or": [
                        {"property.rules.window": True},
                        {"property.rules.window": "True"},
                        {"property.amenities": "cua_so"},
                        {"property.amenities": {"$regex": "cua_so|cửa sổ", "$options": "i"}},
                        {"amenities": {"$regex": "cua_so|cửa sổ", "$options": "i"}},
                    ]
                })
            elif feat == "thu_cung":
                feature_conditions.append({
                    "$or": [
                        {"property.rules.pet_allowed": True},
                        {"property.rules.pet_allowed": "True"},
                        {"property.rules.thu_cung": {"$regex": "Có|Cho|Được", "$options": "i"}},
                        {"property.rules.have_pet": {"$regex": "Có|Cho|Được|Y", "$options": "i"}},
                    ]
                })
            elif feat == "tu_lanh":
                feature_conditions.append({
                    "$or": [
                        {"property.amenities": "tu_lanh"},
                        {"property.amenities": {"$regex": "tu_lanh|tủ lạnh", "$options": "i"}},
                        {"amenities": {"$regex": "tu_lanh|tủ lạnh", "$options": "i"}},
                    ]
                })
            elif feat == "may_giat":
                feature_conditions.append({
                    "$or": [
                        {"property.amenities": "may_giat"},
                        {"property.amenities": {"$regex": "may_giat|máy giặt", "$options": "i"}},
                        {"amenities": {"$regex": "may_giat|máy giặt", "$options": "i"}},
                    ]
                })
        if feature_conditions:
            match_property["$and"] = feature_conditions
        
    if match_property:
        pipeline.append({"$match": match_property})
        
    pipeline.append({"$limit": 6})
    
    try:
        docs = list(col.aggregate(pipeline))
    except Exception:
        docs = []
        
    merged_docs = []
    for doc in docs:
        prop = doc.get("property") or {}
        merged = {**prop, **doc}
        for field in ["address", "fees", "rules", "media", "amenities", "nearby_places"]:
            if field not in doc or not doc[field]:
                merged[field] = prop.get(field)
        merged_docs.append(merged)
        
    return [_normalize_listing(doc) for doc in merged_docs]


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

    filters = _parse_chat_message(message)
    results = _search_listings(filters)
    
    category_name = filters.get("category", "phòng").replace("_", " ").title()
    if category_name == "Can Ho":
        category_name = "căn hộ"
    elif category_name == "Phong Tro":
        category_name = "phòng trọ"
    elif category_name == "Chdv 1Pn":
        category_name = "căn hộ dịch vụ 1PN"
    else:
        category_name = "phòng trọ/căn hộ"
        
    district_name = filters.get("district", "")
    price_limit = filters.get("price_max", None)
    
    if results:
        response_text = f"Dựa trên phân tích yêu cầu của bạn, trợ lý RAG đã lọc được **{len(results)} {category_name}**"
        if district_name:
            response_text += f" tại **{district_name}**"
        if price_limit:
            response_text += f" với giá dưới **{(price_limit / 1000000):.1f} triệu/tháng**"
        response_text += " phù hợp nhất:"
    else:
        response_text = "Tôi chưa tìm thấy phòng trọ nào khớp hoàn toàn với mô tả của bạn. Tuy nhiên, bạn có thể tham khảo một số gợi ý nổi bật dưới đây:"
        results = _load_listings()[:3]
        
    listings_data = []
    for item in results:
        listings_data.append({
            "id": item["id"],
            "title": item["title"],
            "category": item["category"],
            "price_text": item["price_text"],
            "address": item["address"],
            "image": item["image"],
            "area_text": item["area_text"],
            "district": item.get("district", ""),
            "city": item.get("city", ""),
            "available_room_count": item.get("available_room_count"),
            "amenities": item.get("amenities", [])[:4],
        })

    follow_up_chips = [
        "Rẻ hơn",
        "Gần trung tâm hơn",
        "Có ban công",
        "Chỉ ở Quận 10",
        "So sánh 3 phòng này",
    ]
        
    return JsonResponse({
        "success": True,
        "reply": response_text,
        "listings": listings_data,
        "follow_ups": follow_up_chips,
        "filters": {
            "category": filters.get("category"),
            "district": filters.get("district"),
            "price_max": filters.get("price_max"),
            "amenities": filters.get("amenities", []),
        }
    })
