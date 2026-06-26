"""Read-only room repository abstraction."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from .schemas import normalize_room


class RoomRepository(Protocol):
    def search_by_constraints(
        self,
        constraints: dict[str, Any],
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        ...

    def search_by_metadata(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        ...

    def get_by_id(self, room_id: str) -> dict[str, Any] | None:
        ...

    def get_many_by_ids(self, room_ids: list[str]) -> list[dict[str, Any]]:
        ...

    def iter_room_ids(
        self,
        batch_size: int = 100,
        resume_after: str | None = None,
    ) -> Iterable[list[str]]:
        ...


class EmptyRoomRepository:
    """No-data repository for local runs before the crawler/MongoDB exists."""

    def search_by_constraints(self, constraints: dict[str, Any], limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return []

    def search_by_metadata(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def get_by_id(self, room_id: str) -> dict[str, Any] | None:
        return None

    def get_many_by_ids(self, room_ids: list[str]) -> list[dict[str, Any]]:
        return []

    def iter_room_ids(self, batch_size: int = 100, resume_after: str | None = None) -> Iterable[list[str]]:
        return iter(())


class InMemoryRoomRepository:
    """Fake repository for tests. Do not use it for production data."""

    def __init__(self, rooms: list[dict[str, Any]]) -> None:
        self._rooms = {
            item["room_id"]: item
            for item in (normalize_room(raw) for raw in rooms)
            if item and item.get("room_id")
        }

    def search_by_constraints(self, constraints: dict[str, Any], limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        matches = [
            room for room in self._rooms.values()
            if room_matches_constraints(room, constraints)
        ]
        return matches[offset:offset + limit]

    def search_by_metadata(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        from retrieval.metadata_search import extract_metadata_signals, score_metadata_hit
        from config import METADATA_FIELDS

        signals = extract_metadata_signals(query_text)
        scored = []
        for room in self._rooms.values():
            score = score_metadata_hit(room, signals, METADATA_FIELDS)
            if score > 0:
                item = dict(room)
                item["_metadata_score"] = score
                scored.append(item)
        scored.sort(key=lambda item: item.get("_metadata_score", 0), reverse=True)
        return scored[:max(limit, 1)]

    def get_by_id(self, room_id: str) -> dict[str, Any] | None:
        room = self._rooms.get(str(room_id))
        return dict(room) if room else None

    def get_many_by_ids(self, room_ids: list[str]) -> list[dict[str, Any]]:
        found = []
        for room_id in room_ids:
            room = self.get_by_id(room_id)
            if room:
                found.append(room)
        return found

    def iter_room_ids(self, batch_size: int = 100, resume_after: str | None = None) -> Iterable[list[str]]:
        ids = sorted(self._rooms)
        if resume_after:
            ids = [room_id for room_id in ids if room_id > resume_after]
        for idx in range(0, len(ids), batch_size):
            yield ids[idx:idx + batch_size]


class MongoRoomRepository:
    """MongoDB adapter behind the read-only RoomRepository interface.

    Reads from the `rooms` collection which has:
        room_id, house_id, tien_ich_xq, house_remark, embedding_text,
        metadata: { house_name, room_code, province_name, district_name,
                     ward_name, price, status_code, status_desc, allow_sale, has_image }
    """

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        server_selection_timeout_ms: int = 3000,
    ) -> None:
        from pymongo import MongoClient

        self._client = MongoClient(uri, serverSelectionTimeoutMS=server_selection_timeout_ms)
        self._database = self._client[database]
        self._collection = self._database[collection]

    def search_by_constraints(self, constraints: dict[str, Any], limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        query = build_mongo_query(constraints)
        cursor = (
            self._collection
            .find(query)
            .skip(max(offset, 0))
            .limit(max(limit, 1))
        )
        return [item for item in (normalize_room(doc) for doc in cursor) if item]

    def search_by_metadata(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        import re
        from retrieval.metadata_search import extract_metadata_signals, metadata_signal_present, score_metadata_hit
        from config import METADATA_FIELDS

        signals = extract_metadata_signals(query_text)
        if not metadata_signal_present(signals):
            return []

        clauses: list[dict[str, Any]] = []
        for room_id in signals.get("room_id", []) or []:
            clauses.extend(_mongo_id_or_clauses(str(room_id)))
        for snippet in signals.get("district", []) or []:
            escaped = re.escape(str(snippet))
            clauses.extend([
                {"metadata.district_name": {"$regex": escaped, "$options": "i"}},
                {"metadata.ward_name": {"$regex": escaped, "$options": "i"}},
                {"embedding_text": {"$regex": escaped, "$options": "i"}},
            ])
        if not clauses:
            return []

        cursor = self._collection.find({
            "$and": [
                {"metadata.status_code": "0"},
                {"$or": clauses},
            ]
        }).limit(max(limit, 1))
        results = []
        for item in (normalize_room(doc) for doc in cursor):
            if not item:
                continue
            item["_metadata_score"] = score_metadata_hit(item, signals, METADATA_FIELDS)
            results.append(item)
        results.sort(key=lambda item: item.get("_metadata_score", 0), reverse=True)
        return results

    def get_by_id(self, room_id: str) -> dict[str, Any] | None:
        doc = self._collection.find_one({"$or": _mongo_id_or_clauses(str(room_id))})
        return normalize_room(doc)

    def get_many_by_ids(self, room_ids: list[str]) -> list[dict[str, Any]]:
        ids = [str(item) for item in room_ids]
        if not ids:
            return []

        docs = list(self._collection.find({"$or": _mongo_many_id_or_clauses(ids)}))
        by_id: dict[str, dict[str, Any]] = {}
        for doc in docs:
            item = normalize_room(doc)
            if not item:
                continue
            for key in _room_lookup_keys(doc, item):
                by_id.setdefault(key, item)
        return [by_id[item] for item in ids if item in by_id]

    def iter_room_ids(self, batch_size: int = 100, resume_after: str | None = None) -> Iterable[list[str]]:
        query: dict[str, Any] = {}
        if resume_after:
            query["room_id"] = {"$gt": resume_after}
        cursor = self._collection.find(query, {"room_id": 1}).sort("_id", 1)

        batch: list[str] = []
        for doc in cursor:
            room_id = doc.get("room_id")
            if not room_id:
                continue
            batch.append(str(room_id))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def create_room_repository() -> RoomRepository:
    import os as _os
    MONGODB_URI = ""
    MONGODB_DATABASE = ""
    MONGODB_ROOMS_COLLECTION = ""

    try:
        from config import MONGODB_DATABASE as _DB, MONGODB_ROOMS_COLLECTION as _COL, MONGODB_URI as _URI
        # Verify we got the right config module (not Django's config package)
        if _URI and "mongodb" in _URI:
            MONGODB_URI = _URI
            MONGODB_DATABASE = _DB
            MONGODB_ROOMS_COLLECTION = _COL
        else:
            raise ImportError("config module has no MONGODB_URI with 'mongodb'")
    except Exception:
        # Fallback: read directly from environment (covers Django deployment)
        MONGODB_URI = _os.getenv("MONGODB_URI", "")
        MONGODB_DATABASE = _os.getenv("MONGODB_DATABASE", "demo_rag")
        MONGODB_ROOMS_COLLECTION = _os.getenv("MONGODB_ROOMS_COLLECTION", "rooms")

    if MONGODB_URI and MONGODB_DATABASE and MONGODB_ROOMS_COLLECTION:
        try:
            return MongoRoomRepository(
                uri=MONGODB_URI,
                database=MONGODB_DATABASE,
                collection=MONGODB_ROOMS_COLLECTION,
            )
        except Exception:
            return EmptyRoomRepository()
    return EmptyRoomRepository()


def _mongo_id_or_clauses(room_id: str) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [
        {"room_id": room_id},
    ]
    object_id = _to_object_id(room_id)
    if object_id is not None:
        clauses.append({"_id": object_id})
    return clauses


def _mongo_many_id_or_clauses(room_ids: list[str]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [
        {"room_id": {"$in": room_ids}},
    ]
    object_ids = [object_id for item in room_ids if (object_id := _to_object_id(item)) is not None]
    if object_ids:
        clauses.append({"_id": {"$in": object_ids}})
    return clauses


def _to_object_id(value: str) -> Any | None:
    from bson import ObjectId

    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _room_lookup_keys(raw: dict[str, Any], normalized: dict[str, Any]) -> set[str]:
    keys = {
        normalized.get("room_id"),
        raw.get("room_id"),
        raw.get("_id"),
    }
    return {str(key) for key in keys if key}


def build_mongo_query(constraints: dict[str, Any]) -> dict[str, Any]:
    query: dict[str, Any] = {
        "$and": [
            {"metadata.status_code": "0"},  # Phòng trống
        ]
    }

    budget = constraints.get("budget") or {}
    if budget.get("max") is not None:
        if budget.get("max_operator") == "lt":
            query["$and"].append({"metadata.price": {"$lt": budget["max"]}})
        else:
            query["$and"].append({"metadata.price": {"$lte": budget["max"]}})
    if budget.get("min") is not None:
        if budget.get("min_operator") == "gt":
            query["$and"].append({"metadata.price": {"$gt": budget["min"]}})
        else:
            query["$and"].append({"metadata.price": {"$gte": budget["min"]}})

    location = constraints.get("location") or {}
    if location.get("province"):
        query["$and"].append({
            "$or": [
                {"metadata.province_name": {"$regex": location["province"], "$options": "i"}},
                {"embedding_text": {"$regex": location["province"], "$options": "i"}},
            ]
        })
    if location.get("districts"):
        district_clauses = []
        for d in location["districts"]:
            for variant in _location_query_variants(d):
                district_clauses.append({"metadata.district_name": {"$regex": variant, "$options": "i"}})
                district_clauses.append({"embedding_text": {"$regex": variant, "$options": "i"}})
        query["$and"].append({"$or": district_clauses})
    if location.get("wards"):
        ward_clauses = []
        for w in location["wards"]:
            ward_clauses.append({"metadata.ward_name": {"$regex": w, "$options": "i"}})
            ward_clauses.append({"embedding_text": {"$regex": w, "$options": "i"}})
        query["$and"].append({"$or": ward_clauses})

    # Amenities/features — search within embedding_text
    required = constraints.get("amenities_required") or []
    for amenity in required:
        positive_pattern = _amenity_positive_pattern(amenity)
        if positive_pattern:
            query["$and"].append({"embedding_text": {"$regex": positive_pattern, "$options": "i"}})

    excluded = constraints.get("excluded_features") or []
    for feature in excluded:
        positive_pattern = _amenity_positive_pattern(feature)
        if positive_pattern:
            query["$and"].append({"embedding_text": {"$not": {"$regex": positive_pattern, "$options": "i"}}})

    return query


def room_matches_constraints(room: dict[str, Any], constraints: dict[str, Any]) -> bool:
    if not room.get("available"):
        return False

    budget = constraints.get("budget") or {}
    rent = room.get("rent_price")
    if budget.get("max") is not None and rent is not None:
        if budget.get("max_operator") == "lt":
            if rent >= budget["max"]:
                return False
        else:
            if rent > budget["max"]:
                return False
    if budget.get("min") is not None and rent is not None:
        if budget.get("min_operator") == "gt":
            if rent <= budget["min"]:
                return False
        else:
            if rent < budget["min"]:
                return False

    location = constraints.get("location") or {}
    if location.get("districts"):
        room_district = _normalize_location_value(room.get("district"))
        target_districts = {_normalize_location_value(item) for item in location["districts"]}
        if room_district and room_district not in target_districts:
            return False

    # Check required amenities
    required = constraints.get("amenities_required") or []
    room_amenities = room.get("amenities") or []
    required = [
        amenity for amenity in required
        if not _room_has_canonical_amenity(room_amenities, amenity)
    ]
    import re
    for amenity in required:
        if not _room_has_positive_amenity(room, amenity):
            return False

    # Check excluded features
    excluded = constraints.get("excluded_features") or []
    for feature in excluded:
        if _room_has_canonical_amenity(room_amenities, feature):
            return False
        if _room_has_positive_amenity(room, feature):
            return False

    return True


AMENITY_VIETNAMESE_MAP: dict[str, str] = {
    "air_conditioner": "Máy lạnh",
    "balcony": "Ban công",
    "window": "Cửa sổ",
    "washing_machine": "Máy giặt",
    "private_bathroom": "Toilet.*Riêng",
    "mezzanine": "Gác",
    "kitchen": "Kệ bếp",
    "refrigerator": "Tủ lạnh",
    "hot_water": "Nước nóng",
    "bed": "Giường",
    "mattress": "Nệm",
    "wardrobe": "Tủ quần áo",
    "elevator": "Thang máy",
    "wifi": "Wifi",
    "ev_charging": "Xe điện",
    "free_hours": "Giờ giấc.*Tự do",
}


def _amenity_to_vietnamese(amenity: str) -> str | None:
    return AMENITY_VIETNAMESE_MAP.get(amenity)


def _amenity_positive_pattern(amenity: str) -> str | None:
    label = _amenity_to_vietnamese(amenity)
    if not label:
        return None
    if amenity == "private_bathroom":
        return r"Toilet\s*:\s*Riêng"
    if amenity == "free_hours":
        return r"Giờ giấc\s*:\s*Tự do"
    return rf"{label}\s*:\s*(?:Có|Riêng|Tự do|True|Yes|Free)"


def _room_has_positive_amenity(room: dict[str, Any], amenity: str) -> bool:
    import re

    pattern = _amenity_positive_pattern(amenity)
    if not pattern:
        return False
    readable = _amenity_to_vietnamese(amenity) or ""
    room_amenities = room.get("amenities") or []
    for item in room_amenities:
        if re.search(readable, str(item), re.IGNORECASE):
            return True
    return bool(re.search(pattern, str(room.get("embedding_text") or ""), re.IGNORECASE))


def _room_has_canonical_amenity(room_amenities: list[Any], amenity: str) -> bool:
    canonical = str(amenity).strip().lower()
    return any(str(item).strip().lower() == canonical for item in room_amenities)


def _location_query_variants(value: Any) -> list[str]:
    import re

    text = str(value or "").strip()
    variants = [text]
    match = re.fullmatch(r"(?:quan|q\.?)\s*(\d{1,2})", text, flags=re.IGNORECASE)
    if match:
        number = match.group(1)
        variants.extend([f"Quận {number}", f"Q{number}", f"quan {number}"])
    else:
        normalized = _normalize_location_value(text)
        if normalized and normalized != text:
            variants.append(normalized)
        if normalized and not re.fullmatch(r"\d{1,2}", normalized):
            variants.extend([f"Quận {normalized}", f"Huyện {normalized}", f"Thành phố {normalized}"])
        match = re.fullmatch(r"(?:huyen)\s+(.+)", text, flags=re.IGNORECASE)
        if match:
            variants.append(f"Huyện {match.group(1)}")
    unique = list(dict.fromkeys(item for item in variants if item))
    return [_accent_flexible_regex(item) for item in unique]


def _accent_flexible_regex(value: str) -> str:
    import re

    groups = {
        "a": "aàáảãạăằắẳẵặâầấẩẫậ",
        "e": "eèéẻẽẹêềếểễệ",
        "i": "iìíỉĩị",
        "o": "oòóỏõọôồốổỗộơờớởỡợ",
        "u": "uùúủũụưừứửữự",
        "y": "yỳýỷỹỵ",
        "d": "dđ",
    }
    pattern = []
    for char in str(value):
        base = _strip_accents(char).lower()
        if base in groups:
            chars = groups[base]
            pattern.append(f"[{chars}{chars.upper()}]")
        elif char.isspace():
            pattern.append(r"\s+")
        else:
            pattern.append(re.escape(char))
    return "".join(pattern)


def _strip_accents(value: str) -> str:
    import unicodedata

    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(value))
        if unicodedata.category(ch) != "Mn"
    ).replace("đ", "d").replace("Đ", "D")


def _normalize_location_value(value: Any) -> str:
    import re
    import unicodedata

    if value is None:
        return ""
    text = str(value).lower().strip()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"\b(?:quan|district|huyen)\s+|\bq\.?\s*(?=\d)", "", text)
    return re.sub(r"\s+", " ", text).strip()
