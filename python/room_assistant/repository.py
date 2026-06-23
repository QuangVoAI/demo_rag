"""Read-only listing repository abstraction."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from .schemas import normalize_listing


class ListingRepository(Protocol):
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

    def get_by_id(self, listing_id: str) -> dict[str, Any] | None:
        ...

    def get_many_by_ids(self, listing_ids: list[str]) -> list[dict[str, Any]]:
        ...

    def get_current_version(self, listing_id: str) -> int | None:
        ...

    def iter_listing_ids(
        self,
        batch_size: int = 100,
        resume_after: str | None = None,
    ) -> Iterable[list[str]]:
        ...


class EmptyListingRepository:
    """No-data repository for local runs before the crawler/MongoDB exists."""

    def search_by_constraints(self, constraints: dict[str, Any], limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return []

    def search_by_metadata(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def get_by_id(self, listing_id: str) -> dict[str, Any] | None:
        return None

    def get_many_by_ids(self, listing_ids: list[str]) -> list[dict[str, Any]]:
        return []

    def get_current_version(self, listing_id: str) -> int | None:
        return None

    def iter_listing_ids(self, batch_size: int = 100, resume_after: str | None = None) -> Iterable[list[str]]:
        return iter(())


class InMemoryListingRepository:
    """Fake repository for tests. Do not use it for production data."""

    def __init__(self, listings: list[dict[str, Any]]) -> None:
        self._listings = {
            item["listing_id"]: item
            for item in (normalize_listing(raw) for raw in listings)
            if item and item.get("listing_id")
        }

    def search_by_constraints(self, constraints: dict[str, Any], limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        matches = [
            listing for listing in self._listings.values()
            if listing_matches_constraints(listing, constraints)
        ]
        return matches[offset:offset + limit]

    def search_by_metadata(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        from retrieval.metadata_search import extract_metadata_signals, score_metadata_hit
        from config import METADATA_FIELDS

        signals = extract_metadata_signals(query_text)
        scored = []
        for listing in self._listings.values():
            score = score_metadata_hit(listing, signals, METADATA_FIELDS)
            if score > 0:
                item = dict(listing)
                item["_metadata_score"] = score
                scored.append(item)
        scored.sort(key=lambda item: item.get("_metadata_score", 0), reverse=True)
        return scored[:max(limit, 1)]

    def get_by_id(self, listing_id: str) -> dict[str, Any] | None:
        listing = self._listings.get(str(listing_id))
        return dict(listing) if listing else None

    def get_many_by_ids(self, listing_ids: list[str]) -> list[dict[str, Any]]:
        found = []
        for listing_id in listing_ids:
            listing = self.get_by_id(listing_id)
            if listing:
                found.append(listing)
        return found

    def get_current_version(self, listing_id: str) -> int | None:
        listing = self._listings.get(str(listing_id))
        if not listing:
            return None
        return int(listing.get("source_version") or 0)

    def iter_listing_ids(self, batch_size: int = 100, resume_after: str | None = None) -> Iterable[list[str]]:
        ids = sorted(self._listings)
        if resume_after:
            ids = [listing_id for listing_id in ids if listing_id > resume_after]
        for idx in range(0, len(ids), batch_size):
            yield ids[idx:idx + batch_size]


class MongoListingRepository:
    """MongoDB adapter behind the read-only ListingRepository interface."""

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
        self._properties_collection = self._database["properties"]

    def search_by_constraints(self, constraints: dict[str, Any], limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        query = build_mongo_query(constraints)
        self._append_vehicle_query(query, constraints)
        cursor = (
            self._collection
            .find(query)
            .skip(max(offset, 0))
            .limit(max(limit, 1))
        )
        return [item for item in (normalize_listing(doc) for doc in self._enrich_listing_docs(list(cursor))) if item]

    def search_by_metadata(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        import re
        from retrieval.metadata_search import extract_metadata_signals, metadata_signal_present, score_metadata_hit
        from config import METADATA_FIELDS

        signals = extract_metadata_signals(query_text)
        if not metadata_signal_present(signals):
            return []

        clauses: list[dict[str, Any]] = []
        for listing_id in signals.get("listing_id", []) or []:
            clauses.extend(_mongo_id_or_clauses(str(listing_id)))
        for snippet in signals.get("district", []) or []:
            escaped = re.escape(str(snippet))
            clauses.extend([
                {"district": {"$regex": escaped, "$options": "i"}},
                {"location.district": {"$regex": escaped, "$options": "i"}},
                {"title": {"$regex": escaped, "$options": "i"}},
                {"address": {"$regex": escaped, "$options": "i"}},
                {"embedding_text": {"$regex": escaped, "$options": "i"}},
                {"summary": {"$regex": escaped, "$options": "i"}},
            ])
        for arxiv_id in signals.get("arxiv_like", []) or []:
            escaped = re.escape(str(arxiv_id))
            clauses.extend([
                {"arxiv_id": str(arxiv_id)},
                {"title": {"$regex": escaped, "$options": "i"}},
                {"embedding_text": {"$regex": escaped, "$options": "i"}},
            ])
        if not clauses:
            return []

        cursor = self._collection.find({
            "$and": [
                {"available": {"$ne": False}},
                {"$or": [{"status": {"$in": ["active", "published"]}}, {"status": {"$exists": False}}]},
                {"$or": clauses},
            ]
        }).limit(max(limit, 1))
        results = []
        for item in (normalize_listing(doc) for doc in self._enrich_listing_docs(list(cursor))):
            if not item:
                continue
            item["_metadata_score"] = score_metadata_hit(item, signals, METADATA_FIELDS)
            results.append(item)
        results.sort(key=lambda item: item.get("_metadata_score", 0), reverse=True)
        return results

    def get_by_id(self, listing_id: str) -> dict[str, Any] | None:
        doc = self._collection.find_one({"$or": _mongo_id_or_clauses(str(listing_id))})
        doc = self._enrich_listing_doc(doc)
        return normalize_listing(doc)

    def get_many_by_ids(self, listing_ids: list[str]) -> list[dict[str, Any]]:
        ids = [str(item) for item in listing_ids]
        if not ids:
            return []

        docs = self._enrich_listing_docs(list(self._collection.find({"$or": _mongo_many_id_or_clauses(ids)})))
        by_id: dict[str, dict[str, Any]] = {}
        for doc in docs:
            item = normalize_listing(doc)
            if not item:
                continue
            for key in _listing_lookup_keys(doc, item):
                by_id.setdefault(key, item)
        return [by_id[item] for item in ids if item in by_id]

    def get_current_version(self, listing_id: str) -> int | None:
        listing = self.get_by_id(listing_id)
        if not listing:
            return None
        return int(listing.get("source_version") or 0)

    def iter_listing_ids(self, batch_size: int = 100, resume_after: str | None = None) -> Iterable[list[str]]:
        query: dict[str, Any] = {}
        if resume_after:
            object_id = _to_object_id(resume_after)
            if object_id is not None:
                query["_id"] = {"$gt": object_id}
            else:
                query["listing_id"] = {"$gt": resume_after}
        cursor = self._collection.find(query, {"listing_id": 1, "id": 1, "slug": 1}).sort("_id", 1)

        batch: list[str] = []
        for doc in cursor:
            listing_id = doc.get("listing_id") or doc.get("id") or doc.get("slug") or doc.get("_id")
            if not listing_id:
                continue
            batch.append(str(listing_id))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _append_vehicle_query(self, query: dict[str, Any], constraints: dict[str, Any]) -> None:
        vehicles = set(constraints.get("vehicles") or [])
        if not vehicles:
            return

        property_ids = self._property_ids_for_vehicle_constraints(vehicles)
        vehicle_clauses: list[dict[str, Any]] = []
        if "electric_bike" in vehicles:
            vehicle_clauses.extend([
                {"electric_bike_allowed": True},
                {"rules.electric_vehicle_allowed": True},
                {"amenities": {"$in": ["ev_charging", "sac_xe_dien"]}},
            ])
        if "motorbike" in vehicles:
            vehicle_clauses.extend([
                {"shared_parking": True},
                {"rules.shared_parking": True},
                {"vehicles_allowed": {"$in": ["motorbike", "xe_may"]}},
                {"vehicles": {"$in": ["motorbike", "xe_may"]}},
                {"amenities": {"$in": ["parking", "cho_de_xe", "ham_xe"]}},
                {"fees.parking": {"$exists": True}},
            ])
        if property_ids:
            vehicle_clauses.append({"property_id": {"$in": property_ids}})
        if vehicle_clauses:
            query["$and"].append({"$or": vehicle_clauses})

    def _property_ids_for_vehicle_constraints(self, vehicles: set[str]) -> list[Any]:
        if not hasattr(self, "_properties_collection"):
            return []
        clauses = []
        if "motorbike" in vehicles:
            clauses.extend([
                {"rules.shared_parking": True},
                {"fees.parking": {"$exists": True}},
                {"amenities": {"$in": ["parking", "cho_de_xe", "ham_xe"]}},
            ])
        if "electric_bike" in vehicles:
            clauses.extend([
                {"rules.electric_vehicle_allowed": True},
                {"amenities": {"$in": ["ev_charging", "sac_xe_dien"]}},
            ])
        if not clauses:
            return []
        cursor = self._properties_collection.find({"$or": clauses}, {"_id": 1}).limit(1000)
        return [doc["_id"] for doc in cursor if doc.get("_id")]

    def _enrich_listing_docs(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hasattr(self, "_properties_collection"):
            return docs
        property_ids = [doc.get("property_id") for doc in docs if doc and doc.get("property_id")]
        if not property_ids:
            return docs
        properties = {
            prop["_id"]: prop
            for prop in self._properties_collection.find({"_id": {"$in": property_ids}})
            if prop.get("_id")
        }
        return [self._merge_property_doc(doc, properties.get(doc.get("property_id"))) for doc in docs]

    def _enrich_listing_doc(self, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if not doc or not doc.get("property_id") or not hasattr(self, "_properties_collection"):
            return doc
        prop = self._properties_collection.find_one({"_id": doc["property_id"]})
        return self._merge_property_doc(doc, prop)

    def _merge_property_doc(self, listing: dict[str, Any], prop: dict[str, Any] | None) -> dict[str, Any]:
        if not prop:
            return listing
        merged = dict(listing)
        for key in ("amenities", "fees", "media", "nearby_places", "rules"):
            if not merged.get(key) and prop.get(key) is not None:
                merged[key] = prop.get(key)
        if not merged.get("address") and prop.get("address") is not None:
            merged["address"] = prop.get("address")
        if not merged.get("landlord_id") and prop.get("landlord_id") is not None:
            merged["landlord_id"] = prop.get("landlord_id")
        return merged


def create_listing_repository() -> ListingRepository:
    try:
        from config import MONGODB_DATABASE, MONGODB_LISTINGS_COLLECTION, MONGODB_URI
    except Exception:
        MONGODB_URI = ""
        MONGODB_DATABASE = ""
        MONGODB_LISTINGS_COLLECTION = ""

    if MONGODB_URI and MONGODB_DATABASE and MONGODB_LISTINGS_COLLECTION:
        try:
            return MongoListingRepository(
                uri=MONGODB_URI,
                database=MONGODB_DATABASE,
                collection=MONGODB_LISTINGS_COLLECTION,
            )
        except Exception:
            return EmptyListingRepository()
    return EmptyListingRepository()


def _mongo_id_or_clauses(listing_id: str) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [
        {"listing_id": listing_id},
        {"id": listing_id},
        {"slug": listing_id},
    ]
    object_id = _to_object_id(listing_id)
    if object_id is not None:
        clauses.append({"_id": object_id})
    return clauses


def _mongo_many_id_or_clauses(listing_ids: list[str]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [
        {"listing_id": {"$in": listing_ids}},
        {"id": {"$in": listing_ids}},
        {"slug": {"$in": listing_ids}},
    ]
    object_ids = [object_id for item in listing_ids if (object_id := _to_object_id(item)) is not None]
    if object_ids:
        clauses.append({"_id": {"$in": object_ids}})
    return clauses


def _to_object_id(value: str) -> Any | None:
    from bson import ObjectId

    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _listing_lookup_keys(raw: dict[str, Any], normalized: dict[str, Any]) -> set[str]:
    keys = {
        normalized.get("listing_id"),
        raw.get("listing_id"),
        raw.get("id"),
        raw.get("slug"),
        raw.get("_id"),
    }
    return {str(key) for key in keys if key}


def build_mongo_query(constraints: dict[str, Any]) -> dict[str, Any]:
    query: dict[str, Any] = {
        "$and": [
            {"available": {"$ne": False}},
            {"$or": [{"status": {"$in": ["active", "published"]}}, {"status": {"$exists": False}}]},
        ]
    }

    budget = constraints.get("budget") or {}
    if budget.get("max") is not None:
        query["$and"].append({
            "$or": [
                {"rent_price": {"$lte": budget["max"]}},
                {"price.rent": {"$lte": budget["max"]}},
                {"price.monthly": {"$lte": budget["max"]}},
                {"price.min": {"$lte": budget["max"]}},
                {"price.max": {"$lte": budget["max"]}},
            ]
        })
    if budget.get("min") is not None:
        query["$and"].append({
            "$or": [
                {"rent_price": {"$gte": budget["min"]}},
                {"price.rent": {"$gte": budget["min"]}},
                {"price.monthly": {"$gte": budget["min"]}},
                {"price.min": {"$gte": budget["min"]}},
                {"price.max": {"$gte": budget["min"]}},
            ]
        })

    location = constraints.get("location") or {}
    if location.get("province"):
        query["$and"].append({
            "$or": [
                {"province": location["province"]}, 
                {"location.province": location["province"]},
                {"embedding_text": {"$regex": location["province"], "$options": "i"}},
                {"summary": {"$regex": location["province"], "$options": "i"}}
            ]
        })
    if location.get("districts"):
        normalized_districts = [_normalize_location_value(item) for item in location["districts"]]
        district_regexes = [{"embedding_text": {"$regex": d, "$options": "i"}} for d in location["districts"]] + \
                           [{"summary": {"$regex": d, "$options": "i"}} for d in location["districts"]]
        
        # Thêm matching có dấu cho Quận/Huyện
        for d in location["districts"]:
            if d.startswith("quan "):
                num_or_name = d.replace("quan ", "").strip()
                district_regexes.append({"embedding_text": {"$regex": f"Quận {num_or_name}\\b", "$options": "i"}})
                district_regexes.append({"summary": {"$regex": f"Quận {num_or_name}\\b", "$options": "i"}})
            elif d.startswith("huyen "):
                num_or_name = d.replace("huyen ", "").strip()
                district_regexes.append({"embedding_text": {"$regex": f"Huyện {num_or_name}\\b", "$options": "i"}})
                district_regexes.append({"summary": {"$regex": f"Huyện {num_or_name}\\b", "$options": "i"}})

        
        query["$and"].append({
            "$or": [
                {"district": {"$in": location["districts"]}},
                {"location.district": {"$in": location["districts"]}},
                {"district_normalized": {"$in": normalized_districts}},
                {"location.district_normalized": {"$in": normalized_districts}},
            ] + district_regexes
        })
    if location.get("wards"):
        query["$and"].append({"$or": [{"ward": {"$in": location["wards"]}}, {"location.ward": {"$in": location["wards"]}}]})

    if constraints.get("categories"):
        query["$and"].append({"category": {"$in": constraints["categories"]}})
    if constraints.get("occupants"):
        query["$and"].append({"max_occupants": {"$gte": constraints["occupants"]}})
    if constraints.get("pets_required"):
        query["$and"].append({"pets_allowed": True})
    excluded = constraints.get("excluded_features") or []
    if excluded:
        query["$and"].append({"amenities": {"$nin": excluded}})

    
    import json
    # Print clean query for debug
    def serialize_query(q):
        if isinstance(q, dict):
            return {k: serialize_query(v) for k, v in q.items()}
        elif isinstance(q, list):
            return [serialize_query(v) for v in q]
        else:
            return str(q) if hasattr(q, "__class__") and q.__class__.__name__ == "ObjectId" else q
    
    print("MONGO QUERY:", json.dumps(serialize_query(query), ensure_ascii=False))
    return query


def listing_matches_constraints(listing: dict[str, Any], constraints: dict[str, Any]) -> bool:
    if not listing.get("available") or listing.get("status") not in {"active", "published"}:
        return False

    budget = constraints.get("budget") or {}
    rent = listing.get("rent_price")
    if budget.get("max") is not None and rent is not None and rent > budget["max"]:
        return False
    if budget.get("min") is not None and rent is not None and rent < budget["min"]:
        return False

    location = constraints.get("location") or {}
    if location.get("province") and listing.get("province") != location["province"]:
        return False
    if location.get("districts") and _normalize_location_value(listing.get("district")) not in {
        _normalize_location_value(item) for item in location["districts"]
    }:
        return False
    if location.get("wards") and _normalize_location_value(listing.get("ward")) not in {
        _normalize_location_value(item) for item in location["wards"]
    }:
        return False

    if constraints.get("categories") and listing.get("category") not in constraints["categories"]:
        return False

    if constraints.get("occupants") and listing.get("max_occupants") is not None:
        if listing["max_occupants"] < constraints["occupants"]:
            return False
    required = constraints.get("amenities_required") or []
    amenities = set(listing.get("amenities") or [])
    if any(not _amenity_present(amenities, item) for item in required):
        return False
    if constraints.get("pets_required") and listing.get("pets_allowed") is not True:
        return False
    vehicles = set(constraints.get("vehicles") or [])
    if "electric_bike" in vehicles and listing.get("electric_bike_allowed") is not True:
        return False
    if "motorbike" in vehicles and not _supports_motorbike_parking(listing):
        return False
    if any(item in amenities for item in (constraints.get("excluded_features") or [])):
        return False
    return True


AMENITY_EQUIVALENTS: dict[str, set[str]] = {
    "air_conditioner": {"air_conditioner", "may_lanh", "mlanh-y"},
    "may_lanh": {"air_conditioner", "may_lanh", "mlanh-y"},
    "mezzanine": {"mezzanine", "gac", "gac-y"},
    "gac": {"mezzanine", "gac", "gac-y"},
    "kitchen": {"kitchen", "ke_bep", "bep"},
    "refrigerator": {"refrigerator", "tu_lanh"},
    "hot_water": {"hot_water", "nuoc_nong"},
    "bed": {"bed", "giuong"},
    "mattress": {"mattress", "nem"},
    "wardrobe": {"wardrobe", "tu_quan_ao"},
    "elevator": {"elevator", "thang_may"},
    "washing_machine": {"washing_machine", "may_giat"},
    "balcony": {"balcony", "ban_cong", "balcony-y"},
    "window": {"window", "cua_so", "window-y"},
    "free_hours": {"free_hours", "gio_tu_do"},
    "private_bathroom": {"private_bathroom", "toilet_rieng"},
    "ev_charging": {"ev_charging", "sac_xe_dien", "xedien-y"},
}


def _amenity_present(amenities: set[str], required: Any) -> bool:
    required_text = str(required)
    allowed = AMENITY_EQUIVALENTS.get(required_text, {required_text})
    return bool(amenities & allowed)


def _supports_motorbike_parking(listing: dict[str, Any]) -> bool:
    amenities = set(listing.get("amenities") or [])
    vehicles = set(listing.get("vehicles_allowed") or listing.get("vehicles") or [])
    fees = listing.get("fees") if isinstance(listing.get("fees"), dict) else {}
    return (
        listing.get("shared_parking") is True
        or "motorbike" in vehicles
        or "xe_may" in vehicles
        or bool({"parking", "cho_de_xe", "ham_xe"} & amenities)
        or fees.get("parking") is not None
    )


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
    text = re.sub(r"\b(quan|q\.?|district)\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()
