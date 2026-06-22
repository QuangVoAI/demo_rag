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
        self._collection = self._client[database][collection]

    def search_by_constraints(self, constraints: dict[str, Any], limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        query = build_mongo_query(constraints)
        cursor = (
            self._collection
            .find(query)
            .skip(max(offset, 0))
            .limit(max(limit, 1))
        )
        return [item for item in (normalize_listing(doc) for doc in cursor) if item]

    def get_by_id(self, listing_id: str) -> dict[str, Any] | None:
        query = {
            "$or": [
                {"listing_id": str(listing_id)},
                {"id": str(listing_id)},
                {"slug": str(listing_id)},
            ]
        }
        doc = self._collection.find_one(query)
        return normalize_listing(doc)

    def get_many_by_ids(self, listing_ids: list[str]) -> list[dict[str, Any]]:
        ids = [str(item) for item in listing_ids]
        if not ids:
            return []
        cursor = self._collection.find({
            "$or": [
                {"listing_id": {"$in": ids}},
                {"id": {"$in": ids}},
                {"slug": {"$in": ids}},
            ]
        })
        by_id = {
            item["listing_id"]: item
            for item in (normalize_listing(doc) for doc in cursor)
            if item
        }
        return [by_id[item] for item in ids if item in by_id]

    def get_current_version(self, listing_id: str) -> int | None:
        listing = self.get_by_id(listing_id)
        if not listing:
            return None
        return int(listing.get("source_version") or 0)

    def iter_listing_ids(self, batch_size: int = 100, resume_after: str | None = None) -> Iterable[list[str]]:
        query: dict[str, Any] = {}
        if resume_after:
            query["listing_id"] = {"$gt": resume_after}
        cursor = self._collection.find(query, {"listing_id": 1, "id": 1, "slug": 1}).sort("listing_id", 1)

        batch: list[str] = []
        for doc in cursor:
            listing_id = doc.get("listing_id") or doc.get("id") or doc.get("slug")
            if not listing_id:
                continue
            batch.append(str(listing_id))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


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
            ]
        })
    if budget.get("min") is not None:
        query["$and"].append({
            "$or": [
                {"rent_price": {"$gte": budget["min"]}},
                {"price.rent": {"$gte": budget["min"]}},
                {"price.monthly": {"$gte": budget["min"]}},
            ]
        })

    location = constraints.get("location") or {}
    if location.get("province"):
        query["$and"].append({"$or": [{"province": location["province"]}, {"location.province": location["province"]}]})
    if location.get("districts"):
        normalized_districts = [_normalize_location_value(item) for item in location["districts"]]
        query["$and"].append({
            "$or": [
                {"district": {"$in": location["districts"]}},
                {"location.district": {"$in": location["districts"]}},
                {"district_normalized": {"$in": normalized_districts}},
                {"location.district_normalized": {"$in": normalized_districts}},
            ]
        })
    if location.get("wards"):
        query["$and"].append({"$or": [{"ward": {"$in": location["wards"]}}, {"location.ward": {"$in": location["wards"]}}]})

    if constraints.get("occupants"):
        query["$and"].append({"max_occupants": {"$gte": constraints["occupants"]}})
    if constraints.get("amenities_required"):
        query["$and"].append({"amenities": {"$all": constraints["amenities_required"]}})
    if constraints.get("pets_required"):
        query["$and"].append({"pets_allowed": True})
    if "electric_bike" in (constraints.get("vehicles") or []):
        query["$and"].append({"electric_bike_allowed": True})

    excluded = constraints.get("excluded_features") or []
    if excluded:
        query["$and"].append({"amenities": {"$nin": excluded}})

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

    if constraints.get("occupants") and listing.get("max_occupants") is not None:
        if listing["max_occupants"] < constraints["occupants"]:
            return False
    required = constraints.get("amenities_required") or []
    amenities = set(listing.get("amenities") or [])
    if any(item not in amenities for item in required):
        return False
    if constraints.get("pets_required") and listing.get("pets_allowed") is not True:
        return False
    if "electric_bike" in (constraints.get("vehicles") or []) and listing.get("electric_bike_allowed") is not True:
        return False
    if any(item in amenities for item in (constraints.get("excluded_features") or [])):
        return False
    return True


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
