"""Shared schemas and constants for the Nhatrovn room assistant."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict


Intent = Literal[
    "SEARCH_ROOM",
    "REFINE_SEARCH",
    "ASK_ABOUT_ROOM",
    "CALCULATE_COST",
    "COMPARE_ROOMS",
    "FIND_SIMILAR",
    "SUMMARIZE_ROOM",
    "GENERAL_HELP",
    "REQUEST_ACTION",
]

OP_TYPES = {"set", "remove", "append", "replace", "clear"}

INTENTS: tuple[str, ...] = (
    "SEARCH_ROOM",
    "REFINE_SEARCH",
    "ASK_ABOUT_ROOM",
    "CALCULATE_COST",
    "COMPARE_ROOMS",
    "FIND_SIMILAR",
    "SUMMARIZE_ROOM",
    "GENERAL_HELP",
    "REQUEST_ACTION",
)

READ_ONLY_TOOLS: tuple[str, ...] = (
    "search_listings",
    "get_listing_detail",
    "retrieve_listing_context",
    "retrieve_faq",
    "calculate_cost_estimate",
    "compare_listings",
    "find_similar_listings",
)

MAX_READ_TOOL_CALLS_PER_TURN = 3

ALLOWED_OPERATION_PATHS: set[str] = {
    "location.province",
    "location.districts",
    "location.wards",
    "location.near_landmarks",
    "location.max_distance_km",
    "budget.min",
    "budget.max",
    "budget.type",
    "occupants",
    "vehicles",
    "pets_required",
    "amenities_required",
    "amenities_preferred",
    "excluded_features",
    "move_in_date",
}

LIST_OPERATION_PATHS: set[str] = {
    "location.districts",
    "location.wards",
    "location.near_landmarks",
    "vehicles",
    "pets_required",
    "amenities_required",
    "amenities_preferred",
    "excluded_features",
}


class Operation(TypedDict, total=False):
    op: str
    path: str
    value: Any


class ParsedRequest(TypedDict):
    intent: str
    operations: list[Operation]
    current_listing_id: str | None
    referenced_listing_ids: list[str]
    requested_action: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_constraints() -> dict[str, Any]:
    return {
        "location": {
            "province": None,
            "districts": [],
            "wards": [],
            "near_landmarks": [],
            "max_distance_km": None,
        },
        "budget": {
            "min": None,
            "max": None,
            "type": "rent_only",
        },
        "occupants": None,
        "vehicles": [],
        "pets_required": [],
        "amenities_required": [],
        "amenities_preferred": [],
        "excluded_features": [],
        "move_in_date": None,
    }


def default_session_state(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "constraints": default_constraints(),
        "current_listing_id": None,
        "selected_listing_ids": [],
        "last_result_ids": [],
        "last_intent": None,
        "conversation_summary": "",
        "state_version": 1,
        "updated_at": utc_now_iso(),
    }


def clone_session_state(state: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(state)


def normalize_listing(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize listing documents from Mongo/fakes into one read model."""
    if not raw:
        return None

    listing_id = (
        raw.get("listing_id")
        or raw.get("id")
        or raw.get("_id")
        or raw.get("slug")
    )
    if listing_id is None:
        return None

    price = raw.get("price") if isinstance(raw.get("price"), dict) else {}
    fees = raw.get("fees") if isinstance(raw.get("fees"), dict) else {}
    location = raw.get("location") if isinstance(raw.get("location"), dict) else {}

    rent = (
        raw.get("rent_price")
        or raw.get("monthly_rent")
        or price.get("rent")
        or price.get("monthly")
    )

    normalized = dict(raw)
    normalized["listing_id"] = str(listing_id)
    normalized["title"] = raw.get("title") or raw.get("name") or f"Phong {listing_id}"
    normalized["description"] = raw.get("description") or raw.get("summary") or ""
    normalized["status"] = raw.get("status") or ("active" if raw.get("available", True) else "unavailable")
    normalized["available"] = bool(raw.get("available", normalized["status"] in {"active", "published"}))
    normalized["rent_price"] = int(rent) if isinstance(rent, (int, float)) else rent
    normalized["deposit"] = raw.get("deposit") or price.get("deposit")
    normalized["fees"] = fees
    normalized["address"] = raw.get("address") or location.get("address") or ""
    normalized["province"] = raw.get("province") or location.get("province")
    normalized["district"] = raw.get("district") or location.get("district")
    normalized["ward"] = raw.get("ward") or location.get("ward")
    normalized["lat"] = raw.get("lat") or location.get("lat")
    normalized["lng"] = raw.get("lng") or location.get("lng")
    normalized["area_m2"] = raw.get("area_m2") or raw.get("area")
    normalized["amenities"] = list(raw.get("amenities") or [])
    normalized["max_occupants"] = raw.get("max_occupants")
    normalized["pets_allowed"] = raw.get("pets_allowed")
    normalized["vehicles_allowed"] = list(raw.get("vehicles_allowed") or raw.get("vehicles") or [])
    normalized["electric_bike_allowed"] = raw.get("electric_bike_allowed")
    normalized["available_from"] = raw.get("available_from")
    normalized["source_version"] = int(raw.get("source_version") or raw.get("version") or 0)
    normalized["updated_at"] = raw.get("updated_at")
    return normalized


def public_session_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return session state safe for API responses."""
    return {
        "session_id": state.get("session_id"),
        "constraints": state.get("constraints", {}),
        "current_listing_id": state.get("current_listing_id"),
        "selected_listing_ids": state.get("selected_listing_ids", []),
        "last_result_ids": state.get("last_result_ids", []),
        "last_intent": state.get("last_intent"),
        "conversation_summary": state.get("conversation_summary", ""),
        "state_version": state.get("state_version", 1),
        "updated_at": state.get("updated_at"),
    }

