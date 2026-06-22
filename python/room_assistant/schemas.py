"""Schemas và hằng số dùng chung cho Nhatrovn Room Assistant.

Mô tả các intent, operation paths, và cấu trúc session state
phù hợp với luồng tìm phòng trọ trên nhatrovn.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict


Intent = Literal[
    "SEARCH_ROOM",      # Tìm phòng mới theo tiêu chí
    "REFINE_SEARCH",    # Điều chỉnh tiêu chí tìm kiếm đang có
    "ASK_ABOUT_ROOM",   # Hỏi chi tiết về một phòng cụ thể
    "CALCULATE_COST",   # Tính chi phí ban đầu (cọc, phí phát sinh)
    "COMPARE_ROOMS",    # So sánh tối đa 3 phòng với nhau
    "FIND_SIMILAR",     # Tìm phòng tương tự phòng đang xem
    "SUMMARIZE_ROOM",   # Tóm tắt ưu / nhược điểm phòng
    "REQUEST_FAQ",      # Hỏi quy trình thuê, hợp đồng, thủ tục
    "GENERAL_HELP",     # Câu hỏi chung hoặc không xác định được
    "REQUEST_ACTION",   # Yêu cầu thao tác nghiệp vụ (đặt lịch, nhắn chủ...)
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
    "REQUEST_FAQ",
    "GENERAL_HELP",
    "REQUEST_ACTION",
)

# Danh sách tool được phép đăng ký trong ReadOnlyToolRegistry.
# Không được thêm tool có side-effect ghi dữ liệu vào đây.
READ_ONLY_TOOLS: tuple[str, ...] = (
    "search_listings",          # Tìm phòng theo constraint
    "get_listing_detail",       # Lấy chi tiết một phòng
    "retrieve_listing_context", # Lấy context đầy đủ của phòng
    "retrieve_faq",             # Trả lời câu hỏi thường gặp (hợp đồng, thủ tục)
    "calculate_cost_estimate",  # Tính chi phí ban đầu deterministic
    "compare_listings",         # So sánh tối đa 3 phòng
    "find_similar_listings",    # Tìm phòng tương tự
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
    "categories",
    "occupants",
    "vehicles",
    "pets_required",
    "amenities_required",
    "amenities_preferred",
    "excluded_features",
    "move_in_date",
}

LIST_OPERATION_PATHS: set[str] = {
    "categories",
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

    import re

    # Extract ID
    listing_id = (
        raw.get("listing_id")
        or raw.get("id")
        or raw.get("_id")
        or raw.get("slug")
        or raw.get("property_id", {}).get("$oid")
        or raw.get("property_id")
    )
    if listing_id is None:
        return None

    price = raw.get("price") if isinstance(raw.get("price"), dict) else {}
    fees = raw.get("fees") if isinstance(raw.get("fees"), dict) else {}
    location = raw.get("location") if isinstance(raw.get("location"), dict) else {}
    property_info = raw.get("property_info") if isinstance(raw.get("property_info"), dict) else {}

    rent = (
        raw.get("rent_price")
        or raw.get("monthly_rent")
        or price.get("min")
        or price.get("max")
        or price.get("rent")
        or price.get("monthly")
    )

    # Thử parse district/province từ summary hoặc embedding_text nếu không có sẵn
    raw_province = raw.get("province") or location.get("province")
    raw_district = raw.get("district") or location.get("district")
    
    embedding_text = str(raw.get("embedding_text", ""))
    summary = str(raw.get("summary", ""))
    
    if not raw_district:
        # Regex tìm "Quận X" hoặc "Huyện X"
        match = re.search(r"(Quận\s+\d+|Quận\s+[A-Z][a-z]+|Huyện\s+[A-Z][a-z]+)", embedding_text + " " + summary)
        if match:
            raw_district = match.group(1)

    if not raw_province:
        if "Hồ Chí Minh" in embedding_text or "Hồ Chí Minh" in summary:
            raw_province = "Hồ Chí Minh"
        elif "Hà Nội" in embedding_text or "Hà Nội" in summary:
            raw_province = "Hà Nội"

    normalized = dict(raw)
    normalized["listing_id"] = str(listing_id)
    normalized["title"] = raw.get("title") or raw.get("name") or f"Phòng {listing_id}"
    normalized["description"] = raw.get("description") or raw.get("summary") or ""
    normalized["status"] = raw.get("status") or ("active" if raw.get("available", True) else "unavailable")
    normalized["available"] = bool(raw.get("available", normalized["status"] in {"active", "published"}))
    normalized["rent_price"] = int(rent) if isinstance(rent, (int, float)) else rent
    normalized["deposit"] = raw.get("deposit") or price.get("deposit")
    normalized["fees"] = fees
    normalized["address"] = raw.get("address") or location.get("address") or ""
    normalized["province"] = raw_province
    normalized["district"] = raw_district
    normalized["ward"] = raw.get("ward") or location.get("ward")
    normalized["lat"] = raw.get("lat") or location.get("lat")
    normalized["lng"] = raw.get("lng") or location.get("lng")
    normalized["area_m2"] = raw.get("area_m2") or property_info.get("area_m2") or raw.get("area")
    
    # Extract amenities từ embedding_text nếu không có field amenities
    raw_amenities = raw.get("amenities") or []
    if not raw_amenities and "Tiện ích:" in embedding_text:
        try:
            amenities_str = embedding_text.split("Tiện ích:")[1].split("\n")[0]
            raw_amenities = [x.strip() for x in amenities_str.split(",")]
        except Exception:
            pass
            
    normalized["amenities"] = list(raw_amenities)
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


def unknown_listing_fields(listing: dict[str, Any]) -> list[str]:
    fields = ("rent_price", "deposit", "area_m2", "available_from", "pets_allowed")
    return [field for field in fields if listing.get(field) is None]
