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
    "search_rooms",             # Tìm phòng theo constraint
    "get_room_detail",          # Lấy chi tiết một phòng
    "retrieve_room_context",    # Lấy context đầy đủ của phòng
    "retrieve_faq",             # Trả lời câu hỏi thường gặp (hợp đồng, thủ tục)
    "calculate_cost_estimate",  # Tính chi phí ban đầu deterministic
    "compare_rooms",            # So sánh tối đa 3 phòng
    "find_similar_rooms",       # Tìm phòng tương tự
)

MAX_READ_TOOL_CALLS_PER_TURN = 3
RECENT_HISTORY_TURNS = 8

ALLOWED_OPERATION_PATHS: set[str] = {
    "location.province",
    "location.districts",
    "location.wards",
    "location.near_landmarks",
    "location.max_distance_km",
    "budget.min",
    "budget.min_operator",
    "budget.max",
    "budget.max_operator",
    "budget.type",
    "area.preference",
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
    current_room_id: str | None
    referenced_room_ids: list[str]
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
            "min_operator": None,
            "max": None,
            "max_operator": None,
            "type": "rent_only",
        },
        "area": {
            "preference": None,
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
        "current_room_id": None,
        "selected_room_ids": [],
        "last_result_ids": [],
        "last_intent": None,
        "conversation_summary": "",
        "state_version": 1,
        "updated_at": utc_now_iso(),
    }


def clone_session_state(state: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(state)


def normalize_room(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize room documents from MongoDB rooms collection into one read model.

    Rooms collection document structure:
        room_id, house_id, tien_ich_xq, house_remark, embedding_text,
        metadata: { house_name, room_code, province_code, province_name,
                     district_name, ward_name, price, status_code, status_desc,
                     allow_sale, has_image }
    """
    if not raw:
        return None

    import re

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}

    # Extract IDs
    room_id = raw.get("room_id") or raw.get("_id")
    house_id = raw.get("house_id")
    if room_id is None:
        return None

    # Price from metadata
    price = metadata.get("price") if "price" in metadata else raw.get("price", raw.get("rent_price"))

    # Status: "0" = phòng trống (available)
    status_code = metadata.get("status_code")
    if status_code is not None and str(status_code) != "":
        is_available = str(status_code) == "0"
    else:
        is_available = raw.get("available") if "available" in raw else (raw.get("status") == "active")
        if "available" not in raw and "status" not in raw:
            is_available = True

    status_desc = metadata.get("status_desc", "Còn phòng" if is_available else "Hết phòng")

    # Location from metadata
    province = metadata.get("province_name") if "province_name" in metadata else raw.get("province")
    district = metadata.get("district_name") if "district_name" in metadata else raw.get("district")
    ward = metadata.get("ward_name") if "ward_name" in metadata else raw.get("ward")

    # Title: house_name + room_code
    house_name = metadata.get("house_name", "")
    room_code = metadata.get("room_code", "")
    title = f"{house_name} - {room_code}" if house_name and room_code else house_name or room_code or raw.get("title") or f"Phòng {room_id}"

    # Extract address from embedding_text
    embedding_text = str(raw.get("embedding_text", ""))
    address = ""
    addr_match = re.search(r"Địa chỉ:\s*(.+?)(?:\n|$)", embedding_text)
    if addr_match:
        address = addr_match.group(1).strip()

    # Extract amenities from embedding_text "## Tiện ích" section
    amenities_list: list[str] = []
    amenities_section = re.search(r"## Tiện ích\n(.*?)(?:\n##|\Z)", embedding_text, re.DOTALL)
    if amenities_section:
        for line in amenities_section.group(1).strip().split("\n"):
            line = line.strip("- ").strip()
            if ": Có" in line:
                amenity_name = line.split(":")[0].strip()
                amenities_list.append(amenity_name)

    # Extract area from embedding_text
    area_m2 = None
    area_match = re.search(r"Diện tích:\s*(\d+)\s*(?:m2|m²|mét vuông|met vuong)?", embedding_text, re.IGNORECASE)
    if area_match:
        area_m2 = int(area_match.group(1))

    # Extract fees from embedding_text "## Giá & phí" section
    fees: dict[str, Any] = {}
    fees_section = re.search(r"## Giá & phí\n(.*?)(?:\n##|\Z)", embedding_text, re.DOTALL)
    if fees_section:
        for line in fees_section.group(1).strip().split("\n"):
            line = line.strip("- ").strip()
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            if key == "điện":
                fees["electricity"] = val
            elif key == "nước":
                fees["water"] = val
            elif key == "quản lý":
                fees["management"] = val
            elif key == "wifi":
                fees["wifi"] = val
            elif key == "xe":
                fees["parking"] = val
            elif key == "máy giặt":
                fees["washing_machine"] = val

    normalized = dict(raw)
    normalized["room_id"] = str(room_id)
    normalized["house_id"] = str(house_id) if house_id else None
    normalized["title"] = title
    normalized["description"] = raw.get("house_remark") or raw.get("description") or ""
    normalized["status"] = "active" if is_available else "unavailable"
    normalized["status_desc"] = status_desc
    normalized["available"] = is_available
    normalized["allow_sale"] = metadata.get("allow_sale") == "Y"
    normalized["rent_price"] = int(price) if isinstance(price, (int, float)) and price else price
    normalized["deposit"] = raw.get("deposit")
    normalized["fees"] = fees if fees else raw.get("fees", {})
    normalized["address"] = address
    normalized["province"] = province
    normalized["district"] = district
    normalized["ward"] = ward
    normalized["area_m2"] = area_m2 if area_m2 is not None else raw.get("area_m2")
    normalized["amenities"] = amenities_list if amenities_list else raw.get("amenities", [])
    normalized["embedding_text"] = embedding_text
    normalized["tien_ich_xq"] = raw.get("tien_ich_xq", "")
    normalized["has_image"] = metadata.get("has_image", False)
    normalized["updated_at"] = raw.get("updated_at")
    return normalized


def public_session_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return session state safe for API responses."""
    return {
        "session_id": state.get("session_id"),
        "constraints": state.get("constraints", {}),
        "current_room_id": state.get("current_room_id"),
        "selected_room_ids": state.get("selected_room_ids", []),
        "last_result_ids": state.get("last_result_ids", []),
        "last_intent": state.get("last_intent"),
        "conversation_summary": state.get("conversation_summary", ""),
        "state_version": state.get("state_version", 1),
        "updated_at": state.get("updated_at"),
    }


def unknown_room_fields(room: dict[str, Any]) -> list[str]:
    # available_from is not populated in the current normalized room model,
    # so treating it as "unknown" produces a false warning on every answer.
    fields = ("rent_price", "deposit", "area_m2")
    return [field for field in fields if room.get(field) is None]
