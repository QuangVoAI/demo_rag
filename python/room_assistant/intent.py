"""Deterministic intent and constraint patch parser.

This module intentionally returns a structured patch instead of letting a model
overwrite full session state.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .schemas import ALLOWED_OPERATION_PATHS, INTENTS, OP_TYPES, ParsedRequest


ACTION_KEYWORDS = {
    "dat_lich": ("đặt lịch", "dat lich", "hẹn xem", "hen xem", "xem phòng giúp"),
    "message_owner": ("nhắn chủ", "nhan chu", "gửi tin", "gui tin", "liên hệ chủ", "lien he chu"),
    "save_favorite": ("lưu phòng", "luu phong", "yêu thích", "yeu thich"),
    "hold_room": ("giữ chỗ", "giu cho", "giữ phòng", "giu phong"),
    "payment": ("thanh toán", "thanh toan", "đặt cọc", "dat coc", "chuyển khoản", "chuyen khoan"),
    "edit_listing": ("sửa thông tin phòng", "sua thong tin phong", "đổi giá phòng", "doi gia phong"),
    "negotiate": ("thương lượng", "thuong luong", "trả giá", "tra gia", "ép giá", "ep gia"),
}

AMENITY_ALIASES = {
    "máy lạnh": "air_conditioner",
    "may lanh": "air_conditioner",
    "điều hòa": "air_conditioner",
    "dieu hoa": "air_conditioner",
    "ban công": "balcony",
    "ban cong": "balcony",
    "máy giặt": "washing_machine",
    "may giat": "washing_machine",
    "wc riêng": "private_bathroom",
    "toilet riêng": "private_bathroom",
    "gác": "mezzanine",
    "gac": "mezzanine",
    "bếp": "kitchen",
    "bep": "kitchen",
    "cửa sổ": "window",
    "cua so": "window",
}

SOFT_PREFERENCE_ALIASES = {
    "yên tĩnh": "quiet",
    "yen tinh": "quiet",
    "thoáng": "airy",
    "thoang": "airy",
    "sáng": "bright",
    "sang": "bright",
    "gần tiện ích": "near_amenities",
    "gan tien ich": "near_amenities",
    "học tập": "study_friendly",
    "hoc tap": "study_friendly",
}


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(text).lower()).strip()


def _money_to_vnd(raw: str, unit: str | None) -> int:
    raw = raw.strip()
    unit_norm = _norm(unit or "")
    separators = raw.count(".") + raw.count(",")
    if separators > 1:
        return int(re.sub(r"\D", "", raw))
    normalized_raw = raw.replace(",", ".")
    if separators == 1 and not unit_norm:
        whole, frac = re.split(r"[\.,]", raw, maxsplit=1)
        if len(frac) == 3 and len(whole) <= 3:
            return int(whole + frac)
    value = float(normalized_raw)
    if unit_norm in {"tr", "trieu", "million", "m"}:
        return int(value * 1_000_000)
    if unit_norm in {"k", "nghin"}:
        return int(value * 1_000)
    if value < 1000:
        return int(value * 1_000_000)
    return int(value)


def _append_unique(ops: list[dict[str, Any]], op: str, path: str, value: Any = None) -> None:
    if op not in OP_TYPES or path not in ALLOWED_OPERATION_PATHS:
        return
    item: dict[str, Any] = {"op": op, "path": path}
    if op != "clear":
        item["value"] = value
    if item not in ops:
        ops.append(item)


def _extract_listing_ids(text: str) -> list[str]:
    ids: list[str] = []
    patterns = [
        r"#([A-Za-z0-9][A-Za-z0-9_-]{1,40})",
        r"\b(?:listing|phòng|phong|mã|ma)\s+([A-Za-z0-9][A-Za-z0-9_-]{1,40})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            listing_id = match.group(1).strip()
            if listing_id not in ids:
                ids.append(listing_id)
    return ids


def _extract_budget(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    money = r"(\d[\d\.,]*)\s*(triệu|trieu|tr|k|nghìn|nghin|vnd|đ|d)?"
    compact_million = re.search(r"\b(\d+)\s*(?:tr|trieu)\s*(\d+)\b", normalized)
    if compact_million:
        value = f"{compact_million.group(1)}.{compact_million.group(2)}"
        _append_unique(ops, "set", "budget.max", _money_to_vnd(value, "trieu"))
        return

    max_patterns = (
        rf"(?:tối đa|toi da|duoi|dưới|không quá|khong qua|ngân sách|ngan sach|budget).*?{money}",
        rf"{money}\s*(?:đổ lại|do lai|tro xuong|trở xuống)",
    )
    min_patterns = (
        rf"(?:tối thiểu|toi thieu|trên|tren|hơn|hon|từ|tu)\s*{money}",
    )
    for pattern in max_patterns:
        match = re.search(pattern, normalized)
        if match:
            _append_unique(ops, "set", "budget.max", _money_to_vnd(match.group(1), match.group(2)))
            break
    for pattern in min_patterns:
        match = re.search(pattern, normalized)
        if match:
            _append_unique(ops, "set", "budget.min", _money_to_vnd(match.group(1), match.group(2)))
            break

    if "tang ngan sach" in normalized or "tăng ngân sách" in text.lower():
        match = re.search(rf"(?:lên|len)\s*{money}", normalized)
        if match:
            _append_unique(ops, "set", "budget.max", _money_to_vnd(match.group(1), match.group(2)))


def _extract_location(normalized: str, ops: list[dict[str, Any]]) -> None:
    districts = []
    for match in re.finditer(r"\bquan\s+([a-z0-9 ]{1,24})", normalized):
        value = match.group(1).strip()
        value = re.split(r"\b(?:gan|duoi|tren|co|va|,|\.)\b", value)[0].strip()
        if value and value not in districts:
            districts.append(f"quan {value}")
    for district in districts:
        _append_unique(ops, "append", "location.districts", district)

    landmarks = []
    for match in re.finditer(r"\b(?:gan|gần)\s+([a-z0-9 ]{2,40})", normalized):
        value = match.group(1).strip()
        value = re.split(r"\b(?:duoi|tren|co|va|,|\.)\b", value)[0].strip()
        if value and value not in landmarks:
            landmarks.append(value)
    for landmark in landmarks:
        _append_unique(ops, "append", "location.near_landmarks", landmark)


def _extract_people_and_pets(normalized: str, ops: list[dict[str, Any]]) -> None:
    match = re.search(r"(\d+)\s*(?:nguoi|người|ban|bạn)\b", normalized)
    if match:
        _append_unique(ops, "set", "occupants", int(match.group(1)))

    if "nuoi meo" in normalized or "nuôi mèo" in normalized:
        _append_unique(ops, "append", "pets_required", "cat")
    if "nuoi cho" in normalized or "nuôi chó" in normalized:
        _append_unique(ops, "append", "pets_required", "dog")
    if "xe dien" in normalized or "xe điện" in normalized:
        _append_unique(ops, "append", "vehicles", "electric_bike")


def _extract_amenities(text: str, normalized: str, ops: list[dict[str, Any]]) -> None:
    remove_mode = any(token in normalized for token in ("bo ", "khong can", "loai ", "xoa "))
    for alias, canonical in AMENITY_ALIASES.items():
        if _norm(alias) in normalized:
            _append_unique(
                ops,
                "remove" if remove_mode else "append",
                "amenities_required",
                canonical,
            )

    for alias, canonical in SOFT_PREFERENCE_ALIASES.items():
        if _norm(alias) in normalized:
            _append_unique(ops, "append", "amenities_preferred", canonical)

    if any(token in normalized for token in ("xoa het dieu kien", "clear dieu kien", "bo het dieu kien")):
        for path in (
            "location.districts",
            "location.wards",
            "location.near_landmarks",
            "vehicles",
            "pets_required",
            "amenities_required",
            "amenities_preferred",
            "excluded_features",
        ):
            _append_unique(ops, "clear", path)


def _requested_action(normalized: str) -> str | None:
    for action, keywords in ACTION_KEYWORDS.items():
        if any(_norm(keyword) in normalized for keyword in keywords):
            return action
    return None


def _classify_intent(text: str, normalized: str, current_state: dict[str, Any] | None, action: str | None, ids: list[str]) -> str:
    if action:
        return "REQUEST_ACTION"
    if any(token in normalized for token in ("so sanh", "khac nhau", "nen chon phong nao")):
        return "COMPARE_ROOMS"
    if any(token in normalized for token in ("tong chi phi", "chi phi", "tien coc", "phi hang thang", "uoc tinh")):
        return "CALCULATE_COST"
    if any(token in normalized for token in ("tuong tu", "giong phong", "phong giong")):
        return "FIND_SIMILAR"
    if any(token in normalized for token in ("tom tat", "uu diem", "han che", "diem manh")):
        return "SUMMARIZE_ROOM"
    if ids or "phong nay" in normalized or "dang xem" in normalized:
        return "ASK_ABOUT_ROOM"
    if any(token in normalized for token in ("them ", "bo ", "khong can", "tang ngan sach", "giam ngan sach", "doi sang")):
        return "REFINE_SEARCH"
    if any(token in normalized for token in ("tim phong", "phong tro", "nha tro", "can ho", "studio", "thue phong", "phong")):
        return "SEARCH_ROOM"
    if current_state and current_state.get("last_intent") in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        return "REFINE_SEARCH"
    return "GENERAL_HELP"


def parse_intent_and_constraint_patch(
    question: str,
    current_state: dict[str, Any] | None = None,
) -> ParsedRequest:
    """Return a validated structured output for one user turn."""
    text = question or ""
    normalized = _norm(text)
    operations: list[dict[str, Any]] = []

    _extract_budget(text, normalized, operations)
    _extract_location(normalized, operations)
    _extract_people_and_pets(normalized, operations)
    _extract_amenities(text, normalized, operations)

    referenced_listing_ids = _extract_listing_ids(text)
    action = _requested_action(normalized)
    intent = _classify_intent(text, normalized, current_state, action, referenced_listing_ids)
    if intent not in INTENTS:
        intent = "GENERAL_HELP"

    current_listing_id = referenced_listing_ids[0] if referenced_listing_ids else None

    return {
        "intent": intent,
        "operations": operations,
        "current_listing_id": current_listing_id,
        "referenced_listing_ids": referenced_listing_ids,
        "requested_action": action,
    }
