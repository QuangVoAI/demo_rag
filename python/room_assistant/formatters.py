"""Helpers định dạng và trình bày cho room assistant."""

from __future__ import annotations

from typing import Any


AMENITY_LABELS: dict[str, str] = {
    "air_conditioner": "Máy lạnh",
    "balcony": "Ban công",
    "window": "Cửa sổ",
    "washing_machine": "Máy giặt",
    "private_bathroom": "WC riêng",
    "mezzanine": "Gác",
    "kitchen": "Bếp",
    "refrigerator": "Tủ lạnh",
    "hot_water": "Nước nóng",
    "bed": "Giường",
    "mattress": "Nệm",
    "wardrobe": "Tủ quần áo",
    "elevator": "Thang máy",
    "wifi": "Wifi",
    "ev_charging": "Sạc xe điện",
    "free_hours": "Giờ tự do",
    "pets_allowed": "Cho nuôi thú cưng",
}

FEATURE_FACT_LABELS: tuple[str, ...] = (
    "Máy lạnh",
    "Ban công",
    "Cửa sổ",
    "Wifi",
    "Gác",
    "Toilet",
    "Giờ giấc",
    "Máy giặt",
    "Thú cưng",
    "Để xe",
    "Thang máy",
    "Kệ bếp",
    "Nước nóng",
    "Tủ lạnh",
    "Giường",
    "Nệm",
    "Tủ quần áo",
)


def format_vnd(value: Any) -> str:
    if value is None:
        return "chưa rõ"
    try:
        return f"{int(value):,} VND".replace(",", ".")
    except Exception:
        return str(value)


def cost_item_label(name: str) -> str:
    labels = {
        "rent_first_month": "Tiền thuê tháng đầu",
        "deposit": "Tiền cọc",
        "fee_electricity": "Tiền điện",
        "fee_water": "Tiền nước",
        "fee_management": "Phí quản lý",
        "fee_parking": "Phí gửi xe",
        "fee_wifi": "Wifi",
        "fee_washing_machine": "Máy giặt",
    }
    if name.startswith("rent_") and name.endswith("_months"):
        parts = name.split("_")
        if len(parts) >= 2:
            return f"Tiền thuê {parts[1]} tháng"
    if name in labels:
        return labels[name]
    if name.startswith("fee_"):
        return "Phí " + name.removeprefix("fee_").replace("_", " ")
    return name


def extract_feature_status(text: str, label: str) -> str | None:
    if not text:
        return None
    import re

    match = re.search(rf"(?im)^\s*-\s*{re.escape(label)}\s*:\s*([^\n\r]+)", text)
    return match.group(1).strip() if match else None


def room_text_has_amenity(room: dict[str, Any], amenity: str) -> bool:
    label = AMENITY_LABELS.get(amenity)
    if not label:
        return False
    text = str(room.get("embedding_text") or room.get("description") or "")
    if not text:
        return False
    import re

    return bool(re.search(rf"{re.escape(label)}\s*:\s*(?:Có|Riêng|Tự do|True|Yes|Free)", text, re.IGNORECASE))


def verified_amenity_labels(room: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
    room_amenities = {str(item).strip().lower() for item in (room.get("amenities") or [])}
    required = [str(item).strip().lower() for item in (constraints.get("amenities_required") or [])]
    preferred = [str(item).strip().lower() for item in (constraints.get("amenities_preferred") or [])]

    labels: list[str] = []
    for amenity in required + preferred:
        if amenity in room_amenities or room_text_has_amenity(room, amenity):
            label = AMENITY_LABELS.get(amenity, amenity)
            if label not in labels:
                labels.append(label)

    if not labels:
        for amenity in sorted(room_amenities):
            label = AMENITY_LABELS.get(amenity)
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= 4:
                break
    return labels[:4]


def room_feature_facts(room: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    text = str(room.get("embedding_text") or "")
    for label in FEATURE_FACT_LABELS:
        value = extract_feature_status(text, label)
        if value:
            facts.append(f"{label}: {value}")
    if facts:
        return facts
    return [
        AMENITY_LABELS.get(str(item), str(item))
        for item in (room.get("amenities") or [])
        if item
    ]
