"""Comparison helpers cho room assistant."""

from __future__ import annotations

from typing import Any


def _comparison_text(question: str) -> str:
    import unicodedata

    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(question or "").lower())
        if unicodedata.category(ch) != "Mn"
    ).replace("đ", "d")


def _location_quality_score(row: dict[str, Any]) -> float:
    import re

    score = 0.0
    if row.get("district"):
        score += 1.0
    if row.get("ward"):
        score += 1.0
    if row.get("address"):
        score += 1.0
    nearby_text = str(row.get("tien_ich_xq") or "")
    nearby_parts = [
        part.strip()
        for part in re.split(r"[\n,;/|-]+", nearby_text)
        if part and part.strip()
    ]
    score += min(len(nearby_parts), 4)
    score += min(float(row.get("amenities_count") or 0), 4.0) * 0.25
    if row.get("available"):
        score += 0.5
    return score


def comparison_reason(row: dict[str, Any], question: str) -> str:
    normalized = _comparison_text(question)
    if "khu vuc tot hon" in normalized:
        return ", khu vực có nhiều thông tin địa chỉ và tiện ích xung quanh hơn"
    if any(phrase in normalized for phrase in ("rong hon", "lon hon", "dien tich lon hon")):
        return ", phù hợp nếu bạn ưu tiên diện tích rộng hơn"
    if any(phrase in normalized for phrase in ("re hon", "gia tot hon", "tiet kiem hon")):
        return ", phù hợp nếu bạn ưu tiên mức giá tiết kiệm hơn"
    return ""


def best_room_from_comparison(
    rows: list[dict[str, Any]],
    constraints: dict[str, Any],
    question: str = "",
) -> dict[str, Any] | None:
    if not rows:
        return None
    budget = constraints.get("budget") or {}
    max_price = budget.get("max")
    min_price = budget.get("min")
    occupants = constraints.get("occupants")
    normalized_question = _comparison_text(question)
    wants_location = "khu vuc tot hon" in normalized_question
    wants_cheaper = any(phrase in normalized_question for phrase in ("re hon", "gia tot hon", "tiet kiem hon"))
    wants_larger = any(phrase in normalized_question for phrase in ("rong hon", "lon hon", "dien tich lon hon"))
    wants_fit_people = "phu hop hon" in normalized_question and occupants

    def score(row: dict[str, Any]) -> tuple[float, ...]:
        rent = row.get("rent_price")
        area = row.get("area_m2") or 0
        location_score = _location_quality_score(row)
        available = 1.0 if row.get("available") else 0.0
        in_budget = 1
        if rent is not None:
            if max_price is not None and rent > max_price:
                in_budget = 0
            if min_price is not None and rent < min_price:
                in_budget = 0
        cheaper = -(float(rent) if rent is not None else float("inf"))
        if wants_location:
            return (location_score, available, float(in_budget), float(area), cheaper)
        if wants_larger or (constraints.get("area") or {}).get("preference") == "larger":
            return (float(in_budget), float(area), available, location_score, cheaper)
        if wants_cheaper:
            return (float(in_budget), cheaper, available, location_score, float(area))
        if wants_fit_people:
            min_area = max(int(occupants or 1) * 8, 16)
            occupant_fit = 1.0 if area >= min_area else 0.0
            return (occupant_fit, float(in_budget), float(area), available, cheaper)
        return (float(in_budget), available, location_score, float(area), cheaper)

    return max(rows, key=score)
