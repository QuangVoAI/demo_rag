"""Post-generation checks: answer must not contradict grounded room inventory."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# Absolute zero-inventory claims — unsafe when tool_results already returned rooms.
_ZERO_INVENTORY_PHRASES = (
    "khong tim thay phong phu hop",
    "không tìm thấy phòng phù hợp",
    "khong tim thay phong nao",
    "không tìm thấy phòng nào",
    "khong co phong nao",
    "không có phòng nào",
    "chua co phong nao",
    "chưa có phòng nào",
    "khong con phong",
    "không còn phòng",
    "tim moi mat ma chua thay phong nao",
    "tìm mỏi mắt mà chưa thấy phòng nào",
    "khong co phong phu hop",
    "không có phòng phù hợp",
)

_ROOM_ID_PATTERN = re.compile(r"#?([0-9a-f]{24})\b", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def extract_room_ids_from_answer(answer: str) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in _ROOM_ID_PATTERN.finditer(answer or "")))


def _answer_mentions_grounded_room(answer: str, room_ids: list[str]) -> bool:
    if not answer or not room_ids:
        return False
    lower = (answer or "").lower()
    for room_id in room_ids:
        token = str(room_id).strip().lower()
        if token and token in lower:
            return True
    return False


def answer_denies_inventory_while_rooms_exist(answer: str, room_ids: list[str]) -> bool:
    """True when answer claims zero inventory but retrieval already returned rooms."""
    if not room_ids:
        return False
    if _answer_mentions_grounded_room(answer, room_ids):
        return False
    normalized = _normalize_text(answer)
    return any(phrase in normalized for phrase in _ZERO_INVENTORY_PHRASES)


def answer_cites_foreign_room_ids(answer: str, allowed_room_ids: list[str]) -> list[str]:
    """Room IDs mentioned in answer that are not part of the grounded set."""
    allowed = {str(item).strip().lower() for item in allowed_room_ids if item}
    if not allowed:
        return []
    cited = extract_room_ids_from_answer(answer)
    return [room_id for room_id in cited if room_id.lower() not in allowed]


def repair_answer_against_grounding(
    answer: str,
    *,
    intent: str,
    rooms: list[dict[str, Any]],
    template_answer: str,
) -> tuple[str, dict[str, Any]]:
    """Replace contradictory LLM answers with deterministic templates."""
    room_ids = [str(room.get("room_id")) for room in rooms if room.get("room_id")]
    meta: dict[str, Any] = {"repaired": False, "reasons": []}

    if not rooms or not (answer or "").strip():
        return answer, meta

    if answer_denies_inventory_while_rooms_exist(answer, room_ids):
        meta["repaired"] = True
        meta["reasons"].append("denied_inventory_with_rooms")
        return template_answer, meta

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"}:
        foreign = answer_cites_foreign_room_ids(answer, room_ids)
        if foreign:
            meta["repaired"] = True
            meta["reasons"].append(f"foreign_room_ids:{','.join(foreign[:3])}")
            return template_answer, meta

    return answer, meta
