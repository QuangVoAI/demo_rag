"""VND money parsing and answer matching helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def normalize_money_text(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text).lower().replace("đ", "d")).strip()


MILLION_UNIT_ALIASES = frozenset({
    "tr", "trieu", "triệu", "million", "m",
    "cu", "chu",  # củ
    "chieu", "chiệu", "chai",  # chiệu
})

THOUSAND_UNIT_ALIASES = frozenset({"k", "nghin", "nghìn", "ngan"})


def money_to_vnd(raw: str, unit: str | None) -> int:
    """Chuyển chuỗi số tiền + đơn vị sang VND nguyên."""
    raw = raw.strip()
    unit_norm = normalize_money_text(unit or "")
    separators = raw.count(".") + raw.count(",")
    if separators > 1:
        return int(re.sub(r"\D", "", raw))
    normalized_raw = raw.replace(",", ".")
    if separators == 1 and not unit_norm:
        whole, frac = re.split(r"[\.,]", raw, maxsplit=1)
        if len(frac) == 3 and len(whole) <= 3:
            return int(whole + frac)
    value = float(normalized_raw)
    if unit_norm in MILLION_UNIT_ALIASES:
        return int(value * 1_000_000)
    if unit_norm in THOUSAND_UNIT_ALIASES:
        return int(value * 1_000)
    if value < 1000:
        return int(value * 1_000_000)
    return int(value)


def extract_colloquial_budget_vnd(normalized: str) -> int | None:
    """Nhận diện ngân sách kiểu 5 củ / 5m / 5000k / 5 chiệu (tránh nhầm Củ Chi)."""
    patterns = (
        (r"\b(\d[\d\.,]*)\s*c[uủ]\b(?! chi)", "cu"),
        (r"\b(\d[\d\.,]*)\s*(?:chieu|chiệu|chai)\b", "chieu"),
        (r"\b(\d[\d\.,]*)\s*m\b(?!\s*2|\d)", "m"),
        (r"\b(\d[\d\.,]*)\s*k\b", "k"),
    )
    for pattern, unit in patterns:
        match = re.search(pattern, normalized)
        if match:
            return money_to_vnd(match.group(1), unit)
    return None


def parse_money_amount(
    value: Any,
    *,
    fee_name: str | None = None,
    fee_context: bool = False,
) -> int | None:
    """Parse số tiền (int/float/str) sang VND — dùng chung cho budget search và tính phí."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    normalized = raw.lower().replace("đ", "d")
    norm_text = normalize_money_text(raw)

    if fee_context:
        if normalized in {"free", "miễn phí", "mien phi", "0", "0đ", "0d"}:
            return 0
        if normalized in {"không", "khong"} or "không có" in normalized:
            return 0
        if any(unit in normalized for unit in ("/kwh", "/kw", "/m3", "/m³", "/kg")):
            return None

    colloquial = extract_colloquial_budget_vnd(norm_text)
    if colloquial is not None:
        return colloquial

    match = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(tr|triệu|trieu|k|nghìn|nghin|cu|chu|chieu|chai|m)?",
        norm_text,
    )
    if not match:
        return None

    number_text = match.group(1)
    unit = match.group(2) or ""
    if unit == "m" and re.search(rf"\b{re.escape(number_text)}\s*m\s*2\b", norm_text):
        return None

    separators = number_text.count(".") + number_text.count(",")
    if not unit and separators > 1:
        digits = re.sub(r"\D", "", number_text)
        return int(digits) if digits else None

    number = float(number_text.replace(",", "."))
    unit_norm = normalize_money_text(unit)

    if unit_norm in MILLION_UNIT_ALIASES:
        return int(number * 1_000_000)
    if unit_norm in THOUSAND_UNIT_ALIASES:
        return int(number * 1_000)
    if not unit_norm and number < 1000:
        if fee_context and fee_name:
            return int(number * 1_000)
        return int(number * 1_000_000)
    return int(number)


def format_vnd(value: Any) -> str:
    if value is None:
        return "chưa rõ"
    try:
        return f"{int(value):,} VND".replace(",", ".")
    except Exception:
        return str(value)


def answer_mentions_vnd(answer: str, amount_vnd: int, *, tolerance: float = 0.02) -> bool:
    """Kiểm tra câu trả lời có nhắc đúng mức giá (chấp nhận nhiều cách viết VN)."""
    if not answer or amount_vnd <= 0:
        return False

    canonical = format_vnd(amount_vnd)
    if canonical.split()[0] in answer:
        return True

    compact_answer = re.sub(r"\D", "", answer)
    if str(amount_vnd) in compact_answer:
        return True

    millions = amount_vnd / 1_000_000
    text = normalize_money_text(answer)
    million_int = int(millions) if millions == int(millions) else None

    if million_int is not None:
        million_patterns = (
            rf"\b{million_int}[\.,]?\s*(?:trieu|tr|cu|chu|chieu|chai|m)\b",
            rf"\b{million_int}\s*(?:trieu|tr|cu|chu|chieu|chai)\b",
        )
        for pattern in million_patterns:
            if re.search(pattern, text):
                return True

    parsed_amounts = _parse_money_mentions(answer)
    for parsed in parsed_amounts:
        if abs(parsed - amount_vnd) <= max(50_000, amount_vnd * tolerance):
            return True
    return False


def _parse_money_mentions(text: str) -> list[int]:
    normalized = normalize_money_text(text)
    amounts: list[int] = []
    unit_pattern = r"(trieu|tr|cu|chu|chieu|chai|k|nghin|m|vnd|d)"
    for match in re.finditer(rf"\b(\d[\d\.,]*)\s*{unit_pattern}?\b", normalized):
        raw = match.group(1)
        unit = match.group(2) or ""
        if unit == "m" and re.search(rf"\b{re.escape(raw)}\s*m\s*2\b", normalized):
            continue
        try:
            amounts.append(money_to_vnd(raw, unit or None))
        except Exception:
            continue
    return amounts
