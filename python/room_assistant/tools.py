"""Read-only tool registry for the room assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable

from .repository import RoomRepository, normalize_room_code
from .retrieval import RoomSemanticIndex, search_rooms_with_hard_filters
from .schemas import MAX_READ_TOOL_CALLS_PER_TURN, READ_ONLY_TOOLS, unknown_room_fields


class ToolBudgetExceeded(RuntimeError):
    pass


@dataclass
class ToolExecutionContext:
    repository: RoomRepository
    semantic_index: RoomSemanticIndex | None = None
    read_tool_calls: int = 0
    write_tool_calls: int = 0
    tool_latency_ms: dict[str, int] = field(default_factory=dict)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)


class ReadOnlyToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[[dict[str, Any], ToolExecutionContext], Any]] = {
            "search_rooms": search_rooms,
            "get_room_detail": get_room_detail,
            "retrieve_room_context": retrieve_room_context,
            "retrieve_faq": retrieve_faq,
            "calculate_cost_estimate": calculate_cost_estimate,
            "compare_rooms": compare_rooms,
            "find_similar_rooms": find_similar_rooms,
        }
        extra = set(self._tools) - set(READ_ONLY_TOOLS)
        missing = set(READ_ONLY_TOOLS) - set(self._tools)
        if extra or missing:
            raise ValueError(f"Invalid read-only tool registry extra={extra} missing={missing}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute(self, name: str, args: dict[str, Any], context: ToolExecutionContext) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool is not registered or not read-only: {name}")
        if context.read_tool_calls >= MAX_READ_TOOL_CALLS_PER_TURN:
            raise ToolBudgetExceeded("max_read_tool_calls_per_turn exceeded")
        context.read_tool_calls += 1
        return self._tools[name](args, context)


def _bounded_top_k(value: Any, default: int = 5, maximum: int = 20) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))


def search_rooms(args: dict[str, Any], context: ToolExecutionContext) -> list[dict[str, Any]]:
    return search_rooms_with_hard_filters(
        query_text=args.get("query_text", ""),
        constraints=args.get("constraints", {}),
        repository=context.repository,
        semantic_index=context.semantic_index,
        top_k=_bounded_top_k(args.get("top_k", 5)),
        trace=context.retrieval_trace,
    )


def get_room_detail(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any] | None:
    room_id = _normalize_room_reference(args.get("room_id"))
    if not room_id:
        return None
    return _resolve_room_reference(room_id, context)


def retrieve_room_context(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    room = get_room_detail(args, context)
    if not room:
        return {"room": None, "context": "", "unknown": ["room_not_found"]}
    context_parts = [
        room.get("title", ""),
        room.get("description", ""),
        room.get("embedding_text", ""),
        room.get("tien_ich_xq", ""),
        room.get("address", ""),
    ]
    return {
        "room": room,
        "context": "\n".join(part for part in context_parts if part),
        "unknown": _unknown_room_fields(room),
    }


def retrieve_faq(args: dict[str, Any], context: ToolExecutionContext) -> list[dict[str, str]]:
    question = (args.get("question") or "").lower()
    from room_assistant.staff_knowledge import match_staff_faq

    matches = match_staff_faq(question)
    if matches:
        return matches[:3]

    faq = [
        {
            "topic": "booking",
            "keywords": ("đặt lịch", "dat lich", "xem phòng", "xem phong"),
            "answer": "Bạn cần tự thao tác đặt lịch hoặc xem thông tin liên hệ trên giao diện nhatrovn nếu tính năng đó có sẵn.",
        },
        {
            "topic": "deposit",
            "keywords": ("tiền cọc", "tien coc", "đặt cọc", "dat coc", "cọc"),
            "answer": "Tiền cọc phụ thuộc từng phòng. Nếu dữ liệu phòng có trường cọc, mình sẽ dùng đúng số đó; nếu thiếu thì mình sẽ báo chưa có dữ liệu.",
        },
        {
            "topic": "contract",
            "keywords": ("hợp đồng", "hop dong", "thời hạn thuê", "thoi han thue"),
            "answer": "Thông tin hợp đồng và thời hạn thuê cần đối chiếu theo từng phòng hoặc thỏa thuận với chủ nhà. Mình không tự tạo hay xác nhận hợp đồng thay bạn.",
        },
        {
            "topic": "fees",
            "keywords": ("phí", "phi", "điện", "dien", "nước", "nuoc", "wifi", "giữ xe", "giu xe"),
            "answer": "Các khoản phí như điện, nước, wifi, giữ xe chỉ được xem là xác nhận khi có trong dữ liệu phòng hoặc phần ước tính chi phí.",
        },
        {
            "topic": "pets",
            "keywords": ("nuôi mèo", "nuoi meo", "nuôi chó", "nuoi cho", "thú cưng", "thu cung"),
            "answer": "Việc nuôi thú cưng phụ thuộc quy định của từng phòng. Nếu dữ liệu thiếu, mình sẽ nói rõ là chưa có dữ liệu.",
        },
        {
            "topic": "payment",
            "keywords": ("thanh toán", "thanh toan", "chuyển khoản", "chuyen khoan"),
            "answer": "Mình không xử lý thanh toán hay giữ tiền. Bạn chỉ nên thanh toán qua kênh chính thức hoặc sau khi xác minh trực tiếp với bên cho thuê.",
        },
    ]
    matches = [
        {"topic": item["topic"], "answer": item["answer"]}
        for item in faq
        if any(keyword in question for keyword in item["keywords"])
    ]
    if matches:
        return matches[:3]
    return []


def _extract_rate(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    import re
    match = re.search(r"(\d+k|\d{3,}(?:\.\d{3})*)", value.lower())
    if not match:
        return None
    raw = match.group(1).replace(".", "")
    if raw.endswith("k"):
        return int(raw[:-1]) * 1000
    try:
        return int(raw)
    except Exception:
        return None

def _extract_unit(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    val_lower = value.lower()
    if "kwh" in val_lower or "ký" in val_lower or "số" in val_lower:
        return "kWh"
    if "khối" in val_lower or "m3" in val_lower:
        return "m³"
    if "người" in val_lower:
        return "người"
    if "xe" in val_lower or "chiếc" in val_lower:
        return "xe"
    return None

def calculate_cost_estimate(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    room = args.get("room")
    room_id = _normalize_room_reference(args.get("room_id"))
    if not room and room_id:
        room = _resolve_room_reference(room_id, context)
    if not room:
        return {"available": False, "unknown_inputs": ["room_not_found"]}

    rent = room.get("rent_price")
    deposit = room.get("deposit")
    deposit_options = room.get("deposit_options")
    fees = room.get("fees") or {}
    
    fixed_items: list[dict[str, Any]] = []
    variable_items: list[dict[str, Any]] = []
    unknown_inputs: list[str] = []
    
    display_parts = []

    if rent is None:
        unknown_inputs.append("rent_price")
    else:
        rent_amount = _money_value_or_none(rent)
        if rent_amount is None:
            unknown_inputs.append("rent_price")
        else:
            fixed_items.append({"field": "monthly_rent", "amount": rent_amount, "period": "month"})
            display_parts.append(f"{rent_amount:,}".replace(",", "."))

    for name, value in fees.items():
        if value is None:
            unknown_inputs.append(f"fees.{name}")
            continue
        amount = _money_amount_or_none(value, name, args)
        if amount is not None:
            fixed_items.append({"field": name, "amount": amount, "period": "month"})
            display_parts.append(f"{amount:,}".replace(",", "."))
        else:
            rate = _extract_rate(value)
            if rate is not None:
                unit = _extract_unit(value) or "đơn vị"
                variable_items.append({
                    "field": name,
                    "rate": rate,
                    "unit": unit,
                    "quantity_key": f"{name}_qty",
                    "quantity": args.get(f"{name}_qty")
                })
                display_parts.append(f"{rate:,}".replace(",", ".") + f" × số {unit}")
            else:
                unknown_inputs.append(f"fees.{name}")

    display_formula = " + ".join(display_parts) if display_parts else None
    base_subtotal = sum(item["amount"] for item in fixed_items)
    
    initial_payment_options = []
    if deposit_options:
        for opt in deposit_options:
            dep_amt = _money_value_or_none(opt.get("deposit"))
            if dep_amt is not None:
                initial_payment_options.append({
                    "deposit": dep_amt,
                    "hold_days": opt.get("hold_days"),
                    "refundability": opt.get("refundability", "unknown"),
                    "known_initial_subtotal": base_subtotal + dep_amt
                })
    elif deposit is not None:
        dep_amt = _money_value_or_none(deposit)
        if dep_amt is not None:
            initial_payment_options.append({
                "deposit": dep_amt,
                "hold_days": None,
                "refundability": "unknown",
                "known_initial_subtotal": base_subtotal + dep_amt
            })
    else:
        unknown_inputs.append("deposit")

    # Compute legacy compatibility fields
    rental_months = args.get("rental_months") or 1
    monthly_rent_val = 0
    monthly_fees_val = 0
    for item in fixed_items:
        if item["field"] == "monthly_rent":
            monthly_rent_val = item["amount"]
        else:
            monthly_fees_val += item["amount"]

    recurring_fees_for_period = monthly_fees_val * rental_months
    dep_amt_val = 0
    if initial_payment_options:
        dep_amt_val = initial_payment_options[0]["deposit"]
    total_period_cost = (monthly_rent_val * rental_months) + recurring_fees_for_period + dep_amt_val
    total_initial_cost = monthly_rent_val + monthly_fees_val + dep_amt_val

    legacy_unknown = []
    not_calculated_list = []
    for item in unknown_inputs:
        if item.startswith("fees."):
            field_name = item.split(".", 1)[1]
            if field_name in fees:
                not_calculated_list.append({
                    "name": item,
                    "value": fees[field_name]
                })
        else:
            legacy_unknown.append(item)

    for var_item in variable_items:
        if var_item.get("quantity") is None:
            field_name = var_item["field"]
            if not any(x["name"] == f"fees.{field_name}" for x in not_calculated_list):
                not_calculated_list.append({
                    "name": f"fees.{field_name}",
                    "value": fees.get(field_name)
                })

    return {
        "available": True,
        "room_id": room.get("room_id"),
        "house_id": room.get("house_id"),
        "currency": "VND",
        "fixed_items": fixed_items,
        "variable_items": variable_items,
        "exact_monthly_total": base_subtotal if not variable_items else None,
        "display_formula": display_formula,
        "initial_payment_options": initial_payment_options,
        "unknown_inputs": unknown_inputs,
        "note": "Ước tính deterministic. known_initial_subtotal có thể chưa bao gồm các khoản phí biến đổi.",
        # Legacy compatibility fields
        "rental_months": rental_months,
        "total_initial_cost": total_initial_cost,
        "total_period_cost": total_period_cost,
        "recurring_fees_for_period": recurring_fees_for_period,
        "unknown": legacy_unknown,
        "not_calculated": not_calculated_list,
    }



def compare_rooms(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    room_ids = [str(item) for item in (args.get("room_ids") or [])][:3]
    rooms = []
    for room_ref in room_ids:
        room = _resolve_room_reference(room_ref, context)
        if room:
            rooms.append(room)
    rows = []
    for room in rooms:
        rows.append({
            "room_id": room.get("room_id"),
            "house_id": room.get("house_id"),
            "title": room.get("title"),
            "rent_price": room.get("rent_price"),
            "area_m2": room.get("area_m2"),
            "district": room.get("district"),
            "address": room.get("address"),
            "available": room.get("available"),
            "status_desc": room.get("status_desc"),
            "amenities": list(room.get("amenities") or []),
            "fees": dict(room.get("fees") or {}),
            "embedding_text": room.get("embedding_text", ""),
            "unknown": _unknown_room_fields(room),
        })
    return {
        "room_ids": room_ids,
        "rows": rows,
        "max_compared": 3,
        "missing_room_ids": [item for item in room_ids if item not in {row["room_id"] for row in rows}],
    }


def find_similar_rooms(args: dict[str, Any], context: ToolExecutionContext) -> list[dict[str, Any]]:
    room_id = args.get("room_id")
    source = _resolve_room_reference(str(room_id), context) if room_id else None
    if not source:
        return []
    top_k = _bounded_top_k(args.get("top_k", 5))
    constraints = {
        "location": {"districts": [source.get("district")] if source.get("district") else []},
        "budget": {
            "min": int(source["rent_price"] * 0.75) if source.get("rent_price") else None,
            "max": int(source["rent_price"] * 1.25) if source.get("rent_price") else None,
            "type": "rent_only",
        },
        "amenities_required": [],
        "amenities_preferred": [],
    }
    results = search_rooms_with_hard_filters(
        query_text=source.get("embedding_text") or source.get("title") or "",
        constraints=constraints,
        repository=context.repository,
        semantic_index=context.semantic_index,
        top_k=top_k + 1,
        trace=context.retrieval_trace,
    )
    return [item for item in results if item.get("room_id") != source.get("room_id")][:top_k]


def _unknown_room_fields(room: dict[str, Any]) -> list[str]:
    return unknown_room_fields(room)


def _looks_like_mongo_object_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{24}", str(value or "").lower()))


def _looks_like_room_code(value: str) -> bool:
    compact = normalize_room_code(value)
    if not compact or _looks_like_mongo_object_id(compact):
        return False
    return bool(re.fullmatch(r"[A-Z]{1,3}\d{2,5}", compact))


def _resolve_room_reference(room_ref: str, context: ToolExecutionContext) -> dict[str, Any] | None:
    room_ref = _normalize_room_reference(room_ref)
    if not room_ref:
        return None

    if _looks_like_room_code(room_ref):
        try:
            room = context.repository.get_by_room_code(room_ref)
        except Exception:
            room = None
        if room:
            return room

    try:
        room = context.repository.get_by_id(str(room_ref))
    except Exception:
        room = None
    if room:
        return room

    try:
        candidates = context.repository.search_by_metadata(str(room_ref), limit=5)
    except Exception:
        candidates = []
    if not candidates:
        return None
    normalized_ref = normalize_room_code(room_ref)
    for candidate in candidates:
        candidate_room_id = normalize_room_code(candidate.get("room_id") or "")
        candidate_room_code = normalize_room_code(candidate.get("room_code") or "")
        if normalized_ref and normalized_ref in {candidate_room_id, candidate_room_code}:
            return candidate
    if _looks_like_room_code(room_ref):
        return None
    return candidates[0]


def _normalize_room_reference(value: Any) -> str:
    return str(value or "").strip().lstrip("#").strip()


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _money_value_or_none(value: Any) -> int | None:
    from room_assistant.money import parse_money_amount

    return parse_money_amount(value, fee_context=False)


def _money_amount_or_none(value: Any, fee_name: str | None = None, args: dict[str, Any] | None = None) -> int | None:
    from room_assistant.money import parse_money_amount

    return parse_money_amount(value, fee_name=fee_name, fee_context=True)

from enum import Enum

class SufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL_SAFE = "PARTIAL_SAFE"
    INSUFFICIENT = "INSUFFICIENT"

REQUIRED_FIELDS = {
    "price_query": {"monthly_rent"},
    "deposit_query": {"deposit", "hold_days"},
    "room_detail": {"monthly_rent", "status", "location"},
    "availability_query": {"status"},
    "pets_query": {"pets_policy"},
    "utilities_query": {"utilities"},
}

_SENSITIVE_KEYWORD_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("deposit_query", ("tien coc", "tiền cọc", "dat coc", "đặt cọc", "tien dat coc", "tiền đặt cọc")),
    (
        "availability_query",
        ("con phong", "còn phòng", "phong trong", "phòng trống", "het phong", "hết phòng", "con trong khong"),
    ),
    (
        "pets_query",
        ("thu cung", "thú cưng", "nuoi meo", "nuôi mèo", "nuoi cho", "nuôi chó", "cho nuoi", "cho nuôi", "pet"),
    ),
    (
        "utilities_query",
        (
            "tien dien", "tiền điện", "gia dien", "giá điện", "phi dien", "phí điện",
            "tien nuoc", "tiền nước", "gia nuoc", "giá nước", "phi nuoc", "phí nước",
            "phi wifi", "phí wifi", "phi quan ly", "phí quản lý",
            "phi gui xe", "phí gửi xe", "phi giu xe", "phí giữ xe", "gia gui xe", "giá gửi xe",
        ),
    ),
)

_ROOM_PRICE_KEYWORD_PATTERNS: tuple[str, ...] = (
    "gia bao nhieu", "giá bao nhiêu", "gia phong", "giá phòng",
    "bao nhieu tien", "bao nhiêu tiền", "gia thue", "giá thuê", "gia thang", "giá tháng",
)


def _normalize_question_text(question: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFD", (question or "").lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")


def _keyword_sensitive_types(
    question: str,
    *,
    intent: str = "",
    has_room_context: bool = False,
) -> set[str]:
    text = _normalize_question_text(question)
    types: set[str] = set()
    for answer_type, tokens in _SENSITIVE_KEYWORD_PATTERNS:
        if any(token in text for token in tokens):
            types.add(answer_type)
    if has_room_context and intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"}:
        if any(token in text for token in _ROOM_PRICE_KEYWORD_PATTERNS):
            types.add("price_query")
        if any(token in text for token in ("coc", "dat coc", "tien coc", "tien dat coc")):
            types.add("deposit_query")
    return types


def question_asks_price(question: str) -> bool:
    text = _normalize_question_text(question)
    return any(token in text for token in _ROOM_PRICE_KEYWORD_PATTERNS)


def _question_mentions_pets(question: str) -> bool:
    return "pets_query" in _keyword_sensitive_types(question)


def resolve_sensitive_answer_types(
    question: str,
    *,
    intent: str = "",
    constraints: dict[str, Any] | None = None,
    has_room_context: bool = False,
) -> set[str]:
    """Kết hợp slot đã parse + keyword (chỉ khi có ngữ cảnh phòng cho giá/phí)."""
    constraints = constraints or {}
    types = _keyword_sensitive_types(
        question,
        intent=intent,
        has_room_context=has_room_context,
    )

    if constraints.get("pets_required") and _question_mentions_pets(question):
        types.add("pets_query")

    amenities_required = {str(item).strip().lower() for item in (constraints.get("amenities_required") or [])}
    if "pets_allowed" in amenities_required and _question_mentions_pets(question):
        types.add("pets_query")

    if intent == "CALCULATE_COST":
        types.add("deposit_query")
        types.add("price_query")

    return types


def classify_sensitive_question(question: str) -> str | None:
    """Giữ API cũ: trả về một loại đầu tiên (ưu tiên deposit → pets → utilities → price)."""
    priority = ("deposit_query", "pets_query", "availability_query", "utilities_query", "price_query")
    types = resolve_sensitive_answer_types(question, has_room_context=True)
    for item in priority:
        if item in types:
            return item
    return None


def check_sufficiency(room: dict[str, Any], answer_type: str) -> tuple[SufficiencyStatus, set[str]]:
    required = REQUIRED_FIELDS.get(answer_type)
    if not required:
        return SufficiencyStatus.SUFFICIENT, set()

    if answer_type == "pets_query":
        from room_assistant.repository import room_allows_pets

        if room_allows_pets(room):
            return SufficiencyStatus.SUFFICIENT, set()
        return SufficiencyStatus.INSUFFICIENT, {"pets_policy"}

    if answer_type == "utilities_query":
        fees = room.get("fees") or {}
        embedding_text = str(room.get("embedding_text") or "")
        has_fee_data = bool(fees) or "## Giá & phí" in embedding_text
        if has_fee_data:
            return SufficiencyStatus.SUFFICIENT, set()
        return SufficiencyStatus.INSUFFICIENT, {"utilities"}

    if answer_type == "availability_query":
        if room.get("available") is None:
            return SufficiencyStatus.INSUFFICIENT, {"status"}
        return SufficiencyStatus.SUFFICIENT, set()
        
    missing = set()
    for field in required:
        if field == "monthly_rent" and room.get("rent_price") is None:
            missing.add(field)
        elif field == "deposit" and room.get("deposit") is None and not room.get("deposit_options"):
            missing.add(field)
        elif field == "hold_days":
            has_hold = False
            if room.get("deposit_options"):
                has_hold = any(opt.get("hold_days") is not None for opt in room.get("deposit_options", []))
            if not has_hold:
                missing.add(field)
        elif field == "status" and room.get("available") is None:
            missing.add(field)
        elif field == "location" and not room.get("address") and not room.get("district"):
            missing.add(field)
            
    if not missing:
        return SufficiencyStatus.SUFFICIENT, set()
    if answer_type == "deposit_query" and "deposit" in missing:
        return SufficiencyStatus.INSUFFICIENT, missing
    if "monthly_rent" in missing and answer_type in {"price_query", "room_detail"}:
        return SufficiencyStatus.INSUFFICIENT, missing
    return SufficiencyStatus.PARTIAL_SAFE, missing


INSUFFICIENT_FIELD_LABELS: dict[str, str] = {
    "monthly_rent": "giá thuê",
    "deposit": "tiền cọc",
    "hold_days": "thời gian giữ cọc",
    "status": "tình trạng còn phòng",
    "pets_policy": "quy định thú cưng",
    "utilities": "phí điện/nước/wifi",
    "location": "địa chỉ",
}


def evaluate_room_data_sufficiency(
    question: str,
    room: dict[str, Any],
    *,
    intent: str = "",
    constraints: dict[str, Any] | None = None,
) -> tuple[SufficiencyStatus, set[str]]:
    """Đánh giá đủ dữ liệu xác minh cho mọi chủ đề nhạy cảm áp dụng được."""
    answer_types = resolve_sensitive_answer_types(
        question,
        intent=intent,
        constraints=constraints,
        has_room_context=True,
    )
    if not answer_types:
        return SufficiencyStatus.SUFFICIENT, set()

    merged_missing: set[str] = set()
    worst = SufficiencyStatus.SUFFICIENT
    rank = {
        SufficiencyStatus.SUFFICIENT: 0,
        SufficiencyStatus.PARTIAL_SAFE: 1,
        SufficiencyStatus.INSUFFICIENT: 2,
    }
    for answer_type in sorted(answer_types):
        status, missing = check_sufficiency(room, answer_type)
        merged_missing.update(missing)
        if rank[status] > rank[worst]:
            worst = status
    return worst, merged_missing


def format_insufficient_field_labels(missing: set[str]) -> str:
    return ", ".join(
        INSUFFICIENT_FIELD_LABELS.get(item, item)
        for item in sorted(missing)
    )


MAX_TARGETED_FETCH_ATTEMPTS = 1

def targeted_fetch(
    repository: RoomRepository, 
    candidate_ids: list[str], 
    missing_fields: set[str]
) -> list[dict[str, Any]]:
    """Bounded Targeted Fetch loop (`MAX_TARGETED_FETCH_ATTEMPTS = 1`, batch projection)"""
    return repository.get_many_by_ids(candidate_ids)
