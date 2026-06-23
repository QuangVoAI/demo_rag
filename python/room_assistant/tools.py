"""Read-only tool registry for the room assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable

from .repository import RoomRepository
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


def search_rooms(args: dict[str, Any], context: ToolExecutionContext) -> list[dict[str, Any]]:
    return search_rooms_with_hard_filters(
        query_text=args.get("query_text", ""),
        constraints=args.get("constraints", {}),
        repository=context.repository,
        semantic_index=context.semantic_index,
        top_k=int(args.get("top_k", 5)),
        trace=context.retrieval_trace,
    )


def get_room_detail(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any] | None:
    room_id = args.get("room_id")
    if not room_id:
        return None
    return context.repository.get_by_id(str(room_id))


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


def calculate_cost_estimate(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    room = args.get("room")
    if not room and args.get("room_id"):
        room = context.repository.get_by_id(str(args["room_id"]))
    if not room:
        return {"available": False, "unknown": ["room_not_found"], "items": [], "total_initial_cost": None}

    rental_months = _positive_int(args.get("rental_months"))
    rent = room.get("rent_price")
    deposit = room.get("deposit")
    fees = room.get("fees") or {}
    items: list[dict[str, Any]] = []
    period_items: list[dict[str, Any]] = []
    unknown: list[str] = []
    not_calculated: list[dict[str, Any]] = []
    total = 0
    period_total = 0
    recurring_fees_total = 0

    if rent is None:
        unknown.append("rent_price")
    else:
        rent_amount = int(rent)
        items.append({"name": "rent_first_month", "amount": rent_amount, "confirmed": True})
        total += rent_amount
        if rental_months:
            rent_for_period = rent_amount * rental_months
            period_items.append({
                "name": f"rent_{rental_months}_months",
                "amount": rent_for_period,
                "confirmed": True,
            })
            period_total += rent_for_period

    if deposit is None:
        unknown.append("deposit")
    else:
        deposit_amount = int(deposit)
        items.append({"name": "deposit", "amount": deposit_amount, "confirmed": True})
        total += deposit_amount
        if rental_months:
            period_total += deposit_amount

    for name, amount in fees.items():
        if amount is None:
            unknown.append(f"fees.{name}")
            continue
        fee_amount = _money_amount_or_none(amount, name, args)
        if fee_amount is None:
            not_calculated.append({"name": f"fees.{name}", "value": amount})
            continue
        items.append({"name": f"fee_{name}", "amount": fee_amount, "confirmed": True})
        total += fee_amount
        if rental_months:
            recurring_fees_total += fee_amount * rental_months

    if rental_months:
        period_total += recurring_fees_total

    result = {
        "available": True,
        "room_id": room.get("room_id"),
        "house_id": room.get("house_id"),
        "currency": "VND",
        "items": items,
        "total_initial_cost": total if items else None,
        "unknown": unknown,
        "not_calculated": not_calculated,
        "note": "Ước tính deterministic từ giá, cọc và phí đã xác nhận trong dữ liệu phòng.",
    }
    if rental_months:
        result.update({
            "rental_months": rental_months,
            "period_items": period_items,
            "recurring_fees_for_period": recurring_fees_total,
            "total_period_cost": period_total if period_total else None,
        })
    return result


def compare_rooms(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    room_ids = [str(item) for item in (args.get("room_ids") or [])][:3]
    rooms = context.repository.get_many_by_ids(room_ids)
    rows = []
    for room in rooms:
        rows.append({
            "room_id": room.get("room_id"),
            "house_id": room.get("house_id"),
            "title": room.get("title"),
            "rent_price": room.get("rent_price"),
            "area_m2": room.get("area_m2"),
            "district": room.get("district"),
            "available": room.get("available"),
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
    source = context.repository.get_by_id(str(room_id)) if room_id else None
    if not source:
        return []
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
        top_k=int(args.get("top_k", 5)) + 1,
        trace=context.retrieval_trace,
    )
    return [item for item in results if item.get("room_id") != room_id][:int(args.get("top_k", 5))]


def _unknown_room_fields(room: dict[str, Any]) -> list[str]:
    return unknown_room_fields(room)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _money_amount_or_none(value: Any, fee_name: str | None = None, args: dict[str, Any] | None = None) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized in {"free", "miễn phí", "mien phi", "0", "0đ", "0d", "free"}:
        return 0
    if "không có" in normalized or "không" == normalized:
        return 0
    if any(unit in normalized for unit in ("/kwh", "/kw", "/m3", "/m³", "/kg")):
        return None

    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(k|nghìn|nghin|tr|triệu|trieu)?", normalized)
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    unit = match.group(2) or ""
    if unit in {"tr", "triệu", "trieu"}:
        return int(number * 1_000_000)
    if unit in {"k", "nghìn", "nghin"}:
        return int(number * 1_000)
    if number < 1000 and fee_name:
        return int(number * 1_000)
    return int(number)
