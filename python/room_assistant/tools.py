"""Read-only tool registry for the room assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .repository import ListingRepository
from .retrieval import ListingSemanticIndex, search_listings_with_hard_filters
from .schemas import MAX_READ_TOOL_CALLS_PER_TURN, READ_ONLY_TOOLS, unknown_listing_fields


class ToolBudgetExceeded(RuntimeError):
    pass


@dataclass
class ToolExecutionContext:
    repository: ListingRepository
    semantic_index: ListingSemanticIndex | None = None
    read_tool_calls: int = 0
    write_tool_calls: int = 0
    tool_latency_ms: dict[str, int] = field(default_factory=dict)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)


class ReadOnlyToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[[dict[str, Any], ToolExecutionContext], Any]] = {
            "search_listings": search_listings,
            "get_listing_detail": get_listing_detail,
            "retrieve_listing_context": retrieve_listing_context,
            "retrieve_faq": retrieve_faq,
            "calculate_cost_estimate": calculate_cost_estimate,
            "compare_listings": compare_listings,
            "find_similar_listings": find_similar_listings,
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


def search_listings(args: dict[str, Any], context: ToolExecutionContext) -> list[dict[str, Any]]:
    return search_listings_with_hard_filters(
        query_text=args.get("query_text", ""),
        constraints=args.get("constraints", {}),
        repository=context.repository,
        semantic_index=context.semantic_index,
        top_k=int(args.get("top_k", 5)),
        trace=context.retrieval_trace,
    )


def get_listing_detail(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any] | None:
    listing_id = args.get("listing_id")
    if not listing_id:
        return None
    return context.repository.get_by_id(str(listing_id))


def retrieve_listing_context(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    listing = get_listing_detail(args, context)
    if not listing:
        return {"listing": None, "context": "", "unknown": ["listing_not_found"]}
    context_parts = [
        listing.get("title", ""),
        listing.get("description", ""),
        ", ".join(listing.get("amenities") or []),
        listing.get("address", ""),
    ]
    return {
        "listing": listing,
        "context": "\n".join(part for part in context_parts if part),
        "unknown": _unknown_listing_fields(listing),
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
            "answer": "Tiền cọc phụ thuộc từng listing. Nếu dữ liệu phòng có trường cọc, mình sẽ dùng đúng số đó; nếu thiếu thì mình sẽ báo chưa có dữ liệu.",
        },
        {
            "topic": "contract",
            "keywords": ("hợp đồng", "hop dong", "thời hạn thuê", "thoi han thue"),
            "answer": "Thông tin hợp đồng và thời hạn thuê cần đối chiếu theo từng phòng hoặc thỏa thuận với chủ nhà. Mình không tự tạo hay xác nhận hợp đồng thay bạn.",
        },
        {
            "topic": "fees",
            "keywords": ("phí", "phi", "điện", "dien", "nước", "nuoc", "wifi", "giữ xe", "giu xe"),
            "answer": "Các khoản phí như điện, nước, wifi, giữ xe chỉ được xem là xác nhận khi có trong dữ liệu listing hoặc phần ước tính chi phí.",
        },
        {
            "topic": "pets",
            "keywords": ("nuôi mèo", "nuoi meo", "nuôi chó", "nuoi cho", "thú cưng", "thu cung"),
            "answer": "Việc nuôi thú cưng phụ thuộc trường pets_allowed hoặc quy định của từng phòng. Nếu dữ liệu thiếu, mình sẽ nói rõ là chưa có dữ liệu.",
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
    listing = args.get("listing")
    if not listing and args.get("listing_id"):
        listing = context.repository.get_by_id(str(args["listing_id"]))
    if not listing:
        return {"available": False, "unknown": ["listing_not_found"], "items": [], "total_initial_cost": None}

    rent = listing.get("rent_price")
    deposit = listing.get("deposit")
    fees = listing.get("fees") or {}
    items: list[dict[str, Any]] = []
    unknown: list[str] = []
    total = 0

    if rent is None:
        unknown.append("rent_price")
    else:
        items.append({"name": "rent_first_month", "amount": rent, "confirmed": True})
        total += int(rent)

    if deposit is None:
        unknown.append("deposit")
    else:
        items.append({"name": "deposit", "amount": deposit, "confirmed": True})
        total += int(deposit)

    for name, amount in fees.items():
        if amount is None:
            unknown.append(f"fees.{name}")
            continue
        items.append({"name": f"fee_{name}", "amount": amount, "confirmed": True})
        total += int(amount)

    return {
        "available": True,
        "listing_id": listing.get("listing_id"),
        "currency": "VND",
        "items": items,
        "total_initial_cost": total if items else None,
        "unknown": unknown,
        "note": "Ước tính deterministic từ giá, cọc và phí đã xác nhận trong listing.",
    }


def compare_listings(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    listing_ids = [str(item) for item in (args.get("listing_ids") or [])][:3]
    listings = context.repository.get_many_by_ids(listing_ids)
    rows = []
    for listing in listings:
        rows.append({
            "listing_id": listing.get("listing_id"),
            "title": listing.get("title"),
            "rent_price": listing.get("rent_price"),
            "area_m2": listing.get("area_m2"),
            "district": listing.get("district"),
            "available": listing.get("available"),
            "unknown": _unknown_listing_fields(listing),
        })
    return {
        "listing_ids": listing_ids,
        "rows": rows,
        "max_compared": 3,
        "missing_listing_ids": [item for item in listing_ids if item not in {row["listing_id"] for row in rows}],
    }


def find_similar_listings(args: dict[str, Any], context: ToolExecutionContext) -> list[dict[str, Any]]:
    listing_id = args.get("listing_id")
    source = context.repository.get_by_id(str(listing_id)) if listing_id else None
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
        "amenities_preferred": source.get("amenities", [])[:3],
    }
    results = search_listings_with_hard_filters(
        query_text=source.get("description") or source.get("title") or "",
        constraints=constraints,
        repository=context.repository,
        semantic_index=context.semantic_index,
        top_k=int(args.get("top_k", 5)) + 1,
        trace=context.retrieval_trace,
    )
    return [item for item in results if item.get("listing_id") != listing_id][:int(args.get("top_k", 5))]


def _unknown_listing_fields(listing: dict[str, Any]) -> list[str]:
    return unknown_listing_fields(listing)
