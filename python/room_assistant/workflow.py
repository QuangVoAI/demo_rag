"""Production workflow for the read-only Nhatrovn room assistant."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Awaitable

from .intent import parse_intent_and_constraint_patch
from .repository import ListingRepository, create_listing_repository
from .retrieval import ListingSemanticIndex
from .schemas import MAX_READ_TOOL_CALLS_PER_TURN, public_session_state
from .session_store import (
    SessionStore,
    apply_operations,
    create_session_store,
    load_session_state,
    save_session_state,
    update_turn_state,
)
from .tools import ReadOnlyToolRegistry, ToolBudgetExceeded, ToolExecutionContext


_session_store: SessionStore | None = None
_listing_repository: ListingRepository | None = None
_semantic_index: ListingSemanticIndex | None = None
_tool_registry = ReadOnlyToolRegistry()


def _get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = create_session_store()
    return _session_store


def _get_listing_repository() -> ListingRepository:
    global _listing_repository
    if _listing_repository is None:
        _listing_repository = create_listing_repository()
    return _listing_repository


def _get_semantic_index() -> ListingSemanticIndex | None:
    global _semantic_index
    if _semantic_index is not None:
        return _semantic_index
    try:
        from .qdrant_index import QdrantListingSemanticIndex

        _semantic_index = QdrantListingSemanticIndex()
    except Exception:
        _semantic_index = None
    return _semantic_index


async def run_room_assistant(
    question: str,
    history: list[dict[str, Any]] | None = None,
    session_id: str = "",
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
    repository: ListingRepository | None = None,
    session_store: SessionStore | None = None,
    semantic_index: ListingSemanticIndex | None = None,
) -> dict[str, Any]:
    """Run one user turn. Streaming callback is accepted but final-only by design."""
    started = time.time()
    session_id = session_id or str(uuid.uuid4())
    store = session_store or _get_session_store()
    repo = repository or _get_listing_repository()
    semantic_index = semantic_index if semantic_index is not None else _get_semantic_index()
    ttl_seconds = _session_ttl_seconds()

    state_before = load_session_state(session_id, store)
    parsed = parse_intent_and_constraint_patch(question, state_before)
    merged_state, applied_operations = apply_operations(state_before, parsed["operations"])

    context = ToolExecutionContext(repository=repo, semantic_index=semantic_index)
    tool_results: dict[str, Any] = {}
    error_category = None

    try:
        tool_results = _execute_workflow(
            question=question,
            parsed=parsed,
            state=merged_state,
            context=context,
        )
    except ToolBudgetExceeded:
        error_category = "tool_budget_exceeded"
        tool_results = {"error": "tool_budget_exceeded"}

    listings = _extract_listings(tool_results)
    result_ids = [item["listing_id"] for item in listings if item.get("listing_id")]
    current_listing_id = parsed.get("current_listing_id") or _current_listing_from_results(parsed["intent"], listings)

    next_state = update_turn_state(
        merged_state,
        intent=parsed["intent"],
        current_listing_id=current_listing_id,
        referenced_listing_ids=parsed.get("referenced_listing_ids", []),
        result_ids=result_ids,
    )
    _update_summary(next_state, question, parsed["intent"])
    save_session_state(next_state, store, ttl_seconds)

    grounding = _build_grounding_context(parsed, next_state, tool_results)
    answer = _compose_answer(parsed, grounding, tool_results)
    suggested_questions = _suggest_questions(parsed["intent"], listings, current_listing_id)
    processing_time_ms = int((time.time() - started) * 1000)

    return {
        "session_id": session_id,
        "answer": answer,
        "intent": parsed["intent"],
        "session_state": public_session_state(next_state),
        "listings": listings,
        "cost_estimate": tool_results.get("cost_estimate"),
        "comparison": tool_results.get("comparison"),
        "suggested_questions": suggested_questions,
        "sources": grounding["sources"],
        "agent_trace": {
            "workflow": [
                "normalize_input",
                "parse_intent_and_constraint_patch",
                "load_session_state",
                "merge_and_validate_state",
                "route_workflow",
                "execute_read_only_tools",
                "grounding_check",
                "compose_response",
                "persist_state_and_trace",
            ],
            "intent": parsed["intent"],
            "applied_operations": applied_operations,
            "state_version_before": state_before.get("state_version"),
            "state_version_after": next_state.get("state_version"),
            "read_tool_calls": context.read_tool_calls,
            "write_tool_calls": context.write_tool_calls,
            "max_read_tool_calls_per_turn": MAX_READ_TOOL_CALLS_PER_TURN,
            "client_history_messages_seen": min(len(history or []), 8),
            "error_category": error_category,
            "grounding_result": grounding["result"],
        },
        "processing_time_ms": processing_time_ms,
        "is_final": True,
        "chunk_type": None,
    }


def _execute_workflow(
    question: str,
    parsed: dict[str, Any],
    state: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    intent = parsed["intent"]
    constraints = state.get("constraints", {})
    current_listing_id = (
        parsed.get("current_listing_id")
        or state.get("current_listing_id")
        or _first_or_none(state.get("last_result_ids", []))
    )

    if intent in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        listings = _tool_registry.execute(
            "search_listings",
            {"query_text": question, "constraints": constraints, "top_k": 5},
            context,
        )
        if not listings and _has_soft_preferences(constraints) and context.read_tool_calls < MAX_READ_TOOL_CALLS_PER_TURN:
            retry_constraints = dict(constraints)
            retry_constraints["amenities_preferred"] = []
            listings = _tool_registry.execute(
                "search_listings",
                {"query_text": "", "constraints": retry_constraints, "top_k": 5},
                context,
            )
            return {"listings": listings, "retrieval_retry": True}
        return {"listings": listings}

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        detail = _tool_registry.execute(
            "retrieve_listing_context",
            {"listing_id": current_listing_id},
            context,
        )
        return {"listing_context": detail, "listings": [detail["listing"]] if detail.get("listing") else []}

    if intent == "CALCULATE_COST":
        listing = _tool_registry.execute(
            "get_listing_detail",
            {"listing_id": current_listing_id},
            context,
        )
        estimate = _tool_registry.execute(
            "calculate_cost_estimate",
            {"listing": listing},
            context,
        )
        return {"listings": [listing] if listing else [], "cost_estimate": estimate}

    if intent == "COMPARE_ROOMS":
        ids = parsed.get("referenced_listing_ids") or state.get("selected_listing_ids") or state.get("last_result_ids", [])
        comparison = _tool_registry.execute(
            "compare_listings",
            {"listing_ids": ids[:3]},
            context,
        )
        return {"comparison": comparison, "listings": comparison.get("rows", [])}

    if intent == "FIND_SIMILAR":
        listings = _tool_registry.execute(
            "find_similar_listings",
            {"listing_id": current_listing_id, "top_k": 5},
            context,
        )
        return {"listings": listings}

    if intent == "REQUEST_ACTION":
        return {"requested_action": parsed.get("requested_action")}

    return {}


def _build_grounding_context(
    parsed: dict[str, Any],
    state: dict[str, Any],
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    listings = _extract_listings(tool_results)
    sources = []
    source_versions = {}
    unknown: list[str] = []
    for listing in listings:
        listing_id = listing.get("listing_id")
        if not listing_id:
            continue
        source_versions[listing_id] = listing.get("source_version", 0)
        sources.append({
            "type": "listing",
            "listing_id": listing_id,
            "source_version": listing.get("source_version", 0),
            "title": listing.get("title"),
        })
        unknown.extend(f"{listing_id}.{field}" for field in _unknown_fields(listing))

    if tool_results.get("cost_estimate", {}).get("unknown"):
        unknown.extend(tool_results["cost_estimate"]["unknown"])

    return {
        "result": "ok" if not tool_results.get("error") else "error",
        "intent": parsed["intent"],
        "constraints": state.get("constraints", {}),
        "confirmed": {"listing_count": len(listings)},
        "estimated": {"cost_estimate": tool_results.get("cost_estimate")} if tool_results.get("cost_estimate") else {},
        "unknown": sorted(set(unknown)),
        "listings": listings,
        "sources": sources,
        "source_versions": source_versions,
    }


def _compose_answer(
    parsed: dict[str, Any],
    grounding: dict[str, Any],
    tool_results: dict[str, Any],
) -> str:
    intent = parsed["intent"]
    if tool_results.get("error") == "tool_budget_exceeded":
        return "Mình cần giới hạn số lần đọc dữ liệu trong một lượt. Bạn thử hỏi lại hẹp hơn với tối đa 3 phòng hoặc một nhu cầu cụ thể nhé."

    if intent == "REQUEST_ACTION":
        action = parsed.get("requested_action") or "business_action"
        return (
            "Mình chỉ có thể tư vấn và đọc dữ liệu, không thể tự thực hiện thao tác như đặt lịch, nhắn chủ nhà, lưu phòng, giữ chỗ hay thanh toán. "
            f"Với yêu cầu `{action}`, bạn cần tự xác nhận trên giao diện Nhatrovn nếu nút thao tác đó có sẵn."
        )

    listings = grounding["listings"]
    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
        if not listings:
            return "Mình chưa tìm thấy phòng phù hợp trong dữ liệu hiện có. Dữ liệu crawler có thể chưa đầy đủ; bạn có thể nới ngân sách, đổi khu vực hoặc bỏ bớt tiện ích bắt buộc."
        lines = ["Mình tìm được các phòng phù hợp nhất theo dữ liệu đã xác nhận:"]
        for idx, listing in enumerate(listings[:5], 1):
            lines.append(f"{idx}. {listing.get('title')} (#{listing.get('listing_id')}) - {format_vnd(listing.get('rent_price'))}/tháng, {listing.get('district') or 'chưa rõ khu vực'}.")
        lines.append("Giá, trạng thái còn phòng và thông tin chi tiết được lấy lại từ listing repository trước khi trả lời.")
        return "\n".join(lines)

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        if not listings:
            return "Mình chưa xác định được phòng đang xem. Bạn gửi mã phòng hoặc chọn một phòng từ kết quả tìm kiếm trước nhé."
        listing = listings[0]
        unknown = _unknown_fields(listing)
        parts = [
            f"{listing.get('title')} (#{listing.get('listing_id')}) hiện có giá {format_vnd(listing.get('rent_price'))}/tháng.",
            f"Khu vực: {listing.get('address') or listing.get('district') or 'chưa rõ'}.",
            f"Tiện ích đã ghi nhận: {', '.join(listing.get('amenities') or []) or 'chưa có dữ liệu'}.",
        ]
        if unknown:
            parts.append(f"Dữ liệu chưa xác nhận: {', '.join(unknown)}.")
        return "\n".join(parts)

    if intent == "CALCULATE_COST":
        estimate = tool_results.get("cost_estimate") or {}
        if not estimate.get("available"):
            return "Mình chưa có đủ dữ liệu phòng để tính chi phí. Bạn gửi mã phòng cụ thể hơn nhé."
        lines = ["Ước tính chi phí ban đầu bằng tính toán deterministic:"]
        for item in estimate.get("items", []):
            lines.append(f"- {item['name']}: {format_vnd(item.get('amount'))}")
        lines.append(f"Tổng tạm tính: {format_vnd(estimate.get('total_initial_cost'))}.")
        if estimate.get("unknown"):
            lines.append(f"Chưa có dữ liệu cho: {', '.join(estimate['unknown'])}.")
        return "\n".join(lines)

    if intent == "COMPARE_ROOMS":
        comparison = tool_results.get("comparison") or {}
        rows = comparison.get("rows", [])
        if not rows:
            return "Mình cần tối đa 3 mã phòng để so sánh. Bạn gửi dạng `so sánh #A #B #C` nhé."
        lines = ["So sánh tối đa 3 phòng theo dữ liệu đã xác nhận:"]
        for row in rows:
            lines.append(f"- #{row.get('listing_id')}: {format_vnd(row.get('rent_price'))}/tháng, {row.get('area_m2') or 'chưa rõ'} m2, {row.get('district') or 'chưa rõ khu vực'}.")
        return "\n".join(lines)

    return (
        "Mình có thể giúp tìm phòng, lọc lại điều kiện, hỏi đáp về phòng đang xem, tính chi phí, so sánh tối đa 3 phòng và gợi ý phòng tương tự. "
        "Mình không thực hiện các thao tác nghiệp vụ như đặt lịch, nhắn chủ nhà, giữ chỗ hoặc thanh toán."
    )


def _extract_listings(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    listings = tool_results.get("listings") or []
    return [item for item in listings if item]


def _current_listing_from_results(intent: str, listings: list[dict[str, Any]]) -> str | None:
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"} and listings:
        return listings[0].get("listing_id")
    return None


def _unknown_fields(listing: dict[str, Any]) -> list[str]:
    return [field for field in ("rent_price", "deposit", "area_m2", "available_from", "pets_allowed") if listing.get(field) is None]


def _first_or_none(values: list[Any]) -> Any | None:
    return values[0] if values else None


def _has_soft_preferences(constraints: dict[str, Any]) -> bool:
    return bool(constraints.get("amenities_preferred"))


def _suggest_questions(intent: str, listings: list[dict[str, Any]], current_listing_id: str | None) -> list[str]:
    if listings:
        first = current_listing_id or listings[0].get("listing_id")
        return [
            f"Tính tổng chi phí cho #{first}",
            f"Tóm tắt ưu điểm và hạn chế của #{first}",
            "Tìm phòng tương tự nhưng rẻ hơn",
        ]
    if intent == "GENERAL_HELP":
        return [
            "Tìm phòng dưới 5 triệu ở quận Bình Thạnh",
            "So sánh #A #B #C",
            "Phòng này có cho nuôi mèo không?",
        ]
    return ["Nới ngân sách thêm 1 triệu", "Bỏ yêu cầu máy lạnh", "Đổi sang khu vực gần trường hơn"]


def _update_summary(state: dict[str, Any], question: str, intent: str) -> None:
    summary = state.get("conversation_summary", "")
    turn = f"{intent}: {question[:120]}"
    state["conversation_summary"] = (summary + "\n" + turn).strip()[-1000:]


def _session_ttl_seconds() -> int:
    try:
        from config import REDIS_SESSION_TTL_SECONDS
        return int(REDIS_SESSION_TTL_SECONDS)
    except Exception:
        return 24 * 3600


def format_vnd(value: Any) -> str:
    if value is None:
        return "chưa rõ"
    try:
        return f"{int(value):,} VND".replace(",", ".")
    except Exception:
        return str(value)
