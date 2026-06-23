"""Workflow chính cho Nhatrovn Room Assistant (chỉ đọc, không ghi).

Luồng xử lý mỗi lượt người dùng:
  normalize_input → parse_intent_async (regex + LLM fallback)
  → analyze_mood (embedding-based, 0 token)
  → load_session_state → merge_and_validate_state
  → route_workflow → execute_read_only_tools
  → grounding_check → compose_answer (response_writer + reviewer)
  → persist_state_and_trace → END

Assistant KHÔNG thực hiện: đặt lịch, nhắn chủ, giữ chỗ, thanh toán,
chỉnh sửa tin đăng hoặc bất kỳ thao tác ghi nào.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any, Callable, Awaitable

from .intent import parse_intent_and_constraint_patch, parse_intent_async
from .repository import RoomRepository, create_room_repository
from .retrieval import RoomSemanticIndex
from .schemas import MAX_READ_TOOL_CALLS_PER_TURN, public_session_state, unknown_room_fields
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
_room_repository: RoomRepository | None = None
_semantic_index: RoomSemanticIndex | None = None
_init_lock = threading.Lock()
_tool_registry = ReadOnlyToolRegistry()


async def startup(
    repository: RoomRepository | None = None,
    session_store: SessionStore | None = None,
    semantic_index: RoomSemanticIndex | None = None,
) -> None:
    """Initialize shared dependencies once during service startup."""
    global _session_store, _room_repository, _semantic_index
    with _init_lock:
        _session_store = session_store or _session_store or create_session_store()
        _room_repository = repository or _room_repository or create_room_repository()
        if semantic_index is not None:
            _semantic_index = semantic_index
        elif _semantic_index is None:
            try:
                from .qdrant_index import QdrantRoomSemanticIndex
                _semantic_index = QdrantRoomSemanticIndex()
            except Exception:
                _semantic_index = None


def _get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        with _init_lock:
            if _session_store is None:
                _session_store = create_session_store()
    return _session_store


def _get_room_repository() -> RoomRepository:
    global _room_repository
    if _room_repository is None:
        with _init_lock:
            if _room_repository is None:
                _room_repository = create_room_repository()
    return _room_repository


def _get_semantic_index() -> RoomSemanticIndex | None:
    global _semantic_index
    if _semantic_index is not None:
        return _semantic_index
    with _init_lock:
        if _semantic_index is not None:
            return _semantic_index
        try:
            from .qdrant_index import QdrantRoomSemanticIndex
            _semantic_index = QdrantRoomSemanticIndex()
        except Exception:
            _semantic_index = None
    return _semantic_index


async def run_room_assistant(
    question: str,
    history: list[dict[str, Any]] | None = None,
    session_id: str = "",
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
    repository: RoomRepository | None = None,
    session_store: SessionStore | None = None,
    semantic_index: RoomSemanticIndex | None = None,
) -> dict[str, Any]:
    """Chạy một lượt hội thoại của người dùng."""
    started = time.time()
    history = (history or [])[-8:]
    session_id = session_id or str(uuid.uuid4())
    store = session_store or _get_session_store()
    repo = repository or _get_room_repository()
    semantic_index = semantic_index if semantic_index is not None else _get_semantic_index()
    ttl_seconds = _session_ttl_seconds()

    state_before = load_session_state(session_id, store)
    parsed = await parse_intent_async(question, state_before)
    merged_state, applied_operations = apply_operations(state_before, parsed["operations"])

    user_mood = "normal"
    try:
        from agents.sentiment_analyzer import analyze_mood
        user_mood, _ = analyze_mood(question)
    except Exception:
        pass

    context = ToolExecutionContext(repository=repo, semantic_index=semantic_index)
    tool_results: dict[str, Any] = {}
    error_category = None

    try:
        tool_results = await asyncio.to_thread(
            _execute_workflow, question, parsed, merged_state, context,
        )
    except ToolBudgetExceeded:
        error_category = "tool_budget_exceeded"
        tool_results = {"error": "tool_budget_exceeded"}

    rooms = _extract_rooms(tool_results)
    result_ids = [item["room_id"] for item in rooms if item.get("room_id")]
    current_room_id = parsed.get("current_room_id") or _current_room_from_results(parsed["intent"], rooms)

    next_state = update_turn_state(
        merged_state,
        intent=parsed["intent"],
        current_room_id=current_room_id,
        referenced_room_ids=parsed.get("referenced_room_ids", []),
        result_ids=result_ids,
    )
    _update_summary(next_state, question, parsed["intent"])
    save_session_state(next_state, store, ttl_seconds)

    grounding = _build_grounding_context(parsed, next_state, tool_results)
    answer = await _compose_answer_async(question, parsed, grounding, tool_results, history, user_mood)
    suggested_questions = _suggest_questions(parsed["intent"], rooms, current_room_id)
    processing_time_ms = int((time.time() - started) * 1000)

    return {
        "session_id": session_id,
        "answer": answer,
        "intent": parsed["intent"],
        "session_state": public_session_state(next_state),
        "rooms": rooms,
        "cost_estimate": tool_results.get("cost_estimate"),
        "comparison": tool_results.get("comparison"),
        "suggested_questions": suggested_questions,
        "sources": grounding["sources"],
        "retrieval_confidence": context.retrieval_trace.get("retrieval_confidence"),
        "retrieval_low_confidence": context.retrieval_trace.get("retrieval_low_confidence"),
        "retrieval_feedback_retry_count": context.retrieval_trace.get("retrieval_feedback_retry_count", 0),
        "retrieval_attempts": context.retrieval_trace.get("retrieval_attempts", []),
        "agent_trace": {
            "workflow": [
                "normalize_input", "parse_intent_async", "analyze_mood",
                "load_session_state", "merge_and_validate_state",
                "route_workflow", "execute_read_only_tools",
                "grounding_check", "compose_answer_with_review",
                "persist_state_and_trace",
            ],
            "intent": parsed["intent"],
            "applied_operations": applied_operations,
            "state_version_before": state_before.get("state_version"),
            "state_version_after": next_state.get("state_version"),
            "read_tool_calls": context.read_tool_calls,
            "write_tool_calls": context.write_tool_calls,
            "max_read_tool_calls_per_turn": MAX_READ_TOOL_CALLS_PER_TURN,
            "client_history_messages_seen": len(history),
            "error_category": error_category,
            "grounding_result": grounding["result"],
            "retrieval": context.retrieval_trace,
        },
        "processing_time_ms": processing_time_ms,
        "is_final": True,
        "chunk_type": None,
    }


def _execute_workflow(
    question: str, parsed: dict[str, Any],
    state: dict[str, Any], context: ToolExecutionContext,
) -> dict[str, Any]:
    """Route đến tool phù hợp theo intent đã phân loại."""
    intent = parsed["intent"]
    constraints = state.get("constraints", {})
    current_room_id = (
        parsed.get("current_room_id")
        or state.get("current_room_id")
        or _first_or_none(state.get("last_result_ids", []))
    )

    if intent in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        rooms = _tool_registry.execute(
            "search_rooms",
            {"query_text": question, "constraints": constraints, "top_k": 5},
            context,
        )
        if not rooms and _has_soft_preferences(constraints) and context.read_tool_calls < MAX_READ_TOOL_CALLS_PER_TURN:
            retry_constraints = dict(constraints)
            retry_constraints["amenities_preferred"] = []
            rooms = _tool_registry.execute(
                "search_rooms",
                {"query_text": question, "constraints": retry_constraints, "top_k": 5},
                context,
            )
            return {"rooms": rooms, "retrieval_retry": True}
        return {"rooms": rooms}

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        detail = _tool_registry.execute(
            "retrieve_room_context", {"room_id": current_room_id}, context,
        )
        return {"room_context": detail, "rooms": [detail["room"]] if detail.get("room") else []}

    if intent == "CALCULATE_COST":
        room = _tool_registry.execute("get_room_detail", {"room_id": current_room_id}, context)
        estimate = _tool_registry.execute(
            "calculate_cost_estimate",
            {"room": room, "rental_months": _extract_rental_months(question), "constraints": constraints},
            context,
        )
        return {"rooms": [room] if room else [], "cost_estimate": estimate}

    if intent == "COMPARE_ROOMS":
        ids = parsed.get("referenced_room_ids") or state.get("selected_room_ids") or state.get("last_result_ids", [])
        comparison = _tool_registry.execute("compare_rooms", {"room_ids": ids[:3]}, context)
        return {"comparison": comparison, "rooms": comparison.get("rows", [])}

    if intent == "FIND_SIMILAR":
        rooms = _tool_registry.execute(
            "find_similar_rooms", {"room_id": current_room_id, "top_k": 5}, context,
        )
        return {"rooms": rooms}

    if intent == "REQUEST_ACTION":
        return {"requested_action": parsed.get("requested_action")}

    if intent == "REQUEST_FAQ":
        faq_results = _tool_registry.execute("retrieve_faq", {"question": question}, context)
        return {"faq_results": faq_results}

    return {}


def _build_grounding_context(
    parsed: dict[str, Any], state: dict[str, Any], tool_results: dict[str, Any],
) -> dict[str, Any]:
    rooms = _extract_rooms(tool_results)
    sources = []
    unknown: list[str] = []
    for room in rooms:
        room_id = room.get("room_id")
        if not room_id:
            continue
        sources.append({
            "type": "room",
            "room_id": room_id,
            "house_id": room.get("house_id"),
            "title": room.get("title"),
            "score": _preferred_source_score(room),
            "rerank_score": room.get("rerank_score"),
            "combined_score": room.get("combined_score"),
            "rrf_score": room.get("rrf_score"),
            "metadata_score": room.get("metadata_score"),
        })
        unknown.extend(f"{room_id}.{field}" for field in _unknown_fields(room))

    if tool_results.get("cost_estimate", {}).get("unknown"):
        unknown.extend(tool_results["cost_estimate"]["unknown"])

    return {
        "result": "ok" if not tool_results.get("error") else "error",
        "intent": parsed["intent"],
        "constraints": state.get("constraints", {}),
        "confirmed": {"room_count": len(rooms)},
        "estimated": {"cost_estimate": tool_results.get("cost_estimate")} if tool_results.get("cost_estimate") else {},
        "unknown": sorted(set(unknown)),
        "rooms": rooms,
        "sources": sources,
    }


def _build_llm_context(grounding: dict[str, Any], tool_results: dict[str, Any]) -> str:
    """Xây dựng phần [DỮ LIỆU ĐÃ XÁC MINH] để đưa vào prompt LLM."""
    parts: list[str] = []
    rooms = grounding.get("rooms", [])
    if rooms:
        parts.append("Danh sách phòng phù hợp:")
        for room in rooms[:5]:
            rent = format_vnd(room.get("rent_price"))
            parts.append(
                f"- [{room.get('room_id')}] {room.get('title')} | "
                f"{rent}/tháng | {room.get('district') or 'chưa rõ khu vực'} | "
                f"Diện tích: {room.get('area_m2') or '?'} m²"
            )

    estimate = tool_results.get("cost_estimate")
    if estimate and estimate.get("available"):
        parts.append("\nƯớc tính chi phí:")
        for item in estimate.get("items", []):
            parts.append(f"  - {item['name']}: {format_vnd(item.get('amount'))}")
        parts.append(f"  Tổng: {format_vnd(estimate.get('total_initial_cost'))}")
        if estimate.get("rental_months"):
            for item in estimate.get("period_items", []):
                parts.append(f"  - {item['name']}: {format_vnd(item.get('amount'))}")
            parts.append(f"  Tổng {estimate['rental_months']} tháng: {format_vnd(estimate.get('total_period_cost'))}")
        if estimate.get("unknown"):
            parts.append(f"  Chưa có dữ liệu: {', '.join(estimate['unknown'])}")

    comparison = tool_results.get("comparison")
    if comparison and comparison.get("rows"):
        parts.append("\nBảng so sánh:")
        for row in comparison["rows"]:
            parts.append(
                f"  - #{row.get('room_id')}: {format_vnd(row.get('rent_price'))}/tháng, "
                f"{row.get('area_m2') or '?'} m², {row.get('district') or 'chưa rõ'}"
            )

    faq = tool_results.get("faq_results")
    if faq:
        parts.append("\nThông tin FAQ:")
        for item in faq:
            parts.append(f"  [{item.get('topic')}] {item.get('answer')}")

    if grounding.get("unknown"):
        parts.append(f"\nCác trường chưa có dữ liệu: {', '.join(grounding['unknown'])}")

    context = "\n".join(parts) if parts else "Không có dữ liệu phù hợp."
    try:
        from config import EVIDENCE_MAX_CHARS
        return context[:EVIDENCE_MAX_CHARS]
    except Exception:
        return context[:3500]


def _compose_answer_template(
    parsed: dict[str, Any], grounding: dict[str, Any], tool_results: dict[str, Any],
) -> str:
    intent = parsed["intent"]
    if tool_results.get("error") == "tool_budget_exceeded":
        return "Mình cần giới hạn số lần đọc dữ liệu trong một lượt. Bạn thử hỏi lại hẹp hơn với tối đa 3 phòng hoặc một nhu cầu cụ thể nhé."
    if intent == "REQUEST_ACTION":
        action = parsed.get("requested_action") or "thao tác nghiệp vụ"
        return (
            "Mình chỉ có thể tư vấn và đọc dữ liệu — không thể tự thực hiện: đặt lịch, "
            "nhắn chủ nhà, lưu phòng, giữ chỗ hay thanh toán. "
            f"Với yêu cầu '{action}', bạn vui lòng thao tác trực tiếp trên giao diện nhatrovn."
        )
    rooms = grounding["rooms"]
    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
        if not rooms:
            return "Mình chưa tìm thấy phòng phù hợp với điều kiện hiện tại. Bạn thử nới ngân sách, đổi khu vực hoặc bỏ bớt tiện ích bắt buộc nhé."
        lines = ["Dưới đây là các phòng phù hợp nhất theo dữ liệu đã xác nhận trên nhatrovn:"]
        for idx, room in enumerate(rooms[:5], 1):
            lines.append(
                f"{idx}. **{room.get('title')}** (#{room.get('room_id')}) — "
                f"{format_vnd(room.get('rent_price'))}/tháng, "
                f"{room.get('district') or 'chưa rõ khu vực'}."
            )
        lines.append("\n_Giá và trạng thái còn phòng được lấy trực tiếp từ dữ liệu phòng._")
        return "\n".join(lines)
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        if not rooms:
            return "Mình chưa xác định được phòng đang xem. Bạn gửi mã phòng hoặc chọn phòng từ kết quả tìm kiếm nhé."
        room = rooms[0]
        unknown = _unknown_fields(room)
        parts = [
            f"**{room.get('title')}** (#{room.get('room_id')}) — giá {format_vnd(room.get('rent_price'))}/tháng.",
            f"Khu vực: {room.get('address') or room.get('district') or 'chưa rõ'}.",
        ]
        if unknown:
            parts.append(f"_Dữ liệu chưa xác nhận: {', '.join(unknown)}._")
        return "\n".join(parts)
    if intent == "CALCULATE_COST":
        estimate = tool_results.get("cost_estimate") or {}
        if not estimate.get("available"):
            return "Mình chưa có đủ dữ liệu phòng để tính chi phí. Bạn gửi mã phòng cụ thể hơn nhé."
        lines = ["**Ước tính chi phí** (calculator deterministic từ dữ liệu đã xác nhận):"]
        if estimate.get("rental_months"):
            lines.append(f"- Thời gian thuê: {estimate['rental_months']} tháng")
            for item in estimate.get("period_items", []):
                lines.append(f"- {_cost_item_label(item['name'])}: {format_vnd(item.get('amount'))}")
            if estimate.get("recurring_fees_for_period"):
                lines.append(f"- Phí cố định {estimate['rental_months']} tháng: {format_vnd(estimate.get('recurring_fees_for_period'))}")
            lines.append(f"\n**Tổng tạm tính {estimate['rental_months']} tháng:** {format_vnd(estimate.get('total_period_cost'))}")
        else:
            for item in estimate.get("items", []):
                if item.get("amount") == 0:
                    continue
                lines.append(f"- {_cost_item_label(item['name'])}: {format_vnd(item.get('amount'))}")
            lines.append(f"\n**Tổng tạm tính ban đầu:** {format_vnd(estimate.get('total_initial_cost'))}")
        if estimate.get("unknown"):
            lines.append(f"_Chưa có dữ liệu: {', '.join(estimate['unknown'])}._")
        return "\n".join(lines)
    if intent == "COMPARE_ROOMS":
        comparison = tool_results.get("comparison") or {}
        rows = comparison.get("rows", [])
        if not rows:
            return "Mình cần tối đa 3 mã phòng để so sánh. Bạn gửi dạng `so sánh #A #B #C` nhé."
        lines = ["**So sánh phòng** theo dữ liệu đã xác nhận:"]
        for row in rows:
            lines.append(
                f"- **#{row.get('room_id')}**: {format_vnd(row.get('rent_price'))}/tháng, "
                f"{row.get('area_m2') or 'chưa rõ'} m², {row.get('district') or 'chưa rõ khu vực'}."
            )
        return "\n".join(lines)
    if intent == "REQUEST_FAQ":
        faq = tool_results.get("faq_results") or []
        if faq:
            return "\n".join(f"**[{item.get('topic')}]** {item.get('answer')}" for item in faq)
        return "Mình có thể hỗ trợ thông tin về: quy trình thuê phòng, hợp đồng thuê nhà, tiền cọc tiêu chuẩn, và các thủ tục liên quan. Bạn hỏi cụ thể hơn nhé."
    return (
        "Mình có thể giúp tìm phòng, lọc điều kiện, hỏi đáp về phòng đang xem, "
        "tính chi phí, so sánh tối đa 3 phòng và gợi ý phòng tương tự trên nhatrovn. "
        "Mình không thực hiện: đặt lịch, nhắn chủ nhà, giữ chỗ hoặc thanh toán."
    )


async def _compose_answer_async(
    question: str, parsed: dict[str, Any], grounding: dict[str, Any],
    tool_results: dict[str, Any], history: list[dict[str, Any]], user_mood: str = "normal",
) -> str:
    intent = parsed["intent"]
    if intent in {"REQUEST_ACTION", "CALCULATE_COST"} or tool_results.get("error"):
        return _compose_answer_template(parsed, grounding, tool_results)
    rooms = grounding.get("rooms", [])
    faq = tool_results.get("faq_results")
    if not rooms and not faq and intent not in {"GENERAL_HELP", "REQUEST_FAQ"}:
        return _compose_answer_template(parsed, grounding, tool_results)
    verified_data = _build_llm_context(grounding, tool_results)
    answer = ""
    try:
        from agents.response_writer import write_response, write_no_result_response
        if not rooms and intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
            constraints = grounding.get("constraints", {})
            answer = await write_no_result_response(question, constraints, user_mood)
        else:
            answer = await write_response(question=question, verified_context=verified_data, history=history, mood=user_mood)
    except Exception:
        pass
    if not answer or len(answer.strip()) < 20:
        return _compose_answer_template(parsed, grounding, tool_results)
    try:
        from config import ENABLE_REVIEWER
        if not ENABLE_REVIEWER:
            return answer.strip() if answer.strip() else _compose_answer_template(parsed, grounding, tool_results)
        from agents.reviewer import review_with_retry
        answer, _ = await review_with_retry(question=question, answer=answer, room_context=verified_data, max_retries=1)
    except Exception:
        pass
    return answer.strip() if answer.strip() else _compose_answer_template(parsed, grounding, tool_results)


def _extract_rooms(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = tool_results.get("rooms") or []
    return [item for item in rooms if item]


def _current_room_from_results(intent: str, rooms: list[dict[str, Any]]) -> str | None:
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"} and rooms:
        return rooms[0].get("room_id")
    return None


def _unknown_fields(room: dict[str, Any]) -> list[str]:
    return unknown_room_fields(room)


def _preferred_source_score(item: dict[str, Any]) -> float:
    for key in ("rerank_score", "combined_score", "rrf_score", "score"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _first_or_none(values: list[Any]) -> Any | None:
    return values[0] if values else None


def _extract_rental_months(question: str) -> int | None:
    import re
    normalized = question.lower()
    match = re.search(r"\b(\d{1,2})\s*(?:tháng|thang|month|months)\b", normalized)
    if not match:
        return None
    months = int(match.group(1))
    return months if months > 0 else None


def _has_soft_preferences(constraints: dict[str, Any]) -> bool:
    return bool(constraints.get("amenities_preferred"))


def _suggest_questions(intent: str, rooms: list[dict[str, Any]], current_room_id: str | None) -> list[str]:
    if rooms:
        first = current_room_id or rooms[0].get("room_id")
        return [
            f"Tính tổng chi phí cho #{first}",
            f"Tóm tắt ưu điểm và hạn chế của #{first}",
            "Tìm phòng tương tự nhưng rẻ hơn",
        ]
    if intent == "GENERAL_HELP":
        return ["Tìm phòng dưới 5 triệu ở quận Bình Thạnh", "So sánh #A #B #C", "Phòng này có cho nuôi mèo không?"]
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


def _cost_item_label(name: str) -> str:
    labels = {
        "rent_first_month": "Tiền thuê tháng đầu",
        "deposit": "Tiền cọc",
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
