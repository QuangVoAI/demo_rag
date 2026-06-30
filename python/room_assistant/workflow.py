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
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Awaitable

from .comparison import best_room_from_comparison, comparison_reason
from .formatters import (
    AMENITY_LABELS,
    cost_item_label,
    format_vnd,
    room_feature_facts,
    verified_amenity_labels,
)
from .intent import parse_intent_and_constraint_patch, parse_intent_async
from .repository import RoomRepository, create_room_repository, room_matches_constraints
from .retrieval import RoomSemanticIndex
from .schemas import (
    MAX_READ_TOOL_CALLS_PER_TURN,
    RECENT_HISTORY_TURNS,
    public_session_state,
    unknown_room_fields,
)
from .session_store import (
    SessionStore,
    apply_operations,
    create_session_store,
    load_session_state,
    save_session_state,
    update_turn_state,
)
from .tools import ReadOnlyToolRegistry, ToolBudgetExceeded, ToolExecutionContext

try:
    import config as _runtime_config  # noqa: F401  # load .env before Langfuse decorators initialize
except Exception:
    _runtime_config = None

try:
    from langfuse import observe as _observe

    def observe(**kwargs):
        return _observe(**kwargs)
except Exception:
    def observe(**kwargs):
        def decorator(func):
            return func
        return decorator


_session_store: SessionStore | None = None
_room_repository: RoomRepository | None = None
_semantic_index: RoomSemanticIndex | None = None
_init_lock = threading.Lock()
_tool_registry = ReadOnlyToolRegistry()
_logger = logging.getLogger(__name__)
_best_room_from_comparison = best_room_from_comparison


def _build_semantic_index() -> RoomSemanticIndex | None:
    try:
        from .qdrant_index import QdrantRoomSemanticIndex

        return QdrantRoomSemanticIndex()
    except Exception:
        return None


async def startup(
    repository: RoomRepository | None = None,
    session_store: SessionStore | None = None,
    semantic_index: RoomSemanticIndex | None = None,
) -> None:
    """Initialize shared dependencies once during service startup."""
    global _session_store, _room_repository, _semantic_index
    try:
        from config import validate_runtime_config
        validation = validate_runtime_config()
        for item in validation.get("warnings", []):
            _logger.warning("room_assistant_config_warning %s", item)
    except Exception as exc:
        _logger.warning("room_assistant_config_validation_failed %s", exc)
    with _init_lock:
        if session_store is not None:
            _session_store = session_store
        elif _session_store is None:
            _session_store = create_session_store()

        if repository is not None:
            _room_repository = repository
        elif _room_repository is None:
            _room_repository = create_room_repository()

        if semantic_index is not None:
            _semantic_index = semantic_index
        elif _semantic_index is None:
            _semantic_index = _build_semantic_index()


def _get_session_store() -> SessionStore:
    global _session_store
    with _init_lock:
        if _session_store is None:
            _session_store = create_session_store()
        return _session_store


def _get_room_repository() -> RoomRepository:
    global _room_repository
    with _init_lock:
        if _room_repository is None:
            _room_repository = create_room_repository()
        return _room_repository


def _get_semantic_index() -> RoomSemanticIndex | None:
    global _semantic_index
    with _init_lock:
        if _semantic_index is None:
            _semantic_index = _build_semantic_index()
        return _semantic_index


@observe(name="room_assistant_turn", capture_input=False, capture_output=False)
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
    question = str(question or "").strip()
    history = (history or [])[-RECENT_HISTORY_TURNS:]
    session_id = session_id or str(uuid.uuid4())
    question_hash = _question_hash(question)
    _update_langfuse_turn_span(
        span_input={
            "question_hash": question_hash,
            "question_chars": len(question),
            "history_messages": len(history),
        },
        metadata={
            "session_hash": _question_hash(session_id),
            "capture_policy": "sanitized_root_summary",
        },
    )
    store = session_store or _get_session_store()
    repo = repository or _get_room_repository()
    ttl_seconds = _session_ttl_seconds()

    state_before = load_session_state(session_id, store)
    max_question_chars = _max_question_chars()
    if len(question) > max_question_chars:
        processing_time_ms = int((time.time() - started) * 1000)
        result = _input_too_long_result(
            session_id=session_id,
            state_before=state_before,
            max_question_chars=max_question_chars,
            processing_time_ms=processing_time_ms,
        )
        _update_langfuse_turn_span(
            span_output=_langfuse_safe_turn_output(result),
            metadata=_langfuse_safe_turn_metadata(result, question_hash),
            level="WARNING",
            status_message="input_too_long",
        )
        _log_turn_summary(result, question_hash=question_hash)
        return result

    semantic_index = semantic_index if semantic_index is not None else _get_semantic_index()
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
    retrieval_started = time.time()
    retrieval_ms = 0

    try:
        tool_results = await asyncio.to_thread(
            _execute_workflow, question, parsed, merged_state, context,
        )
    except ToolBudgetExceeded:
        error_category = "tool_budget_exceeded"
        tool_results = {"error": "tool_budget_exceeded"}
    finally:
        retrieval_ms = int((time.time() - retrieval_started) * 1000)

    rooms = _sanitize_result_rooms(
        parsed["intent"],
        _extract_rooms(tool_results),
        merged_state.get("constraints", {}),
        context.retrieval_trace,
    )
    tool_results["rooms"] = rooms
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

    grounding = _build_grounding_context(parsed, next_state, tool_results, context.retrieval_trace)
    llm_started = time.time()
    answer = await _compose_answer_async(
        question,
        parsed,
        grounding,
        tool_results,
        history,
        user_mood,
        stream_callback=stream_callback,
    )
    llm_generate_ms = int((time.time() - llm_started) * 1000)
    suggested_questions = _suggest_questions(parsed["intent"], rooms, current_room_id)
    processing_time_ms = int((time.time() - started) * 1000)

    result = {
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
        "retrieval_fallback_strategy": context.retrieval_trace.get("fallback_strategy", "original"),
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
            "retrieval_ms": retrieval_ms,
            "llm_generate_ms": llm_generate_ms,
        },
        "processing_time_ms": processing_time_ms,
        "is_final": True,
        "chunk_type": None,
    }
    _update_langfuse_turn_span(
        span_output=_langfuse_safe_turn_output(result),
        metadata=_langfuse_safe_turn_metadata(result, question_hash),
        level="WARNING" if error_category else None,
        status_message=error_category,
    )
    _log_turn_summary(result, question_hash=question_hash)
    return result


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
        followup_room_question = _resolve_inline_room_followup(question, constraints, context.repository)
        if not rooms and _has_soft_preferences(constraints) and context.read_tool_calls < MAX_READ_TOOL_CALLS_PER_TURN:
            retry_constraints = dict(constraints)
            retry_constraints["amenities_preferred"] = []
            rooms = _tool_registry.execute(
                "search_rooms",
                {"query_text": question, "constraints": retry_constraints, "top_k": 5},
                context,
            )
            return {"rooms": rooms, "retrieval_retry": True, "followup_room_question": followup_room_question}
        return {"rooms": rooms, "followup_room_question": followup_room_question}

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        detail = _tool_registry.execute(
            "retrieve_room_context", {"room_id": current_room_id, "query_text": question}, context,
        )
        return {"room_context": detail, "rooms": [detail["room"]] if detail.get("room") else []}

    if intent == "CALCULATE_COST":
        room = _tool_registry.execute("get_room_detail", {"room_id": current_room_id, "query_text": question}, context)
        estimate = _tool_registry.execute(
            "calculate_cost_estimate",
            {"room": room, "rental_months": _extract_rental_months(question), "constraints": constraints},
            context,
        )
        return {"rooms": [room] if room else [], "cost_estimate": estimate}

    if intent == "COMPARE_ROOMS":
        ids = parsed.get("referenced_room_ids") or []
        selected_ids = state.get("selected_room_ids") or []
        last_result_ids = state.get("last_result_ids") or []
        if len(ids) < 2 and _asks_to_compare_result_set(question):
            ids = last_result_ids or selected_ids or ids
        if not ids:
            ids = selected_ids if len(selected_ids) >= 2 else last_result_ids
        comparison = _tool_registry.execute("compare_rooms", {"room_ids": ids[:3]}, context)
        if len(ids) > 3:
            comparison["not_compared_room_ids"] = ids[3:]
        return {"comparison": comparison, "rooms": comparison.get("rows", [])}

    if intent == "FIND_SIMILAR":
        rooms = _tool_registry.execute(
            "find_similar_rooms",
            {"room_id": current_room_id, "top_k": 5, "question": question, "constraints": constraints},
            context,
        )
        return {"rooms": rooms}

    if intent == "REQUEST_ACTION":
        return {"requested_action": parsed.get("requested_action")}

    if intent == "REQUEST_FAQ":
        faq_results = _tool_registry.execute("retrieve_faq", {"question": question}, context)
        return {"faq_results": faq_results}

    return {}


def _input_too_long_result(
    session_id: str,
    state_before: dict[str, Any],
    max_question_chars: int,
    processing_time_ms: int,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "answer": (
            f"Câu hỏi hơi dài nên mình chưa xử lý để tránh sai lệch dữ liệu. "
            f"Bạn rút gọn dưới {max_question_chars} ký tự và gửi lại giúp mình nhé."
        ),
        "intent": "GENERAL_HELP",
        "session_state": public_session_state(state_before),
        "rooms": [],
        "cost_estimate": None,
        "comparison": None,
        "suggested_questions": [
            "Tìm phòng dưới 5 triệu ở Bình Thạnh",
            "So sánh #A101 #B202",
            "Phòng này có máy lạnh không?",
        ],
        "sources": [],
        "retrieval_confidence": None,
        "retrieval_low_confidence": None,
        "retrieval_feedback_retry_count": 0,
        "retrieval_attempts": [],
        "agent_trace": {
            "workflow": ["normalize_input", "input_limit"],
            "intent": "GENERAL_HELP",
            "applied_operations": [],
            "state_version_before": state_before.get("state_version"),
            "state_version_after": state_before.get("state_version"),
            "read_tool_calls": 0,
            "write_tool_calls": 0,
            "max_read_tool_calls_per_turn": MAX_READ_TOOL_CALLS_PER_TURN,
            "client_history_messages_seen": 0,
            "error_category": "input_too_long",
            "grounding_result": "skipped",
            "retrieval": {},
        },
        "processing_time_ms": processing_time_ms,
        "is_final": True,
        "chunk_type": None,
    }


def _build_grounding_context(
    parsed: dict[str, Any], state: dict[str, Any], tool_results: dict[str, Any],
    retrieval_trace: dict[str, Any] | None = None,
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
        "retrieval": dict(retrieval_trace or {}),
    }


def _build_llm_context(grounding: dict[str, Any], tool_results: dict[str, Any]) -> str:
    """Xây dựng phần [DỮ LIỆU ĐÃ XÁC MINH] để đưa vào prompt LLM."""
    parts: list[str] = []
    rooms = grounding.get("rooms", [])
    constraints = grounding.get("constraints", {})
    if rooms:
        parts.append("Danh sách phòng phù hợp:")
        for room in rooms[:5]:
            rent = format_vnd(room.get("rent_price"))
            details = [
                f"[{room.get('room_id')}] {room.get('title')}",
                f"{rent}/tháng",
                room.get("district") or "chưa rõ khu vực",
                f"Diện tích: {room.get('area_m2') or '?'} m²",
            ]
            amenities = verified_amenity_labels(room, constraints)
            if amenities:
                details.append(f"Tiện ích xác minh: {', '.join(amenities)}")
            feature_facts = room_feature_facts(room)
            if feature_facts:
                details.append(f"Thông tin phòng: {', '.join(feature_facts[:8])}")
            if room.get("available") is not None:
                details.append("Trạng thái: còn phòng" if room.get("available") else "Trạng thái: hết phòng")
            parts.append(
                "- " + " | ".join(str(item) for item in details if item)
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
        if estimate.get("not_calculated"):
            not_calculated = [
                f"{item.get('name')}: {item.get('value')}"
                for item in estimate["not_calculated"]
            ]
            parts.append(f"  Có dữ liệu nhưng chưa tính vào tổng: {', '.join(not_calculated)}")

    comparison = tool_results.get("comparison")
    if comparison and comparison.get("rows"):
        parts.append("\nBảng so sánh:")
        for row in comparison["rows"]:
            parts.append(
                f"  - #{row.get('room_id')}: {format_vnd(row.get('rent_price'))}/tháng, "
                f"{row.get('area_m2') or '?'} m², {row.get('district') or 'chưa rõ'}"
            )
        if comparison.get("missing_room_ids"):
            parts.append(f"  Chưa có dữ liệu: {', '.join('#' + item for item in comparison['missing_room_ids'])}")
        if comparison.get("not_compared_room_ids"):
            parts.append(f"  Chưa so sánh do giới hạn tối đa 3 phòng: {', '.join('#' + item for item in comparison['not_compared_room_ids'])}")

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
    parsed: dict[str, Any], grounding: dict[str, Any], tool_results: dict[str, Any], question: str = "",
) -> str:
    intent = parsed["intent"]
    constraints = grounding.get("constraints", {}) or {}
    if tool_results.get("error") == "tool_budget_exceeded":
        return "Mình cần giới hạn số lần đọc dữ liệu trong một lượt. Bạn thử hỏi lại hẹp hơn với tối đa 3 phòng hoặc một nhu cầu cụ thể nhé."
    if intent == "REQUEST_ACTION":
        action = parsed.get("requested_action") or "thao tác nghiệp vụ"
        if action == "dat_lich":
            return (
                "Dạ được ạ. Em chưa tự chốt lịch ngay trong chat, nhưng nếu anh/chị đã ưng căn nào thì mình bấm "
                "`Đặt lịch xem phòng` để mở form booking sẵn trên giao diện. "
                "Anh/chị tiện đi xem buổi sáng hay buổi chiều để em gợi ý bước tiếp theo cho nhanh ạ?"
            )
        return (
            "Dạ em có thể tư vấn và đọc dữ liệu giúp mình, nhưng chưa tự thao tác thay anh/chị như đặt lịch, "
            "nhắn chủ nhà, lưu phòng, giữ chỗ hay thanh toán. "
            f"Với yêu cầu '{action}', anh/chị vui lòng thao tác trực tiếp trên giao diện nhatrovn giúp em nhé."
        )
    rooms = grounding["rooms"]
    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
        if not rooms:
            return (
                "Dạ em chưa tìm thấy căn nào khớp hoàn toàn với điều kiện hiện tại ạ. "
                "Anh/chị muốn em lọc lại theo hướng nới nhẹ ngân sách, đổi sang khu lân cận, hay bớt 1 tiện ích bắt buộc để ra phòng sát hơn ạ?"
            )
        if _is_broad_new_lead(question, constraints):
            first = rooms[0]
            return (
                f"Dạ bên em vẫn còn phòng ạ 😊 Hiện có căn từ khoảng {format_vnd(first.get('rent_price'))}/tháng theo dữ liệu đang còn trống. "
                "Anh/chị đang ưu tiên khu vực nào và khoảng mấy người ở để em lọc đúng căn hợp nhất cho mình ạ?"
            )
        lines = ["Dạ em lọc được vài căn đang còn phòng theo dữ liệu đã xác minh trên nhatrovn:"]
        fallback_strategy = (grounding.get("retrieval") or {}).get("fallback_strategy")
        if fallback_strategy in {"nearby_location", "relax_price_nearby_location"}:
            lines.append("Khu anh/chị chọn hiện chưa còn căn khớp hoàn toàn, nên em lấy thêm các căn khu lân cận gần nhất để mình cân nhắc ạ.")
        for idx, room in enumerate(rooms[:3], 1):
            lines.append(
                f"{idx}. **{room.get('title')}** (#{room.get('room_id')}) — "
                f"{format_vnd(room.get('rent_price'))}/tháng, "
                f"{room.get('district') or 'chưa rõ khu vực'}."
            )
        lines.append("\n_Giá và trạng thái còn phòng được lấy trực tiếp từ dữ liệu phòng._")
        lines.append("Nếu anh/chị thấy căn nào ổn, mình bấm `Đặt lịch xem phòng` để qua bước xem thực tế nhanh hơn nha. Anh/chị thích căn số mấy nhất ạ?")
        followup_text = _compose_inline_followup_answer(tool_results.get("followup_room_question") or {})
        if followup_text:
            lines.append("")
            lines.append(followup_text)
        return "\n".join(lines)
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        if not rooms:
            return "Dạ em chưa xác định được đúng phòng mình đang hỏi. Anh/chị gửi mã phòng hoặc chọn lại từ danh sách để em tư vấn chính xác hơn nha."
        room = rooms[0]
        unknown = _unknown_fields(room)
        parts = [
            "Dạ em gửi anh/chị thông tin đã xác minh của căn này ạ:",
            f"**{room.get('title')}** (#{room.get('room_id')}) — giá {format_vnd(room.get('rent_price'))}/tháng.",
            f"Khu vực: {room.get('address') or room.get('district') or 'chưa rõ'}.",
            f"Diện tích: {room.get('area_m2') or 'chưa rõ'} m².",
        ]
        feature_facts = room_feature_facts(room)
        if feature_facts:
            parts.append(f"Tiện ích và thông tin phòng đã xác minh: {', '.join(feature_facts)}.")
        if unknown:
            parts.append(f"_Dữ liệu chưa xác nhận: {', '.join(unknown)}._")
        parts.append("Nếu căn này đang khá hợp nhu cầu, anh/chị có thể bấm `Đặt lịch xem phòng` để qua xem thực tế cho yên tâm ạ.")
        return "\n".join(parts)
    if intent == "CALCULATE_COST":
        estimate = tool_results.get("cost_estimate") or {}
        if not estimate.get("available"):
            return "Mình chưa có đủ dữ liệu phòng để tính chi phí. Bạn gửi mã phòng cụ thể hơn nhé."
        lines = ["**Ước tính chi phí** (calculator deterministic từ dữ liệu đã xác nhận):"]
        if estimate.get("rental_months"):
            lines.append(f"- Thời gian thuê: {estimate['rental_months']} tháng")
            for item in estimate.get("period_items", []):
                lines.append(f"- {cost_item_label(item['name'])}: {format_vnd(item.get('amount'))}")
            if estimate.get("recurring_fees_for_period"):
                lines.append(f"- Phí cố định {estimate['rental_months']} tháng: {format_vnd(estimate.get('recurring_fees_for_period'))}")
            lines.append(f"\n**Tổng tạm tính {estimate['rental_months']} tháng:** {format_vnd(estimate.get('total_period_cost'))}")
        else:
            for item in estimate.get("items", []):
                if item.get("amount") == 0:
                    continue
                lines.append(f"- {cost_item_label(item['name'])}: {format_vnd(item.get('amount'))}")
            lines.append(f"\n**Tổng tạm tính ban đầu:** {format_vnd(estimate.get('total_initial_cost'))}")
        if estimate.get("unknown"):
            lines.append(f"_Chưa có dữ liệu: {', '.join(estimate['unknown'])}._")
        if estimate.get("not_calculated"):
            details = [
                f"{cost_item_label('fee_' + str(item.get('name', '')).removeprefix('fees.'))}: {item.get('value')}"
                for item in estimate["not_calculated"]
            ]
            lines.append(f"_Có dữ liệu nhưng chưa tính vào tổng: {', '.join(details)}._")
        return "\n".join(lines)
    if intent == "COMPARE_ROOMS":
        comparison = tool_results.get("comparison") or {}
        rows = comparison.get("rows", [])
        if not rows:
            missing = comparison.get("missing_room_ids") or []
            if missing:
                return f"Mình chưa tìm thấy dữ liệu cho: {', '.join('#' + item for item in missing)}. Bạn kiểm tra lại mã phòng hoặc chọn phòng từ danh sách kết quả nhé."
            return "Mình cần tối đa 3 mã phòng để so sánh. Bạn gửi dạng `so sánh #A #B #C` nhé."
        lines = ["**So sánh phòng** theo dữ liệu đã xác nhận:"]
        for row in rows:
            lines.append(
                f"- **#{row.get('room_id')}**: {format_vnd(row.get('rent_price'))}/tháng, "
                f"{row.get('area_m2') or 'chưa rõ'} m², {row.get('district') or 'chưa rõ khu vực'}."
            )
        best = best_room_from_comparison(rows, grounding.get("constraints", {}), question)
        if best:
            area = f", diện tích {best.get('area_m2')} m²" if best.get("area_m2") else ""
            reason = comparison_reason(best, question)
            lines.append(
                f"\n**Gợi ý phù hợp nhất:** #{best.get('room_id')} "
                f"với giá {format_vnd(best.get('rent_price'))}/tháng{area}{reason}."
            )
        missing = comparison.get("missing_room_ids") or []
        if missing:
            lines.append(f"_Chưa có dữ liệu cho: {', '.join('#' + item for item in missing)}._")
        not_compared = comparison.get("not_compared_room_ids") or []
        if not_compared:
            lines.append(f"_Mình chỉ so sánh tối đa 3 phòng/lượt nên chưa so sánh: {', '.join('#' + item for item in not_compared)}._")
        return "\n".join(lines)
    if intent == "REQUEST_FAQ":
        faq = tool_results.get("faq_results") or []
        if faq:
            lines = ["Dạ em trả lời theo thông tin hiện có ạ:"]
            lines.extend(f"**[{item.get('topic')}]** {item.get('answer')}" for item in faq)
            lines.append("Nếu anh/chị đã ưng căn nào rồi, mình có thể bấm `Đặt lịch xem phòng` để qua bước xem thực tế nhé.")
            return "\n".join(lines)
        return "Dạ em có thể hỗ trợ thông tin về quy trình thuê, hợp đồng, tiền cọc và các thủ tục liên quan. Anh/chị muốn hỏi cụ thể phần nào để em trả lời đúng ý hơn ạ?"
    return (
        "Mình có thể giúp tìm phòng, lọc điều kiện, hỏi đáp về phòng đang xem, "
        "tính chi phí, so sánh tối đa 3 phòng và gợi ý phòng tương tự trên nhatrovn. "
        "Mình không thực hiện: đặt lịch, nhắn chủ nhà, giữ chỗ hoặc thanh toán."
    )


async def _compose_answer_async(
    question: str, parsed: dict[str, Any], grounding: dict[str, Any],
    tool_results: dict[str, Any], history: list[dict[str, Any]], user_mood: str = "normal",
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    intent = parsed["intent"]
    rooms = grounding.get("rooms", [])
    faq = tool_results.get("faq_results")

    if _should_use_template_response(intent, question, grounding, tool_results, faq):
        answer = _compose_answer_template(parsed, grounding, tool_results, question)
        if stream_callback is not None:
            await _stream_text_chunks(answer, stream_callback)
        return answer
    verified_data = _build_llm_context(grounding, tool_results)
    answer = ""
    try:
        from agents.response_writer import (
            write_no_result_response,
            write_response,
        )
        if not rooms and intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
            constraints = grounding.get("constraints", {})
            answer = await write_no_result_response(question, constraints, user_mood)
            if stream_callback is not None:
                await _stream_text_chunks(answer, stream_callback)
        else:
            answer = await write_response(question=question, verified_context=verified_data, history=history, mood=user_mood)
            if stream_callback is not None:
                await _stream_text_chunks(answer, stream_callback)
    except Exception as e:
        import traceback
        traceback.print_exc()
    if not answer or len(answer.strip()) < 20:
        answer = _compose_answer_template(parsed, grounding, tool_results, question)
        if stream_callback is not None:
            await _stream_text_chunks(answer, stream_callback)
        return answer
    if stream_callback is not None:
        return answer.strip()
    try:
        from config import ENABLE_REVIEWER
        if not ENABLE_REVIEWER:
            return answer.strip() if answer.strip() else _compose_answer_template(parsed, grounding, tool_results, question)
        from agents.reviewer import review_with_retry
        answer, _ = await review_with_retry(question=question, answer=answer, room_context=verified_data, max_retries=1)
    except Exception:
        pass
    return answer.strip() if answer.strip() else _compose_answer_template(parsed, grounding, tool_results, question)


async def _stream_text_chunks(
    text: str,
    stream_callback: Callable[[str], Awaitable[None]],
    words_per_chunk: int = 4,
) -> None:
    """Emit deterministic answers in small chunks so SSE UX matches LLM replies."""
    import re

    pieces = re.findall(r"\S+\s*|\n+", str(text or ""))
    if not pieces:
        return

    chunk_pieces: list[str] = []
    visible_words = 0
    for piece in pieces:
        chunk_pieces.append(piece)
        if piece.strip() and "\n" not in piece:
            visible_words += 1
        if visible_words >= words_per_chunk:
            await stream_callback("".join(chunk_pieces))
            chunk_pieces = []
            visible_words = 0
            await asyncio.sleep(0.02)

    if chunk_pieces:
        await stream_callback("".join(chunk_pieces))


def _should_use_template_response(
    intent: str,
    question: str,
    grounding: dict[str, Any],
    tool_results: dict[str, Any],
    faq: list[dict[str, Any]] | None,
) -> bool:
    if intent in {"REQUEST_ACTION", "CALCULATE_COST", "COMPARE_ROOMS", "ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        return True
    if tool_results.get("error"):
        return True
    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"} and grounding.get("rooms"):
        return True
    if grounding.get("rooms") or faq:
        return False
    return intent not in {"GENERAL_HELP", "REQUEST_FAQ", "SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}


def _extract_rooms(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = tool_results.get("rooms") or []
    return [item for item in rooms if item]


def _current_room_from_results(intent: str, rooms: list[dict[str, Any]]) -> str | None:
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"} and rooms:
        return rooms[0].get("room_id")
    return None


def _unknown_fields(room: dict[str, Any]) -> list[str]:
    return unknown_room_fields(room)


def _asks_about_amenities(question: str) -> bool:
    import unicodedata
    text = unicodedata.normalize("NFD", question.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return any(
        phrase in text
        for phrase in (
            "tien ich", "co gi", "may lanh", "ban cong", "cua so",
            "wifi", "gac", "toilet", "gio giac", "thu cung", "de xe",
            "may giat", "nuoc nong", "tu lanh",
        )
    )


def _asks_to_compare_result_set(question: str) -> bool:
    import re
    import unicodedata

    text = unicodedata.normalize("NFD", question.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")
    if not re.search(r"\bso\s*sanh\b", text):
        return False
    return bool(
        re.search(r"\b(?:3|ba)\s*phong\b", text)
        or re.search(r"\b(?:cac|nhung|may)\s+phong\b", text)
        or re.search(r"\bphong\s+(?:nay|tren|vua|dau tien)\b", text)
    )


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
    import unicodedata

    normalized = question.lower()
    ascii_text = "".join(
        ch for ch in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(ch) != "Mn"
    ).replace("đ", "d")
    if re.search(r"\b(?:nua nam|nửa năm)\b", ascii_text):
        return 6
    match = re.search(r"\b(\d{1,2})\s*(?:nam|năm|year|years)\b", ascii_text)
    if match:
        years = int(match.group(1))
        return years * 12 if years > 0 else None
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
            f"Chi phí tháng đầu của #{first} khoảng bao nhiêu?",
            f"Phòng #{first} hợp mấy người ở?",
            "Nếu ổn thì mình xem phòng buổi sáng hay chiều?",
        ]
    if intent == "GENERAL_HELP":
        return ["Tìm phòng dưới 5 triệu ở quận Bình Thạnh", "Phòng này có cho nuôi mèo không?", "Nếu hợp thì mình đặt lịch xem luôn được không?"]
    return ["Mình ưu tiên gần hơn hay rẻ hơn ạ?", "Nới ngân sách thêm 500k được không?", "Em lọc sang khu lân cận cho mình nhé?"]


def _resolve_inline_room_followup(
    question: str,
    constraints: dict[str, Any],
    repository: RoomRepository,
) -> dict[str, Any] | None:
    import re

    normalized = _normalize_sales_text(question)
    match = re.search(
        r"phong\s+(duong\s+so\s+\d+[a-z]?)\s+co\s+(may\s+lanh|ban\s+cong|cua\s+so|wifi|gac)\s+khong",
        normalized,
    )
    if not match:
        return None

    amenity_map = {
        "may lanh": "air_conditioner",
        "ban cong": "balcony",
        "cua so": "window",
        "wifi": "wifi",
        "gac": "mezzanine",
    }
    street_ref = match.group(1).strip()
    amenity = amenity_map.get(match.group(2).strip())
    if not amenity:
        return None

    location = constraints.get("location") or {}
    followup_constraints = {
        "location": {
            "province": location.get("province"),
            "districts": list(location.get("districts") or []),
            "wards": list(location.get("wards") or []),
            "near_landmarks": [street_ref],
            "max_distance_km": None,
        },
        "budget": {"min": None, "min_operator": None, "max": None, "max_operator": None, "type": "rent_only"},
        "area": {"preference": None},
        "occupants": None,
        "vehicles": [],
        "pets_required": [],
        "amenities_required": [],
        "amenities_preferred": [],
        "excluded_features": [],
        "move_in_date": None,
    }
    candidates = repository.search_by_constraints(followup_constraints, limit=5, offset=0)
    if not candidates:
        candidates = _search_inline_followup_candidates(repository, street_ref, location)
    return {"street_ref": street_ref, "amenity": amenity, "candidates": candidates}


def _compose_inline_followup_answer(followup: dict[str, Any]) -> str:
    if not followup:
        return ""
    street_ref = followup.get("street_ref") or "địa chỉ này"
    amenity = followup.get("amenity") or ""
    candidates = followup.get("candidates") or []
    amenity_labels = {
        "air_conditioner": "máy lạnh",
        "balcony": "ban công",
        "window": "cửa sổ",
        "wifi": "wifi",
        "mezzanine": "gác",
    }
    amenity_label = amenity_labels.get(amenity, amenity)
    if not candidates:
        return f"Về ý sau, em chưa xác định được phòng {street_ref} cụ thể nào trong dữ liệu hiện có để xác nhận {amenity_label} cho mình ạ."

    positive = next((room for room in candidates if _room_has_positive_amenity(room, amenity)), None)
    if positive:
        return (
            f"Về ý sau, em có thấy căn gần {street_ref}: **{positive.get('title')}** "
            f"(#{positive.get('room_id')}) và căn này **có {amenity_label}** ạ."
        )

    room = candidates[0]
    return (
        f"Về ý sau, em có thấy căn gần {street_ref}: **{room.get('title')}** "
        f"(#{room.get('room_id')}) nhưng dữ liệu xác minh hiện tại cho thấy căn này **không có {amenity_label}** ạ."
    )


def _room_has_positive_amenity(room: dict[str, Any], amenity: str) -> bool:
    import re

    pattern_map = {
        "air_conditioner": r"Máy lạnh\s*:\s*(?:Có|Riêng|Tự do|True|Yes|Free)",
        "balcony": r"Ban công\s*:\s*(?:Có|True|Yes)",
        "window": r"Cửa sổ\s*:\s*(?:Có|True|Yes)",
        "wifi": r"Wifi\s*:\s*(?:Có|Free|True|Yes)",
        "mezzanine": r"Gác\s*:\s*(?:Có|True|Yes)",
    }
    pattern = pattern_map.get(amenity)
    if not pattern:
        return False
    searchable = " ".join(
        str(part or "")
        for part in (room.get("embedding_text"), room.get("description"), " ".join(room.get("amenities") or []))
    )
    return bool(re.search(pattern, searchable, re.IGNORECASE))


def _search_inline_followup_candidates(
    repository: RoomRepository,
    street_ref: str,
    location: dict[str, Any],
) -> list[dict[str, Any]]:
    import re

    collection = getattr(repository, "_collection", None)
    if collection is None:
        return []

    street_number_match = re.search(r"(\d+[a-z]?)", street_ref, re.IGNORECASE)
    if not street_number_match:
        return []
    street_number = street_number_match.group(1)

    clauses: list[dict[str, Any]] = [
        {"metadata.house_name": {"$regex": street_number, "$options": "i"}},
        {"metadata.room_code": {"$regex": street_number, "$options": "i"}},
        {"embedding_text": {"$regex": rf"(?:duong|đường)\\s*(?:so|số)?\\s*{re.escape(street_number)}", "$options": "i"}},
    ]

    area_filters: list[dict[str, Any]] = []
    for ward in location.get("wards") or []:
        area_filters.append({"metadata.ward_name": {"$regex": _accent_flexible_location_pattern(ward), "$options": "i"}})
    for district in location.get("districts") or []:
        area_filters.append({"metadata.district_name": {"$regex": _accent_flexible_location_pattern(district), "$options": "i"}})

    query = {
        "$and": [
            {"metadata.status_code": {"$in": ["0", ""]}},
            {"$or": clauses},
        ]
    }
    if area_filters:
        query["$and"].append({"$or": area_filters})

    docs = list(collection.find(query).limit(5))
    normalizer = globals().get("normalize_room")
    if normalizer is None:
        from .schemas import normalize_room as normalizer
    return [item for item in (normalizer(doc) for doc in docs) if item]


def _accent_flexible_location_pattern(value: str) -> str:
    from .repository import _accent_flexible_regex, _normalize_location_value

    normalized = _normalize_location_value(value)
    if not normalized:
        return ""
    return _accent_flexible_regex(normalized)


def _sanitize_result_rooms(
    intent: str,
    rooms: list[dict[str, Any]],
    constraints: dict[str, Any],
    retrieval_trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if intent not in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"} or not rooms:
        return rooms
    trace = retrieval_trace or {}
    fallback_strategy = trace.get("fallback_strategy")
    if fallback_strategy in {"nearby_location", "relax_price_nearby_location"}:
        return rooms
    matched = [room for room in rooms if room_matches_constraints(room, constraints)]
    return matched


def _is_broad_new_lead(question: str, constraints: dict[str, Any]) -> bool:
    normalized = _normalize_sales_text(question)
    budget = constraints.get("budget") or {}
    location = constraints.get("location") or {}
    has_budget = budget.get("min") is not None or budget.get("max") is not None
    has_location = bool(location.get("province") or location.get("districts") or location.get("wards") or location.get("near_landmarks"))
    has_required_amenities = bool(constraints.get("amenities_required"))
    if has_budget or has_location or has_required_amenities:
        return False
    return any(
        phrase in normalized
        for phrase in ("con phong khong", "con phong trong khong", "xin gia", "phong o dau", "gia bao nhieu")
    )


def _normalize_sales_text(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFD", str(text or "").lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return normalized.replace("đ", "d")


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


def _max_question_chars() -> int:
    try:
        from config import MAX_USER_QUESTION_CHARS
        return max(100, int(MAX_USER_QUESTION_CHARS))
    except Exception:
        return 1200


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _update_langfuse_turn_span(
    *,
    span_input: dict[str, Any] | None = None,
    span_output: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> None:
    """Attach a bounded, non-sensitive summary to the active Langfuse span."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return
    kwargs: dict[str, Any] = {}
    if span_input is not None:
        kwargs["input"] = span_input
    if span_output is not None:
        kwargs["output"] = span_output
    if metadata is not None:
        kwargs["metadata"] = metadata
    if level is not None:
        kwargs["level"] = level
    if status_message is not None:
        kwargs["status_message"] = status_message
    if not kwargs:
        return
    try:
        from langfuse import get_client
        get_client().update_current_span(**kwargs)
    except Exception as exc:
        _logger.debug("langfuse_turn_span_update_failed %s", exc)


def _langfuse_safe_turn_output(result: dict[str, Any]) -> dict[str, Any]:
    trace = result.get("agent_trace") or {}
    room_ids = [
        str(room.get("room_id"))
        for room in (result.get("rooms") or [])[:5]
        if isinstance(room, dict) and room.get("room_id")
    ]
    return {
        "intent": result.get("intent"),
        "room_ids": room_ids,
        "read_tool_calls": trace.get("read_tool_calls", 0),
        "write_tool_calls": trace.get("write_tool_calls", 0),
        "retrieval_low_confidence": bool(result.get("retrieval_low_confidence")),
        "retrieval_feedback_retry_count": result.get("retrieval_feedback_retry_count", 0),
        "processing_time_ms": result.get("processing_time_ms"),
        "error_category": trace.get("error_category"),
    }


def _langfuse_safe_turn_metadata(result: dict[str, Any], question_hash: str) -> dict[str, Any]:
    return {
        "session_hash": _question_hash(str(result.get("session_id") or "")),
        "question_hash": question_hash,
        "source_count": len(result.get("sources") or []),
        "suggested_question_count": len(result.get("suggested_questions") or []),
        "capture_policy": "sanitized_root_summary",
    }


def _log_turn_summary(result: dict[str, Any], question_hash: str) -> None:
    trace = result.get("agent_trace") or {}
    record = {
        "event": "room_assistant_turn",
        "session_id": result.get("session_id"),
        "question_hash": question_hash,
        "intent": result.get("intent"),
        "room_count": len(result.get("rooms") or []),
        "read_tool_calls": trace.get("read_tool_calls", 0),
        "write_tool_calls": trace.get("write_tool_calls", 0),
        "retrieval_confidence": result.get("retrieval_confidence"),
        "retrieval_low_confidence": result.get("retrieval_low_confidence"),
        "error_category": trace.get("error_category"),
        "processing_time_ms": result.get("processing_time_ms"),
        "retrieval_ms": trace.get("retrieval_ms"),
        "llm_generate_ms": trace.get("llm_generate_ms"),
    }
    try:
        _logger.info("room_assistant_turn %s", json.dumps(record, ensure_ascii=False, sort_keys=True))
    except Exception:
        _logger.info("room_assistant_turn intent=%s error=%s", record["intent"], record["error_category"])


def format_vnd(value: Any) -> str:
    if value is None:
        return "chưa rõ"
    try:
        return f"{int(value):,} VND".replace(",", ".")
    except Exception:
        return str(value)
