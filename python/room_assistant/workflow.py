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

from .intent import parse_intent_and_constraint_patch, parse_intent_async
from .landmark_aliases import expand_landmark_tokens, primary_display_token
from .repository import RoomRepository, create_room_repository
from .retrieval import RoomSemanticIndex, build_retrieval_explanation
from .schemas import MAX_READ_TOOL_CALLS_PER_TURN, public_session_state, unknown_room_fields
from .session_store import (
    SessionStore,
    apply_operations,
    create_session_store,
    load_session_state,
    save_session_state,
    update_turn_state,
)
from .prompts import (
    AMENITY_LABELS,
    ASK_ROOM_AMENITIES,
    ASK_ROOM_HEADER,
    ASK_ROOM_LOCATION,
    ASK_ROOM_MISSING_ID,
    ASK_ROOM_UNKNOWN_FIELDS,
    CALCULATE_COST_DEPOSIT,
    CALCULATE_COST_LINE,
    CALCULATE_COST_MISSING_INFO,
    CALCULATE_COST_NOT_CALCULATED,
    CALCULATE_COST_OPENING,
    CALCULATE_COST_RECURRING_FEES,
    CALCULATE_COST_RENTAL_PERIOD,
    CALCULATE_COST_TOTAL_INITIAL,
    CALCULATE_COST_TOTAL_PERIOD,
    CALCULATE_COST_UNKNOWN,
    COMPARE_BEST_PICK,
    COMPARE_BEST_PICK_AREA_SUFFIX,
    COMPARE_INSIGHT_CHEAPER,
    COMPARE_INSIGHT_FIRST_ADVANTAGE,
    COMPARE_INSIGHT_LARGER,
    COMPARE_INSIGHT_SECOND_ADVANTAGE,
    COMPARE_INSIGHT_TIE,
    COMPARE_MISSING_DATA,
    COMPARE_MISSING_ROOMS,
    COMPARE_NEED_ROOM_IDS,
    COMPARE_NOT_COMPARED,
    COMPARE_ORDINAL_UNRESOLVED,
    COMPARE_OPENING,
    COMPARE_ROW,
    COMPARE_STATUS_AVAILABLE,
    COMPARE_STATUS_UNKNOWN,
    COMPARE_UNKNOWN_AREA,
    COST_FIXED_FIELD_LABELS,
    COST_ITEM_LABELS,
    FEATURE_FACT_LABELS,
    GENERAL_HELP_DEFAULT,
    GENERAL_HELP_OFF_TOPIC,
    GENERAL_HELP_PRICE_OBJECTION,
    INPUT_TOO_LONG_ANSWER,
    INSUFFICIENT_VERIFIED_DATA,
    LANDMARK_HINT_SUFFIX,
    LANDMARK_NEAR_HINT,
    LLM_CONTEXT_COMPARE_HEADER,
    LLM_CONTEXT_COMPARE_MISSING,
    LLM_CONTEXT_COMPARE_NOT_COMPARED,
    LLM_CONTEXT_COMPARE_ROW,
    LLM_CONTEXT_COST_HEADER,
    LLM_CONTEXT_COST_LINE,
    LLM_CONTEXT_COST_NOT_CALCULATED,
    LLM_CONTEXT_COST_TOTAL_INITIAL,
    LLM_CONTEXT_COST_TOTAL_PERIOD,
    LLM_CONTEXT_COST_UNKNOWN,
    ORDINAL_OUT_OF_RANGE,
    LLM_CONTEXT_FAQ_HEADER,
    LLM_CONTEXT_FAQ_LINE,
    LLM_CONTEXT_ROOM_FEATURES,
    LLM_CONTEXT_ROOM_LIST_HEADER,
    LLM_CONTEXT_ROOM_STATUS_AVAILABLE,
    LLM_CONTEXT_ROOM_STATUS_UNAVAILABLE,
    LLM_CONTEXT_UNKNOWN_FIELDS,
    LLM_CONTEXT_VERIFIED_AMENITIES,
    RELAXED_NOTE_DEFAULT,
    RELAXED_NOTE_WITH_FIELDS,
    RELAX_FIELD_LABELS,
    REQUEST_FAQ_FALLBACK,
    SEARCH_ALTERNATIVE_CTA,
    format_request_action_answer,
    SEARCH_ALTERNATIVE_OPENING,
    SEARCH_ALTERNATIVE_ROOM_LINE,
    SEARCH_NO_RESULT_BUDGET_FRUSTRATED_PREFIX,
    SEARCH_NO_RESULT_BUDGET_LINE,
    SEARCH_NO_RESULT_BUDGET_ONLY,
    SEARCH_NO_RESULT_BUDGET_ONLY_EMPATHY_PREFIX,
    SEARCH_NO_RESULT_BUDGET_URGENT_PREFIX,
    SEARCH_NO_RESULT_DEFAULT,
    SEARCH_NO_RESULT_FRUSTRATED,
    SEARCH_NO_RESULT_SALES_HANDOFF,
    SEARCH_NO_RESULT_URGENT,
    SEARCH_RELAXED_OPENING,
    SEARCH_ROOM_LINE,
    SEARCH_SUCCESS_CTA,
    SEARCH_SUCCESS_OPENING,
    SUGGESTED_QUESTIONS_GENERAL_HELP,
    SUGGESTED_QUESTIONS_INPUT_TOO_LONG,
    SUGGESTED_QUESTIONS_NO_RESULT,
    SUGGESTED_QUESTIONS_WITH_ROOM,
    TOOL_BUDGET_EXCEEDED_ANSWER,
    UNKNOWN_DISTRICT,
    UNKNOWN_LOCATION,
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
_UNSET = object()
_init_lock = threading.Lock()
_tool_registry = ReadOnlyToolRegistry()
_logger = logging.getLogger(__name__)


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


@observe(name="room_assistant_turn", capture_input=False, capture_output=False)
async def run_room_assistant(
    question: str,
    history: list[dict[str, Any]] | None = None,
    session_id: str = "",
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
    repository: RoomRepository | None = None,
    session_store: SessionStore | None = None,
    semantic_index: Any = _UNSET,
) -> dict[str, Any]:
    """Chạy một lượt hội thoại của người dùng."""
    started = time.time()
    question = str(question or "").strip()
    history = (history or [])[-8:]
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

    if stream_callback:
        await stream_callback("[status:Phân tích|Hệ thống] Đang phân tích yêu cầu...\n")

    if semantic_index is _UNSET:
        semantic_index = _get_semantic_index()
    parsed = await parse_intent_async(question, state_before)
    merged_state, applied_operations = apply_operations(state_before, parsed["operations"])

    user_mood = "normal"
    try:
        from agents.sentiment_analyzer import analyze_mood
        user_mood, _ = analyze_mood(question)
    except Exception:
        pass

    if stream_callback:
        await stream_callback("[status:Truy vấn|Cơ sở dữ liệu] Đang tìm kiếm các phòng phù hợp...\n")

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

    if stream_callback:
        await stream_callback("[status:Tổng hợp|Trợ lý AI] Đang tổng hợp câu trả lời...\n")

    rooms = _extract_rooms(tool_results)
    if tool_results.get("relaxed_search"):
        rooms = [dict(room, relaxed_search=True) for room in rooms]
    result_ids = [item["room_id"] for item in rooms if item.get("room_id")]
    if parsed.get("ordinal_out_of_range"):
        current_room_id = None
    else:
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

    grounding = _build_grounding_context(parsed, next_state, tool_results, question=question)
    composed = await _compose_answer_async(
        question, parsed, grounding, tool_results, history, user_mood, stream_callback,
    )
    answer = composed.get("answer", "")
    abstain = bool(composed.get("abstain"))
    abstain_reason = composed.get("abstain_reason") or ""
    verification = composed.get("verification") or {}
    suggested_questions = _suggest_questions(parsed["intent"], rooms, current_room_id)
    processing_time_ms = int((time.time() - started) * 1000)
    retrieval_explanation = build_retrieval_explanation(
        context.retrieval_trace,
        next_state.get("constraints"),
        relaxed_fields=tool_results.get("relaxed_fields") or [],
        result_count=len(rooms),
    )

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
        "abstain": abstain,
        "abstain_reason": abstain_reason if abstain else None,
        "verification": verification,
        "retrieval_confidence": context.retrieval_trace.get("retrieval_confidence"),
        "retrieval_low_confidence": context.retrieval_trace.get("retrieval_low_confidence"),
        "retrieval_feedback_retry_count": context.retrieval_trace.get("retrieval_feedback_retry_count", 0),
        "retrieval_attempts": context.retrieval_trace.get("retrieval_attempts", []),
        "retrieval_explanation": retrieval_explanation,
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
            "retrieval_explanation": retrieval_explanation,
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
    if parsed.get("ordinal_out_of_range"):
        current_room_id = None
    else:
        current_room_id = (
            parsed.get("current_room_id")
            or state.get("current_room_id")
            or _first_or_none(state.get("last_result_ids", []))
        )

    # 1. Direct Fast Path (disambiguate room_id vs room_code) — chỉ cho hỏi chi tiết, không chặn tính phí/so sánh.
    exact_ref = parsed.get("exact_reference") or parsed.get("exact_room_reference")
    if exact_ref and intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        room_id = exact_ref.get("room_id") if isinstance(exact_ref, dict) else getattr(exact_ref, "room_id", None)
        room_code = exact_ref.get("room_code") if isinstance(exact_ref, dict) else getattr(exact_ref, "room_code", None)
        lookup_ref = room_id or room_code
        if lookup_ref:
            detail = _tool_registry.execute(
                "retrieve_room_context", {"room_id": lookup_ref}, context,
            )
            if detail.get("room"):
                return {"room_context": detail, "rooms": [detail["room"]], "fast_path": True}

    # 2. Simple Search vs Complex Planner routing
    # If the user asks a simple question or planner is disabled
    is_complex = parsed.get("complex_planning", False)
    
    if intent in {"SEARCH_ROOM", "REFINE_SEARCH"}:
        if is_complex:
            # Route to planner (simulated here by standard search but indicating complex branch)
            context.planner_invoked = True

        rooms = _tool_registry.execute(
            "search_rooms",
            {"query_text": question, "constraints": constraints, "top_k": 5},
            context,
        )
        if rooms:
            rooms = _filter_rooms_by_budget(rooms, constraints)
            if rooms:
                return {"rooms": rooms}

        has_district = bool((constraints.get("location") or {}).get("districts"))
        has_budget = bool((constraints.get("budget") or {}).get("max") or (constraints.get("budget") or {}).get("min"))

        # Bước 1 — nới lỏng các bộ lọc địa lý "mờ" (mốc gần, phường) + tiện ích ưu tiên.
        # Giữ nguyên quận + ngân sách để vẫn đúng khu vực và túi tiền của khách.
        soft_constraints, soft_dropped = _relax_soft_filters(constraints)
        if soft_dropped and context.read_tool_calls < MAX_READ_TOOL_CALLS_PER_TURN:
            rooms = _tool_registry.execute(
                "search_rooms",
                {"query_text": question, "constraints": soft_constraints, "top_k": 5},
                context,
            )
            rooms = _filter_rooms_by_budget(rooms, constraints)
            if rooms:
                return {"rooms": rooms, "relaxed_search": True, "relaxed_fields": soft_dropped}
        else:
            soft_constraints = dict(constraints)

        # Bước 2 — nới tiện ích bắt buộc. Giữ ngân sách + quận.
        hard_constraints, hard_dropped = _relax_hard_filters(soft_constraints)
        all_dropped = soft_dropped + hard_dropped
        if hard_dropped and context.read_tool_calls < MAX_READ_TOOL_CALLS_PER_TURN:
            alt_rooms = _tool_registry.execute(
                "search_rooms",
                {"query_text": question, "constraints": hard_constraints, "top_k": 5},
                context,
            )
            alt_rooms = _filter_rooms_by_budget(alt_rooms, constraints)
            if alt_rooms:
                if has_district:
                    return {"rooms": alt_rooms, "relaxed_search": True, "relaxed_fields": all_dropped}
                return {"rooms": [], "alternative_rooms": alt_rooms, "relaxed_fields": all_dropped}

        if has_budget or has_district:
            return {"rooms": [], "budget_or_district_miss": True, "public_inventory_miss": True}
        return {"rooms": [], "public_inventory_miss": True}

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        if parsed.get("ordinal_out_of_range"):
            return {"rooms": [], "ordinal_out_of_range": True}
        detail = _tool_registry.execute(
            "retrieve_room_context", {"room_id": current_room_id}, context,
        )
        payload: dict[str, Any] = {
            "room_context": detail,
            "rooms": [detail["room"]] if detail.get("room") else [],
        }
        try:
            from room_assistant.staff_knowledge import match_staff_faq
            if match_staff_faq(question):
                payload["faq_results"] = _tool_registry.execute(
                    "retrieve_faq", {"question": question}, context,
                )
        except Exception:
            pass
        return payload

    if intent == "CALCULATE_COST":
        room = _tool_registry.execute("get_room_detail", {"room_id": current_room_id}, context)
        estimate = _tool_registry.execute(
            "calculate_cost_estimate",
            {"room": room, "rental_months": _extract_rental_months(question), "constraints": constraints},
            context,
        )
        return {"rooms": [room] if room else [], "cost_estimate": estimate}

    if intent == "COMPARE_ROOMS":
        if parsed.get("compare_unresolved"):
            return {"comparison": {"rows": [], "compare_unresolved": True}, "rooms": []}
        ids = parsed.get("referenced_room_ids") or []
        selected_ids = state.get("selected_room_ids") or []
        last_result_ids = state.get("last_result_ids") or []
        if len(ids) < 2 and _asks_to_compare_result_set(question):
            pool = last_result_ids or selected_ids or ids
            requested_count = _requested_result_set_count(question, len(pool))
            ids = pool[:requested_count] if requested_count else pool
        if not ids:
            ids = selected_ids if len(selected_ids) >= 2 else last_result_ids
        comparison = _tool_registry.execute("compare_rooms", {"room_ids": ids[:3]}, context)
        if len(ids) > 3:
            comparison["not_compared_room_ids"] = ids[3:]
        return {"comparison": comparison, "rooms": comparison.get("rows", [])}

    if intent == "FIND_SIMILAR":
        if not current_room_id:
            return {"rooms": [], "find_similar_missing_source": True}
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


def _input_too_long_result(
    session_id: str,
    state_before: dict[str, Any],
    max_question_chars: int,
    processing_time_ms: int,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "answer": INPUT_TOO_LONG_ANSWER.format(max_chars=max_question_chars),
        "intent": "GENERAL_HELP",
        "session_state": public_session_state(state_before),
        "rooms": [],
        "cost_estimate": None,
        "comparison": None,
        "suggested_questions": list(SUGGESTED_QUESTIONS_INPUT_TOO_LONG),
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
    question: str = "",
) -> dict[str, Any]:
    rooms = _extract_rooms(tool_results)
    from .sources import build_room_sources

    sources = build_room_sources(rooms, query=question)
    unknown: list[str] = []
    for room in rooms:
        room_id = room.get("room_id")
        if not room_id:
            continue
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
    constraints = grounding.get("constraints", {})
    if rooms:
        parts.append(LLM_CONTEXT_ROOM_LIST_HEADER)
        for room in rooms[:5]:
            rent = format_vnd(room.get("rent_price"))
            details = [
                f"[{room.get('room_id')}] {room.get('title')}",
                f"{rent}/tháng",
                room.get("district") or UNKNOWN_DISTRICT,
                f"Diện tích: {room.get('area_m2') or '?'} m²",
            ]
            amenities = _verified_amenity_labels(room, constraints)
            if amenities:
                details.append(LLM_CONTEXT_VERIFIED_AMENITIES.format(amenities=", ".join(amenities)))
            feature_facts = _room_feature_facts(room)
            if feature_facts:
                details.append(LLM_CONTEXT_ROOM_FEATURES.format(features=", ".join(feature_facts[:8])))
            if room.get("available") is not None:
                details.append(
                    LLM_CONTEXT_ROOM_STATUS_AVAILABLE
                    if room.get("available")
                    else LLM_CONTEXT_ROOM_STATUS_UNAVAILABLE
                )
            parts.append(
                "- " + " | ".join(str(item) for item in details if item)
            )

    estimate = tool_results.get("cost_estimate")
    if estimate and estimate.get("available"):
        parts.append(LLM_CONTEXT_COST_HEADER)
        for item in estimate.get("items", []):
            parts.append(LLM_CONTEXT_COST_LINE.format(name=item["name"], amount=format_vnd(item.get("amount"))))
        parts.append(LLM_CONTEXT_COST_TOTAL_INITIAL.format(amount=format_vnd(estimate.get("total_initial_cost"))))
        if estimate.get("rental_months"):
            for item in estimate.get("period_items", []):
                parts.append(LLM_CONTEXT_COST_LINE.format(name=item["name"], amount=format_vnd(item.get("amount"))))
            parts.append(
                LLM_CONTEXT_COST_TOTAL_PERIOD.format(
                    months=estimate["rental_months"],
                    amount=format_vnd(estimate.get("total_period_cost")),
                )
            )
        if estimate.get("unknown"):
            parts.append(LLM_CONTEXT_COST_UNKNOWN.format(fields=", ".join(estimate["unknown"])))
        if estimate.get("not_calculated"):
            not_calculated = [
                f"{item.get('name')}: {item.get('value')}"
                for item in estimate["not_calculated"]
            ]
            parts.append(LLM_CONTEXT_COST_NOT_CALCULATED.format(details=", ".join(not_calculated)))

    comparison = tool_results.get("comparison")
    if comparison and comparison.get("rows"):
        parts.append(LLM_CONTEXT_COMPARE_HEADER)
        for row in comparison["rows"]:
            parts.append(
                LLM_CONTEXT_COMPARE_ROW.format(
                    room_id=row.get("room_id"),
                    rent=format_vnd(row.get("rent_price")),
                    area=row.get("area_m2") or "?",
                    district=row.get("district") or UNKNOWN_LOCATION,
                )
            )
        if comparison.get("missing_room_ids"):
            parts.append(
                LLM_CONTEXT_COMPARE_MISSING.format(
                    room_ids=", ".join("#" + item for item in comparison["missing_room_ids"])
                )
            )
        if comparison.get("not_compared_room_ids"):
            parts.append(
                LLM_CONTEXT_COMPARE_NOT_COMPARED.format(
                    room_ids=", ".join("#" + item for item in comparison["not_compared_room_ids"])
                )
            )

    faq = tool_results.get("faq_results")
    if faq:
        parts.append(LLM_CONTEXT_FAQ_HEADER)
        for item in faq:
            parts.append(LLM_CONTEXT_FAQ_LINE.format(topic=item.get("topic"), answer=item.get("answer")))

    if grounding.get("unknown"):
        parts.append(LLM_CONTEXT_UNKNOWN_FIELDS.format(fields=", ".join(grounding["unknown"])))

    context = "\n".join(parts) if parts else LLM_CONTEXT_EMPTY
    try:
        from config import EVIDENCE_MAX_CHARS
        return context[:EVIDENCE_MAX_CHARS]
    except Exception:
        return context[:3500]


def _search_no_result_message(
    user_mood: str = "normal",
    constraints: dict[str, Any] | None = None,
    *,
    sales_handoff: bool = False,
) -> str:
    if sales_handoff:
        return SEARCH_NO_RESULT_SALES_HANDOFF.get(
            user_mood,
            SEARCH_NO_RESULT_SALES_HANDOFF["normal"],
        )
    budget = (constraints or {}).get("budget") or {}
    districts = ((constraints or {}).get("location") or {}).get("districts") or []
    max_price = budget.get("max")
    district_label = districts[0] if districts else ""

    if max_price and district_label:
        area = district_label.replace("_", " ")
        budget_line = SEARCH_NO_RESULT_BUDGET_LINE.format(
            area=area,
            budget=format_vnd(max_price),
        )
        if user_mood == "urgent":
            return SEARCH_NO_RESULT_BUDGET_URGENT_PREFIX.format(body=budget_line)
        if user_mood == "frustrated":
            return SEARCH_NO_RESULT_BUDGET_FRUSTRATED_PREFIX.format(body=budget_line)
        return budget_line

    if max_price:
        budget_only = SEARCH_NO_RESULT_BUDGET_ONLY.format(budget=format_vnd(max_price))
        if user_mood in {"urgent", "frustrated"}:
            return SEARCH_NO_RESULT_BUDGET_ONLY_EMPATHY_PREFIX.format(body=budget_only)
        return budget_only

    if user_mood == "urgent":
        return SEARCH_NO_RESULT_URGENT
    if user_mood == "frustrated":
        return SEARCH_NO_RESULT_FRUSTRATED
    return SEARCH_NO_RESULT_DEFAULT


def _search_alternative_opening(user_mood: str = "normal") -> str:
    return SEARCH_ALTERNATIVE_OPENING.get(user_mood, SEARCH_ALTERNATIVE_OPENING["normal"])


def _compose_answer_template(
    parsed: dict[str, Any],
    grounding: dict[str, Any],
    tool_results: dict[str, Any],
    *,
    question: str = "",
    user_mood: str = "normal",
) -> str:
    intent = parsed["intent"]
    if parsed.get("ordinal_out_of_range"):
        requested = parsed.get("ordinal_requested") or 0
        available = int(parsed.get("ordinal_available_count") or 0)
        return ORDINAL_OUT_OF_RANGE.format(requested=requested, available=available)
    if tool_results.get("find_similar_missing_source"):
        from room_assistant.prompts import FIND_SIMILAR_MISSING_SOURCE
        return FIND_SIMILAR_MISSING_SOURCE
    if parsed.get("compare_unresolved") and intent == "COMPARE_ROOMS":
        return "Dạ em chưa đủ phòng trong danh sách để so sánh theo thứ tự anh/chị yêu cầu. Anh/chị chọn lại giúp em nha!"
    if tool_results.get("error") == "tool_budget_exceeded":
        return TOOL_BUDGET_EXCEEDED_ANSWER
    if intent == "REQUEST_ACTION":
        return format_request_action_answer(parsed.get("requested_action"))
    rooms = grounding["rooms"]
    constraints = grounding.get("constraints", {})
    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
        if not rooms:
            alternative_rooms = [item for item in (tool_results.get("alternative_rooms") or []) if item]
            if alternative_rooms:
                alt_lines = [_search_alternative_opening(user_mood)]
                for idx, room in enumerate(alternative_rooms[:5], 1):
                    alt_lines.append(
                        SEARCH_ALTERNATIVE_ROOM_LINE.format(
                            index=idx,
                            title=room.get("title"),
                            room_id=room.get("room_id"),
                            rent=format_vnd(room.get("rent_price")),
                            district=room.get("district") or UNKNOWN_DISTRICT,
                        )
                    )
                alt_lines.append(SEARCH_ALTERNATIVE_CTA)
                return "\n".join(alt_lines)
            return _search_no_result_message(
                user_mood,
                constraints,
                sales_handoff=bool(
                    tool_results.get("public_inventory_miss")
                    or tool_results.get("budget_or_district_miss")
                ),
            )
        if tool_results.get("relaxed_search"):
            lines = [
                SEARCH_RELAXED_OPENING.format(
                    relaxed_note=_relaxed_note(tool_results.get("relaxed_fields") or []),
                )
            ]
        else:
            lines = [SEARCH_SUCCESS_OPENING]
        landmark_hints = _matching_landmark_hints(rooms, grounding.get("constraints", {}))
        for idx, room in enumerate(rooms[:5], 1):
            landmark_suffix = (
                LANDMARK_HINT_SUFFIX.format(hint=landmark_hints.get(room.get("room_id")))
                if room.get("room_id") in landmark_hints
                else ""
            )
            lines.append(
                SEARCH_ROOM_LINE.format(
                    index=idx,
                    title=room.get("title"),
                    room_id=room.get("room_id"),
                    rent=format_vnd(room.get("rent_price")),
                    district=room.get("district") or UNKNOWN_DISTRICT,
                    landmark_suffix=landmark_suffix,
                )
            )
        lines.append(SEARCH_SUCCESS_CTA)
        return "\n".join(lines)
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        if not rooms:
            return ASK_ROOM_MISSING_ID
        room = rooms[0]
        insufficient = _verified_data_insufficient_message(
            question,
            room,
            intent=intent,
            constraints=constraints,
        )
        if insufficient:
            return insufficient
        unknown = _unknown_fields(room)
        parts = [
            ASK_ROOM_HEADER.format(
                title=room.get("title"),
                room_id=room.get("room_id"),
                rent=format_vnd(room.get("rent_price")),
            ),
            ASK_ROOM_LOCATION.format(
                location=room.get("address") or room.get("district") or UNKNOWN_LOCATION,
            ),
        ]
        feature_facts = _room_feature_facts(room)
        if feature_facts:
            parts.append(ASK_ROOM_AMENITIES.format(features=", ".join(feature_facts)))
        if unknown:
            parts.append(ASK_ROOM_UNKNOWN_FIELDS.format(fields=", ".join(unknown)))
        faq = tool_results.get("faq_results") or []
        if faq:
            parts.append(faq[0].get("answer", ""))
        return "\n".join(parts)
    if intent == "CALCULATE_COST":
        estimate = tool_results.get("cost_estimate") or {}
        if not estimate.get("available"):
            return CALCULATE_COST_MISSING_INFO
        lines = [CALCULATE_COST_OPENING]
        fixed_items = estimate.get("fixed_items") or []
        if fixed_items:
            for item in fixed_items:
                amount = item.get("amount")
                if amount is None or amount == 0:
                    continue
                label = COST_FIXED_FIELD_LABELS.get(str(item.get("field")), _cost_item_label(f"fee_{item.get('field')}"))
                if item.get("field") == "monthly_rent":
                    label = COST_FIXED_FIELD_LABELS["monthly_rent"]
                lines.append(CALCULATE_COST_LINE.format(label=label, amount=format_vnd(amount)))
        initial_options = estimate.get("initial_payment_options") or []
        if initial_options and initial_options[0].get("deposit") is not None:
            lines.append(CALCULATE_COST_DEPOSIT.format(amount=format_vnd(initial_options[0].get("deposit"))))
        if estimate.get("rental_months"):
            lines.append(CALCULATE_COST_RENTAL_PERIOD.format(months=estimate["rental_months"]))
            for item in estimate.get("period_items", []):
                lines.append(
                    CALCULATE_COST_LINE.format(
                        label=_cost_item_label(item["name"]),
                        amount=format_vnd(item.get("amount")),
                    )
                )
            if estimate.get("recurring_fees_for_period"):
                lines.append(
                    CALCULATE_COST_RECURRING_FEES.format(
                        months=estimate["rental_months"],
                        amount=format_vnd(estimate.get("recurring_fees_for_period")),
                    )
                )
            lines.append(
                CALCULATE_COST_TOTAL_PERIOD.format(
                    months=estimate["rental_months"],
                    amount=format_vnd(estimate.get("total_period_cost")),
                )
            )
        else:
            for item in estimate.get("items", []):
                if item.get("amount") == 0:
                    continue
                lines.append(
                    CALCULATE_COST_LINE.format(
                        label=_cost_item_label(item["name"]),
                        amount=format_vnd(item.get("amount")),
                    )
                )
            lines.append(CALCULATE_COST_TOTAL_INITIAL.format(amount=format_vnd(estimate.get("total_initial_cost"))))
        if estimate.get("unknown"):
            lines.append(CALCULATE_COST_UNKNOWN.format(fields=", ".join(estimate["unknown"])))
        if estimate.get("not_calculated"):
            details = [
                f"{_cost_item_label('fee_' + str(item.get('name', '')).removeprefix('fees.'))}: {item.get('value')}"
                for item in estimate["not_calculated"]
            ]
            lines.append(CALCULATE_COST_NOT_CALCULATED.format(details=", ".join(details)))
        return "\n".join(lines)
    if intent == "COMPARE_ROOMS":
        comparison = tool_results.get("comparison") or {}
        if comparison.get("compare_unresolved"):
            return COMPARE_ORDINAL_UNRESOLVED
        rows = comparison.get("rows", [])
        if not rows:
            missing = comparison.get("missing_room_ids") or []
            if missing:
                return COMPARE_MISSING_ROOMS.format(
                    room_ids=", ".join("#" + item for item in missing),
                )
            return COMPARE_NEED_ROOM_IDS
        lines = [COMPARE_OPENING]
        for row in rows:
            lines.append(
                COMPARE_ROW.format(
                    title=row.get("title") or ("#" + str(row.get("room_id"))),
                    room_id=row.get("room_id"),
                    rent=format_vnd(row.get("rent_price")),
                    area=row.get("area_m2") or COMPARE_UNKNOWN_AREA,
                    district=row.get("district") or UNKNOWN_DISTRICT,
                    status=row.get("status_desc") or (
                        COMPARE_STATUS_AVAILABLE if row.get("available") else COMPARE_STATUS_UNKNOWN
                    ),
                )
            )
        for insight in _comparison_insights(rows):
            lines.append(insight)
        best = _best_room_from_comparison(rows, grounding.get("constraints", {}))
        if best:
            area_suffix = (
                COMPARE_BEST_PICK_AREA_SUFFIX.format(area=best.get("area_m2"))
                if best.get("area_m2")
                else ""
            )
            lines.append(
                COMPARE_BEST_PICK.format(
                    title=best.get("title") or ("#" + str(best.get("room_id"))),
                    room_id=best.get("room_id"),
                    rent=format_vnd(best.get("rent_price")),
                    area_suffix=area_suffix,
                )
            )
        missing = comparison.get("missing_room_ids") or []
        if missing:
            lines.append(COMPARE_MISSING_DATA.format(room_ids=", ".join("#" + item for item in missing)))
        not_compared = comparison.get("not_compared_room_ids") or []
        if not_compared:
            lines.append(COMPARE_NOT_COMPARED.format(room_ids=", ".join("#" + item for item in not_compared)))
        return "\n".join(lines)
    if intent == "REQUEST_FAQ":
        faq = tool_results.get("faq_results") or []
        if faq:
            answers = [item.get("answer", "").strip() for item in faq if item.get("answer")]
            body = "\n\n".join(answers[:2])
            try:
                from room_assistant.staff_knowledge import staff_cta_line
                return f"{body}\n\n{staff_cta_line()}"
            except Exception:
                return body
        return REQUEST_FAQ_FALLBACK
    if intent == "GENERAL_HELP" and _is_price_objection(question):
        return GENERAL_HELP_PRICE_OBJECTION
    if intent == "GENERAL_HELP" and _is_off_topic_question(question):
        return GENERAL_HELP_OFF_TOPIC
    return GENERAL_HELP_DEFAULT


async def _compose_answer_async(
    question: str, parsed: dict[str, Any], grounding: dict[str, Any],
    tool_results: dict[str, Any], history: list[dict[str, Any]], user_mood: str = "normal",
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    intent = parsed["intent"]
    rooms = grounding.get("rooms", [])
    verification: dict[str, Any] = {"reviewed": False, "approved": True}

    def _template_answer() -> str:
        return _compose_answer_template(parsed, grounding, tool_results, question=question, user_mood=user_mood)

    if parsed.get("ordinal_out_of_range") or parsed.get("compare_unresolved"):
        answer = _template_answer()
        return await _finalize_composed_answer_async(answer, False, "", verification, stream_callback)

    if intent in {
        "REQUEST_ACTION",
        "CALCULATE_COST",
        "COMPARE_ROOMS",
        "REQUEST_FAQ",
        "GENERAL_HELP",
    } or tool_results.get("error"):
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return await _finalize_composed_answer_async(answer, abstain, reason, verification, stream_callback)

    if _is_off_topic_question(question):
        answer = _compose_answer_template(
            {"intent": "GENERAL_HELP"},
            grounding,
            tool_results,
            question=question,
            user_mood=user_mood,
        )
        return await _finalize_composed_answer_async(answer, False, "", verification, stream_callback)

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"} and rooms:
        insufficient = _verified_data_insufficient_message(
            question,
            rooms[0],
            intent=intent,
            constraints=grounding.get("constraints"),
        )
        if insufficient:
            return await _finalize_composed_answer_async(insufficient, False, "", verification, stream_callback)

    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"} and rooms:
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return await _finalize_composed_answer_async(answer, abstain, reason, verification, stream_callback)

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"} and _asks_about_amenities(question):
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return await _finalize_composed_answer_async(answer, abstain, reason, verification, stream_callback)

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"} and _asks_about_price(question) and rooms:
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return await _finalize_composed_answer_async(answer, abstain, reason, verification, stream_callback)

    faq = tool_results.get("faq_results")
    if not rooms and not faq and intent not in {"GENERAL_HELP", "REQUEST_FAQ"}:
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return await _finalize_composed_answer_async(answer, abstain, reason, verification, stream_callback)

    verified_data = _build_llm_context(grounding, tool_results)
    sales_handoff = bool(
        tool_results.get("public_inventory_miss")
        or tool_results.get("budget_or_district_miss")
    )
    answer = ""
    try:
        from agents.response_writer import write_response, write_no_result_response
        if not rooms and intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
            constraints = grounding.get("constraints", {})
            alt_rooms = tool_results.get("alternative_rooms", [])
            answer = await write_no_result_response(
                question,
                constraints,
                user_mood,
                alt_rooms,
                stream_callback=None,
                sales_handoff=sales_handoff and not alt_rooms,
            )
        else:
            answer = await write_response(
                question=question,
                verified_context=verified_data,
                history=history,
                mood=user_mood,
                stream_callback=None,
            )
    except Exception:
        pass
    if not answer or len(answer.strip()) < 20:
        answer = _template_answer()

    try:
        from config import ENABLE_REVIEWER
        if ENABLE_REVIEWER:
            from agents.reviewer import review_with_retry
            answer, review_result = await review_with_retry(
                question=question,
                answer=answer,
                room_context=verified_data,
                max_retries=1,
            )
            verification = {
                "reviewed": True,
                "approved": bool(review_result.get("is_approved", True)),
                "issues": list(review_result.get("issues") or []),
                "corrected_answer_used": bool(review_result.get("used_corrected_answer")),
            }
            if not verification["approved"]:
                abstain, reason = True, "reviewer_rejected"
                return await _finalize_composed_answer_async(answer, abstain, reason, verification, stream_callback)
    except Exception:
        pass

    abstain, reason = _evaluate_abstain(question, intent, grounding, tool_results, answer)
    if abstain and reason == "unverified_claims":
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
    if abstain and reason == "insufficient_verified_data" and rooms:
        specific = _verified_data_insufficient_message(
            question,
            rooms[0],
            intent=intent,
            constraints=grounding.get("constraints"),
        )
        if specific:
            return await _finalize_composed_answer_async(specific, False, "", verification, stream_callback)
    return await _finalize_composed_answer_async(answer, abstain, reason, verification, stream_callback)


def _evaluate_abstain(
    question: str,
    intent: str,
    grounding: dict[str, Any],
    tool_results: dict[str, Any],
    answer: str,
    *,
    from_template: bool = False,
) -> tuple[bool, str]:
    try:
        from agents.reviewer import should_abstain
        abstain, reason = should_abstain(question, intent, grounding, tool_results, answer)
        if from_template and reason == "unverified_claims":
            return False, ""
        return abstain, reason
    except Exception:
        return False, ""


async def _maybe_stream_answer(
    answer: str,
    stream_callback: Callable[[str], Awaitable[None]] | None,
) -> None:
    if stream_callback and answer and not answer.startswith("[status:"):
        await stream_callback(answer)


def _finalize_composed_answer(
    answer: str,
    abstain: bool,
    abstain_reason: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    if abstain:
        from agents.reviewer import ABSTAIN_USER_MESSAGE
        answer = ABSTAIN_USER_MESSAGE
    return {
        "answer": answer.strip(),
        "abstain": abstain,
        "abstain_reason": abstain_reason or None,
        "verification": verification,
    }


async def _finalize_composed_answer_async(
    answer: str,
    abstain: bool,
    abstain_reason: str,
    verification: dict[str, Any],
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    result = _finalize_composed_answer(answer, abstain, abstain_reason, verification)
    await _maybe_stream_answer(result["answer"], stream_callback)
    return result


def _extract_rooms(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = tool_results.get("rooms") or []
    return [item for item in rooms if item]


def _current_room_from_results(intent: str, rooms: list[dict[str, Any]]) -> str | None:
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"} and rooms:
        return rooms[0].get("room_id")
    return None


def _unknown_fields(room: dict[str, Any]) -> list[str]:
    return unknown_room_fields(room)


def _verified_data_insufficient_message(
    question: str,
    room: dict[str, Any],
    *,
    intent: str = "",
    constraints: dict[str, Any] | None = None,
) -> str | None:
    from room_assistant.tools import (
        SufficiencyStatus,
        evaluate_room_data_sufficiency,
        format_insufficient_field_labels,
    )

    status, missing = evaluate_room_data_sufficiency(
        question,
        room,
        intent=intent,
        constraints=constraints,
    )
    if status != SufficiencyStatus.INSUFFICIENT or not missing:
        return None
    return INSUFFICIENT_VERIFIED_DATA.format(fields=format_insufficient_field_labels(missing))


def _verified_amenity_labels(room: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
    room_amenities = {str(item).strip().lower() for item in (room.get("amenities") or [])}
    required = [str(item).strip().lower() for item in (constraints.get("amenities_required") or [])]
    preferred = [str(item).strip().lower() for item in (constraints.get("amenities_preferred") or [])]

    labels: list[str] = []
    for amenity in required + preferred:
        if amenity in room_amenities or _room_text_has_amenity(room, amenity):
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


def _room_feature_facts(room: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    text = str(room.get("embedding_text") or "")
    for label in FEATURE_FACT_LABELS:
        value = _extract_feature_status(text, label)
        if value:
            facts.append(f"{label}: {value}")
    if facts:
        return facts
    return [
        AMENITY_LABELS.get(str(item), str(item))
        for item in (room.get("amenities") or [])
        if item
    ]


def _extract_feature_status(text: str, label: str) -> str | None:
    if not text:
        return None
    import re
    match = re.search(rf"(?im)^\s*-\s*{re.escape(label)}\s*:\s*([^\n\r]+)", text)
    return match.group(1).strip() if match else None


def _room_text_has_amenity(room: dict[str, Any], amenity: str) -> bool:
    label = AMENITY_LABELS.get(amenity)
    if not label:
        return False
    text = str(room.get("embedding_text") or room.get("description") or "")
    if not text:
        return False
    import re
    return bool(re.search(rf"{re.escape(label)}\s*:\s*(?:Có|Riêng|Tự do|True|Yes|Free)", text, re.IGNORECASE))


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


def _asks_about_price(question: str) -> bool:
    from .tools import question_asks_price

    return question_asks_price(question)


def _asks_to_compare_result_set(question: str) -> bool:
    import re
    import unicodedata

    text = unicodedata.normalize("NFD", question.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")
    if not re.search(r"\bso\s*sanh\b", text):
        return False
    return bool(
        re.search(r"\b(?:3|ba)\s*phong\b", text)
        or re.search(r"\b(?:2|hai)\s*phong\b", text)
        or re.search(r"\b(?:cac|nhung|may)\s+phong\b", text)
        or re.search(r"\bphong\s+(?:nay|tren|vua|dau tien)\b", text)
    )


def _requested_result_set_count(question: str, available_count: int) -> int:
    import re
    import unicodedata

    text = unicodedata.normalize("NFD", question.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")
    if match := re.search(r"\b(\d{1,2})\s*phong\b", text):
        count = int(match.group(1))
        return max(1, min(count, available_count, 3))
    word_map = {
        "mot": 1,
        "một": 1,
        "hai": 2,
        "ba": 3,
    }
    if match := re.search(r"\b(mot|một|hai|ba)\s*phong\b", text):
        count = word_map.get(match.group(1), available_count)
        return max(1, min(count, available_count, 3))
    if "dau tien" in text or "đầu tiên" in question.lower():
        return max(1, min(available_count, 3))
    return max(1, min(available_count, 3))


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


def _best_room_from_comparison(rows: list[dict[str, Any]], constraints: dict[str, Any]) -> dict[str, Any] | None:
    if not rows:
        return None
    budget = constraints.get("budget") or {}
    max_price = budget.get("max")
    min_price = budget.get("min")

    def score(row: dict[str, Any]) -> tuple[int, float, float]:
        rent = row.get("rent_price")
        area = row.get("area_m2") or 0
        in_budget = 1
        if rent is not None:
            if max_price is not None and rent > max_price:
                in_budget = 0
            if min_price is not None and rent < min_price:
                in_budget = 0
        cheaper = -(float(rent) if rent is not None else float("inf"))
        return (in_budget, float(area), cheaper)

    return max(rows, key=score)


def _comparison_insights(rows: list[dict[str, Any]]) -> list[str]:
    if len(rows) < 2:
        return []

    insights: list[str] = []
    first, second = rows[0], rows[1]

    first_title = first.get("title") or f"#{first.get('room_id')}"
    second_title = second.get("title") or f"#{second.get('room_id')}"

    first_rent = first.get("rent_price")
    second_rent = second.get("rent_price")
    if isinstance(first_rent, (int, float)) and isinstance(second_rent, (int, float)) and first_rent != second_rent:
        cheaper, pricier = (first, second) if first_rent < second_rent else (second, first)
        diff = abs(int(first_rent) - int(second_rent))
        insights.append(
            COMPARE_INSIGHT_CHEAPER.format(
                cheaper_title=cheaper.get("title") or ("#" + str(cheaper.get("room_id"))),
                pricier_title=pricier.get("title") or ("#" + str(pricier.get("room_id"))),
                diff=format_vnd(diff),
            )
        )

    first_area = first.get("area_m2")
    second_area = second.get("area_m2")
    if isinstance(first_area, (int, float)) and isinstance(second_area, (int, float)) and first_area != second_area:
        larger, smaller = (first, second) if first_area > second_area else (second, first)
        diff_area = abs(float(first_area) - float(second_area))
        diff_text = int(diff_area) if diff_area.is_integer() else diff_area
        insights.append(
            COMPARE_INSIGHT_LARGER.format(
                larger_title=larger.get("title") or ("#" + str(larger.get("room_id"))),
                smaller_title=smaller.get("title") or ("#" + str(smaller.get("room_id"))),
                diff=diff_text,
            )
        )

    first_features = set(_comparison_feature_labels(first))
    second_features = set(_comparison_feature_labels(second))
    first_only = sorted(first_features - second_features)
    second_only = sorted(second_features - first_features)
    if first_only:
        insights.append(
            COMPARE_INSIGHT_FIRST_ADVANTAGE.format(
                title=first_title,
                features=", ".join(first_only[:5]),
            )
        )
    if second_only:
        insights.append(
            COMPARE_INSIGHT_SECOND_ADVANTAGE.format(
                title=second_title,
                features=", ".join(second_only[:5]),
            )
        )

    if not insights:
        insights.append(COMPARE_INSIGHT_TIE)
    return insights


def _comparison_feature_labels(room: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for fact in _room_feature_facts(room):
        if ":" not in fact:
            continue
        name, value = fact.split(":", 1)
        value_norm = str(value).strip().lower()
        if value_norm in {"có", "free", "riêng", "tự do"}:
            label = name.strip()
            if label not in labels:
                labels.append(label)

    for amenity in room.get("amenities") or []:
        label = AMENITY_LABELS.get(str(amenity).strip().lower(), str(amenity).strip())
        if label and label not in labels:
            labels.append(label)
    return labels


def _matching_landmark_hints(rooms: list[dict[str, Any]], constraints: dict[str, Any]) -> dict[str, str]:
    location = constraints.get("location") or {}
    landmarks = [str(item).strip() for item in (location.get("near_landmarks") or []) if item]
    if not landmarks:
        return {}

    hints: dict[str, str] = {}
    for room in rooms:
        room_id = str(room.get("room_id") or "").strip()
        if not room_id:
            continue
        text = " ".join(
            str(part or "")
            for part in (
                room.get("embedding_text"),
                room.get("tien_ich_xq"),
                room.get("address"),
                room.get("title"),
            )
        ).lower()
        matched = []
        for landmark in landmarks:
            for token in expand_landmark_tokens(str(landmark)):
                needle = str(token).strip().lower()
                if needle and needle in text:
                    label = needle.upper() if len(needle) <= 6 else needle.title()
                    if label not in matched:
                        matched.append(label)
                    break
        if not matched:
            display = primary_display_token(str(landmarks[0]))
            if display.lower() in text:
                matched.append(display)
        if matched:
            hints[room_id] = LANDMARK_NEAR_HINT.format(landmarks=" / ".join(matched))
    return hints


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


def _is_price_objection(question: str) -> bool:
    import unicodedata

    text = unicodedata.normalize("NFD", str(question or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")
    return any(
        phrase in text
        for phrase in (
            "gia cao",
            "dat qua",
            "mac qua",
            "qua tam tien",
            "thue noi",
            "thue khong noi",
            "sinh vien",
        )
    )


def _is_off_topic_question(question: str) -> bool:
    import unicodedata

    text = unicodedata.normalize("NFD", str(question or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")
    off_topic_phrases = (
        "giai bai",
        "giai bai tap",
        "lam bai tap",
        "lam ho bai",
        "viet code",
        "code giup",
        "lap trinh giup",
        "debug code",
        "fix bug",
        "viet bai van",
        "viet essay",
        "dich doan van",
        "dich bai",
        "lam slide",
        "lam powerpoint",
        "lam cv",
        "viet cv",
        "viet email",
        "tom tat tai lieu",
        "giai toan",
        "giai ly",
        "giai hoa",
        "lam de thi",
        "xem boi",
        "coi tarot",
        "tu van tinh cam",
    )
    domain_phrases = (
        "phong",
        "nha tro",
        "phong tro",
        "can ho",
        "chdv",
        "thue",
        "gia phong",
        "tien coc",
        "tien ich",
        "quan",
        "phuong",
        "dia chi",
        "xem phong",
        "dat lich",
        "chu nha",
        "hop dong",
    )
    return any(phrase in text for phrase in off_topic_phrases) and not any(phrase in text for phrase in domain_phrases)


def _has_soft_preferences(constraints: dict[str, Any]) -> bool:
    return bool(constraints.get("amenities_preferred"))


def _relax_soft_filters(constraints: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Bỏ các điều kiện "mềm"/mờ: mốc gần, phường, tiện ích ưu tiên.

    Đây là các bộ lọc hay khiến tìm kiếm trả về rỗng vì khớp chuỗi quá chặt
    (tên đường, viết tắt địa danh) dù khu vực vẫn còn phòng.
    """
    relaxed = dict(constraints)
    dropped: list[str] = []
    location = dict(relaxed.get("location") or {})
    if location.get("near_landmarks"):
        location["near_landmarks"] = []
        dropped.append("near_landmarks")
    if location.get("wards"):
        location["wards"] = []
        dropped.append("wards")
    relaxed["location"] = location
    if relaxed.get("amenities_preferred"):
        relaxed["amenities_preferred"] = []
        dropped.append("amenities_preferred")
    return relaxed, dropped


def _relax_hard_filters(constraints: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Nới tiện ích bắt buộc / loại trừ. Giữ ngân sách và quận để không gợi ý phòng lệch giá."""
    relaxed = dict(constraints)
    dropped: list[str] = []
    if relaxed.get("amenities_required"):
        relaxed["amenities_required"] = []
        dropped.append("amenities_required")
    if relaxed.get("excluded_features"):
        relaxed["excluded_features"] = []
        dropped.append("excluded_features")
    return relaxed, dropped


def _room_matches_budget(room: dict[str, Any], constraints: dict[str, Any]) -> bool:
    budget = constraints.get("budget") or {}
    rent = room.get("rent_price")
    if rent is None:
        return True
    max_price = budget.get("max")
    min_price = budget.get("min")
    if max_price is not None:
        if budget.get("max_operator") == "lt":
            if rent >= max_price:
                return False
        elif rent > max_price:
            return False
    if min_price is not None:
        if budget.get("min_operator") == "gt":
            if rent <= min_price:
                return False
        elif rent < min_price:
            return False
    return True


def _filter_rooms_by_budget(
    rooms: list[dict[str, Any]] | None,
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    if not rooms:
        return []
    return [room for room in rooms if _room_matches_budget(room, constraints)]


def _relaxed_note(dropped: list[str]) -> str:
    labels = [RELAX_FIELD_LABELS[item] for item in dropped if item in RELAX_FIELD_LABELS]
    if not labels:
        return RELAXED_NOTE_DEFAULT
    return RELAXED_NOTE_WITH_FIELDS.format(fields=", ".join(labels))


def _suggest_questions(intent: str, rooms: list[dict[str, Any]], current_room_id: str | None) -> list[str]:
    if rooms:
        first = current_room_id or rooms[0].get("room_id")
        return [item.format(room_id=first) for item in SUGGESTED_QUESTIONS_WITH_ROOM]
    if intent == "GENERAL_HELP":
        return list(SUGGESTED_QUESTIONS_GENERAL_HELP)
    return list(SUGGESTED_QUESTIONS_NO_RESULT)


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
        from langfuse.decorators import langfuse_context
        langfuse_context.update_current_observation(**kwargs)
        if metadata and "session_hash" in metadata:
            langfuse_context.update_current_trace(session_id=metadata["session_hash"])
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
    }
    try:
        _logger.info("room_assistant_turn %s", json.dumps(record, ensure_ascii=False, sort_keys=True))
    except Exception:
        _logger.info("room_assistant_turn intent=%s error=%s", record["intent"], record["error_category"])


def format_vnd(value: Any) -> str:
    from .money import format_vnd as _format_vnd

    return _format_vnd(value)


def _cost_item_label(name: str) -> str:
    if name.startswith("rent_") and name.endswith("_months"):
        parts = name.split("_")
        if len(parts) >= 2:
            return f"Tiền thuê {parts[1]} tháng"
    if name in COST_ITEM_LABELS:
        return COST_ITEM_LABELS[name]
    if name.startswith("fee_"):
        return "Phí " + name.removeprefix("fee_").replace("_", " ")
    return name
