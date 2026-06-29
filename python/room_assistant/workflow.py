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
    current_room_id = (
        parsed.get("current_room_id")
        or state.get("current_room_id")
        or _first_or_none(state.get("last_result_ids", []))
    )

    # 1. Direct Fast Path (disambiguate room_id vs room_code)
    # Check if exact room reference is found
    exact_ref = parsed.get("exact_room_reference")
    if exact_ref and getattr(exact_ref, "room_id", None):
        current_room_id = exact_ref.room_id
        detail = _tool_registry.execute(
            "retrieve_room_context", {"room_id": current_room_id}, context,
        )
        return {"room_context": detail, "rooms": [detail["room"]] if detail.get("room") else [], "fast_path": True}
    elif exact_ref and getattr(exact_ref, "room_code", None):
        # Ambiguous room_code -> need simple search for exact room
        constraints["room_code"] = exact_ref.room_code
        intent = "SEARCH_ROOM"

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
            return {"rooms": [], "budget_or_district_miss": True}
        return {"rooms": []}

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
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
        parts.append("Danh sách phòng phù hợp:")
        for room in rooms[:5]:
            rent = format_vnd(room.get("rent_price"))
            details = [
                f"[{room.get('room_id')}] {room.get('title')}",
                f"{rent}/tháng",
                room.get("district") or "chưa rõ khu vực",
                f"Diện tích: {room.get('area_m2') or '?'} m²",
            ]
            amenities = _verified_amenity_labels(room, constraints)
            if amenities:
                details.append(f"Tiện ích xác minh: {', '.join(amenities)}")
            feature_facts = _room_feature_facts(room)
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


def _search_no_result_message(
    user_mood: str = "normal",
    constraints: dict[str, Any] | None = None,
) -> str:
    budget = (constraints or {}).get("budget") or {}
    districts = ((constraints or {}).get("location") or {}).get("districts") or []
    max_price = budget.get("max")
    district_label = districts[0] if districts else ""

    if max_price and district_label:
        area = district_label.replace("_", " ")
        budget_line = (
            f"Dạ em tìm trong **{area}** với ngân sách **{format_vnd(max_price)}/tháng** "
            f"mà chưa thấy căn trống khớp ạ. "
            f"Anh/chị thử nới thêm khoảng 500k–1 triệu hoặc xem khu lân cận, em lọc lại ngay nha."
        )
        if user_mood == "urgent":
            return f"Dạ em hiểu mình cần gấp ạ. {budget_line}"
        if user_mood == "frustrated":
            return f"Dạ em hiểu mình tìm mãi cũng mệt ạ. {budget_line}"
        return budget_line

    if max_price:
        budget_only = (
            f"Dạ em chưa thấy căn nào trong tầm **{format_vnd(max_price)}/tháng** ạ. "
            "Anh/chị cho em biết khu vực ưu tiên hoặc nới ngân sách thêm chút, em lọc lại liền nha."
        )
        if user_mood in {"urgent", "frustrated"}:
            return f"Dạ em hiểu mà ạ. {budget_only}"
        return budget_only

    if user_mood == "urgent":
        return (
            "Dạ em hiểu mình đang cần gấp ạ. Em chưa thấy căn khớp 100% ngay, "
            "nhưng nếu mình nới ngân sách một chút hoặc mở rộng khu vực, em lọc lại liền "
            "để tìm phòng còn trống sớm nhất cho mình nha."
        )
    if user_mood == "frustrated":
        return (
            "Dạ em hiểu mình tìm mãi cũng hơi mệt ạ. Em chưa thấy căn khớp trọn điều kiện, "
            "nhưng mình thử nới ngân sách hoặc bỏ bớt 1–2 tiêu chí, em lọc lại ngay — "
            "chắc chắn sẽ có thêm lựa chọn phù hợp hơn ạ."
        )
    return (
        "Dạ em tìm mỏi mắt mà chưa thấy phòng nào khớp 100% điều kiện của mình ạ. "
        "Anh/chị thử nới ngân sách hoặc mở rộng khu vực giúp em nhé, đảm bảo sẽ có nhiều căn đẹp lắm ạ!"
    )


def _search_alternative_opening(user_mood: str = "normal") -> str:
    if user_mood == "frustrated":
        return (
            "Dạ em hiểu điều kiện hơi khó nên mình hơi mệt khi chưa thấy căn ưng ý ạ. "
            "Em gợi ý mấy căn gần đúng nhất để mình tham khảo nha:"
        )
    if user_mood == "urgent":
        return (
            "Dạ em hiểu mình cần gấp ạ. Chưa có căn khớp 100% nhưng em tìm được vài căn "
            "gần đúng nhất để mình xem trước nha:"
        )
    return (
        "Dạ điều kiện hiện tại hơi khó nên em chưa thấy căn khớp 100% ạ. "
        "Em gợi ý mấy căn gần đúng nhất để mình tham khảo nha:"
    )


def _compose_answer_template(
    parsed: dict[str, Any],
    grounding: dict[str, Any],
    tool_results: dict[str, Any],
    *,
    question: str = "",
    user_mood: str = "normal",
) -> str:
    intent = parsed["intent"]
    if tool_results.get("error") == "tool_budget_exceeded":
        return "Mình cần giới hạn số lần đọc dữ liệu trong một lượt. Bạn thử hỏi lại hẹp hơn với tối đa 3 phòng hoặc một nhu cầu cụ thể nhé."
    if intent == "REQUEST_ACTION":
        action = parsed.get("requested_action") or "thao tác nghiệp vụ"
        return (
            "Dạ tính năng thao tác tự động em chưa được học ạ. "
            f"Với yêu cầu '{action}', anh/chị thao tác trực tiếp trên giao diện giúp em nha! "
            "Nhưng nếu ưng phòng rồi, chiều nay ghé xem thực tế luôn cho tiện anh/chị nhỉ?"
        )
    rooms = grounding["rooms"]
    constraints = grounding.get("constraints", {})
    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
        if not rooms:
            alternative_rooms = [item for item in (tool_results.get("alternative_rooms") or []) if item]
            if alternative_rooms:
                alt_lines = [_search_alternative_opening(user_mood)]
                for idx, room in enumerate(alternative_rooms[:5], 1):
                    alt_lines.append(
                        f"{idx}. **{room.get('title')}** (#{room.get('room_id')}) — "
                        f"chỉ {format_vnd(room.get('rent_price'))}/tháng, "
                        f"{room.get('district') or 'chưa rõ khu vực'}."
                    )
                alt_lines.append("\nNếu mình nới ngân sách hoặc bỏ bớt 1–2 tiêu chí, em sẽ tìm được nhiều căn ưng hơn ạ 😊")
                return "\n".join(alt_lines)
            return _search_no_result_message(user_mood, constraints)
        if tool_results.get("relaxed_search"):
            lines = [f"Dạ {_relaxed_note(tool_results.get('relaxed_fields') or [])} Mấy căn cùng khu vực vẫn ngon mà hợp lý nè:"]
        else:
            lines = ["Dạ còn phòng ạ! Em vừa lọc ra mấy căn sạch đẹp, giá cực tốt cho mình đây:"]
        landmark_hints = _matching_landmark_hints(rooms, grounding.get("constraints", {}))
        for idx, room in enumerate(rooms[:5], 1):
            landmark_suffix = f", {landmark_hints.get(room.get('room_id'))}" if room.get("room_id") in landmark_hints else ""
            lines.append(
                f"{idx}. **{room.get('title')}** (#{room.get('room_id')}) — "
                f"chỉ {format_vnd(room.get('rent_price'))}/tháng, "
                f"{room.get('district') or 'chưa rõ khu vực'}{landmark_suffix}."
            )
        lines.append("\nAnh/chị ưng căn nào chưa ạ? Nếu rảnh thì sắp xếp ghé qua xem thực tế nha, phòng bên ngoài đẹp hơn ảnh nhiều ạ 😊")
        return "\n".join(lines)
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"}:
        if not rooms:
            return "Dạ em chưa rõ anh/chị đang quan tâm căn nào. Anh/chị gửi mã phòng cho em nha!"
        room = rooms[0]
        unknown = _unknown_fields(room)
        parts = [
            f"**{room.get('title')}** (#{room.get('room_id')}) — giá {format_vnd(room.get('rent_price'))}/tháng.",
            f"Khu vực: {room.get('address') or room.get('district') or 'chưa rõ'}.",
        ]
        feature_facts = _room_feature_facts(room)
        if feature_facts:
            parts.append(f"Dạ tiện ích có đủ: {', '.join(feature_facts)}. Mình dọn vào là ở thoải mái luôn ạ.")
        if unknown:
            parts.append(f"_Dữ liệu chưa xác nhận: {', '.join(unknown)}._")
        faq = tool_results.get("faq_results") or []
        if faq:
            parts.append(faq[0].get("answer", ""))
        return "\n".join(parts)
    if intent == "CALCULATE_COST":
        estimate = tool_results.get("cost_estimate") or {}
        if not estimate.get("available"):
            return "Dạ em chưa đủ thông tin tính chi phí căn này. Anh/chị cho em xin mã phòng nhé!"
        lines = ["Dạ em tính sương sương chi phí dự kiến cho anh/chị nhé:"]
        fixed_items = estimate.get("fixed_items") or []
        if fixed_items:
            field_labels = {
                "monthly_rent": "Tiền thuê mỗi tháng",
                "parking": "Phí gửi xe",
                "management": "Phí quản lý",
                "water": "Tiền nước",
                "wifi": "Wifi",
                "washing_machine": "Máy giặt",
            }
            for item in fixed_items:
                amount = item.get("amount")
                if amount is None or amount == 0:
                    continue
                label = field_labels.get(str(item.get("field")), _cost_item_label(f"fee_{item.get('field')}"))
                if item.get("field") == "monthly_rent":
                    label = "Tiền thuê mỗi tháng"
                lines.append(f"- {label}: {format_vnd(amount)}")
        initial_options = estimate.get("initial_payment_options") or []
        if initial_options and initial_options[0].get("deposit") is not None:
            lines.append(f"- Tiền cọc: {format_vnd(initial_options[0].get('deposit'))}")
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
        if estimate.get("not_calculated"):
            details = [
                f"{_cost_item_label('fee_' + str(item.get('name', '')).removeprefix('fees.'))}: {item.get('value')}"
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
                return f"Dạ em chưa tìm thấy dữ liệu phòng: {', '.join('#' + item for item in missing)} ạ. Anh/chị kiểm tra lại mã giúp em nha."
            return "Dạ để em so sánh chuẩn xác, anh/chị gửi giúp em tối đa 3 mã phòng nha (ví dụ: `so sánh #A #B`)."
        lines = ["Dạ em gửi anh/chị bảng so sánh chi tiết:"]
        for row in rows:
            lines.append(
                f"- **{row.get('title') or ('#' + str(row.get('room_id')))}** "
                f"(#{row.get('room_id')}): {format_vnd(row.get('rent_price'))}/tháng, "
                f"{row.get('area_m2') or 'chưa rõ'} m², "
                f"{row.get('district') or 'chưa rõ khu vực'}, "
                f"{row.get('status_desc') or ('Còn phòng' if row.get('available') else 'chưa rõ trạng thái')}."
            )
        for insight in _comparison_insights(rows):
            lines.append(insight)
        best = _best_room_from_comparison(rows, grounding.get("constraints", {}))
        if best:
            area = f", diện tích {best.get('area_m2')} m²" if best.get("area_m2") else ""
            lines.append(
                f"\n✨ **Gợi ý cực hợp lý:** Căn **{best.get('title') or ('#' + str(best.get('room_id')))}** "
                f"(#{best.get('room_id')}) với giá {format_vnd(best.get('rent_price'))}/tháng{area}."
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
            answers = [item.get("answer", "").strip() for item in faq if item.get("answer")]
            body = "\n\n".join(answers[:2])
            try:
                from room_assistant.staff_knowledge import staff_cta_line
                return f"{body}\n\n{staff_cta_line()}"
            except Exception:
                return body
        return "Dạ anh/chị cần hỏi thêm về quy trình thuê, hợp đồng hay tiền cọc không ạ? Anh/chị cứ nhắn, em tư vấn kỹ cho nha."
    if intent == "GENERAL_HELP" and _is_price_objection(question):
        return (
            "Dạ em hiểu mà ạ, tầm giá này với sinh viên thì mình phải cân lên đặt xuống dữ lắm. "
            "Nếu mình ưu tiên tiết kiệm, em có thể lọc giúp các căn mềm hơn một chút hoặc tìm khu vực lân cận để giá dễ chịu hơn.\n"
            "Mình nói em mức ngân sách dễ thở nhất với khu anh/chị muốn ở, em lọc lại ngay mấy căn hợp túi tiền cho mình nha 😊"
        )
    if intent == "GENERAL_HELP" and _is_off_topic_question(question):
        return (
            "Dạ em chỉ hỗ trợ tư vấn phòng trọ, giá thuê, chi phí, tiện ích và khu vực phù hợp thôi ạ. "
            "Mấy việc như giải bài, viết code hay xử lý nội dung ngoài thuê phòng thì em chưa hỗ trợ được.\n"
            "Nếu anh/chị đang cần tìm phòng, cứ nhắn khu vực, ngân sách hoặc tiện ích mong muốn, em lọc ngay cho mình nha 😊"
        )
    return (
        "Dạ em có thể tìm phòng, so sánh giá, tư vấn chi phí và tiện ích chi tiết ạ. "
        "Anh/chị đang cần tìm phòng quanh khu vực nào để em hỗ trợ gửi phòng đẹp ngay nhé 😊"
    )


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
        return _finalize_composed_answer(answer, abstain, reason, verification)

    if _is_off_topic_question(question):
        answer = _compose_answer_template(
            {"intent": "GENERAL_HELP"},
            grounding,
            tool_results,
            question=question,
            user_mood=user_mood,
        )
        return _finalize_composed_answer(answer, False, "", verification)

    if intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"} and rooms:
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return _finalize_composed_answer(answer, abstain, reason, verification)

    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM"} and _asks_about_amenities(question):
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return _finalize_composed_answer(answer, abstain, reason, verification)

    faq = tool_results.get("faq_results")
    if not rooms and not faq and intent not in {"GENERAL_HELP", "REQUEST_FAQ"}:
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
        return _finalize_composed_answer(answer, abstain, reason, verification)

    verified_data = _build_llm_context(grounding, tool_results)
    answer = ""
    try:
        from agents.response_writer import write_response, write_no_result_response
        if not rooms and intent in {"SEARCH_ROOM", "REFINE_SEARCH", "FIND_SIMILAR"}:
            constraints = grounding.get("constraints", {})
            alt_rooms = tool_results.get("alternative_rooms", [])
            answer = await write_no_result_response(
                question, constraints, user_mood, alt_rooms, stream_callback=stream_callback,
            )
        else:
            answer = await write_response(
                question=question,
                verified_context=verified_data,
                history=history,
                mood=user_mood,
                stream_callback=stream_callback,
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
            }
            if not verification["approved"]:
                abstain, reason = True, "reviewer_rejected"
                return _finalize_composed_answer(answer, abstain, reason, verification)
    except Exception:
        pass

    abstain, reason = _evaluate_abstain(question, intent, grounding, tool_results, answer)
    if abstain and reason == "unverified_claims":
        answer = _template_answer()
        abstain, reason = _evaluate_abstain(
            question, intent, grounding, tool_results, answer, from_template=True,
        )
    return _finalize_composed_answer(answer, abstain, reason, verification)


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


def _extract_rooms(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = tool_results.get("rooms") or []
    return [item for item in rooms if item]


def _current_room_from_results(intent: str, rooms: list[dict[str, Any]]) -> str | None:
    if intent in {"ASK_ABOUT_ROOM", "SUMMARIZE_ROOM", "CALCULATE_COST"} and rooms:
        return rooms[0].get("room_id")
    return None


def _unknown_fields(room: dict[str, Any]) -> list[str]:
    return unknown_room_fields(room)


AMENITY_LABELS: dict[str, str] = {
    "air_conditioner": "Máy lạnh",
    "balcony": "Ban công",
    "window": "Cửa sổ",
    "washing_machine": "Máy giặt",
    "private_bathroom": "WC riêng",
    "mezzanine": "Gác",
    "kitchen": "Bếp",
    "refrigerator": "Tủ lạnh",
    "hot_water": "Nước nóng",
    "bed": "Giường",
    "mattress": "Nệm",
    "wardrobe": "Tủ quần áo",
    "elevator": "Thang máy",
    "wifi": "Wifi",
    "ev_charging": "Sạc xe điện",
    "free_hours": "Giờ tự do",
    "pets_allowed": "Cho nuôi thú cưng",
}

FEATURE_FACT_LABELS: tuple[str, ...] = (
    "Máy lạnh",
    "Ban công",
    "Cửa sổ",
    "Wifi",
    "Gác",
    "Toilet",
    "Giờ giấc",
    "Máy giặt",
    "Thú cưng",
    "Để xe",
    "Thang máy",
    "Kệ bếp",
    "Nước nóng",
    "Tủ lạnh",
    "Giường",
    "Nệm",
    "Tủ quần áo",
)


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
            f"- **Giá:** {cheaper.get('title') or ('#' + str(cheaper.get('room_id')))} rẻ hơn "
            f"{pricier.get('title') or ('#' + str(pricier.get('room_id')))} khoảng {format_vnd(diff)}/tháng."
        )

    first_area = first.get("area_m2")
    second_area = second.get("area_m2")
    if isinstance(first_area, (int, float)) and isinstance(second_area, (int, float)) and first_area != second_area:
        larger, smaller = (first, second) if first_area > second_area else (second, first)
        diff_area = abs(float(first_area) - float(second_area))
        diff_text = int(diff_area) if diff_area.is_integer() else diff_area
        insights.append(
            f"- **Diện tích:** {larger.get('title') or ('#' + str(larger.get('room_id')))} rộng hơn "
            f"{smaller.get('title') or ('#' + str(smaller.get('room_id')))} khoảng {diff_text} m²."
        )

    first_features = set(_comparison_feature_labels(first))
    second_features = set(_comparison_feature_labels(second))
    first_only = sorted(first_features - second_features)
    second_only = sorted(second_features - first_features)
    if first_only:
        insights.append(f"- **Ưu điểm {first_title}:** có thêm {', '.join(first_only[:5])}.")
    if second_only:
        insights.append(f"- **Ưu điểm {second_title}:** có thêm {', '.join(second_only[:5])}.")

    if not insights:
        insights.append("- Hai căn này khá ngang nhau về dữ liệu hiện có; nếu cần mình có thể đào sâu thêm vào phí, nội thất và tiện ích chi tiết.")
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
            hints[room_id] = f"gần {' / '.join(matched)}"
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


_RELAX_FIELD_LABELS: dict[str, str] = {
    "near_landmarks": "vị trí gần mốc bạn nói",
    "wards": "phường bạn chọn",
    "amenities_preferred": "vài tiện nghi ưu tiên",
    "amenities_required": "đủ tiện nghi yêu cầu",
    "excluded_features": "điều kiện loại trừ",
    "budget": "mức ngân sách",
}


def _relaxed_note(dropped: list[str]) -> str:
    labels = [_RELAX_FIELD_LABELS[item] for item in dropped if item in _RELAX_FIELD_LABELS]
    if not labels:
        return "em chưa thấy căn khớp đúng 100% nên xin phép nới nhẹ tiêu chí cho mình ạ."
    return (
        "em chưa thấy căn khớp đúng "
        + ", ".join(labels)
        + " nên em xin phép gợi ý mấy căn gần đúng nhất nha."
    )


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
        "fee_electricity": "Tiền điện",
        "fee_water": "Tiền nước",
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
