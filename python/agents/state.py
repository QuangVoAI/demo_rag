"""Legacy shared types — kept for backward compatibility only.

Runtime chính dùng ``room_assistant.session_store`` và ``room_assistant.schemas``,
không còn LangGraph state machine.
"""

from __future__ import annotations

from typing import Any

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class NhatrovnAgentState(TypedDict, total=False):
    """TypedDict tối giản — chỉ còn cho type hint tương thích."""

    session_id: str
    question: str
    history: list[dict]
    user_mood: str
    user_mood_score: float
    answer: str
    agent_trace: dict[str, Any]
