"""Compatibility entry point for the Nhatrovn read-only room assistant.

The legacy support graph is intentionally no longer imported here.
Production flow:
START -> normalize_input -> parse_intent_and_constraint_patch
      -> load_session_state -> merge_and_validate_state -> route_workflow
      -> execute_read_only_tools -> grounding_check -> compose_response
      -> persist_state_and_trace -> END
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Awaitable, Callable, Optional

sys.path.append(str(Path(__file__).parent.parent))

from room_assistant.schemas import RECENT_HISTORY_TURNS
from room_assistant.workflow import run_room_assistant


async def run_streaming(
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict:
    """Run final-only grounded response generation.

    The callback is accepted for API compatibility, but no ungrounded token is
    streamed before the final response is composed and validated.
    """
    return await run_room_assistant(
        question=question,
        history=(history or [])[-RECENT_HISTORY_TURNS:],
        session_id=session_id,
        stream_callback=stream_callback,
    )
