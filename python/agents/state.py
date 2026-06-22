"""
Agent State — Shared state definition cho LangGraph EmpathAI pipeline.
"""
from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State chung cho EmpathAI LangGraph pipeline."""

    # --- Input ---
    session_id: str
    question: str               # Tin nhan cua khach hang
    history: list[dict]         # Chat history [{role, content}, ...]

    # --- Router Output ---
    intent: str                 # "COMPLAINT" | "INQUIRY" | "CASUAL"

    # --- Sentiment Analysis Output ---
    sentiment: str              # "toxic" | "frustrated" | "disappointed" | "neutral"
    sentiment_score: float      # 0.0 - 1.0

    # --- Retrieval Output ---
    evidence: list[dict]        # Retrieved & reranked policy chunks
    evidence_text: str          # Formatted policy context cho LLM
    policy_context: str         # Chinh sach ap dung cu the
    compensation: str           # Goi y boi thuong tu RAG

    # --- Rewrite Loop ---
    rewrite_count: int
    is_evidence_sufficient: bool
    translated_query: str       # Query da duoc rewrite (khong dich, chi rewrite)

    # --- Generation Output ---
    answer: str                 # Phan hoi thau cam cuoi cung

    # --- Reviewer Output ---
    reviewer_triggered: bool
    reviewer_result: dict       # {is_approved, issues, retry_count}

    # --- Metadata ---
    agent_trace: dict
    processing_time_ms: int

    # --- Streaming ---
    stream_callback: Any
