"""
Grader Agent — Đánh giá chất lượng policy documents sau Retrieve + Rerank.
KHÔNG sử dụng LLM (0 token, 0ms).
"""
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import GRADE_SCORE_THRESHOLD, MIN_GOOD_DOCS
from agents.state import AgentState
from utils.console import console


def grade_documents_node(state: AgentState) -> dict:
    """
    Node: Đánh giá chất lượng tài liệu policy retrieved.
    Dựa trên rerank_score (output của BGE-Reranker cross-encoder).
    """
    t0 = time.time()
    evidence = state.get("evidence", [])
    rewrite_count = state.get("rewrite_count", 0)

    good_docs = []
    for doc in evidence:
        score = doc.get("rerank_score", doc.get("rrf_score", 0))
        if score >= GRADE_SCORE_THRESHOLD:
            good_docs.append(doc)

    is_sufficient = len(good_docs) >= MIN_GOOD_DOCS

    # Extract compensation info from good docs
    compensation = ""
    for doc in good_docs:
        limit = doc.get("compensation_limit", 0)
        if limit and limit > 0:
            compensation = f"Hạn mức bồi thường: {limit:,}đ"
            break

    decision = "GOOD" if is_sufficient else "BAD"
    elapsed = int((time.time() - t0) * 1000)

    console.print(
        f"[dim]  Grader: {len(good_docs)}/{len(evidence)} docs above "
        f"threshold={GRADE_SCORE_THRESHOLD} -> {decision} "
        f"(rewrite #{rewrite_count}) ({elapsed}ms)[/]"
    )

    return {
        "is_evidence_sufficient": is_sufficient,
        "compensation": compensation,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "grade_good_docs": len(good_docs),
            "grade_total_docs": len(evidence),
            "grade_threshold": GRADE_SCORE_THRESHOLD,
            "grade_decision": decision,
            "grade_rewrite_count": rewrite_count,
            "grade_ms": elapsed,
        },
    }
