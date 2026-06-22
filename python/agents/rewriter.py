"""
Rewriter Agent — Viết lại query khi policy search không đủ.
Sử dụng Groq FAST để tinh chỉnh query tìm chính sách phù hợp.
"""
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.state import AgentState
from agents.llm_client import groq_complete, vertex_custom_complete, GROQ_MODEL_FAST
from config import EMPATHY_MODE
from utils.console import console

REWRITE_SYSTEM_PROMPT = """\
You are a query rewriting expert for customer service policy search.
Given a customer complaint that returned poor policy results, rewrite the search query.

Strategies:
1. Extract the CORE ISSUE (delivery delay, broken product, wrong item, etc.)
2. Add specific policy-related terms (hoàn tiền, đổi trả, bồi thường, voucher, bảo hành)
3. Include the CATEGORY (vận chuyển, sản phẩm, dịch vụ, thanh toán)
4. Be concise and specific

RULES:
- Output ONLY the rewritten query in Vietnamese, nothing else
- Keep it concise (1-2 sentences max)
- Focus on finding the RIGHT POLICY to resolve the complaint
"""


async def rewrite_query_node(state: AgentState) -> dict:
    """Node: Rewrite query để tìm chính sách phù hợp hơn."""
    t0 = time.time()
    original_query = state.get("translated_query", state["question"])
    rewrite_count = state.get("rewrite_count", 0)
    evidence = state.get("evidence", [])
    sentiment = state.get("sentiment", "")

    evidence_context = ""
    if evidence:
        titles = [doc.get("doc_title", "")[:80] for doc in evidence[:3]]
        evidence_context = (
            f"\nPrevious search returned these partially relevant policies:\n"
            + "\n".join(f"- {t}" for t in titles if t)
            + "\nRewrite to find MORE relevant policies."
        )

    prompt = (
        f"Customer complaint: {state['question']}\n"
        f"Sentiment: {sentiment}\n"
        f"Original query (attempt #{rewrite_count + 1}): {original_query}\n"
        f"{evidence_context}\n\n"
        f"Rewrite this query to find the best matching CSKH policy:"
    )

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    if EMPATHY_MODE == "vertex":
        try:
            rewritten = await vertex_custom_complete(
                messages=messages,
                max_tokens=128,
                temperature=0.3,
            )
        except Exception as e:
            print(f"Vertex AI rewrite error: {e}, falling back to Groq")
            rewritten = await groq_complete(
                prompt=prompt,
                system_prompt=REWRITE_SYSTEM_PROMPT,
                model=GROQ_MODEL_FAST,
                max_tokens=128,
                temperature=0.3,
            )
    else:
        rewritten = await groq_complete(
            prompt=prompt,
            system_prompt=REWRITE_SYSTEM_PROMPT,
            model=GROQ_MODEL_FAST,
            max_tokens=128,
            temperature=0.3,
        )

    rewritten = rewritten.strip().strip('"').strip("'")

    elapsed = int((time.time() - t0) * 1000)
    console.print(
        f"[yellow]  Rewrite #{rewrite_count + 1}: "
        f"'{original_query[:40]}...' -> '{rewritten[:40]}...' ({elapsed}ms)[/]"
    )

    return {
        "translated_query": rewritten,
        "rewrite_count": rewrite_count + 1,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            f"rewrite_{rewrite_count + 1}_from": original_query,
            f"rewrite_{rewrite_count + 1}_to": rewritten,
            f"rewrite_{rewrite_count + 1}_ms": elapsed,
        },
    }
