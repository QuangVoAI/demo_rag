"""
EmpathAI evaluation for the single production architecture.

The benchmark calls the same LangGraph pipeline used by the Kafka worker:
Router -> Sentiment/Retrieval -> Grader/Rewriter -> Writer -> Reviewer.

Usage:
  pip install -r evaluation/requirements.txt
  python evaluation/evaluate.py [--limit N] [--delay SECONDS]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
PYTHON_DIR = PROJECT_ROOT / "python"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PYTHON_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from agents.graph import run_streaming  # noqa: E402
from metrics import compute_all  # noqa: E402

ARCH_LABEL = "EmpathAI | LangGraph + Fine-tuned RAG"


async def run_empathai(question: str) -> tuple[str, list[str], dict, float]:
    start = time.perf_counter()
    final_state = await run_streaming(question=question, history=[], session_id="evaluation")
    latency = time.perf_counter() - start
    evidence = final_state.get("evidence", []) or []
    retrieved_ids = [
        doc.get("policy_id") or doc.get("doc_id") or ""
        for doc in evidence
    ]
    return final_state.get("answer", ""), retrieved_ids, final_state.get("agent_trace", {}), latency


async def evaluate(questions: list[dict], delay_s: float) -> dict:
    hypotheses: list[str] = []
    retrieved_ids: list[list[str]] = []
    traces: list[dict] = []
    latencies: list[float] = []

    print(f"\nEvaluating: {ARCH_LABEL}")
    print(f"Questions : {len(questions)}")

    for i, item in enumerate(questions, 1):
        question = item["question"]
        try:
            answer, ids, trace, latency = await run_empathai(question)
        except Exception as exc:
            print(f"  [Q{i:02d} ERROR] {exc}")
            answer, ids, trace, latency = "", [], {}, 0.0

        hypotheses.append(answer)
        retrieved_ids.append(ids)
        traces.append(trace)
        latencies.append(latency)

        status = "OK" if answer else "FAIL"
        print(f"  [{status}] Q{i:02d}/{len(questions)} {latency:.1f}s {question[:55]}...")

        if delay_s > 0 and i < len(questions):
            await asyncio.sleep(delay_s)

    references = [q["reference"] for q in questions]
    relevant_ids = [q["relevant_policy"] for q in questions]
    metrics = compute_all(
        hypotheses=hypotheses,
        references=references,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
    )
    metrics["Avg latency (s)"] = round(sum(latencies) / max(len(latencies), 1), 2)

    return {
        "label": ARCH_LABEL,
        "hypotheses": hypotheses,
        "retrieved_ids": retrieved_ids,
        "traces": traces,
        "latencies": latencies,
        "metrics": metrics,
    }


def save_results(run: dict, questions: list[dict], timestamp: str) -> None:
    summary_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Architecture", "BLEU", "ROUGE-L", "BERTScore", "Recall@5", "Avg latency (s)"])
        metrics = run["metrics"]
        writer.writerow([
            run["label"],
            metrics.get("BLEU"),
            metrics.get("ROUGE-L"),
            metrics.get("BERTScore"),
            metrics.get("Recall@5"),
            metrics.get("Avg latency (s)"),
        ])
    print(f"Summary saved -> {summary_path}")

    detail_path = RESULTS_DIR / f"detail_{timestamp}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "question", "reference", "empathai_answer", "retrieved_policy_ids", "latency_s"])
        for i, q in enumerate(questions):
            writer.writerow([
                q["id"],
                q["category"],
                q["question"],
                q["reference"],
                run["hypotheses"][i],
                "|".join(run["retrieved_ids"][i]),
                round(run["latencies"][i], 2),
            ])
    print(f"Detail saved -> {detail_path}")

    human_path = RESULTS_DIR / f"human_eval_{timestamp}.csv"
    with open(human_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "question", "empathai_answer",
            "empathy(1-5)", "accuracy(1-5)", "natural(1-5)", "notes",
        ])
        for i, q in enumerate(questions):
            writer.writerow([q["id"], q["category"], q["question"], run["hypotheses"][i], "", "", "", ""])
    print(f"Human eval template saved -> {human_path}")


async def main(limit: Optional[int], delay: float) -> None:
    test_set = json.loads((EVAL_DIR / "test_set.json").read_text(encoding="utf-8"))
    questions = test_set["questions"]
    if limit:
        questions = questions[:limit]

    run = await evaluate(questions, delay_s=delay)

    print("\nEvaluation results")
    for key, value in run["metrics"].items():
        print(f"  {key}: {value}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(run, questions, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmpathAI single-architecture evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between questions in seconds")
    parser.add_argument("--arch", default="empathai", help="Compatibility flag; only 'empathai' or 'all' is supported")
    args = parser.parse_args()

    if args.arch not in ("empathai", "all", "4"):
        raise SystemExit("Only the EmpathAI architecture is available.")

    asyncio.run(main(limit=args.limit, delay=args.delay))
