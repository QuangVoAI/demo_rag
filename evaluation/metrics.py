"""
Metric computation for EmpathAI evaluation.

- BLEU: sacrebleu with char-level tokenization.
- ROUGE-L: longest common subsequence F1.
- BERTScore: semantic similarity.
- Recall@5: whether the relevant policy appears in the top retrieved docs.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level BLEU on a 0-100 scale."""
    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize="char")
    result = bleu.corpus_score(hypotheses, [references])
    return round(result.score, 2)


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Average ROUGE-L F1 on a 0-100 scale."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    return round(float(np.mean(scores)) * 100, 2)


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str = "vi",
    model_type: Optional[str] = None,
) -> float:
    """Average BERTScore F1 on a 0-100 scale."""
    from bert_score import score as bs

    kwargs: dict = {"lang": lang, "verbose": False}
    kwargs["model_type"] = model_type or "bert-base-multilingual-cased"

    safe_hypotheses = [h if h and h.strip() else "empty" for h in hypotheses]

    _, _, f1 = bs(safe_hypotheses, references, **kwargs)
    return round(float(f1.mean()) * 100, 2)


def compute_recall_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids: List[str],
    k: int = 5,
) -> float:
    """Recall@k for the retrieved policy ids."""
    hits = sum(
        1
        for retrieved, relevant in zip(retrieved_ids, relevant_ids)
        if relevant in retrieved[:k]
    )
    return round(hits / max(len(relevant_ids), 1) * 100, 2)


def compute_all(
    hypotheses: List[str],
    references: List[str],
    retrieved_ids: Optional[List[List[str]]] = None,
    relevant_ids: Optional[List[str]] = None,
) -> dict:
    """Compute all configured metrics."""
    results: dict = {
        "BLEU": compute_bleu(hypotheses, references),
        "ROUGE-L": compute_rouge_l(hypotheses, references),
    }
    try:
        results["BERTScore"] = compute_bertscore(hypotheses, references)
    except Exception:
        results["BERTScore"] = "N/A"
    if retrieved_ids is not None and relevant_ids is not None:
        results["Recall@5"] = compute_recall_at_k(retrieved_ids, relevant_ids, k=5)
    else:
        results["Recall@5"] = None
    return results
