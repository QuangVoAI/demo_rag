#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for LLM Data Processing Pipeline
"""

import concurrent.futures as _futures
import json
import re
from pathlib import Path
from typing import Dict, List

from config import PROJECT_ID, LOCATION, MODEL_NAME, PROJECT_ROOT, DATA_DIR, \
    DPO_TRAIN_FILE, DPO_VAL_FILE, DPO_CLEANED_FILE, DPO_FINAL_FILE, \
    SFT_TRAIN_FILE, SFT_DEV_FILE, SFT_TEST_FILE
from prompts import FORBIDDEN_KEYWORDS, DPO_BOUNDARY_KEYWORDS

# ─── Tuning constants ────────────────────────────────────────────────────────
MIN_RESPONSE_LEN = 50
MAX_SIMILARITY   = 0.85
MIN_BAD_SIGNAL   = 1

# Vietnamese toxic patterns redacted to neutral placeholders.
# This prevents Gemini's safety filter from blocking requests that quote
# verbatim toxic customer messages.
_SANITIZE_RULES = [
    (r'\b(đm|đcm|dm|đéo|clm|vãi|mẹ mày|con mẹ|thằng chó|đồ chó)\b', '[ngôn ngữ xúc phạm]'),
    (r'\b(kiện|tố cáo|đăng phốt|bóc phốt|khởi kiện)\b', '[đe dọa pháp lý]'),
    (r'\b(đ[uú]t|cút|biến|xéo)\b', '[xúc phạm]'),
]


# ─── Gemini safety helpers ────────────────────────────────────────────────────

def sanitize_for_gemini(text: str) -> str:
    """
    Lightly redact toxic Vietnamese patterns so Gemini's safety filter does not
    block requests containing raw customer complaints.  The scenario context is
    preserved — only exact trigger phrases are replaced with category labels.
    """
    result = text
    for pattern, placeholder in _SANITIZE_RULES:
        result = re.sub(pattern, placeholder, result, flags=re.IGNORECASE)
    return result


def is_safety_blocked(response) -> bool:
    """
    Return True when Gemini refused the request due to its safety filters.

    Vertex AI signals a block in two ways:
      1. response.text is empty or raises ValueError on access.
      2. candidates[0].finish_reason == SAFETY (enum value / string "3").

    Both are treated as a block.
    """
    try:
        text = response.text
        if not text or not text.strip():
            return True
    except (ValueError, AttributeError):
        return True

    try:
        if response.candidates:
            reason = str(response.candidates[0].finish_reason)
            if "SAFETY" in reason or reason == "3":
                return True
    except (AttributeError, IndexError):
        pass

    return False


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def strip_json_markdown(text: str) -> str:
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def clean_json_response(text: str):
    """
    Parse JSON from Gemini output.
    Uses JSONDecoder.raw_decode() to correctly handle nested objects/arrays
    (replaces the old greedy r'{.*}' regex that swallowed everything).
    """
    cleaned = strip_json_markdown(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in range(len(cleaned)):
        if cleaned[start] not in ('{', '['):
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned, start)
            return obj
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("No valid JSON found in response", cleaned, 0)


# ─── File I/O ────────────────────────────────────────────────────────────────

def load_jsonl(file_path: Path) -> List[Dict]:
    data = []
    print("Reading {}...".format(file_path))
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("Warning: Line {} decode error: {}".format(i + 1, e))
    print("Loaded {} records from {}".format(len(data), file_path))
    return data


def write_jsonl(file_path: Path, data: List[Dict]) -> None:
    print("Writing {} records to {}...".format(len(data), file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("Done → {}".format(file_path))


# ─── DPO pair quality ─────────────────────────────────────────────────────────

def _extract_assistant_text(messages: List[Dict]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "").lower()
    return ""


def _jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def is_good_pair(chosen: List[Dict], rejected: List[Dict]) -> bool:
    """
    A good DPO pair satisfies ALL four conditions:
      1. Both responses >= MIN_RESPONSE_LEN characters.
      2. chosen does NOT contain sycophantic/forbidden keywords.
      3. rejected DOES contain >= MIN_BAD_SIGNAL bad-signal keywords.
      4. Jaccard similarity between chosen and rejected < MAX_SIMILARITY.
    """
    chosen_text   = _extract_assistant_text(chosen)
    rejected_text = _extract_assistant_text(rejected)

    if len(chosen_text) < MIN_RESPONSE_LEN:
        print("  ✗ chosen too short ({} chars)".format(len(chosen_text)))
        return False
    if len(rejected_text) < MIN_RESPONSE_LEN:
        print("  ✗ rejected too short ({} chars)".format(len(rejected_text)))
        return False

    all_bad = FORBIDDEN_KEYWORDS + DPO_BOUNDARY_KEYWORDS
    for kw in all_bad:
        if kw in chosen_text:
            print("  ✗ chosen contains bad keyword '{}'".format(kw))
            return False

    bad_signals = sum(1 for kw in all_bad if kw in rejected_text)
    if bad_signals < MIN_BAD_SIGNAL:
        print("  ✗ rejected has no bad-signal markers")
        return False

    sim = _jaccard_similarity(chosen_text, rejected_text)
    if sim >= MAX_SIMILARITY:
        print("  ✗ too similar (Jaccard={:.2f})".format(sim))
        return False

    print("  ✓ good pair (Jaccard={:.2f}, bad_signals={})".format(sim, bad_signals))
    return True


# ─── Timeout-safe generation ─────────────────────────────────────────────────

_CALL_TIMEOUT = 30  # seconds before a generate_content call is considered hung


def generate_with_timeout(model, prompt, generation_config, timeout: int = _CALL_TIMEOUT):
    """
    Call model.generate_content() in a thread with a hard timeout.
    Raises TimeoutError if the call does not return within `timeout` seconds.

    NOTE: do NOT use 'with executor' — its __exit__ calls shutdown(wait=True),
    which re-blocks the main thread even after the timeout fires.
    Instead we call shutdown(wait=False) immediately so abandoned threads are
    left to finish on their own without blocking the caller.
    """
    executor = _futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(model.generate_content, prompt, generation_config=generation_config)
    executor.shutdown(wait=False)
    try:
        return future.result(timeout=timeout)
    except _futures.TimeoutError:
        raise TimeoutError(
            "generate_content timed out after {}s".format(timeout)
        )


# ─── Vertex AI client ────────────────────────────────────────────────────────

def get_gemini_client():
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        print("Vertex AI ready: {}/{}/{}".format(PROJECT_ID, LOCATION, MODEL_NAME))
        return model
    except Exception as e:
        print("ERROR: Vertex AI init failed: {}".format(e))
        raise