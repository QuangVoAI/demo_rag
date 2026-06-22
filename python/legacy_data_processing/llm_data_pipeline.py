#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Data Processing Pipeline - 4-stage DPO/SFT pipeline

Pipeline stages:
1. Merge & Clean DPO Data
2. Edge Cases Data Augmentation
3. DPO to SFT Format Conversion
4. Train/Dev/Test Split
"""

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, ChatSession
except ImportError:
    print("ERROR: vertexai not installed.  pip install vertexai")
    exit(1)

from config import (
    PROJECT_ID, LOCATION, MODEL_NAME,
    DPO_TRAIN_FILE, DPO_VAL_FILE, DPO_CLEANED_FILE, DPO_FINAL_FILE,
    SFT_TRAIN_FILE, SFT_DEV_FILE, SFT_TEST_FILE, DEBUG_MODE,
    DPO_TRAIN_SPLIT_FILE, DPO_DEV_SPLIT_FILE, DPO_TEST_SPLIT_FILE,
    FAILED_RECORDS_FILE,
)
from prompts import (
    SCENARIOS, SCENARIO_SYSTEM_PROMPS,
    REWRITE_PROMPT, AUGMENTATION_PROMPT, SINGLE_PAIR_PROMPT, SITUATION_HINTS,
)
from utils import (
    sanitize_for_gemini, is_safety_blocked,
    strip_json_markdown, clean_json_response,
    load_jsonl, write_jsonl, is_good_pair, get_gemini_client,
    generate_with_timeout,
)

# ─── Shared generation config ────────────────────────────────────────────────
_GEN_CFG_REWRITE = {
    "temperature": 0.7,
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
}
_GEN_CFG_AUG = {
    "temperature": 0.9,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

CHECKPOINT_EVERY = 100  # save cleaned_data to disk every N processed records
_AUG_TIMEOUT    = 60   # augmentation generates 10 pairs (~8192 tokens) — needs more time

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _scenario_meta(scenario_id: Optional[int]) -> Dict:
    """Return scenario dict, falling back to a generic profile when id is None."""
    if scenario_id and scenario_id in SCENARIOS:
        return SCENARIOS[scenario_id]
    return {
        "description": "Tổng quát",
        "context":     "Khách hàng phản ánh vấn đề với sản phẩm/dịch vụ.",
        "chosen_rule": (
            "Thấu cảm thực sự, nhượng bộ thông minh, KHÔNG BAO GIỜ dùng "
            "văn mẫu như 'Chúng tôi rất tiếc' hay 'Theo chính sách công ty'. "
            "Kết thúc bằng câu hỏi mở."
        ),
        "rejected_rule": (
            "Phản hồi kém, chứa ít nhất một từ: 'xin lỗi', 'ôi trời', "
            "'mình hiểu lắm', 'bồi thường', hoặc 'voucher'."
        ),
    }


def _generate_single_pair(
    sc: Dict,
    model: GenerativeModel,
    hint_index: int = 0,
) -> Optional[Dict]:
    """
    Generate exactly one DPO pair using SINGLE_PAIR_PROMPT.
    Used as:
      - Fallback when REWRITE_PROMPT is safety-blocked.
      - Padding when AUGMENTATION_PROMPT returns fewer than target pairs.
    Returns None on failure.
    """
    hint = SITUATION_HINTS[hint_index % len(SITUATION_HINTS)]
    prompt = SINGLE_PAIR_PROMPT.format(
        scenario_description=sc["description"],
        scenario_context=sc["context"],
        chosen_rule=sc["chosen_rule"],
        rejected_rule=sc["rejected_rule"],
        situation_hint=hint,
    )

    response_text = "<no response yet>"
    try:
        response      = generate_with_timeout(model, prompt, _GEN_CFG_REWRITE, timeout=30)
        response_text = response.text

        if is_safety_blocked(response):
            print("    ✗ SINGLE_PAIR_PROMPT blocked by safety filter")
            return None

        pair = clean_json_response(response_text)
        if "prompt" in pair and "chosen" in pair and "rejected" in pair:
            return pair
        # Salvage: model returned chosen+rejected without prompt — synthesize one
        if "chosen" in pair and "rejected" in pair:
            synth_prompt = [
                {"role": "system",  "content": "Bạn là EmpathAI, trợ lý CSKH. {}".format(sc["context"])},
                {"role": "user",    "content": hint},
            ]
            chosen_val   = pair["chosen"]
            rejected_val = pair["rejected"]
            chosen_msg   = chosen_val   if isinstance(chosen_val,   list) else [{"role": "assistant", "content": chosen_val}]
            rejected_msg = rejected_val if isinstance(rejected_val, list) else [{"role": "assistant", "content": rejected_val}]
            return {"prompt": synth_prompt, "chosen": chosen_msg, "rejected": rejected_msg}
        print("    ✗ SINGLE_PAIR_PROMPT: unexpected schema")
        return None

    except Exception as e:
        print("    ✗ SINGLE_PAIR_PROMPT error: {} | response: {}".format(e, response_text[:120]))
        return None


# ─── Stage 1 helpers ─────────────────────────────────────────────────────────

def rewrite_bad_pair(
    prompt: List[Dict],
    chosen: List[Dict],
    rejected: List[Dict],
    model: GenerativeModel,
    scenario_id: Optional[int] = None,
) -> Optional[Dict]:
    """
    Attempt to rewrite a bad DPO pair.

    Fix (Stage 1 / Safety Filter):
      - prompt_text is sanitized via sanitize_for_gemini() before interpolation.
      - Added "[DỮ LIỆU HUẤN LUYỆN AI]" framing (lives in REWRITE_PROMPT).
      - On safety block → falls back to _generate_single_pair() so Stage 1
        never silently drops the record.

    Returns the rewritten (or freshly generated) pair, or None on full failure.
    """
    sc = _scenario_meta(scenario_id)

    # Extract and sanitize the customer message
    raw_prompt_text = next(
        (msg.get("content", "") for msg in prompt if msg.get("role") == "user"), ""
    )
    safe_prompt_text = sanitize_for_gemini(raw_prompt_text)

    rewriting_prompt = REWRITE_PROMPT.format(
        scenario_description=sc["description"],
        scenario_context=sc["context"],
        chosen_rule=sc["chosen_rule"],
        rejected_rule=sc["rejected_rule"],
        prompt_text=safe_prompt_text,
    )

    response_text = "<no response yet>"
    try:
        response      = generate_with_timeout(model, rewriting_prompt, _GEN_CFG_REWRITE, timeout=30)
        response_text = response.text

        if is_safety_blocked(response):
            print("  ⚠ REWRITE_PROMPT safety-blocked → generating fresh pair instead")
            fresh = _generate_single_pair(sc, model)
            return fresh  # may be None; caller decides what to do

        result = clean_json_response(response_text)
        if "chosen" not in result or "rejected" not in result:
            raise ValueError("Missing 'chosen' or 'rejected' in response")

        return {
            "prompt":   prompt,
            "chosen":   [{"role": "assistant", "content": result["chosen"]}],
            "rejected": [{"role": "assistant", "content": result["rejected"]}],
        }

    except Exception as e:
        print("  ✗ rewrite failed: {} | response: {}".format(e, response_text[:120]))
        # Last-chance fallback
        try:
            print("  ↩ trying fresh pair fallback")
            return _generate_single_pair(sc, model)
        except Exception:
            return None


# ─── Stage 1 ──────────────────────────────────────────────────────────────────

def stage1_merge_and_clean() -> List[Dict]:
    """Merge dpo_train + dpo_val, filter and rewrite bad pairs."""
    print("=== STAGE 1: Merge & Clean DPO Data ===")

    train_data = load_jsonl(DPO_TRAIN_FILE)
    val_data   = load_jsonl(DPO_VAL_FILE)
    all_data   = train_data + val_data
    print("Merged {} records".format(len(all_data)))

    if DEBUG_MODE:
        print("DEBUG_MODE: limiting to first 10 records")
        all_data = all_data[:10]

    gemini_model   = get_gemini_client()
    cleaned_data   = []
    bad_pair_count = 0
    rewritten_ok   = 0
    fallback_ok    = 0

    for i, item in enumerate(all_data):
        prompt   = item.get("prompt",   [])
        chosen   = item.get("chosen",   [])
        rejected = item.get("rejected", [])

        if not prompt or not chosen or not rejected:
            print("Record {}: skipping (missing fields)".format(i + 1))
            continue

        if is_good_pair(chosen, rejected):
            cleaned_data.append(item)
            print("Record {}: ✓ good pair, keeping".format(i + 1))
        else:
            print("Record {}: ✗ bad pair, rewriting...".format(i + 1))
            bad_pair_count += 1

            result = rewrite_bad_pair(prompt, chosen, rejected, gemini_model)

            if result is None:
                print("Record {}: all fallbacks failed, skipping".format(i + 1))
                with open(FAILED_RECORDS_FILE, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif result.get("prompt") is prompt:
                cleaned_data.append(result)
                rewritten_ok += 1
                print("Record {}: rewritten".format(i + 1))
            else:
                # Fresh pair from fallback — prompt field differs
                cleaned_data.append(result)
                fallback_ok += 1
                print("Record {}: replaced with fresh pair".format(i + 1))

            time.sleep(2.5)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print("  [Checkpoint] Saving {} records after record {}...".format(len(cleaned_data), i + 1))
            write_jsonl(DPO_CLEANED_FILE, cleaned_data)

    print("\nStage 1: {} cleaned records".format(len(cleaned_data)))
    print("  bad detected={} rewritten={} fresh_fallback={}".format(
        bad_pair_count, rewritten_ok, fallback_ok))

    write_jsonl(DPO_CLEANED_FILE, cleaned_data)
    return cleaned_data


# ─── Stage 2 helpers ─────────────────────────────────────────────────────────

def _parse_pairs_from_result(result) -> List[Dict]:
    """Extract a list of DPO pairs from whatever shape Gemini returned."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        if "pairs" in result and isinstance(result["pairs"], list):
            return result["pairs"]
        if "prompt" in result and "chosen" in result and "rejected" in result:
            return [result]
    return []


def _fetch_pairs_batch(
    sc: Dict,
    n: int,
    model: GenerativeModel,
    attempt: int = 1,
) -> List[Dict]:
    """
    Ask Gemini to generate n pairs in one call.
    Returns however many valid pairs came back (may be < n).
    """
    prompt = AUGMENTATION_PROMPT.format(
        scenario_description=sc["description"],
        scenario_context=sc["context"],
        chosen_rule=sc["chosen_rule"],
        rejected_rule=sc["rejected_rule"],
        n=n,
    )

    response_text = "<no response yet>"
    try:
        response      = generate_with_timeout(model, prompt, _GEN_CFG_AUG, timeout=_AUG_TIMEOUT)
        response_text = response.text

        if is_safety_blocked(response):
            print("    ✗ AUGMENTATION_PROMPT safety-blocked (attempt {})".format(attempt))
            return []

        result = clean_json_response(response_text)
        pairs  = _parse_pairs_from_result(result)
        print("    Attempt {}: asked={} got={}".format(attempt, n, len(pairs)))
        return pairs[:n]

    except Exception as e:
        print("    ✗ Batch fetch error (attempt {}): {} | {}".format(
            attempt, e, response_text[:120]))
        return []


# ─── Stage 2 ──────────────────────────────────────────────────────────────────

def stage2_data_augmentation(existing_data: List[Dict]) -> List[Dict]:
    """
    Generate DPO pairs for 6 scenarios.

    Fix (Stage 2 / Lazy Gemini):
      - AUGMENTATION_PROMPT now passes {n} explicitly and repeats the count
        requirement.  temperature bumped to 0.9 for more variety.
      - After each batch call: if returned < target, retry once with the
        shortfall amount.
      - If still short: pad one-by-one via _generate_single_pair() with
        cycling SITUATION_HINTS so every pair has a different context.
    """
    print("=== STAGE 2: Edge Cases Data Augmentation ===")

    gemini_model = get_gemini_client()
    augmented_data = []

    iterations_per_scenario = 1 if DEBUG_MODE else 30
    pairs_per_call = 10  # target per API call

    if DEBUG_MODE:
        print("DEBUG_MODE: 1 iteration per scenario")

    hint_counter = 0  # global index into SITUATION_HINTS for padding

    for scenario_id, sc in SCENARIOS.items():
        print("\nScenario {}: {}".format(scenario_id, sc["description"]))
        scenario_data = []

        for iteration in range(iterations_per_scenario):
            print("  Iteration {}/{}".format(iteration + 1, iterations_per_scenario))

            # ── First attempt ────────────────────────────────────────────────
            pairs = _fetch_pairs_batch(sc, pairs_per_call, gemini_model, attempt=1)
            time.sleep(1.5)

            # ── Retry if short ───────────────────────────────────────────────
            shortfall = pairs_per_call - len(pairs)
            if shortfall > 0:
                print("    Short by {}, retrying batch...".format(shortfall))
                extra = _fetch_pairs_batch(sc, shortfall, gemini_model, attempt=2)
                pairs.extend(extra)
                if extra:
                    time.sleep(1.5)

            # ── Pad one-by-one if still short ────────────────────────────────
            remaining = pairs_per_call - len(pairs)
            if remaining > 0:
                print("    Still short by {}, padding one-by-one...".format(remaining))
                for _ in range(remaining):
                    single = _generate_single_pair(sc, gemini_model, hint_counter)
                    hint_counter += 1
                    if single:
                        pairs.append(single)
                    time.sleep(1.0)

            scenario_data.extend(pairs)
            print("  → Iteration total: {} pairs (running: {})".format(
                len(pairs), len(scenario_data)))

        print("Scenario {} done: {} pairs".format(scenario_id, len(scenario_data)))
        augmented_data.extend(scenario_data)

        # ── Checkpoint after each scenario ───────────────────────────────────
        combined_so_far = existing_data + augmented_data
        print("  [Checkpoint] Saving {} records (scenario {} complete)...".format(
            len(combined_so_far), scenario_id))
        write_jsonl(DPO_FINAL_FILE, combined_so_far)

    print("\nTotal augmented: {}".format(len(augmented_data)))
    combined = existing_data + augmented_data
    print("Combined with existing: {}".format(len(combined)))

    write_jsonl(DPO_FINAL_FILE, combined)
    return combined


# ─── Stage 3 ──────────────────────────────────────────────────────────────────

def stage3_dpo_to_sft_conversion(dpo_data: List[Dict]) -> List[Dict]:
    """Convert DPO pairs to SFT format — keep chosen, drop rejected."""
    print("=== STAGE 3: DPO to SFT Conversion ===")

    sft_data = []
    for i, item in enumerate(dpo_data):
        prompt = item.get("prompt", [])
        chosen = item.get("chosen", [])

        system_msg = user_msg = assistant_msg = None

        for msg in prompt:
            if msg.get("role") == "system":
                system_msg = {"role": "system", "content": msg.get("content", "")}
            elif msg.get("role") == "user":
                user_msg   = {"role": "user",   "content": msg.get("content", "")}

        for msg in chosen:
            if msg.get("role") == "assistant":
                assistant_msg = {"role": "assistant", "content": msg.get("content", "")}

        if system_msg and user_msg and assistant_msg:
            sft_data.append({"messages": [system_msg, user_msg, assistant_msg]})
        else:
            print("  Record {}: missing messages, skipping".format(i + 1))

    print("Converted {} SFT records".format(len(sft_data)))
    return sft_data


# ─── Stage 4 ──────────────────────────────────────────────────────────────────

def _split_80_10_10(data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Shuffle and split into 80 / 10 / 10."""
    data = list(data)
    random.shuffle(data)
    total     = len(data)
    train_end = int(total * 0.8)
    dev_end   = train_end + int(total * 0.1)
    return data[:train_end], data[train_end:dev_end], data[dev_end:]


def stage4_train_dev_test_split(
    sft_data: List[Dict],
    dpo_data: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Shuffle then split both SFT and DPO 80 / 10 / 10."""
    print("=== STAGE 4: Train/Dev/Test Split ===")

    # ── SFT split ────────────────────────────────────────────────────────────
    train_data, dev_data, test_data = _split_80_10_10(sft_data)
    total = len(sft_data)
    print("  SFT  — Train: {} ({:.1f}%)  Dev: {} ({:.1f}%)  Test: {} ({:.1f}%)".format(
        len(train_data), len(train_data) / total * 100,
        len(dev_data),   len(dev_data)   / total * 100,
        len(test_data),  len(test_data)  / total * 100,
    ))
    write_jsonl(SFT_TRAIN_FILE, train_data)
    write_jsonl(SFT_DEV_FILE,   dev_data)
    write_jsonl(SFT_TEST_FILE,  test_data)

    # ── DPO split ────────────────────────────────────────────────────────────
    dpo_train, dpo_dev, dpo_test = _split_80_10_10(dpo_data)
    dtotal = len(dpo_data)
    print("  DPO  — Train: {} ({:.1f}%)  Dev: {} ({:.1f}%)  Test: {} ({:.1f}%)".format(
        len(dpo_train), len(dpo_train) / dtotal * 100,
        len(dpo_dev),   len(dpo_dev)   / dtotal * 100,
        len(dpo_test),  len(dpo_test)  / dtotal * 100,
    ))
    write_jsonl(DPO_TRAIN_SPLIT_FILE, dpo_train)
    write_jsonl(DPO_DEV_SPLIT_FILE,   dpo_dev)
    write_jsonl(DPO_TEST_SPLIT_FILE,  dpo_test)

    return train_data, dev_data, test_data


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== LLM Data Processing Pipeline ===")
    print("Project={} | Location={} | Model={}".format(PROJECT_ID, LOCATION, MODEL_NAME))
    if DEBUG_MODE:
        print("⚠ DEBUG_MODE is ON")

    print("\n--- Stage 1 (skipped: loading from existing dpo_cleaned.jsonl) ---")
    cleaned_data = load_jsonl(DPO_CLEANED_FILE)

    print("\n--- Stage 2 ---")
    final_data = stage2_data_augmentation(cleaned_data)

    print("\n--- Stage 3 ---")
    sft_data = stage3_dpo_to_sft_conversion(final_data)

    print("\n--- Stage 4 ---")
    train_data, dev_data, test_data = stage4_train_dev_test_split(sft_data, final_data)

    print("\n=== Pipeline Complete ===")
    print("Output files:")
    for label, path in [
        ("DPO cleaned",    DPO_CLEANED_FILE),
        ("DPO final",      DPO_FINAL_FILE),
        ("DPO train",      DPO_TRAIN_SPLIT_FILE),
        ("DPO dev",        DPO_DEV_SPLIT_FILE),
        ("DPO test",       DPO_TEST_SPLIT_FILE),
        ("SFT train",      SFT_TRAIN_FILE),
        ("SFT dev",        SFT_DEV_FILE),
        ("SFT test",       SFT_TEST_FILE),
    ]:
        print("  {:<12}: {}".format(label, path))

    print("\nRecord counts:")
    print("  Stage 1: {}  Stage 2: {}  Stage 3: {}".format(
        len(cleaned_data), len(final_data), len(sft_data)))
    print("  Train={} Dev={} Test={}".format(
        len(train_data), len(dev_data), len(test_data)))


if __name__ == "__main__":
    main()