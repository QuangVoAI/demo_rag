#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Data Augmentation Script for MyKingdom Policies
Generates synthetic JSONL entries using Vertex AI (Gemini 2.5 Flash)

Usage:
    cd python/data_processing
    python rag_augment.py                    # full run
    python rag_augment.py --fix-only         # only fix policy file encoding
    python rag_augment.py --entries N        # N entries per section (default: 3)
    python rag_augment.py --dry-run          # print prompts, no API calls
"""

import argparse
import concurrent.futures as _futures
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Windows UTF-8 stdout fix ──────────────────────────────────────────────────
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
    from vertexai.generative_models import GenerativeModel
except ImportError:
    print("ERROR: vertexai not installed.  pip install vertexai")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "empathai-494308"
LOCATION   = "us-central1"
MODEL_NAME = "gemini-2.5-flash"

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
POLICY_FILE  = DATA_DIR / "mykingdom_policies.json"
OUTPUT_FILE  = DATA_DIR / "mykingdom_rag_augmented.jsonl"

GIT_POLICY_REF = "fdfd7c6:data/mykingdom_policies.json"

DEFAULT_ENTRIES_PER_SECTION = 3
CROSS_SCENARIO_PAIRS = [
    ("dieu_kien_dieu_khoan_thanh_vien", "chinh_sach_bao_hanh_doi_tra"),
    ("chinh_sach_giao_hang",            "dieu_kien_dieu_khoan_thanh_vien"),
    ("chinh_sach_dong_goi_kiem_hang",   "chinh_sach_bao_hanh_doi_tra"),
    ("chinh_sach_thanh_toan",           "chinh_sach_bao_hanh_doi_tra"),
    ("dieu_kien_dieu_khoan_thanh_vien", "chinh_sach_giao_hang"),
]
API_DELAY   = 2    # seconds between calls
MAX_RETRIES = 3
TIMEOUT_S   = 60

GEN_CFG = {
    "temperature": 0.8,
    "max_output_tokens": 8192,
}

# ── Encoding fix ──────────────────────────────────────────────────────────────

def fix_policy_encoding() -> bool:
    """
    Restore mykingdom_policies.json from git with correct UTF-8 encoding,
    then apply MilkyBloom → MyKingdom brand replacement.
    Returns True if fix was applied, False if file already looks valid.
    """
    try:
        with open(POLICY_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        json.loads(content)
        if 'MyKingdom' in content and 'Công ty' in content:
            print("[encoding] Policy file looks valid — no fix needed.")
            return False
    except Exception:
        pass

    print("[encoding] Restoring clean copy from git ref {}...".format(GIT_POLICY_REF))
    result = subprocess.run(
        ['git', 'show', GIT_POLICY_REF],
        capture_output=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("ERROR: git show failed: {}".format(result.stderr.decode('utf-8', errors='replace')))
        return False

    content = result.stdout.decode('utf-8')

    replacements = [
        ('MilkyBloom', 'MyKingdom'),
        ('hotro@MyKingdom.com.vn', 'hotro@mykingdom.com.vn'),
        ('https://www.MyKingdom.com.vn', 'https://www.mykingdom.com.vn'),
        ('"company": "Công ty Cổ phần Việt Tinh Anh"',
         '"company": "Công ty Cổ phần Thương mại Quốc tế MyKingdom"'),
    ]
    for old, new in replacements:
        content = content.replace(old, new)

    with open(POLICY_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    print("[encoding] Policy file fixed and saved as UTF-8.")
    return True


# ── Vertex AI client ──────────────────────────────────────────────────────────

def get_model() -> GenerativeModel:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    print("[vertex] Ready: {}/{}/{}".format(PROJECT_ID, LOCATION, MODEL_NAME))
    return model


def _generate_with_timeout(model: GenerativeModel, prompt: str, timeout: int = TIMEOUT_S):
    executor = _futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(model.generate_content, prompt, generation_config=GEN_CFG)
    executor.shutdown(wait=False)
    try:
        return future.result(timeout=timeout)
    except _futures.TimeoutError:
        raise TimeoutError("generate_content timed out after {}s".format(timeout))


def _strip_markdown(text: str) -> str:
    import re
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json_array(raw: str) -> List[Dict]:
    cleaned = _strip_markdown(raw)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    import re
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON array found in response:\n{}".format(cleaned[:300]))


# ── Prompt builders ───────────────────────────────────────────────────────────

def _section_prompt(
    policy_id: str,
    policy_title: str,
    heading: str,
    content: str,
    n: int,
    id_start: int,
    existing_ids: List[str],
) -> str:
    prefix = "RAG_{}_".format(policy_id.upper()[:5].replace("_", ""))
    ids_hint = ", ".join("{}{}".format(prefix, str(id_start + i).zfill(3)) for i in range(n))
    existing_str = ", ".join(existing_ids[-10:]) if existing_ids else "chưa có"
    return (
        "Bạn là Data Engineer tạo dữ liệu RAG cho chatbot CSKH của MyKingdom (chuỗi cửa hàng đồ chơi).\n"
        "Nhiệm vụ: CHỈ trích xuất ĐIỀU KIỆN và KẾT QUẢ cứng (If-Then rules). "
        "TUYỆT ĐỐI không thêm câu xin lỗi, xoa dịu, hay hướng dẫn cảm xúc.\n\n"
        "CHÍNH SÁCH: {title}\n"
        "MỤC: {heading}\n"
        "NỘI DUNG:\n{content}\n\n"
        "Tạo ĐÚNG {n} RAG entries dưới dạng JSON ARRAY. "
        "Mỗi entry khai thác một khía cạnh/edge case KHÁC NHAU (ưu tiên trường hợp TỪ CHỐI / NGOẠI LỆ).\n"
        "context_id gợi ý: {ids}\n"
        "context_id đã tồn tại (không dùng lại): {existing}\n\n"
        "Cấu trúc mỗi object:\n"
        '{{\n'
        '  "context_id": "string",\n'
        '  "topic": "string ngắn gọn",\n'
        '  "refined_knowledge": "Bối cảnh + If-Then rules cứng, không có câu xoa dịu",\n'
        '  "expected_action": "Chấp nhận / Từ chối / Yêu cầu thêm thông tin + lý do",\n'
        '  "synthetic_queries": ["5 câu đa dạng: chuẩn + vội/giận + từ lóng (ship/freeship/bóc seal/unbox/hộp mù/mã giảm giá)"],\n'
        '  "metadata": {{\n'
        '    "urgency": "high/medium/low",\n'
        '    "linked_policies": ["{pid}"],\n'
        '    "tags": ["keyword", "slang"]\n'
        '  }}\n'
        '}}\n\n'
        "Output: chỉ JSON array [{n} objects], không có text giải thích hay markdown."
    ).format(
        title=policy_title, heading=heading, content=content,
        n=n, ids=ids_hint, existing=existing_str, pid=policy_id,
    )


def _cross_prompt(
    p1: Dict, p2: Dict, n: int, id_start: int, existing_ids: List[str]
) -> str:
    existing_str = ", ".join(existing_ids[-10:]) if existing_ids else "chưa có"
    ids_hint = ", ".join("RAG_CROSS_{}".format(str(id_start + i).zfill(3)) for i in range(n))
    return (
        "Bạn là Data Engineer tạo dữ liệu RAG cho chatbot CSKH của MyKingdom.\n"
        "Tạo {n} kịch bản GIAO THOA (cross-scenario) giữa 2 chính sách sau:\n\n"
        "Chính sách A ({pid1}): {sum1}\n"
        "Chính sách B ({pid2}): {sum2}\n\n"
        "Ví dụ kịch bản giao thoa:\n"
        "  - Khách dùng voucher sinh nhật mua POP MART → bị từ chối → muốn đổi sang SP khác\n"
        "  - Khách Diamond đặt ship hỏa tốc → hỏi freeship\n"
        "  - Khách mua bằng điểm MyPoints → quá 7 ngày → xử lý thế nào\n"
        "  - Khách không có video unbox → quá 72h → yêu cầu đổi hàng lỗi\n\n"
        "context_id gợi ý: {ids}\n"
        "context_id đã tồn tại (không dùng lại): {existing}\n\n"
        "Cấu trúc mỗi object (JSON array {n} items):\n"
        '{{\n'
        '  "context_id": "string",\n'
        '  "topic": "Kịch bản: [mô tả ngắn]",\n'
        '  "refined_knowledge": "If-Then rules cho kịch bản giao thoa, không có câu xoa dịu",\n'
        '  "expected_action": "Hành động cuối cùng hệ thống cần thực hiện",\n'
        '  "synthetic_queries": ["5 câu đa dạng phản ánh kịch bản phức tạp"],\n'
        '  "metadata": {{\n'
        '    "urgency": "high/medium/low",\n'
        '    "linked_policies": ["{pid1}", "{pid2}"],\n'
        '    "tags": ["cross-scenario", "tag1", "tag2"]\n'
        '  }}\n'
        '}}\n\n'
        "Output: chỉ JSON array, không có text hay markdown."
    ).format(
        n=n, pid1=p1["id"], sum1=p1.get("summary", ""),
        pid2=p2["id"], sum2=p2.get("summary", ""),
        ids=ids_hint, existing=existing_str,
    )


# ── Generation helpers ────────────────────────────────────────────────────────

def load_existing_ids() -> List[str]:
    if not OUTPUT_FILE.exists():
        return []
    ids = []
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.append(obj.get("context_id", ""))
            except json.JSONDecodeError:
                pass
    return ids


def _call_with_retry(
    model: GenerativeModel,
    prompt: str,
    label: str,
    dry_run: bool = False,
) -> Optional[List[Dict]]:
    if dry_run:
        print("[dry-run] Prompt for {}:\n{}\n".format(label, prompt[:200]))
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _generate_with_timeout(model, prompt)
            entries = _parse_json_array(response.text)
            print("  [ok] {} → {} entries".format(label, len(entries)))
            return entries
        except TimeoutError:
            print("  [timeout] {} attempt {}/{}".format(label, attempt, MAX_RETRIES))
        except Exception as e:
            print("  [error] {} attempt {}/{}: {}".format(label, attempt, MAX_RETRIES, e))
            if attempt < MAX_RETRIES:
                traceback.print_exc()
        time.sleep(API_DELAY * attempt)
    return None


def _append_entries(entries: List[Dict], label: str) -> int:
    written = 0
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not entry.get("context_id") or not entry.get("refined_knowledge"):
                print("  [skip] Malformed entry in {}".format(label))
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            written += 1
    return written


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_augmentation(
    entries_per_section: int = DEFAULT_ENTRIES_PER_SECTION,
    dry_run: bool = False,
) -> None:
    with open(POLICY_FILE, 'r', encoding='utf-8') as f:
        policies_data = json.load(f)

    policies: List[Dict] = policies_data.get("policies", [])
    print("[info] Loaded {} policies from {}".format(len(policies), POLICY_FILE.name))

    model = None if dry_run else get_model()

    existing_ids = load_existing_ids()
    print("[info] {} existing entries in output file".format(len(existing_ids)))

    total_written = 0
    id_counter = len(existing_ids) + 100

    # ── Phase 1: Per-section augmentation ─────────────────────────────────────
    print("\n[phase 1] Per-section augmentation ({} entries/section)".format(entries_per_section))
    for policy in policies:
        pid   = policy.get("id", "unknown")
        title = policy.get("title", "")
        sections = policy.get("sections", [])
        print("\n  Policy: {}".format(pid))

        for section in sections:
            heading = section.get("heading", "")
            content = section.get("content", "")
            label   = "{} / {}".format(pid, heading[:40])

            prompt = _section_prompt(
                policy_id=pid,
                policy_title=title,
                heading=heading,
                content=content,
                n=entries_per_section,
                id_start=id_counter,
                existing_ids=existing_ids,
            )
            entries = _call_with_retry(model, prompt, label, dry_run)
            if entries:
                written = _append_entries(entries, label)
                total_written += written
                existing_ids.extend(e.get("context_id", "") for e in entries)
                id_counter += written

            if not dry_run:
                time.sleep(API_DELAY)

    # ── Phase 2: Cross-scenario augmentation ──────────────────────────────────
    print("\n[phase 2] Cross-scenario augmentation ({} pairs)".format(len(CROSS_SCENARIO_PAIRS)))
    policy_map = {p["id"]: p for p in policies}

    for pid1, pid2 in CROSS_SCENARIO_PAIRS:
        p1 = policy_map.get(pid1)
        p2 = policy_map.get(pid2)
        if not p1 or not p2:
            print("  [skip] Unknown policy id: {} or {}".format(pid1, pid2))
            continue

        label = "CROSS: {} x {}".format(pid1[:20], pid2[:20])
        prompt = _cross_prompt(p1, p2, n=2, id_start=id_counter, existing_ids=existing_ids)
        entries = _call_with_retry(model, prompt, label, dry_run)
        if entries:
            written = _append_entries(entries, label)
            total_written += written
            existing_ids.extend(e.get("context_id", "") for e in entries)
            id_counter += written

        if not dry_run:
            time.sleep(API_DELAY)

    print("\n[done] Total new entries written: {}".format(total_written))
    print("[done] Output: {}".format(OUTPUT_FILE))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG augmentation for MyKingdom policies")
    parser.add_argument("--fix-only",  action="store_true",
                        help="Only fix policy file encoding, skip augmentation")
    parser.add_argument("--entries",   type=int, default=DEFAULT_ENTRIES_PER_SECTION,
                        help="Entries to generate per section (default: {})".format(
                            DEFAULT_ENTRIES_PER_SECTION))
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print prompts without making API calls")
    args = parser.parse_args()

    print("=" * 60)
    print("MyKingdom RAG Augmentation Script")
    print("=" * 60)

    fix_policy_encoding()

    if args.fix_only:
        print("[done] Encoding fix complete.")
        return

    run_augmentation(entries_per_section=args.entries, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
