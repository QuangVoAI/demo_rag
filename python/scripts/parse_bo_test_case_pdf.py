"""Parse Bo_Test_Case_Chat.pdf text extract into JSON fixtures."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = ROOT / "python" / "tests" / "_bo_test_case_extracted.txt"
OUT = ROOT / "python" / "tests" / "fixtures" / "bo_test_case_chat.json"


def _extract_sends(body: str) -> list[str]:
    sends: list[str] = []
    patterns = [
        r'(?:Gửi|gửi|B1 gửi|B2 gửi|B3 gửi)\s*:\s*"([^"]+)"',
        r'(?:Gửi|gửi|B1 gửi|B2 gửi|B3 gửi)\s*:\s*“([^”]+)”',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, body, flags=re.IGNORECASE):
            msg = re.sub(r"\s+", " ", match.group(1)).strip()
            if msg and msg not in sends:
                sends.append(msg)

    if sends:
        return sends

    # Multiline: Gửi: then quoted lines broken across PDF layout
    for match in re.finditer(
        r"(?:Gửi|gửi|B1 gửi|B2 gửi|B3 gửi)\s*:\s*\n([\s\S]*?)(?:\nĐạt|\nTC_CHAT_|\Z)",
        body,
        flags=re.IGNORECASE,
    ):
        chunk = match.group(1)
        chunk = chunk.replace('"', "").replace("“", "").replace("”", "")
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk and len(chunk) >= 4:
            if chunk not in sends:
                sends.append(chunk)
    return sends


def parse_cases(text: str) -> list[dict]:
    normalized = re.sub(r"TC_CHAT_0\s*\n\s*(\d{2})", r"TC_CHAT_\1", text)
    parts = re.split(r"(?=TC_CHAT_\d{2}\s)", normalized)
    cases: list[dict] = []
    for part in parts:
        match = re.match(r"TC_CHAT_(\d{2})\s*(.*)", part, re.DOTALL)
        if not match:
            continue
        num, body = match.group(1), match.group(2)
        case_id = f"TC_CHAT_{num}"
        group_match = re.search(r"\n([A-Za-z]+)\s*\n", body)
        group = group_match.group(1) if group_match else ""
        sends = _extract_sends(body)
        expected_note = ""
        if "Đạt" in body:
            note_match = re.search(r"Đạt\s*\n([\s\S]*?)(?:\nTC_CHAT_|\Z)", body)
            if note_match:
                expected_note = re.sub(r"\s+", " ", note_match.group(1)[:800]).strip()
        cases.append({
            "id": case_id,
            "group": group,
            "sends": sends,
            "expected_note": expected_note,
            "ui_only": group in {"Smoke", "Identity", "Validation", "Session", "Streaming", "Payload", "UI", "Layout"},
        })
    return cases


def main() -> int:
    if not EXTRACT.exists():
        print(f"Missing extract: {EXTRACT}", file=sys.stderr)
        return 1
    text = EXTRACT.read_text(encoding="utf-8")
    cases = parse_cases(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    with_sends = sum(1 for item in cases if item["sends"])
    print(f"Wrote {len(cases)} cases ({with_sends} with send messages) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
