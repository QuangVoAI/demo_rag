"""Mine landmark phrases from MongoDB — grouped by province, safer regex."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import MONGODB_DATABASE, MONGODB_ROOMS_COLLECTION, MONGODB_URI
from pymongo import MongoClient

from room_assistant.landmark_aliases import _norm_landmark

PATTERNS = [
    (
        r"(?i)(?:gan|gần|cach|cách|khu vuc|khu vực|đối diện|doi dien|sát|sat)\s+"
        r"(?P<place>[^,\n\.]{3,50})",
        "near",
    ),
    (r"(?i)(?:duong|đường|d\.)\s*(?P<place>[^,\n\.]{3,40})", "street"),
    (r"(?i)(?:tttm|trung tam thuong mai|trung tâm thương mại)\s*(?P<place>[^,\n\.]{0,40})", "mall"),
    (r"(?i)(?:truong|trường|dai hoc|đại học|dh|đh)\s+(?P<place>[^,\n\.]{3,40})", "school"),
    (r"(?i)(?:benh vien|bệnh viện|bv)\s+(?P<place>[^,\n\.]{3,40})", "hospital"),
    (
        r"(?i)(?:\bchợ\b|\bcho\b(?!\s+thuê\b)(?!\s+thue\b)(?!\s+sinh\b)(?!\s+nu\b)(?!\s+nam\b))"
        r"\s+(?P<place>[^,\n\.]{3,30})",
        "market",
    ),
    (
        r"(?i)(vincom|aeon|lotte|big c|coopmart|saigon centre|landmark\s*81|estella|"
        r"sc vivo|thiso|gigamall|van hanh(?:\s+mall)?)\s*(?P<place>[^,\n\.]{0,30})",
        "brand_mall",
    ),
    (
        r"(?i)\b(tdtu|tdt|hutech|uel|ueh|hcmus|hcmut|huflit|hcmussh|ctu|dhct|hust|dut)\b",
        "uni_abbr",
    ),
    (r"(?i)\b(kcn|vsip|kcx)\s+(?P<place>[^,\n\.]{3,40})", "industrial"),
    (r"(?i)san bay\s+(?P<place>[^,\n\.]{3,30})", "airport"),
]

_STOP_NEAR_TAIL = re.compile(
    r"\b(?:\d+\s*(?:m|km|p|phut|phút)|thuan tien|tiện|thoai mai|o to|oto)\b.*$",
    re.IGNORECASE,
)

UNI_ABBRS = sorted(
    {"tdtu", "tdt", "hutech", "uel", "ueh", "hcmus", "hcmut", "huflit", "hcmussh", "ctu", "dhct", "hust", "dut"},
    key=len,
    reverse=True,
)


def _clean_phrase(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())[:80]
    text = _STOP_NEAR_TAIL.sub("", text).strip(" ,.;")
    return text


def _split_tien_ich(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;|]|\s+/\s+", str(value)) if part.strip()]


def main() -> None:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
    col = client[MONGODB_DATABASE][MONGODB_ROOMS_COLLECTION]
    counters: dict[str, Counter] = {key: Counter() for _, key in PATTERNS}
    room_sets: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    tien_ich = Counter()
    seen_rooms: dict[str, set] = defaultdict(set)

    cursor = col.find(
        {},
        {
            "embedding_text": 1,
            "tien_ich_xq": 1,
            "metadata.house_name": 1,
            "metadata.province_name": 1,
            "metadata.district_name": 1,
        },
    )

    for doc in cursor:
        room_id = str(doc.get("_id"))
        province = _norm_landmark((doc.get("metadata") or {}).get("province_name") or "unknown")
        text = " ".join(
            filter(
                None,
                [
                    doc.get("embedding_text"),
                    doc.get("tien_ich_xq"),
                    (doc.get("metadata") or {}).get("house_name"),
                ],
            )
        )
        if doc.get("tien_ich_xq"):
            for part in _split_tien_ich(str(doc["tien_ich_xq"])):
                if len(part) >= 3:
                    phrase = part[:80]
                    tien_ich[phrase] += 1
                    seen_rooms[f"tien:{phrase}"].add(room_id)
        for pat, key in PATTERNS:
            for match in re.finditer(pat, text):
                phrase = _clean_phrase(match.group("place") if "place" in match.groupdict() else match.group(0))
                if len(phrase) >= 3:
                    bucket = f"{province}::{phrase}"
                    counters[key][bucket] += 1
                    room_sets[key][bucket].add(room_id)

    print("=== tien_ich_xq top 30 (global) ===")
    for phrase, count in tien_ich.most_common(30):
        rooms = len(seen_rooms.get(f"tien:{phrase}", set()))
        print(f"{count:4d} occ | {rooms:4d} rooms | {phrase[:70]}")

    for key in counters:
        print(f"\n=== {key} top 20 by province ===")
        for bucket, count in counters[key].most_common(20):
            rooms = len(room_sets[key][bucket])
            print(f"{count:4d} occ | {rooms:4d} rooms | {bucket[:90]}")


def keyword_counts() -> None:
    keywords = [
        "tdtu", "ctu", "dhct", "hust", "dut", "huit", "hcmut", "vincom", "aeon",
        "landmark 81", "san bay can tho", "san bay phu quoc", "tan son nhat",
    ]
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
    col = client[MONGODB_DATABASE][MONGODB_ROOMS_COLLECTION]
    print("=== keyword room counts (tien_ich_xq or embedding) ===")
    for kw in keywords:
        n = col.count_documents({
            "$or": [
                {"embedding_text": {"$regex": re.escape(kw), "$options": "i"}},
                {"tien_ich_xq": {"$regex": re.escape(kw), "$options": "i"}},
            ],
        })
        if n:
            print(f"{n:4d} | {kw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine landmark phrases from MongoDB")
    parser.add_argument("--keywords", action="store_true", help="Print keyword room counts")
    args = parser.parse_args()
    main()
    if args.keywords:
        keyword_counts()
