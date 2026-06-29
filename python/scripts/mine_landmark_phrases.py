"""One-off: mine landmark/location phrases from MongoDB for alias building."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import MONGODB_DATABASE, MONGODB_ROOMS_COLLECTION, MONGODB_URI
from pymongo import MongoClient

PATTERNS = [
    (r"(?i)(?:gan|gần|cach|cách|khu vuc|khu vực|đối diện|doi dien|sát|sat)\s+([^,\n\.]{3,50})", "near"),
    (r"(?i)(?:duong|đường|d\.)\s*([^,\n\.]{3,40})", "street"),
    (r"(?i)(tttm|trung tam thuong mai|trung tâm thương mại)\s*([^,\n\.]{0,40})", "mall"),
    (r"(?i)(truong|trường|dai hoc|đại học|dh|đh)\s+([^,\n\.]{3,40})", "school"),
    (r"(?i)(benh vien|bệnh viện|bv)\s+([^,\n\.]{3,40})", "hospital"),
    (r"(?i)(cho|chợ)\s+([^,\n\.]{3,30})", "market"),
    (
        r"(?i)(vincom|aeon|lotte|big c|coopmart|saigon centre|landmark|estella|sc vivo|thiso|gigamall|van hanh|van hanh mall)\s*([^,\n\.]{0,30})",
        "brand_mall",
    ),
    (r"(?i)\b(tdtu|tdt|hutech|uel|ueh|hcmus|hcmut|ftu|ufm|huflit|hcmute|hcmussh|hcmussh|hcmussh|hcmussh)\b", "uni_abbr"),
]


def main() -> None:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    col = client[MONGODB_DATABASE][MONGODB_ROOMS_COLLECTION]
    counters = {key: Counter() for _, key in PATTERNS}
    tien_ich = Counter()

    cursor = col.find(
        {"$or": [{"metadata.status_code": "0"}, {"metadata.status_code": ""}]},
        {"embedding_text": 1, "tien_ich_xq": 1, "metadata.house_name": 1},
    )

    for doc in cursor:
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
            for part in re.split(r"[,;|/]", str(doc["tien_ich_xq"])):
                p = part.strip()
                if len(p) >= 3:
                    tien_ich[p[:80]] += 1
        for pat, key in PATTERNS:
            for m in re.finditer(pat, text):
                phrase = m.group(0).strip()[:80]
                counters[key][phrase] += 1

    print("=== tien_ich_xq top 100 ===")
    for phrase, count in tien_ich.most_common(100):
        print(f"{count:4d} | {phrase}")

    for key in counters:
        print(f"\n=== {key} top 50 ===")
        for phrase, count in counters[key].most_common(50):
            print(f"{count:4d} | {phrase}")


def keyword_counts() -> None:
    keywords = [
        "tdtu", "tdt", "ton duc thang", "hutech", "uel", "ueh", "hcmus", "hcmut",
        "van hien", "cong thuong", "huit", "fpt", "nguyen huu tho", "xvnt",
        "xuan vinh nguyen", "landmark 81", "vincom", "aeon", "lotte", "thiso",
        "sc vivo", "estella", "gigamall", "van hanh", "nguyen van cu",
        "cach mang thang 8", "le van luong", "phan van hon", "tan ky tan quy",
        "duong 3/2", "nguyen thi thap", "huynh tan phat", "dhqg", "rmtt", "rmit",
        "nttu", "nguyen tat thanh", "hoa sen", "hong bang", "gtvt", "y duoc",
        "pasteur", "benh vien cho ray", "cho ray",
    ]
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    col = client[MONGODB_DATABASE][MONGODB_ROOMS_COLLECTION]
    base = {"$or": [{"metadata.status_code": "0"}, {"metadata.status_code": ""}]}
    print("=== keyword room counts ===")
    for kw in keywords:
        n = col.count_documents({
            **base,
            "$or": [
                {"embedding_text": {"$regex": kw, "$options": "i"}},
                {"tien_ich_xq": {"$regex": kw, "$options": "i"}},
            ],
        })
        if n:
            print(f"{n:4d} | {kw}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine landmark phrases from MongoDB")
    parser.add_argument("--keywords", action="store_true", help="Print keyword room counts")
    args = parser.parse_args()
    main()
    if args.keywords:
        keyword_counts()
