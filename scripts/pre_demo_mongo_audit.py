"""Pre-demo audit: MongoDB data quality + live Q&A risk scan."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

sys.path.insert(0, str(ROOT / "python"))

from apps.rooms.services import get_collection, get_mongodb_db  # noqa: E402
from agents.graph import run_streaming  # noqa: E402

DEMO_QUESTIONS = [
    ("search_q7", "Tìm phòng ở quận 7 dưới 5 triệu"),
    ("search_q7_repeat", "Tìm phòng ở quận 7"),
    ("search_q7_street", "tìm phòng quận 7 ở đường Nguyễn Hữu Thọ"),
    ("search_tdtu", "Căn nào gần TDTU"),
    ("search_q1", "Tìm phòng quận 1 dưới 6 triệu"),
    ("refine_budget", "dưới 3 triệu thôi"),
    ("off_topic", "Viết code Python giúp tôi"),
    ("empathy", "Giá cao thế, sinh viên sao thuê nổi"),
    ("compare", "So sánh 2 phòng đầu tiên"),
    ("detail_no_context", "Phòng này cọc bao nhiêu?"),
    ("amenity", "Có nuôi mèo được không?"),
    ("multi_district", "Tìm phòng quận 7 hoặc quận 10"),
]

NO_RESULT_MARKERS = (
    "tìm mỏi mắt",
    "chưa thấy phòng nào khớp 100%",
    "không tìm thấy phòng phù hợp",
)


def audit_mongo_rooms() -> dict:
    col = get_collection("rooms")
    total = col.count_documents({})
    available = col.count_documents({
        "$or": [{"metadata.status_code": "0"}, {"metadata.status_code": ""}],
    })

    issues: list[dict] = []
    district_counts: Counter = Counter()
    missing_price = 0
    missing_district = 0
    missing_ward = 0
    missing_embedding = 0
    zero_price_available = 0
    q7_available = 0

    sample_cursor = col.find({}, {
        "room_id": 1,
        "metadata.price": 1,
        "metadata.district_name": 1,
        "metadata.ward_name": 1,
        "metadata.status_code": 1,
        "embedding_text": 1,
    }).limit(5000)

    for doc in sample_cursor:
        meta = doc.get("metadata") or {}
        district = str(meta.get("district_name") or "").strip()
        if district:
            district_counts[district.lower()] += 1
        else:
            missing_district += 1

        price = meta.get("price")
        status = str(meta.get("status_code") or "")
        is_avail = status in {"0", ""}
        if is_avail:
            if not price or int(price or 0) <= 0:
                zero_price_available += 1
            if "quận 7" in district.lower() or "quan 7" in district.lower():
                q7_available += 1

        if price is None or int(price or 0) <= 0:
            missing_price += 1
        if not str(meta.get("ward_name") or "").strip():
            missing_ward += 1
        if not str(doc.get("embedding_text") or "").strip():
            missing_embedding += 1

    top_districts = district_counts.most_common(15)

    if q7_available == 0:
        issues.append({
            "severity": "critical",
            "code": "no_q7_available",
            "message": "Không có phòng available nào ghi district Quận 7 — demo tìm Q7 sẽ fail.",
        })
    if zero_price_available > 0:
        issues.append({
            "severity": "warning",
            "code": "zero_price_available",
            "message": f"{zero_price_available} phòng available có price=0/null — bot có thể trả 'Liên hệ' hoặc lọc sai.",
            "count": zero_price_available,
        })
    if missing_district > total * 0.05:
        issues.append({
            "severity": "warning",
            "code": "missing_district",
            "message": f"{missing_district}/{total} phòng thiếu district_name — lọc theo quận có thể miss.",
            "count": missing_district,
        })
    if missing_embedding > total * 0.1:
        issues.append({
            "severity": "warning",
            "code": "missing_embedding_text",
            "message": f"{missing_embedding} phòng thiếu embedding_text — Qdrant/landmark search yếu.",
            "count": missing_embedding,
        })

    return {
        "total_rooms": total,
        "available_rooms": available,
        "q7_available": q7_available,
        "missing_price": missing_price,
        "missing_district": missing_district,
        "missing_ward": missing_ward,
        "missing_embedding_text": missing_embedding,
        "top_districts": top_districts,
        "issues": issues,
    }


def audit_contacts_chat() -> dict:
    contacts = get_collection("contacts")
    chats = get_collection("chat_history")
    issues: list[dict] = []

    contact_count = contacts.count_documents({})
    chat_count = chats.count_documents({})
    dup_phones = list(contacts.aggregate([
        {"$match": {"phone_number": {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": "$phone_number", "n": {"$sum": 1}}},
        {"$match": {"n": {"$gt": 1}}},
        {"$limit": 10},
    ]))

    if dup_phones:
        issues.append({
            "severity": "warning",
            "code": "duplicate_phone_contacts",
            "message": f"Có {len(dup_phones)} SĐT trùng trong contacts (unique index có thể fail khi upsert).",
            "samples": dup_phones[:3],
        })

    empty_threads = chats.count_documents({"$or": [{"messages": {"$exists": False}}, {"messages": {"$size": 0}}]})

    return {
        "contacts": contact_count,
        "chat_history": chat_count,
        "empty_threads": empty_threads,
        "duplicate_phone_samples": dup_phones[:5],
        "issues": issues,
    }


def _check_answer(name: str, question: str, result: dict) -> list[dict]:
    flags: list[dict] = []
    answer = (result.get("answer") or "").lower()
    rooms = result.get("rooms") or []
    intent = result.get("intent") or ""
    abstain = bool(result.get("abstain"))

    if name.startswith("search_") and not name.endswith("_repeat"):
        if not rooms and any(m in answer for m in NO_RESULT_MARKERS):
            flags.append({
                "severity": "high",
                "case": name,
                "question": question,
                "issue": "Trả template 0 kết quả dù Mongo có data.",
            })
        if rooms:
            for room in rooms[:5]:
                price = room.get("rent_price") or 0
                if "dưới 5 triệu" in question.lower() and price > 5_500_000:
                    flags.append({
                        "severity": "high",
                        "case": name,
                        "question": question,
                        "issue": f"Phòng {room.get('room_id')} giá {price} vượt ngân sách 5tr.",
                    })
                if "dưới 3 triệu" in question.lower() and price > 3_200_000:
                    flags.append({
                        "severity": "high",
                        "case": name,
                        "question": question,
                        "issue": f"Phòng {room.get('room_id')} giá {price} vượt ngân sách 3tr.",
                    })

    if name == "off_topic" and "phòng trọ" not in answer:
        flags.append({"severity": "medium", "case": name, "question": question, "issue": "Không từ chối off-topic đúng."})

    if name == "detail_no_context" and not abstain and "cọc" in answer and not rooms:
        flags.append({
            "severity": "high",
            "case": name,
            "question": question,
            "issue": "Trả lời cọc khi chưa chọn phòng — nguy cơ hallucination.",
            "answer_snippet": answer[:200],
        })

    if name == "compare" and rooms and "so sánh" not in answer and "rẻ hơn" not in answer and "ưu điểm" not in answer:
        flags.append({"severity": "medium", "case": name, "question": question, "issue": "Compare intent nhưng câu trả lời không giống so sánh."})

    if abstain:
        flags.append({
            "severity": "info",
            "case": name,
            "question": question,
            "issue": f"Abstain: {result.get('abstain_reason') or 'unknown'}",
        })

    return flags


async def audit_live_qa(session_id: str) -> dict:
    history: list[dict] = []
    turns: list[dict] = []
    flags: list[dict] = []

    for name, question in DEMO_QUESTIONS:
        try:
            result = await run_streaming(question=question, history=history, session_id=session_id)
        except Exception as exc:
            flags.append({
                "severity": "critical",
                "case": name,
                "question": question,
                "issue": f"Exception: {exc}",
            })
            continue

        turn_flags = _check_answer(name, question, result)
        flags.extend(turn_flags)
        turns.append({
            "case": name,
            "question": question,
            "intent": result.get("intent"),
            "room_count": len(result.get("rooms") or []),
            "abstain": bool(result.get("abstain")),
            "relaxed": any(bool(r.get("relaxed_search")) for r in (result.get("rooms") or [])),
            "answer_preview": (result.get("answer") or "")[:240],
            "room_ids": [r.get("room_id") for r in (result.get("rooms") or [])[:5]],
        })
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.get("answer") or ""})

    return {"session_id": session_id, "turns": turns, "flags": flags}


async def main() -> int:
    db_name = get_mongodb_db().name
    print(f"Mongo DB: {db_name}")

    mongo_rooms = audit_mongo_rooms()
    mongo_chat = audit_contacts_chat()
    session_id = f"pre-demo-{uuid.uuid4().hex[:8]}"
    print(f"Live Q&A session: {session_id} (sequential, ~2-3 min)...")
    live_qa = await audit_live_qa(session_id)

    all_issues = (
        mongo_rooms.get("issues", [])
        + mongo_chat.get("issues", [])
        + live_qa.get("flags", [])
    )
    by_severity = Counter(item.get("severity", "?") for item in all_issues)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mongodb_database": db_name,
        "mongo_rooms": mongo_rooms,
        "mongo_chat": mongo_chat,
        "live_qa": live_qa,
        "summary": {
            "total_flags": len(all_issues),
            "by_severity": dict(by_severity),
            "critical_high": sum(1 for i in all_issues if i.get("severity") in {"critical", "high"}),
        },
    }

    out_json = ROOT / "pre-demo-audit.json"
    out_md = ROOT / "pre-demo-audit.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Pre-demo audit (Mongo + live Q&A)",
        f"\nDB: `{db_name}` | {report['generated_at']}",
        f"\nPhòng: **{mongo_rooms['total_rooms']}** total, **{mongo_rooms['available_rooms']}** available, **{mongo_rooms['q7_available']}** Q7 available",
        f"\nContacts: {mongo_chat['contacts']} | Chat threads: {mongo_chat['chat_history']}",
        f"\nFlags: **{report['summary']['critical_high']}** critical/high / {report['summary']['total_flags']} total",
        "",
        "## Top districts (available sample)",
    ]
    for dist, cnt in mongo_rooms.get("top_districts", [])[:10]:
        lines.append(f"- {dist}: {cnt}")

    lines.append("\n## Live Q&A turns")
    for t in live_qa["turns"]:
        icon = "⚠️" if t["room_count"] == 0 and t["case"].startswith("search") else "✅"
        lines.append(f"- {icon} **{t['case']}**: {t['question'][:50]} → {t['room_count']} phòng, intent={t['intent']}, abstain={t['abstain']}")
        if t.get("room_ids"):
            lines.append(f"  - IDs: {', '.join(str(x) for x in t['room_ids'])}")

    if all_issues:
        lines.append("\n## Issues / risks")
        for item in all_issues:
            lines.append(f"- **[{item.get('severity')}]** {item.get('case') or item.get('code')}: {item.get('issue') or item.get('message')}")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Report: {out_md}")
    return 0 if report["summary"]["critical_high"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
