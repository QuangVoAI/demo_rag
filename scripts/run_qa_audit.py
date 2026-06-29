"""Full Q&A audit: replay production transcript + logic checks, emit JSON report."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Transcript from production test (Jun 2026)
TRANSCRIPT_TURNS = [
    "tìm cho tôi phòng quận 7 ở đường Nguyễn Hữu Thọ",
    "Có gợi ý phòng nào gần đó k?",
    "Vậy phòng ở phường tân hưng",
    "Tìm phòng ở quận 7",
    "Ngân sách 4tr",
    "Tìm cho tôi phòng ở quận 1",
    "Tìm cho tôi phòng ở quận 7 dưới 5 triệu",
    "dưới 4 triệu đi",
    "Căn nào mà gần TDTU á",
]

EXTRA_SCENARIOS = [
  {
    "name": "off_topic_code",
    "turns": ["Viết code Python giúp tôi"],
    "expect": lambda r: "phòng trọ" in r["answer"].lower() and "code" in r["answer"].lower(),
    "note": "Từ chối off-topic, quay về tư vấn phòng",
  },
  {
    "name": "empathy_price",
    "turns": ["Giá cao thế, sinh viên sao thuê nổi"],
    "expect": lambda r: "hiểu" in r["answer"].lower() or "ngân sách" in r["answer"].lower(),
    "note": "Phản hồi đồng cảm khi phàn nàn giá",
  },
  {
    "name": "compare_rooms",
    "turns": [
      "Tìm cho tôi phòng ở quận 7 dưới 5 triệu",
      "So sánh 3 phòng đầu tiên",
    ],
    "expect": lambda r: "so sánh" in r["answer"].lower() or "rẻ hơn" in r["answer"].lower() or "ưu điểm" in r["answer"].lower(),
    "note": "So sánh phòng sau khi đã có kết quả",
  },
  {
    "name": "landmark_tdt_full_name",
    "turns": ["mình muốn tìm phòng ở gần trường đại học TDT"],
    "expect": lambda r: bool(r.get("rooms")) and not any(
        marker in (r.get("answer") or "").lower() for marker in NO_RESULT_MARKERS
    ),
    "note": "Tên đầy đủ 'trường đại học TDT' phải khớp phòng ghi Gan TDTU trong DB",
  },
]

NO_RESULT_MARKERS = (
    "tìm mỏi mắt",
    "chưa thấy phòng nào khớp 100%",
    "không tìm thấy phòng phù hợp",
)


@dataclass
class TurnResult:
    scenario: str
    turn_index: int
    question: str
    answer: str
    room_count: int
    room_ids: list[str]
    intent: str
    abstain: bool
    relaxed_search: bool
    constraints: dict
    processing_time_ms: int
    issues: list[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class ScenarioReport:
    name: str
    turns: list[TurnResult]
    passed: bool
    notes: str = ""


def _location(state: dict) -> dict:
    return ((state or {}).get("constraints") or {}).get("location") or {}


def _is_no_result(answer: str, rooms: list) -> bool:
    text = (answer or "").lower()
    if rooms:
        return False
    return any(marker in text for marker in NO_RESULT_MARKERS)


def _audit_transcript_turn(turn_idx: int, question: str, result: dict) -> list[str]:
    issues: list[str] = []
    answer = result.get("answer") or ""
    rooms = result.get("rooms") or []
    loc = _location(result.get("session_state") or {})

    if turn_idx == 1:
        if _is_no_result(answer, rooms):
            issues.append("Tìm quận 7 + đường Nguyễn Hữu Thọ không trả phòng dù có data Q7.")
        landmarks = [str(x).lower() for x in (loc.get("near_landmarks") or [])]
        if any(x in {"do", "đó"} for x in landmarks):
            issues.append('Landmark rác "đó/do" từ câu "gần đó" vẫn còn trong session.')

    if turn_idx == 3:
        if _is_no_result(answer, rooms):
            issues.append("Tìm theo phường Tân Hưng (quận 7) không trả phòng dù DB có data Q7.")

    if turn_idx == 3:
        wards = [str(x).lower() for x in (loc.get("wards") or [])]
        if not wards and _is_no_result(answer, rooms):
            issues.append("Không parse/ghi ward 'tân hưng' vào constraints.")

    if turn_idx == 4:
        if _is_no_result(answer, rooms):
            issues.append(
                "REGRESSION: 'Tìm phòng ở quận 7' trả 0 kết quả — lọc vị trí cũ (ward/landmark) chưa được xóa."
            )
        stale_landmarks = loc.get("near_landmarks") or []
        stale_wards = loc.get("wards") or []
        if rooms and (stale_landmarks or stale_wards):
            issues.append(
                f"Trả được phòng nhưng session vẫn giữ ward/landmark cũ: wards={stale_wards}, landmarks={stale_landmarks}"
            )

    if turn_idx == 6:
        districts = [str(x).lower() for x in (loc.get("districts") or [])]
        if "quan 1" not in " ".join(districts) and "quận 1" not in " ".join(districts):
            if not _is_no_result(answer, rooms):
                pass
            elif not districts:
                issues.append("Đổi sang quận 1 nhưng constraints district trống.")

    if turn_idx == 7:
        if _is_no_result(answer, rooms):
            issues.append("Câu có quận 7 + ngân sách 5 triệu phải trả phòng (đã từng OK trên prod).")
        prices = [r.get("rent_price") for r in rooms if r.get("rent_price")]
        if prices and any(p > 5_500_000 for p in prices):
            issues.append(f"Có phòng vượt ngân sách 5 triệu: {prices[:3]}")

    if turn_idx == 8:
        if rooms:
            over = [r for r in rooms if (r.get("rent_price") or 0) > 4_200_000]
            if over:
                issues.append(f"Lọc 'dưới 4 triệu' nhưng có phòng >4.2tr: {[x.get('room_id') for x in over[:3]]}")
        elif _is_no_result(answer, rooms):
            issues.append("Refine 'dưới 4 triệu' không trả phòng dù YUHOME 2.3tr đã từng match.")

    if turn_idx == 9:
        if _is_no_result(answer, rooms):
            issues.append(
                "REGRESSION: 'gần TDTU á' trả 0 kết quả — landmark TDTU nên relax trong quận 7."
            )
        landmarks = [str(x).lower() for x in (loc.get("near_landmarks") or [])]
        if any("tdtu a" == lm or lm == "a" for lm in landmarks):
            issues.append(f"Landmark TDTU bị suffix rác: {landmarks}")

    return issues


async def _run_turn(question: str, history: list[dict], session_id: str, **kwargs) -> dict:
    if kwargs.get("use_streaming"):
        from agents.graph import run_streaming
        return await run_streaming(question=question, history=history, session_id=session_id)

    from room_assistant.session_store import InMemorySessionStore
    from room_assistant.workflow import run_room_assistant

    return await run_room_assistant(
        question=question,
        history=history,
        session_id=session_id,
        repository=kwargs.get("repository"),
        session_store=kwargs.get("session_store") or InMemorySessionStore(),
        semantic_index=None,
    )


def _build_local_q7_repository():
    from room_assistant.repository import InMemoryRoomRepository

    def room(
        room_id: str,
        price: int,
        district: str = "Quận 7",
        ward: str = "Phú Thuận",
        street: str = "",
        extras: str = "",
    ) -> dict:
        address = street or f"Địa chỉ {ward}, {district}"
        return {
            "room_id": room_id,
            "metadata": {
                "house_name": room_id.split("-")[0].strip(),
                "room_code": room_id,
                "price": price,
                "status_code": "0",
                "district_name": district,
                "ward_name": ward,
            },
            "embedding_text": f"## Thông tin nhà\n- Địa chỉ: {address}\n{extras}",
            "available": True,
            "status": "active",
            "title": room_id,
        }

    rooms = [
        room("61ea636e3048d576be90729b", 4_800_000, ward="Phú Thuận", street="C2 - C3 Hoàng Quốc Việt, Phường Phú Thuận, Quận 7"),
        room("61ea636e3048d576be90729f", 4_800_000, ward="Phú Thuận", street="C2 - C3 Hoàng Quốc Việt, Phường Phú Thuận, Quận 7"),
        room("62963aae137e2a3d7e03c9d0", 2_300_000, ward="Phú Mỹ", street="73H Lê Văn Lương, Phường Phú Mỹ, Quận 7", extras="Gan TDTU, di hoc tien"),
        room("62cc0e17ff6aae63cefbeda7", 4_500_000, ward="Tân Quy", street="Số 1 đường 71, Phường Tân Quy, Quận 7"),
        room("6541fa2fdea752782de864ae", 4_200_000, ward="Phú Mỹ", street="1283/17c Huỳnh Tấn Phát, Phường Phú Mỹ, Quận 7"),
        room("6399d72f07985f204285aed9", 3_400_000, ward="Bình Thuận", street="17/21A Tân Thuận Tây, Phường Bình Thuận, Quận 7"),
        room("6399d72f07985f204285aee4", 3_400_000, ward="Bình Thuận", street="17/21A Tân Thuận Tây, Phường Bình Thuận, Quận 7"),
        room("639ee917c9d7e004afed99ce", 3_900_000, ward="Bình Thuận", street="300/23/23A NGUYỄN VĂN LINH, Phường Bình Thuận, Quận 7"),
        room("635b410eef688d562f34f0e0", 3_800_000, ward="Phú Thuận", street="945 Huỳnh Tấn Phát, Phường Phú Thuận, Quận 7"),
        room("nguyen-huu-tho-q7", 4_600_000, ward="Tân Hưng", street="Nguyễn Hữu Thọ, Phường Tân Hưng, Quận 7"),
        room("q1-studio", 4_900_000, district="Quận 1", ward="Bến Nghé", street="Quận 1, TP.HCM"),
    ]
    return InMemoryRoomRepository(rooms)


async def replay_transcript(session_id: str, run_kwargs: dict) -> ScenarioReport:
    history: list[dict] = []
    turns: list[TurnResult] = []

    for idx, question in enumerate(TRANSCRIPT_TURNS, start=1):
        started = time.time()
        result = await _run_turn(question, history, session_id, **run_kwargs)
        elapsed = int((time.time() - started) * 1000)
        answer = result.get("answer") or ""
        rooms = result.get("rooms") or []
        issues = _audit_transcript_turn(idx, question, result)
        tr = TurnResult(
            scenario="production_transcript",
            turn_index=idx,
            question=question,
            answer=answer[:500],
            room_count=len(rooms),
            room_ids=[str(r.get("room_id") or r.get("id") or "") for r in rooms[:5]],
            intent=str(result.get("intent") or ""),
            abstain=bool(result.get("abstain")),
            relaxed_search=any(bool(r.get("relaxed_search")) for r in rooms),
            constraints=_location(result.get("session_state") or {}),
            processing_time_ms=elapsed,
            issues=issues,
            passed=not issues,
        )
        turns.append(tr)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

    return ScenarioReport(
        name="production_transcript",
        turns=turns,
        passed=all(t.passed for t in turns),
        notes="Replay 9 lượt từ transcript test thực tế (Jun 2026)",
    )


async def replay_extra(session_id: str, spec: dict, run_kwargs: dict) -> ScenarioReport:
    history: list[dict] = []
    turns: list[TurnResult] = []
    for idx, question in enumerate(spec["turns"], start=1):
        started = time.time()
        result = await _run_turn(question, history, session_id, **run_kwargs)
        elapsed = int((time.time() - started) * 1000)
        answer = result.get("answer") or ""
        rooms = result.get("rooms") or []
        issues: list[str] = []
        if idx == len(spec["turns"]) and not spec["expect"](result):
            issues.append(f"Không đạt kỳ vọng: {spec['note']}")
        tr = TurnResult(
            scenario=spec["name"],
            turn_index=idx,
            question=question,
            answer=answer[:500],
            room_count=len(rooms),
            room_ids=[str(r.get("room_id") or "") for r in rooms[:5]],
            intent=str(result.get("intent") or ""),
            abstain=bool(result.get("abstain")),
            relaxed_search=any(bool(r.get("relaxed_search")) for r in rooms),
            constraints=_location(result.get("session_state") or {}),
            processing_time_ms=elapsed,
            issues=issues,
            passed=not issues,
        )
        turns.append(tr)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
    return ScenarioReport(
        name=spec["name"],
        turns=turns,
        passed=all(t.passed for t in turns),
        notes=spec.get("note", ""),
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "live"], default="local")
    args = parser.parse_args()

    run_kwargs: dict = {}
    if args.mode == "local":
        from room_assistant.session_store import InMemorySessionStore
        run_kwargs = {
            "repository": _build_local_q7_repository(),
            "session_store": InMemorySessionStore(),
        }
    else:
        run_kwargs = {"use_streaming": True}

    reports: list[ScenarioReport] = []
    base_session = f"qa-audit-{uuid.uuid4().hex[:8]}"

    print(f"=== Q&A AUDIT ({args.mode}) production transcript ===", flush=True)
    reports.append(await replay_transcript(base_session, run_kwargs))

    for spec in EXTRA_SCENARIOS:
        sid = f"qa-audit-{spec['name']}-{uuid.uuid4().hex[:6]}"
        print(f"=== Q&A AUDIT: {spec['name']} ===", flush=True)
        reports.append(await replay_extra(sid, spec, run_kwargs))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "session_base": base_session,
        "summary": {
            "scenarios_total": len(reports),
            "scenarios_passed": sum(1 for r in reports if r.passed),
            "turns_total": sum(len(r.turns) for r in reports),
            "turns_failed": sum(1 for r in reports for t in r.turns if not t.passed),
        },
        "scenarios": [
            {
                "name": r.name,
                "passed": r.passed,
                "notes": r.notes,
                "turns": [asdict(t) for t in r.turns],
            }
            for r in reports
        ],
    }

    report_path = ROOT / "qa-audit-report.json"
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Báo cáo kiểm tra Q&A",
        f"\nThời gian: {out['generated_at']}",
        f"\nTổng: {out['summary']['turns_failed']} lượt lỗi / {out['summary']['turns_total']} lượt",
        "",
    ]
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        md_lines.append(f"## [{status}] {r.name}")
        if r.notes:
            md_lines.append(f"_{r.notes}_\n")
        for t in r.turns:
            icon = "✅" if t.passed else "❌"
            md_lines.append(f"### {icon} Lượt {t.turn_index}: {t.question}")
            md_lines.append(f"- Phòng: **{t.room_count}** | Intent: `{t.intent}` | {t.processing_time_ms}ms")
            if t.room_ids:
                md_lines.append(f"- IDs: {', '.join(t.room_ids)}")
            if t.relaxed_search:
                md_lines.append("- Có nới điều kiện (relaxed_search)")
            if t.constraints:
                md_lines.append(f"- Constraints: `{json.dumps(t.constraints, ensure_ascii=False)}`")
            md_lines.append(f"- Trả lời: {t.answer[:280]}{'…' if len(t.answer)>280 else ''}")
            for issue in t.issues:
                md_lines.append(f"- **Lỗi:** {issue}")
            md_lines.append("")

    md_path = ROOT / "qa-audit-report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(out["summary"], ensure_ascii=False))
    print(f"Report: {md_path}")
    return 0 if out["summary"]["turns_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
