"""Full web Q&A — Django /api/rag/query/ + /api/chat/ trên Mongo+Qdrant thật."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "tests"))

from mongo_test_helpers import load_dotenv

load_dotenv()

BASE = os.getenv("WEB_QA_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("RAG_API_KEY", "").strip()


def _post_json(path: str, body: dict) -> tuple[int, dict]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = urllib.request.Request(f"{BASE}{path}", data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def _get_csrf() -> tuple[str, str]:
    req = urllib.request.Request(f"{BASE}/", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        cookie = resp.headers.get("Set-Cookie", "")
        html = resp.read().decode("utf-8", errors="replace")
    import re

    match = re.search(r'csrfmiddlewaretoken" value="([^"]+)"', html)
    if not match:
        match = re.search(r"name=['\"]csrfmiddlewaretoken['\"] value=['\"]([^'\"]+)", html)
    token = match.group(1) if match else ""
    parts = [p.strip() for p in cookie.split(";") if p.strip()]
    session = next((p for p in parts if p.startswith("sessionid=")), "")
    csrf_c = next((p for p in parts if p.startswith("csrftoken=")), "")
    cookie_header = "; ".join(x for x in (session, csrf_c) if x)
    return token, cookie_header


def _read_sse_final(path: str, body: dict, csrf_token: str, cookie: str) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-CSRFToken": csrf_token,
            "Cookie": cookie,
            "Referer": f"{BASE}/",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        events = []
        final = {}
        for raw_line in resp.read().decode("utf-8", errors="replace").splitlines():
            if not raw_line.startswith("data: "):
                continue
            event = json.loads(raw_line[6:])
            events.append(event)
            if event.get("type") == "final":
                final = event.get("payload") or {}
        final["_sse_events"] = events
        return final


def _query(message: str, session_id: str) -> dict:
    status, payload = _post_json("/api/rag/query/", {
        "message": message,
        "session_id": session_id,
    })
    payload["_http_status"] = status
    return payload


def _run_case(name: str, check, runner) -> bool:
    started = time.perf_counter()
    try:
        ok = bool(check(runner()))
    except Exception as exc:
        print(f"\n[FAIL] {name} — exception: {exc}")
        return False
    elapsed = time.perf_counter() - started
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name} ({elapsed:.1f}s)")
    return ok


def main() -> int:
    print("=" * 72)
    print("FULL WEB QA — Mongo + Qdrant + Django (không mock)")
    print(f"BASE={BASE}")
    print("=" * 72)

    try:
        with urllib.request.urlopen(f"{BASE}/api/health/", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        print(f"health: {health.get('status')}")
    except Exception as exc:
        print(f"FAIL health: {exc}")
        return 1

    home_req = urllib.request.Request(f"{BASE}/", method="GET")
    with urllib.request.urlopen(home_req, timeout=30) as resp:
        home_html = resp.read().decode("utf-8", errors="replace")
    print(f"home: {resp.status} | chat_ui={'/api/chat/' in home_html}")

    passed = 0
    total = 0

    def record(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        if detail:
            print(f"       {detail}")

    # --- Single-turn /api/rag/query/ ---
    cases = [
        (
            "P.305 room_code + giá template",
            lambda: _query("phòng P.305 giá bao nhiêu", "web-full-p305"),
            lambda p: (
                p.get("success")
                and p.get("intent") == "ASK_ABOUT_ROOM"
                and bool(p.get("rooms"))
                and "VND" in (p.get("answer") or "")
            ),
        ),
        (
            "Studio quận 7",
            lambda: _query("Tìm studio quận 7", "web-full-studio"),
            lambda p: p.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} and bool(p.get("rooms")),
        ),
        (
            "Nuôi mèo Q7",
            lambda: _query("Tìm phòng quận 7 cho nuôi mèo", "web-full-pets"),
            lambda p: (
                p.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"}
                and (
                    not p.get("rooms")
                    or all("thú cưng: không" not in (r.get("embedding_text") or "").lower() for r in p["rooms"])
                )
            ),
        ),
        (
            "Search Q7 dưới 5 triệu",
            lambda: _query("quận 7 dưới 5 triệu", "web-full-q7"),
            lambda p: p.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} and bool(p.get("rooms")),
        ),
        (
            "Search generic Q7",
            lambda: _query("Tìm phòng quận 7", "web-full-generic"),
            lambda p: p.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} and bool(p.get("rooms")),
        ),
        (
            "Sales handoff (ngân sách cực thấp)",
            lambda: _query("Tìm phòng quận 7 dưới 100 nghìn", "web-full-sales"),
            lambda p: (
                not p.get("rooms")
                and "tìm mỏi mắt" in (p.get("answer") or "").lower()
                and ("sales" in (p.get("answer") or "").lower() or "tư vấn" in (p.get("answer") or "").lower())
            ),
        ),
        (
            "Ngân sách colloquial 5 củ",
            lambda: _query("Tìm phòng quận 7 dưới 5 củ", "web-full-cu"),
            lambda p: p.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} and bool(p.get("rooms")),
        ),
    ]

    for name, runner, check in cases:
        print()
        p = runner()
        ok = check(p)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        print(f"  Q: {p.get('_question', '')}")
        print(f"  intent={p.get('intent')} rooms={len(p.get('rooms') or [])}")
        ans = (p.get("answer") or p.get("reply") or "")[:160]
        print(f"  A: {ans}...")
        record(name, ok)

    # --- Multi-turn ordinal / compare ---
    print()
    sid = "web-full-multiturn"
    r1 = _query("quận 7 dưới 5 triệu", sid)
    r2 = _query("Giá phòng số 99 bao nhiêu?", sid)
    ordinal_ok = (
        "phòng số 99" in (r2.get("answer") or "").lower()
        or "chỉ có" in (r2.get("answer") or "").lower()
        or "chưa có phòng số" in (r2.get("answer") or "").lower()
    )
    print(f"[{'PASS' if ordinal_ok else 'FAIL'}] Ordinal vượt list (multi-turn, phòng #99)")
    print(f"  intent={r2.get('intent')} rooms={len(r2.get('rooms') or [])}")
    print(f"  A: {(r2.get('answer') or '')[:160]}...")
    record("Ordinal vượt list", ordinal_ok)

    sid2 = "web-full-compare"
    _query("quận 7 dưới 5 triệu", sid2)
    r3 = _query("So sánh phòng đầu tiên và phòng thứ ba", sid2)
    compare_ok = len((r3.get("comparison") or {}).get("rows") or []) >= 2
    print(f"\n[{'PASS' if compare_ok else 'FAIL'}] So sánh phòng 1 và 3 (multi-turn)")
    print(f"  intent={r3.get('intent')} compare_rows={len((r3.get('comparison') or {}).get('rows') or [])}")
    print(f"  A: {(r3.get('answer') or '')[:160]}...")
    record("So sánh phòng 1 và 3", compare_ok)

    # Detail không bẩn filter
    sid3 = "web-full-detail"
    _query("quận 7 dưới 5 triệu", sid3)
    r4 = _query("Phòng này có máy lạnh không?", sid3)
    detail_ok = (
        r4.get("intent") == "ASK_ABOUT_ROOM"
        and "air_conditioner" not in (
            (r4.get("session_state") or {}).get("constraints", {}).get("amenities_required") or []
        )
    )
    print(f"\n[{'PASS' if detail_ok else 'FAIL'}] Detail AC không bẩn session filter")
    print(f"  amenities={(r4.get('session_state') or {}).get('constraints', {}).get('amenities_required')}")
    record("Detail AC không bẩn filter", detail_ok)

    # --- /api/chat/ SSE (UI path) ---
    print()
    try:
        csrf, cookie = _get_csrf()
        chat = _read_sse_final(
            "/api/chat/",
            {
                "message": "phòng P.305 giá bao nhiêu",
                "contact_name": "Web Full QA",
                "contact_phone": "0900111222",
                "demo_role": "customer",
                "new_chat": "1",
            },
            csrf,
            cookie,
        )
        status_events = [e for e in chat.get("_sse_events", []) if e.get("type") == "status"]
        chat_ok = bool(chat.get("rooms")) and "VND" in (chat.get("reply") or "")
        print(f"[{'PASS' if chat_ok else 'FAIL'}] /api/chat/ SSE P.305 (UI path)")
        print(f"  intent={chat.get('intent')} rooms={len(chat.get('rooms') or [])} status_events={len(status_events)}")
        print(f"  A: {(chat.get('reply') or '')[:160]}...")
        record("/api/chat/ SSE P.305", chat_ok)

        chat2 = _read_sse_final(
            "/api/chat/",
            {
                "message": "Tìm studio quận 7",
                "contact_name": "Web Full QA",
                "contact_phone": "0900111222",
                "demo_role": "customer",
                "new_chat": "1",
            },
            csrf,
            cookie,
        )
        chat_studio_ok = bool(chat2.get("rooms"))
        print(f"\n[{'PASS' if chat_studio_ok else 'FAIL'}] /api/chat/ SSE studio Q7")
        print(f"  rooms={len(chat2.get('rooms') or [])}")
        record("/api/chat/ SSE studio Q7", chat_studio_ok)
    except Exception as exc:
        print(f"[FAIL] /api/chat/ SSE: {exc}")
        record("/api/chat/ SSE P.305", False)
        record("/api/chat/ SSE studio Q7", False)

    print("\n" + "=" * 72)
    print(f"Kết quả full web QA: {passed}/{total}")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
