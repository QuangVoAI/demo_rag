"""Smoke Q&A qua Django HTTP (cùng path web: /api/rag/query/ + /api/chat/)."""

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


def _post_json(path: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    if API_KEY:
        req_headers["X-API-Key"] = API_KEY
    req = urllib.request.Request(f"{BASE}{path}", data=payload, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


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
        final = {}
        for raw_line in resp.read().decode("utf-8", errors="replace").splitlines():
            if not raw_line.startswith("data: "):
                continue
            event = json.loads(raw_line[6:])
            if event.get("type") == "final":
                final = event.get("payload") or {}
        return final


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
    session = ""
    for part in cookie.split(";"):
        if part.strip().startswith("sessionid="):
            session = part.strip()
            break
    cookie_header = session
    if "csrftoken=" in cookie:
        for part in cookie.split(";"):
            if part.strip().startswith("csrftoken="):
                cookie_header = f"{session}; {part.strip()}" if session else part.strip()
                break
    return token, cookie_header


SCENARIOS = [
    {
        "name": "P.305 giá",
        "message": "phòng P.305 giá bao nhiêu",
        "check": lambda p: p.get("intent") == "ASK_ABOUT_ROOM" and bool(p.get("rooms")),
    },
    {
        "name": "Sales handoff",
        "message": "Tìm phòng quận 7 dưới 1 triệu",
        "check": lambda p: (
            "tìm mỏi mắt" in (p.get("answer") or p.get("reply") or "").lower()
            or (
                p.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"}
                and bool(p.get("rooms"))
                and all((r.get("rent_price") or 0) <= 1_100_000 for r in (p.get("rooms") or []))
            )
        ),
    },
    {
        "name": "Search Q7",
        "message": "quận 7 dưới 5 triệu",
        "check": lambda p: p.get("intent") in {"SEARCH_ROOM", "REFINE_SEARCH"} and bool(p.get("rooms")),
    },
    {
        "name": "Studio Q7",
        "message": "Tìm studio quận 7",
        "check": lambda p: p.get("intent") == "SEARCH_ROOM" and bool(p.get("rooms")),
    },
]


def main() -> int:
    print("=" * 72)
    print("WEB QA SMOKE — Django HTTP (engine thật, không mock)")
    print(f"BASE={BASE}")
    print("=" * 72)

    try:
        with urllib.request.urlopen(f"{BASE}/api/health/", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        print(f"health: {health.get('status')}")
    except Exception as exc:
        print(f"FAIL health: {exc}")
        print("Chạy server: python manage.py runserver")
        return 1

    passed = 0
    for idx, scenario in enumerate(SCENARIOS, 1):
        sid = f"web-smoke-{idx}"
        started = time.perf_counter()
        status, payload = _post_json("/api/rag/query/", {
            "message": scenario["message"],
            "session_id": sid,
        })
        elapsed = time.perf_counter() - started
        answer = payload.get("answer") or payload.get("reply") or ""
        ok = status == 200 and payload.get("success") and scenario["check"](payload)
        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] {scenario['name']} ({elapsed:.1f}s)")
        print(f"  Q: {scenario['message']}")
        print(f"  intent={payload.get('intent')} rooms={len(payload.get('rooms') or [])}")
        print(f"  A: {answer[:180]}...")
        if ok:
            passed += 1

    # Chat drawer path (/api/chat/ SSE) — giống UI demo
    try:
        csrf, cookie = _get_csrf()
        started = time.perf_counter()
        chat_payload = _read_sse_final(
            "/api/chat/",
            {
                "message": "phòng P.305 giá bao nhiêu",
                "contact_name": "Web Smoke",
                "contact_phone": "0900000001",
                "demo_role": "customer",
                "new_chat": "1",
            },
            csrf,
            cookie,
        )
        elapsed = time.perf_counter() - started
        answer = chat_payload.get("answer") or chat_payload.get("reply") or ""
        chat_ok = bool(chat_payload.get("rooms")) and (
            chat_payload.get("intent") == "ASK_ABOUT_ROOM" or chat_payload.get("intent") is None
        )
        mark = "PASS" if chat_ok else "FAIL"
        print(f"\n[{mark}] Web /api/chat/ SSE P.305 ({elapsed:.1f}s)")
        print(f"  intent={chat_payload.get('intent')} rooms={len(chat_payload.get('rooms') or [])}")
        print(f"  A: {answer[:180]}...")
        if chat_ok:
            passed += 1
    except Exception as exc:
        print(f"\n[FAIL] Web /api/chat/ SSE: {exc}")

    total = len(SCENARIOS) + 1
    print("\n" + "=" * 72)
    print(f"Kết quả web smoke: {passed}/{total}")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
