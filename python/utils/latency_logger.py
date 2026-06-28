from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import threading
from typing import Any


_lock = threading.Lock()


def _default_log_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "latency.jsonl"


def log_latency_event(event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    log_path = _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
