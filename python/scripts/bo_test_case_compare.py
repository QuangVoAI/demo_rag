"""Compare Bo test case audit: current code vs baseline git commit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
TESTS = PYTHON / "tests"
sys.path.insert(0, str(PYTHON))
sys.path.insert(0, str(TESTS))

from room_assistant.repository import create_room_repository
from room_assistant.workflow import run_room_assistant

from bo_test_case_auditor import run_all_cases, summarize


def _audit_with_runner(runner, label: str) -> dict:
    repo = create_room_repository()
    audits = run_all_cases(repo, runner=runner)
    summary = summarize(audits)
    summary["label"] = label
    summary["live_mongo"] = repo.__class__.__name__
    return summary


def _audit_in_worktree(commit: str, label: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "baseline"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(worktree / "python") + os.pathsep + str(TESTS)
            snippet = f"""
import json, sys
sys.path.insert(0, r"{worktree / 'python'}")
sys.path.insert(0, r"{TESTS}")
from room_assistant.repository import create_room_repository
from room_assistant.workflow import run_room_assistant
from bo_test_case_auditor import run_all_cases, summarize

repo = create_room_repository()
summary = summarize(run_all_cases(repo, runner=run_room_assistant))
summary["label"] = {label!r}
summary["live_mongo"] = repo.__class__.__name__
print(json.dumps(summary, ensure_ascii=False))
"""
            proc = subprocess.run(
                [sys.executable, "-c", snippet],
                cwd=worktree / "python",
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout).strip())
            payload_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
            if not payload_lines:
                raise RuntimeError(proc.stdout or "empty baseline audit output")
            return json.loads(payload_lines[-1])
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="e2f635a", help="Git commit for baseline model")
    parser.add_argument("--out", default=str(PYTHON / "tests" / "fixtures" / "bo_test_case_compare.json"))
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    current = None if args.baseline_only else _audit_with_runner(run_room_assistant, "current")
    baseline = None
    if not args.current_only:
        try:
            baseline = _audit_in_worktree(args.baseline, f"baseline:{args.baseline}")
        except Exception as exc:
            baseline = {"label": f"baseline:{args.baseline}", "error": str(exc)}

    comparison = {
        "current": current,
        "baseline": baseline,
        "delta_passed": (
            current.get("passed", 0) - baseline.get("passed", 0)
            if current and baseline and "error" not in baseline
            else None
        ),
        "current_must_win": (
            current.get("failed", 999) <= baseline.get("failed", 999)
            and current.get("passed", 0) >= baseline.get("passed", 0)
            if current and baseline and "error" not in baseline
            else None
        ),
    }
    Path(args.out).write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if baseline and "error" in baseline:
        return 2
    if current and current.get("failed", 0) > 0:
        return 1
    if current and baseline and current.get("passed", 0) < baseline.get("passed", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
