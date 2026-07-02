"""Run Bo_Test_Case_Chat.pdf scenarios against room assistant (live Mongo when available)."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from room_assistant.repository import create_room_repository

from bo_test_case_auditor import load_cases, run_all_cases, summarize


class BoTestCaseChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = create_room_repository()
        cls.using_live_mongo = os.getenv("BO_TEST_FORCE_INMEMORY", "").lower() not in {"1", "true", "yes"}
        try:
            from room_assistant.repository import EmptyRoomRepository, MongoRoomRepository
            cls.using_live_mongo = isinstance(cls.repository, MongoRoomRepository)
            if isinstance(cls.repository, EmptyRoomRepository):
                cls.using_live_mongo = False
        except Exception:
            cls.using_live_mongo = False

        cls.audits = run_all_cases(cls.repository)
        cls.summary = summarize(cls.audits)

    def test_bo_fixture_has_72_cases(self):
        self.assertEqual(len(load_cases()), 72)

    def test_rag_scenarios_pass(self):
        if not self.using_live_mongo:
            self.skipTest("MongoDB unavailable — BO PDF cases require live inventory")
        failed = self.summary["failed"]
        report_path = Path(__file__).parent / "fixtures" / "bo_test_case_latest_report.json"
        report_path.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(
            failed,
            0,
            msg=json.dumps(self.summary.get("failures", [])[:10], ensure_ascii=False, indent=2),
        )


if __name__ == "__main__":
    unittest.main()
