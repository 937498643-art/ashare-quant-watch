"""Test day-level candidate-selection deduplication without running the scanner."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.tracking import _deduplicate_history, enrich_tracking
from storage.database import CandidateDatabase


REPORT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "selection_day_dedup_check.json"
TEST_DB_DIR = PROJECT_ROOT / "data" / "diagnostics"


def candidate_frame(code: str, score: float = 60.0) -> pd.DataFrame:
    """Return a minimal candidate row for deterministic persistence tests."""
    return pd.DataFrame([{
        "code": code,
        "name": f"test-{code}",
        "source_type": "active_watchlist",
        "price": 10.0,
        "pct_chg": 1.0,
        "amount": 200_000_000.0,
        "turnover": 3.0,
        "volume_ratio": 1.2,
        "score": score,
        "level": "watch",
        "risk_summary": "",
    }])


def simulate_selection(db_path: Path, code: str, selection_date: str, runs: int) -> pd.DataFrame:
    """Simulate repeated scans on one effective trade date."""
    database = CandidateDatabase(db_path)
    last = pd.DataFrame()
    for index in range(runs):
        enriched = enrich_tracking(candidate_frame(code, 60.0 + index), db_path, selection_date)
        scan_time = datetime.strptime(f"{selection_date} 10:{index:02d}:00", "%Y-%m-%d %H:%M:%S")
        database.save_candidates(enriched, scan_time)
        last = enriched
    return last


def selection_days(db_path: Path, code: str) -> pd.DataFrame:
    """Read effective records only, not append-only scan snapshots."""
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM candidate_selection_days WHERE code = ? ORDER BY selection_trade_date",
            connection,
            params=[code],
        )


def main() -> int:
    """Exercise same-day, interrupted, and Friday-to-Monday sequences."""
    TEST_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = TEST_DB_DIR / f"selection_day_dedup_test_{datetime.now():%Y%m%d_%H%M%S_%f}.db"
    primary_code = "600001"

    day_one = simulate_selection(db_path, primary_code, "2026-07-14", 10)
    first_days = selection_days(db_path, primary_code)
    same_day_result = {
        "code": primary_code,
        "simulated_run_count": 10,
        "unique_selection_days": int(len(first_days)),
        "selection_days_total": int(day_one.iloc[0]["selection_days_total"]),
        "consecutive_selection_days": int(day_one.iloc[0]["consecutive_selection_days"]),
        "duplicate_same_day_ignored_count": 9,
    }

    day_two = simulate_selection(db_path, primary_code, "2026-07-15", 5)
    two_days = selection_days(db_path, primary_code)
    two_trade_days_result = {
        "code": primary_code,
        "simulated_run_count": 15,
        "unique_selection_days": int(len(two_days)),
        "selection_days_total": int(day_two.iloc[0]["selection_days_total"]),
        "consecutive_selection_days": int(day_two.iloc[0]["consecutive_selection_days"]),
        "duplicate_same_day_ignored_count": 13,
    }

    interrupted = enrich_tracking(candidate_frame(primary_code), db_path, "2026-07-17")
    simulate_selection(db_path, primary_code, "2026-07-17", 1)
    interrupted_sequence_result = {
        "code": primary_code,
        "missing_trade_date": "2026-07-16",
        "selection_days_total": int(interrupted.iloc[0]["selection_days_total"]),
        "consecutive_selection_days": int(interrupted.iloc[0]["consecutive_selection_days"]),
    }

    weekend_code = "600002"
    simulate_selection(db_path, weekend_code, "2026-07-17", 1)
    monday = simulate_selection(db_path, weekend_code, "2026-07-20", 1)
    weekend_continuity_result = {
        "code": weekend_code,
        "friday": "2026-07-17",
        "monday": "2026-07-20",
        "selection_days_total": int(monday.iloc[0]["selection_days_total"]),
        "consecutive_selection_days": int(monday.iloc[0]["consecutive_selection_days"]),
    }

    legacy_duplicates = pd.DataFrame([
        {"code": "600003", "selection_trade_date": "2026-07-14", "latest_scan_time": "2026-07-14 10:00:00"},
        {"code": "600003", "selection_trade_date": "2026-07-14", "latest_scan_time": "2026-07-14 14:00:00"},
    ])
    deduplicated_legacy = _deduplicate_history(legacy_duplicates)
    historical_duplicate_handling_passed = len(deduplicated_legacy) == 1

    dedup_passed = (
        same_day_result["unique_selection_days"] == 1
        and same_day_result["selection_days_total"] == 1
        and two_trade_days_result["unique_selection_days"] == 2
        and two_trade_days_result["selection_days_total"] == 2
        and historical_duplicate_handling_passed
    )
    consecutive_logic_passed = (
        two_trade_days_result["consecutive_selection_days"] == 2
        and interrupted_sequence_result["consecutive_selection_days"] == 1
        and weekend_continuity_result["consecutive_selection_days"] == 2
    )
    report = {
        "same_day_10_runs_result": same_day_result,
        "two_trade_days_result": two_trade_days_result,
        "interrupted_sequence_result": interrupted_sequence_result,
        "weekend_continuity_result": weekend_continuity_result,
        "historical_duplicate_handling_passed": historical_duplicate_handling_passed,
        "dedup_passed": dedup_passed,
        "consecutive_logic_passed": consecutive_logic_passed,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if dedup_passed and consecutive_logic_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
