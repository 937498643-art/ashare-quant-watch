"""Validate the local Tushare daily_basic float-share cache.

The cache is read-only market/basic reference data. This script does not
access broker software, accounts, positions, or execution functions.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.tushare_cache_source import load_cache  # noqa: E402


CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "tushare" / "daily_basic_latest.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "float_share_cache_check.json"
REQUIRED_COLUMNS = [
    "ts_code",
    "trade_date",
    "close",
    "float_share",
    "turnover_rate",
    "volume_ratio",
]


def main() -> int:
    report = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cache_path": str(CACHE_PATH),
        "cache_found": CACHE_PATH.exists(),
        "float_share_unit": "万股",
        "rows": 0,
        "columns": [],
        "missing_columns": [],
        "float_share_valid_count": 0,
        "float_share_coverage": 0.0,
        "duplicate_ts_code_count": 0,
        "trade_dates": [],
        "trade_date_consistent": False,
        "quality_passed": False,
        "error_message": "",
    }

    if not CACHE_PATH.exists():
        report["error_message"] = "daily_basic_latest.csv was not found."
        _save_report(report)
        print("daily_basic_latest.csv not found")
        return 1

    data = load_cache(CACHE_PATH)
    if data.empty:
        report["error_message"] = "daily_basic_latest.csv is empty or unreadable."
        _save_report(report)
        print("daily_basic_latest.csv is empty or unreadable")
        return 1

    report["rows"] = int(len(data))
    report["columns"] = [str(column) for column in data.columns]
    report["missing_columns"] = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    float_share = pd.to_numeric(data.get("float_share"), errors="coerce")
    valid_float_share = float_share > 0 if float_share is not None else pd.Series(False, index=data.index)
    report["float_share_valid_count"] = int(valid_float_share.sum())
    report["float_share_coverage"] = round(report["float_share_valid_count"] / len(data), 6)
    report["duplicate_ts_code_count"] = int(data["ts_code"].astype(str).duplicated().sum()) if "ts_code" in data else len(data)
    trade_dates = sorted(data["trade_date"].dropna().astype(str).unique().tolist()) if "trade_date" in data else []
    report["trade_dates"] = trade_dates
    report["trade_date_consistent"] = len(trade_dates) == 1
    report["quality_passed"] = bool(
        len(data) > 5000
        and not report["missing_columns"]
        and report["float_share_coverage"] >= 0.95
        and report["duplicate_ts_code_count"] == 0
        and report["trade_date_consistent"]
    )

    _save_report(report)
    print(f"cache: {CACHE_PATH}")
    print(f"rows: {report['rows']}")
    print(f"float_share_valid_count: {report['float_share_valid_count']}")
    print(f"float_share_coverage: {report['float_share_coverage']:.2%}")
    print(f"duplicate_ts_code_count: {report['duplicate_ts_code_count']}")
    print(f"trade_dates: {trade_dates}")
    print("float_share_unit: 万股")
    print("sample:")
    sample = [column for column in REQUIRED_COLUMNS if column in data.columns]
    print(data[sample].head(10).to_string(index=False))
    print(f"quality_passed: {report['quality_passed']}")
    return 0 if report["quality_passed"] else 1


def _save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
