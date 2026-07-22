"""Validate the local Tushare K-line cache pipeline without network requests."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.tushare_kline_cache_reader import normalize_code, read_tushare_kline_from_cache


TEST_STOCKS = {
    "002384": "Dongshan Precision",
    "600183": "Shengyi Technology",
}
REPORT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "kline_data_pipeline_check.json"


def inspect_stock(code: str, name: str) -> dict[str, object]:
    data = read_tushare_kline_from_cache(code, days=60)
    dates = pd.to_datetime(data.get("trade_date"), errors="coerce") if not data.empty else pd.Series(dtype="datetime64[ns]")
    numeric = data.copy()
    for column in ["open", "high", "low", "close"]:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    valid_ohlc = numeric.dropna(subset=["open", "high", "low", "close"])
    valid_ohlc = valid_ohlc[(valid_ohlc[["open", "high", "low", "close"]] > 0).all(axis=1)]
    source_codes = data.get("ts_code", pd.Series(dtype="string")).map(normalize_code)
    other_stock_mixed_count = int((source_codes != normalize_code(code)).sum()) if not source_codes.empty else 0
    matched_rows = int(len(data))

    return {
        "code": code,
        "name": name,
        "matched_rows": matched_rows,
        "first_trade_date": None if dates.dropna().empty else dates.min().strftime("%Y-%m-%d"),
        "last_trade_date": None if dates.dropna().empty else dates.max().strftime("%Y-%m-%d"),
        "date_sorted": bool(dates.dropna().is_monotonic_increasing),
        "duplicate_trade_date_count": int(dates.dropna().duplicated().sum()),
        "valid_ohlc_count": int(len(valid_ohlc)),
        "ma5_valid_count": int(data.get("ma5", pd.Series(dtype="float64")).notna().sum()),
        "ma10_valid_count": int(data.get("ma10", pd.Series(dtype="float64")).notna().sum()),
        "ma20_valid_count": int(data.get("ma20", pd.Series(dtype="float64")).notna().sum()),
        "expected_ma5_valid_count": max(matched_rows - 4, 0),
        "expected_ma10_valid_count": max(matched_rows - 9, 0),
        "expected_ma20_valid_count": max(matched_rows - 19, 0),
        "other_stock_mixed_count": other_stock_mixed_count,
    }


def is_valid(result: dict[str, object]) -> bool:
    first_date = result["first_trade_date"]
    last_date = result["last_trade_date"]
    return bool(
        result["matched_rows"] >= 5
        and result["date_sorted"]
        and result["duplicate_trade_date_count"] == 0
        and result["other_stock_mixed_count"] == 0
        and result["valid_ohlc_count"] == result["matched_rows"]
        and first_date is not None
        and last_date is not None
        and first_date < last_date
        and result["ma5_valid_count"] == result["expected_ma5_valid_count"]
        and result["ma10_valid_count"] == result["expected_ma10_valid_count"]
        and result["ma20_valid_count"] == result["expected_ma20_valid_count"]
    )


def main() -> int:
    results = [inspect_stock(code, name) for code, name in TEST_STOCKS.items()]
    report = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": results,
        "passed": all(is_valid(result) for result in results),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
