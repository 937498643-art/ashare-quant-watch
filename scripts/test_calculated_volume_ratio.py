"""Independent validation of calculated intraday volume ratio.

This script reads public Sina quotes and existing local Tushare cache files.
It never changes scanner, dashboard, strategy, or trading behavior.
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

from data_sources.calculated_volume_ratio_source import (  # noqa: E402
    HISTORICAL_VOLUME_UNIT,
    REALTIME_VOLUME_UNIT,
    attach_reference_volume_ratio,
    calculate_realtime_volume_ratio,
    fetch_sina_quotes_for_volume_ratio,
    load_recent_daily_volume,
    normalize_code,
    trading_minutes_at,
)


DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
SAMPLE_PATH = DIAGNOSTICS_DIR / "calculated_volume_ratio_sample.csv"
JSON_PATH = DIAGNOSTICS_DIR / "calculated_volume_ratio_check.json"
MANUAL_PATH = DIAGNOSTICS_DIR / "manual_volume_ratio_benchmark.csv"
LATEST_CANDIDATES_PATH = PROJECT_ROOT / "data" / "latest_candidates.csv"


def active_watch_codes() -> list[str]:
    """Return up to 30 current active-watch codes, preserving scanner order."""
    if not LATEST_CANDIDATES_PATH.exists():
        return []
    try:
        data = pd.read_csv(LATEST_CANDIDATES_PATH, encoding="utf-8-sig", dtype={"code": "string"})
    except Exception:
        return []
    if "source_type" in data.columns:
        data = data[data["source_type"].eq("active_watchlist")]
    return data.get("code", pd.Series(dtype="string")).map(normalize_code).loc[lambda codes: codes.ne("")].head(30).tolist()


def build_manual_template(data: pd.DataFrame) -> pd.DataFrame:
    """Create a safe blank broker-comparison sheet for manual input."""
    columns = [
        "code", "name", "calculation_time", "calculated_volume_ratio", "volume_ratio_ref",
    ]
    template = data[[column for column in columns if column in data.columns]].copy()
    template["broker_volume_ratio_manual"] = pd.NA
    template["abs_error"] = pd.NA
    return template


def main() -> int:
    """Run the isolated calculation and persist transparent diagnostics."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    check_time = datetime.now()
    elapsed_minutes, market_phase = trading_minutes_at(check_time)
    history, trade_dates = load_recent_daily_volume()
    used_trade_dates = trade_dates[:5]
    raw, quotes = fetch_sina_quotes_for_volume_ratio()

    quotes = quotes.copy()
    if quotes.empty:
        calculated = pd.DataFrame()
    else:
        calculated = calculate_realtime_volume_ratio(quotes, history, trade_dates, elapsed_minutes)
        calculated = attach_reference_volume_ratio(calculated)
        calculated["calculation_time"] = check_time.strftime("%Y-%m-%d %H:%M:%S")

    watch_codes = active_watch_codes()
    if watch_codes and not calculated.empty:
        sample = calculated[calculated["code"].isin(watch_codes)].copy()
    else:
        sample = calculated.head(30).copy()

    output_columns = [
        "code", "name", "realtime_volume", "realtime_volume_unit", "elapsed_trading_minutes",
        "avg_5d_volume", "avg_5d_volume_unit", "calculated_volume_ratio",
        "calculated_volume_ratio_display", "volume_ratio_ref", "reference_trade_date",
        "volume_ratio_source", "volume_ratio_calculated_at", "is_realtime_volume_ratio",
        "volume_ratio_validation_status", "calculation_time",
    ]
    sample = sample[[column for column in output_columns if column in sample.columns]].copy()
    sample.to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
    build_manual_template(sample).to_csv(MANUAL_PATH, index=False, encoding="utf-8-sig")

    matched_count = int(calculated.get("avg_5d_volume", pd.Series(dtype="float64")).notna().sum())
    calculated_count = int(calculated.get("is_realtime_volume_ratio", pd.Series(dtype="bool")).fillna(False).sum())
    sina_rows = int(len(quotes))
    coverage = round(calculated_count / sina_rows, 6) if sina_rows else 0.0
    history_complete = len(used_trade_dates) == 5
    values = pd.to_numeric(calculated.get("calculated_volume_ratio"), errors="coerce")
    not_all_same = bool(values.dropna().nunique() > 1)
    validation_passed = bool(
        history_complete
        and elapsed_minutes > 0
        and sina_rows > 0
        and matched_count / sina_rows >= 0.95
        and coverage >= 0.95
        and not_all_same
    )

    warning = ""
    if not history_complete:
        warning = f"Historical daily cache is insufficient: found {len(trade_dates)} of 5 required trade dates."
    elif elapsed_minutes <= 0:
        warning = f"Market is not in an effective trading period ({market_phase}); realtime ratio is not calculated."

    report = {
        "check_time": check_time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_market_time": check_time.strftime("%Y-%m-%d %H:%M:%S"),
        "market_phase": market_phase,
        "elapsed_trading_minutes": elapsed_minutes,
        "sina_rows": sina_rows,
        "sina_raw_rows": int(len(raw)),
        "historical_trade_dates": used_trade_dates,
        "historical_day_count": len(used_trade_dates),
        "history_complete": history_complete,
        "matched_count": matched_count,
        "calculated_volume_ratio_count": calculated_count,
        "calculated_volume_ratio_coverage": coverage,
        "detected_realtime_volume_unit": REALTIME_VOLUME_UNIT,
        "detected_historical_volume_unit": HISTORICAL_VOLUME_UNIT,
        "historical_volume_conversion": "Tushare daily vol is hands and is multiplied by 100 to shares.",
        "validation_passed": validation_passed,
        "warning_message": warning,
        "recommended_next_step": (
            "Build four additional distinct daily caches before validating intraday volume ratio."
            if not history_complete else "Fill manual_volume_ratio_benchmark.csv with same-time broker values for final review."
        ),
    }
    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Calculated intraday volume-ratio diagnosis")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if sample.empty:
        print("No sample rows available.")
    else:
        print(sample.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
