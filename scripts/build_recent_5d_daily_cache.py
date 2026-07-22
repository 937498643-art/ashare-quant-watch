"""Build only missing Tushare daily caches for intraday volume-ratio validation.

The script is read-only with respect to market providers. It writes local cache
files only and never starts the main scanner or any execution capability.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.tushare_cache_source import (  # noqa: E402
    fetch_daily,
    fetch_trade_calendar,
    get_errors,
    load_cache,
    reset_errors,
    save_cache,
)


CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tushare"
MARKET_STATUS_PATH = PROJECT_ROOT / "data" / "latest_market_status.json"
CALENDAR_CACHE_PATH = CACHE_DIR / "trade_cal.csv"
STATUS_PATH = CACHE_DIR / "recent_5d_daily_cache_status.json"
VOLUME_RATIO_TEST_PATH = PROJECT_ROOT / "scripts" / "test_calculated_volume_ratio.py"
REQUIRED_DAILY_COLUMNS = {
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file without stopping the cache-build process."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def normalize_date(value: Any) -> str:
    """Convert date-like values to YYYYMMDD."""
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def resolve_quote_trade_date() -> tuple[str | None, str]:
    """Prefer the newest successful market-status scan date over system date."""
    status = read_json(MARKET_STATUS_PATH)
    scan_time = normalize_date(status.get("scan_time"))
    spot_count = pd.to_numeric(status.get("spot_count"), errors="coerce")
    if scan_time and pd.notna(spot_count) and spot_count > 0:
        return scan_time, "latest_market_status.scan_time"

    latest_daily = sorted(CACHE_DIR.glob("daily_????????.csv"))
    if latest_daily:
        cached_date = normalize_date(latest_daily[-1].stem.removeprefix("daily_"))
        if cached_date:
            return cached_date, "latest_daily_cache"
    return None, "unresolved"


def load_or_fetch_calendar(quote_date: str) -> tuple[pd.DataFrame, str]:
    """Use a local calendar when sufficient, otherwise make one bounded request."""
    cached = load_cache(CALENDAR_CACHE_PATH)
    if _calendar_covers(cached, quote_date):
        return cached, "local_cache"

    start_date = (datetime.strptime(quote_date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d")
    calendar = fetch_trade_calendar(start_date, quote_date)
    if not calendar.empty:
        save_cache(calendar, CALENDAR_CACHE_PATH)
        return calendar, "tushare_trade_cal"
    return cached, "unavailable"


def _calendar_covers(calendar: pd.DataFrame, quote_date: str) -> bool:
    if calendar.empty or "cal_date" not in calendar.columns:
        return False
    dates = calendar["cal_date"].map(normalize_date)
    return quote_date in set(dates)


def required_previous_trade_dates(calendar: pd.DataFrame, quote_date: str) -> list[str]:
    """Return the five completed exchange days immediately before quote_date."""
    if calendar.empty or not {"cal_date", "is_open"}.issubset(calendar.columns):
        return []
    open_days = calendar[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)]["cal_date"].map(normalize_date)
    return sorted({day for day in open_days if day and day < quote_date}, reverse=True)[:5]


def valid_daily_cache(path: Path, trade_date: str) -> bool:
    """Require one full-market daily cache for the exact requested trade date."""
    data = load_cache(path)
    if data.empty or not REQUIRED_DAILY_COLUMNS.issubset(data.columns):
        return False
    dates = set(data["trade_date"].map(normalize_date))
    return dates == {trade_date}


def is_rate_limited(errors: list[str]) -> bool:
    text = " ".join(errors).lower()
    markers = ("rate", "limit", "frequency", "频率", "次数", "每小时")
    return any(marker in text for marker in markers)


def write_status(status: dict[str, Any]) -> None:
    """Persist resumable cache-build state."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """Build missing caches once per date, stopping safely after an API failure."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reset_errors()
    now = datetime.now()
    quote_date, quote_source = resolve_quote_trade_date()
    status: dict[str, Any] = {
        "build_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "quote_trade_date": quote_date,
        "quote_trade_date_source": quote_source,
        "required_trade_dates": [],
        "existing_trade_dates_before_run": [],
        "successfully_built_dates": [],
        "skipped_existing_dates": [],
        "failed_dates": {},
        "missing_dates_after_run": [],
        "complete_5d_history": False,
        "next_retry_time": None,
        "warning_message": "",
    }
    if not quote_date:
        status["warning_message"] = "Unable to resolve the current quote trade date from local market status."
        write_status(status)
        print(status["warning_message"])
        return 1

    calendar, calendar_source = load_or_fetch_calendar(quote_date)
    required_dates = required_previous_trade_dates(calendar, quote_date)
    status["trade_calendar_source"] = calendar_source
    status["required_trade_dates"] = required_dates
    if len(required_dates) != 5:
        status["warning_message"] = "Unable to obtain five completed trade dates from the local or Tushare calendar."
        status["failed_dates"] = {"trade_calendar": "; ".join(get_errors()) or "calendar unavailable"}
        write_status(status)
        print(status["warning_message"])
        return 1

    existing = []
    missing = []
    for trade_date in required_dates:
        path = CACHE_DIR / f"daily_{trade_date}.csv"
        if valid_daily_cache(path, trade_date):
            existing.append(trade_date)
        else:
            missing.append(trade_date)
    status["existing_trade_dates_before_run"] = existing

    stop_after_failure = False
    for trade_date in missing:
        if stop_after_failure:
            status["failed_dates"][trade_date] = "not_attempted_after_previous_api_failure"
            continue
        daily = fetch_daily(trade_date)
        errors = get_errors()
        path = CACHE_DIR / f"daily_{trade_date}.csv"
        if daily.empty or not REQUIRED_DAILY_COLUMNS.issubset(daily.columns):
            reason = "; ".join(errors) if errors else "daily returned empty or missing required columns"
            status["failed_dates"][trade_date] = reason
            stop_after_failure = True
            if is_rate_limited(errors):
                status["next_retry_time"] = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            continue
        if not save_cache(daily, path):
            status["failed_dates"][trade_date] = "; ".join(get_errors()) or "cache write failed"
            stop_after_failure = True
            continue
        status["successfully_built_dates"].append(trade_date)

    status["skipped_existing_dates"] = existing
    status["missing_dates_after_run"] = [
        trade_date for trade_date in required_dates
        if not valid_daily_cache(CACHE_DIR / f"daily_{trade_date}.csv", trade_date)
    ]
    status["complete_5d_history"] = not status["missing_dates_after_run"]
    if status["failed_dates"] and not status["warning_message"]:
        status["warning_message"] = "Cache build stopped after an API or cache failure; rerun later to resume missing dates."
    write_status(status)

    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["complete_5d_history"]:
        completed = subprocess.run([sys.executable, str(VOLUME_RATIO_TEST_PATH)], cwd=PROJECT_ROOT, check=False)
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
