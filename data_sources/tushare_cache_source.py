"""Tushare Pro read-only cache helpers.

This module only fetches market/basic data and saves local cache files.
It does not contain any trading, order, account, or broker-operation logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

try:
    import tushare as ts
except Exception:  # pragma: no cover - handled at runtime for clear diagnostics.
    ts = None

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

LAST_ERRORS: List[str] = []


def _record_error(message: str) -> None:
    LAST_ERRORS.append(message)
    LOGGER.error(message)


def reset_errors() -> None:
    """Clear errors captured by the latest cache build run."""

    LAST_ERRORS.clear()


def get_errors() -> List[str]:
    """Return captured fetch/save/load errors."""

    return list(LAST_ERRORS)


def get_pro_api():
    """Initialize Tushare Pro API from .env without printing the token."""

    if ts is None:
        _record_error("tushare is not installed. Please install tushare first.")
        return None

    load_dotenv(dotenv_path=ENV_PATH)

    import os

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        _record_error("TUSHARE_TOKEN is missing. Please configure it in .env.")
        return None

    try:
        return ts.pro_api(token)
    except Exception as exc:
        _record_error(f"Failed to initialize Tushare Pro API: {type(exc).__name__}: {exc}")
        return None


def fetch_stock_basic() -> pd.DataFrame:
    """Fetch listed A-share basic information."""

    pro = get_pro_api()
    if pro is None:
        return pd.DataFrame()

    fields = "ts_code,symbol,name,area,industry,market,list_date"
    try:
        return pro.stock_basic(exchange="", list_status="L", fields=fields)
    except Exception as exc:
        _record_error(f"fetch_stock_basic failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def fetch_daily_basic(trade_date: str) -> pd.DataFrame:
    """Fetch daily valuation/activity fields for one trade date."""

    pro = get_pro_api()
    if pro is None:
        return pd.DataFrame()

    fields = (
        "ts_code,trade_date,close,turnover_rate,volume_ratio,"
        "total_share,float_share,free_share,total_mv,circ_mv"
    )
    try:
        return pro.daily_basic(trade_date=trade_date, fields=fields)
    except Exception as exc:
        _record_error(f"fetch_daily_basic({trade_date}) failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def fetch_latest_open_trade_date(end_date: str | None = None) -> str | None:
    """Find one recent open trading day without probing daily_basic repeatedly."""

    pro = get_pro_api()
    if pro is None:
        return None

    try:
        end = datetime.strptime(end_date, "%Y%m%d") if end_date else datetime.now()
        # Daily basic data is end-of-day data; before the afternoon close, use
        # the prior calendar day as the latest likely completed market day.
        if end_date is None and end.hour < 17:
            end -= timedelta(days=1)
        start = end - timedelta(days=21)
        calendar = pro.trade_cal(
            exchange="",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="cal_date,is_open",
        )
        if calendar is None or calendar.empty:
            _record_error("fetch_latest_open_trade_date returned empty trade calendar data.")
            return None

        open_days = calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce") == 1, "cal_date"]
        if open_days.empty:
            _record_error("fetch_latest_open_trade_date found no open day in the recent calendar range.")
            return None
        return str(open_days.astype(str).max())
    except Exception as exc:
        _record_error(f"fetch_latest_open_trade_date failed: {type(exc).__name__}: {exc}")
        return None


def fetch_daily(trade_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV market data for one trade date."""

    pro = get_pro_api()
    if pro is None:
        return pd.DataFrame()

    fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    try:
        return pro.daily(trade_date=trade_date, fields=fields)
    except Exception as exc:
        _record_error(f"fetch_daily({trade_date}) failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def fetch_trade_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch the exchange calendar once for local cache planning."""

    pro = get_pro_api()
    if pro is None:
        return pd.DataFrame()

    try:
        return pro.trade_cal(
            exchange="",
            start_date=start_date,
            end_date=end_date,
            fields="exchange,cal_date,is_open,pretrade_date",
        )
    except Exception as exc:
        _record_error(f"fetch_trade_calendar({start_date}, {end_date}) failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def save_cache(df: pd.DataFrame, path: str | Path) -> bool:
    """Save a DataFrame cache as UTF-8 CSV."""

    cache_path = Path(path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return True
    except Exception as exc:
        _record_error(f"save_cache failed for {cache_path}: {type(exc).__name__}: {exc}")
        return False


def load_cache(path: str | Path) -> pd.DataFrame:
    """Load a CSV cache file. Returns an empty DataFrame if unavailable."""

    cache_path = Path(path)
    if not cache_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(cache_path, dtype={"ts_code": str, "trade_date": str})
    except Exception as exc:
        _record_error(f"load_cache failed for {cache_path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def has_recent_success(
    status: dict,
    interface_name: str,
    now,
    max_age_seconds: int = 3600,
) -> bool:
    """Return whether an interface succeeded recently enough to skip refetch."""

    last_success_times = status.get("last_success_times", {})
    last_success = last_success_times.get(interface_name)
    if not last_success:
        return False

    try:
        last_dt = pd.to_datetime(last_success)
        return (now - last_dt).total_seconds() < max_age_seconds
    except Exception:
        return False
