"""Build local Tushare Pro cache files.

This script is independent from the main watch workflow. It only reads
Tushare market/basic data and writes local CSV caches. It contains no trading,
order, broker operation, or account-password logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.tushare_cache_source import (  # noqa: E402
    fetch_daily,
    fetch_daily_basic,
    fetch_latest_open_trade_date,
    fetch_stock_basic,
    get_errors,
    load_cache,
    reset_errors,
    save_cache,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tushare"
STATUS_PATH = CACHE_DIR / "tushare_cache_status.json"
DAILY_BASIC_LATEST_PATH = CACHE_DIR / "daily_basic_latest.csv"
INTERFACE_OPTIONS = ("stock-basic", "daily", "daily-basic", "all")


def load_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(status: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def today_trade_date() -> str:
    return datetime.now().strftime("%Y%m%d")


def latest_daily_cache_date() -> str | None:
    """Return the latest YYYYMMDD suffix from an existing daily cache file."""
    dates = []
    for path in CACHE_DIR.glob("daily_*.csv"):
        suffix = path.stem.removeprefix("daily_")
        if len(suffix) == 8 and suffix.isdigit():
            dates.append(suffix)
    return max(dates) if dates else None


def resolve_latest_daily_basic_date() -> tuple[str, str]:
    """Resolve one likely latest completed trading day without daily_basic probing."""
    calendar_date = fetch_latest_open_trade_date()
    if calendar_date:
        return calendar_date, "trade_calendar"

    cached_date = latest_daily_cache_date()
    if cached_date:
        return cached_date, "daily_cache"

    fallback = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    return fallback, "calendar_fallback"


def parse_dt(value: str | None):
    if not value:
        return None
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def get_last_success_time(previous_status: dict, interface_key: str) -> str | None:
    direct_key = f"{interface_key}_last_success_time"
    if previous_status.get(direct_key):
        return previous_status.get(direct_key)

    legacy_times = previous_status.get("last_success_times", {})
    return legacy_times.get(interface_key)


def is_recent_success(previous_status: dict, interface_key: str, now: datetime, max_age_seconds: int = 3600) -> bool:
    last_success = parse_dt(get_last_success_time(previous_status, interface_key))
    if last_success is None:
        return False
    return (now - last_success).total_seconds() < max_age_seconds


def selected_interfaces(only: str) -> List[str]:
    if only == "all":
        return ["stock_basic", "daily_basic", "daily"]
    if only == "stock-basic":
        return ["stock_basic"]
    if only == "daily-basic":
        return ["daily_basic"]
    return ["daily"]


def cache_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_interface(
    interface_key: str,
    cache_path: Path,
    fetcher: Callable[[], pd.DataFrame],
    previous_status: dict,
    now: datetime,
    force: bool,
) -> Tuple[pd.DataFrame, bool, bool, str, str | None]:
    """Fetch one interface, skip existing cache, or respect per-interface guard.

    Returns: dataframe, success, skipped, warning, fail_reason.
    """

    if not force and cache_exists(cache_path):
        cached = load_cache(cache_path)
        if not cached.empty:
            warning = f"{interface_key} 缓存已存在，本次跳过请求。"
            print(warning)
            return cached, True, True, warning, None

        warning = f"{interface_key} 缓存文件存在但读取为空，本次保守跳过；如需重建请使用 --force。"
        print(warning)
        return pd.DataFrame(), False, True, warning, "cache_empty"

    if not force and is_recent_success(previous_status, interface_key, now):
        warning = f"{interface_key} 1 小时内已成功调用，为避免触发限频，本次不重复请求。"
        print(warning)
        return pd.DataFrame(), False, True, warning, "rate_limit_guard"

    df = fetcher()
    if df.empty:
        warning = f"{interface_key} 返回空数据，可能是非交易日、数据尚未更新、权限不足或频率限制。"
        print(warning)
        return df, False, False, warning, "empty_or_failed"

    save_cache(df, cache_path)
    print(f"{interface_key} 获取成功，行数: {len(df)}，缓存: {cache_path}")
    return df, True, False, "", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local Tushare cache files.")
    parser.add_argument("--trade-date", help="Trade date in YYYYMMDD format. Defaults to today.")
    parser.add_argument(
        "--latest-daily-basic",
        action="store_true",
        help="Resolve one latest completed trading day and build only its daily_basic cache.",
    )
    parser.add_argument("--force", action="store_true", help="Force API calls even when cache exists or guard applies.")
    parser.add_argument("--only", choices=INTERFACE_OPTIONS, default="all", help="Build only one cache group.")
    args = parser.parse_args()

    now = datetime.now()
    build_time = now.strftime("%Y-%m-%d %H:%M:%S")
    previous_status = load_status()
    reset_errors()

    latest_date_source = None
    if args.latest_daily_basic:
        if args.trade_date:
            trade_date = args.trade_date
            latest_date_source = "explicit_trade_date"
        else:
            trade_date, latest_date_source = resolve_latest_daily_basic_date()
        args.only = "daily-basic"
    else:
        trade_date = args.trade_date or today_trade_date()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stock_basic_path = CACHE_DIR / "stock_basic.csv"
    daily_basic_path = CACHE_DIR / f"daily_basic_{trade_date}.csv"
    daily_path = CACHE_DIR / f"daily_{trade_date}.csv"

    paths = {
        "stock_basic": stock_basic_path,
        "daily_basic": daily_basic_path,
        "daily": daily_path,
    }
    fetchers = {
        "stock_basic": fetch_stock_basic,
        "daily_basic": lambda: fetch_daily_basic(trade_date),
        "daily": lambda: fetch_daily(trade_date),
    }

    targets = selected_interfaces(args.only)
    warnings: List[str] = []
    skipped_interfaces: List[str] = []
    failed_interfaces: List[str] = []
    dataframes: Dict[str, pd.DataFrame] = {
        "stock_basic": load_cache(stock_basic_path),
        "daily_basic": load_cache(daily_basic_path),
        "daily": load_cache(daily_path),
    }
    success = {
        "stock_basic": not dataframes["stock_basic"].empty,
        "daily_basic": not dataframes["daily_basic"].empty,
        "daily": not dataframes["daily"].empty,
    }

    last_success = {
        "stock_basic": get_last_success_time(previous_status, "stock_basic"),
        "daily_basic": get_last_success_time(previous_status, "daily_basic"),
        "daily": get_last_success_time(previous_status, "daily"),
    }

    for interface_key in targets:
        df, ok, skipped, warning, fail_reason = run_interface(
            interface_key,
            paths[interface_key],
            fetchers[interface_key],
            previous_status,
            now,
            args.force,
        )
        dataframes[interface_key] = df
        success[interface_key] = ok
        if warning:
            warnings.append(warning)
        if skipped:
            skipped_interfaces.append(interface_key)
        if not ok and not skipped:
            failed_interfaces.append(interface_key)
        if ok and not skipped and fail_reason is None:
            last_success[interface_key] = build_time

    if success["daily_basic"] and not dataframes["daily_basic"].empty:
        # This alias is intentionally a local copy, so the dashboard/main
        # never need to call Tushare merely to locate the newest cache.
        save_cache(dataframes["daily_basic"], DAILY_BASIC_LATEST_PATH)

    latest_daily_basic_df = dataframes["daily_basic"]
    if latest_daily_basic_df.empty:
        latest_daily_basic_df = load_cache(DAILY_BASIC_LATEST_PATH)

    errors = get_errors()
    stock_basic_df = dataframes["stock_basic"]
    daily_basic_df = dataframes["daily_basic"]
    daily_df = dataframes["daily"]

    float_share = pd.to_numeric(latest_daily_basic_df.get("float_share"), errors="coerce")
    float_share_valid_count = int((float_share > 0).sum()) if float_share is not None else 0
    float_share_coverage = (
        round(float_share_valid_count / len(latest_daily_basic_df), 6)
        if len(latest_daily_basic_df)
        else 0.0
    )
    daily_basic_failed = "daily_basic" in failed_interfaces
    daily_basic_next_retry_time = (
        (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        if daily_basic_failed and any("频率" in error or "rate" in error.lower() for error in errors)
        else None
    )

    status = {
        "build_time": build_time,
        "trade_date": trade_date,
        "last_run_only": args.only,
        "stock_basic_success": bool(success["stock_basic"]),
        "stock_basic_rows": int(len(stock_basic_df)),
        "daily_basic_success": bool(not latest_daily_basic_df.empty),
        "daily_basic_rows": int(len(latest_daily_basic_df)),
        "daily_success": bool(success["daily"]),
        "daily_rows": int(len(daily_df)),
        "stock_basic_cache_exists": cache_exists(stock_basic_path),
        "daily_cache_exists": cache_exists(daily_path),
        "daily_basic_cache_exists": cache_exists(daily_basic_path),
        "daily_basic_latest_exists": cache_exists(DAILY_BASIC_LATEST_PATH),
        "stock_basic_cache_path": str(stock_basic_path),
        "daily_cache_path": str(daily_path),
        "daily_basic_cache_path": str(daily_basic_path),
        "daily_basic_latest_path": str(DAILY_BASIC_LATEST_PATH),
        "daily_basic_trade_date": trade_date,
        "daily_basic_trade_date_source": latest_date_source,
        "stock_basic_last_success_time": last_success["stock_basic"],
        "daily_last_success_time": last_success["daily"],
        "daily_basic_last_success_time": last_success["daily_basic"],
        "has_turnover_rate": "turnover_rate" in latest_daily_basic_df.columns,
        "has_volume_ratio": "volume_ratio" in latest_daily_basic_df.columns,
        "has_float_share": "float_share" in latest_daily_basic_df.columns,
        "has_free_share": "free_share" in latest_daily_basic_df.columns,
        "float_share_valid_count": float_share_valid_count,
        "float_share_coverage": float_share_coverage,
        "float_share_unit": "万股",
        "daily_basic_last_attempt_time": build_time if "daily_basic" in targets else previous_status.get("daily_basic_last_attempt_time"),
        "daily_basic_next_retry_time": daily_basic_next_retry_time,
        "has_daily_ohlc": all(col in daily_df.columns for col in ["open", "high", "low", "close"]),
        "skipped_interfaces": skipped_interfaces,
        "failed_interfaces": failed_interfaces,
        "error_message": "; ".join(errors),
        "warning_message": "; ".join(warnings),
        "last_success_times": {
            "stock_basic": last_success["stock_basic"],
            "daily_basic": last_success["daily_basic"],
            "daily": last_success["daily"],
        },
        "cache_files": {
            "stock_basic": str(stock_basic_path),
            "daily_basic": str(daily_basic_path),
            "daily": str(daily_path),
            "daily_basic_latest": str(DAILY_BASIC_LATEST_PATH),
        },
        "skipped_by_rate_limit_guard": {
            "stock_basic": "stock_basic" in skipped_interfaces and not cache_exists(stock_basic_path),
            "daily_basic": "daily_basic" in skipped_interfaces and not cache_exists(daily_basic_path),
            "daily": "daily" in skipped_interfaces and not cache_exists(daily_path),
        },
    }

    write_status(status)

    print("\nTushare 缓存构建结果")
    print(f"交易日期: {trade_date}")
    print(f"本次范围: {args.only}")
    print(f"stock_basic: 成功={success['stock_basic']} 行数={len(stock_basic_df)} 缓存存在={status['stock_basic_cache_exists']}")
    print(f"daily_basic: 成功={success['daily_basic']} 行数={len(daily_basic_df)} 缓存存在={status['daily_basic_cache_exists']}")
    print(f"daily: 成功={success['daily']} 行数={len(daily_df)} 缓存存在={status['daily_cache_exists']}")
    print(f"daily_basic 包含 turnover_rate: {'是' if status['has_turnover_rate'] else '否'}")
    print(f"daily_basic 包含 volume_ratio: {'是' if status['has_volume_ratio'] else '否'}")
    print(f"daily 包含 OHLC: {'是' if status['has_daily_ohlc'] else '否'}")
    if skipped_interfaces:
        print(f"跳过接口: {', '.join(skipped_interfaces)}")
    if failed_interfaces:
        print(f"失败接口: {', '.join(failed_interfaces)}")
    if warnings:
        print(f"提示: {status['warning_message']}")
    if errors:
        print(f"错误: {status['error_message']}")
    print(f"状态文件: {STATUS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
