#!/usr/bin/env python3
"""Build resumable full A-share daily and daily_basic history from Sina data."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_history_cache import (  # noqa: E402
    DAILY_BASIC_FIELDS,
    DAILY_FIELDS,
    HISTORY_ROOT,
    _fetch_sina_history,
    _merge_existing_daily,
    _merge_existing_daily_basic,
    _read_csv,
)


PROGRESS_PATH = HISTORY_ROOT / "full_a_share_history_progress.json"
FAILURES_PATH = HISTORY_ROOT / "full_a_share_history_failures.json"


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default.copy()
    return data if isinstance(data, dict) else default.copy()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _latest_industry_file() -> Path | None:
    files = sorted((HISTORY_ROOT / "industry").glob("industry_*.csv"))
    return files[-1] if files else None


def _load_all_codes() -> list[str]:
    """Use the already cached full industry universe as the source of truth."""
    industry_file = _latest_industry_file()
    if industry_file is None:
        raise FileNotFoundError(
            "未找到 data/history/industry/industry_*.csv，无法确认全市场股票清单。"
        )

    industry = pd.read_csv(industry_file, dtype={"ts_code": str})
    if "ts_code" not in industry.columns:
        raise ValueError(f"{industry_file} 缺少 ts_code 字段。")
    codes = (
        industry["ts_code"]
        .dropna()
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not codes:
        raise ValueError(f"{industry_file} 未提供有效股票代码。")
    return codes


def _partition(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _code6(value: object) -> str:
    extracted = pd.Series([str(value)]).str.extract(r"(\d{6})", expand=False).iloc[0]
    return str(extracted) if pd.notna(extracted) else str(value)


def _fetch_batch(
    codes: list[str],
    start_date: str,
    end_date: str,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_sina_history, code, start, end): code
            for code in codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                frame, error = future.result()
                if frame is None or frame.empty:
                    errors[code] = error or "新浪历史接口未返回有效行情。"
                else:
                    frames.append(frame)
            except Exception as exc:  # pragma: no cover - network data source
                errors[code] = f"{type(exc).__name__}: {exc}"
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        errors,
    )


def _persist_batch(frame: pd.DataFrame) -> set[str]:
    """Merge each batch into per-trading-day files immediately."""
    if frame.empty:
        return set()

    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    successful_codes: set[str] = set()
    daily_dir = HISTORY_ROOT / "daily"
    basic_dir = HISTORY_ROOT / "daily_basic"
    daily_dir.mkdir(parents=True, exist_ok=True)
    basic_dir.mkdir(parents=True, exist_ok=True)

    for trade_date, day_frame in frame.groupby("trade_date", sort=True):
        daily = day_frame.loc[:, DAILY_FIELDS].copy()
        daily_path = daily_dir / f"daily_{trade_date}.csv"
        _merge_existing_daily(daily_path, daily).to_csv(daily_path, index=False)

        basic = day_frame.loc[:, DAILY_BASIC_FIELDS].copy()
        basic_path = basic_dir / f"daily_basic_{trade_date}.csv"
        _merge_existing_daily_basic(basic_path, basic).to_csv(
            basic_path, index=False
        )
        successful_codes.update(_code6(code) for code in day_frame["ts_code"])
    return successful_codes


def _coverage(daily_dir: Path, basic_dir: Path, min_universe: int) -> dict[str, Any]:
    dates = sorted(
        path.stem.removeprefix("daily_")
        for path in daily_dir.glob("daily_*.csv")
        if path.stem.removeprefix("daily_").isdigit()
    )
    under_daily: list[str] = []
    under_basic: list[str] = []
    for trade_date in dates:
        daily = _read_csv(daily_dir / f"daily_{trade_date}.csv")
        basic = _read_csv(basic_dir / f"daily_basic_{trade_date}.csv")
        if len(daily) < min_universe:
            under_daily.append(trade_date)
        if len(basic) < min_universe:
            under_basic.append(trade_date)
    return {
        "daily_file_count": len(dates),
        "daily_undercovered_dates": under_daily,
        "daily_basic_undercovered_dates": under_basic,
        "min_universe": min_universe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从新浪历史行情构建可断点续传的全A股 daily 与 daily_basic 缓存。"
    )
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-07-17")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--max-codes",
        type=int,
        default=0,
        help="仅处理尚未完成的前 N 只股票；0 表示处理全部。",
    )
    parser.add_argument("--min-universe", type=int, default=4000)
    args = parser.parse_args()

    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers 和 --batch-size 必须大于 0。")

    all_codes = _load_all_codes()
    progress = _read_json(
        PROGRESS_PATH,
        {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "universe_size": len(all_codes),
            "completed_codes": [],
            "failed_codes": {},
        },
    )
    previous_range = (progress.get("start_date"), progress.get("end_date"))
    requested_range = (args.start_date, args.end_date)
    if previous_range != requested_range:
        progress = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "universe_size": len(all_codes),
            "completed_codes": [],
            "failed_codes": {},
        }

    completed = {_code6(code) for code in progress.get("completed_codes", [])}
    raw_failures = progress.get("failed_codes", {})
    normalized_failures = {
        _code6(code): str(message)
        for code, message in raw_failures.items()
        if _code6(code) not in completed
    }
    progress["completed_codes"] = sorted(completed)
    progress["failed_codes"] = normalized_failures
    pending = [code for code in all_codes if code not in completed]
    if args.max_codes > 0:
        pending = pending[: args.max_codes]

    print(
        "全市场缓存构建："
        f" universe={len(all_codes)}"
        f" completed={len(completed)}"
        f" pending_this_run={len(pending)}"
    )
    for batch_index, codes in enumerate(_partition(pending, args.batch_size), start=1):
        frame, errors = _fetch_batch(
            codes, args.start_date, args.end_date, args.workers
        )
        successful = _persist_batch(frame)
        completed.update(successful)
        failed = dict(progress.get("failed_codes", {}))
        for code in successful:
            failed.pop(code, None)
        failed.update(errors)
        progress.update(
            {
                "completed_codes": sorted(completed),
                "failed_codes": failed,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _write_json(PROGRESS_PATH, progress)
        print(
            f"批次 {batch_index}: requested={len(codes)}"
            f" success={len(successful)} errors={len(errors)}"
            f" completed_total={len(completed)}"
        )

    coverage = _coverage(
        HISTORY_ROOT / "daily",
        HISTORY_ROOT / "daily_basic",
        args.min_universe,
    )
    failures = {
        "source": "sina via akshare.stock_zh_a_daily",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "failed_codes": progress.get("failed_codes", {}),
        "daily_undercovered_dates": coverage["daily_undercovered_dates"],
        "daily_basic_undercovered_dates": coverage[
            "daily_basic_undercovered_dates"
        ],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(FAILURES_PATH, failures)
    print(
        "完成状态："
        f" completed_codes={len(completed)}/{len(all_codes)}"
        f" daily_files={coverage['daily_file_count']}"
        f" daily_undercovered={len(coverage['daily_undercovered_dates'])}"
        f" daily_basic_undercovered={len(coverage['daily_basic_undercovered_dates'])}"
        f" failure_log={FAILURES_PATH.relative_to(PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
