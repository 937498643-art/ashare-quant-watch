"""Build a resumable history-cache layout for V3 backtests.

``--source local`` only normalizes existing local files and is safe offline.
``--source sina`` populates the cache from AKShare's Sina historical endpoint.
The backtest itself reads only ``data/history`` and never makes network calls.
Missing historical fields are recorded explicitly rather than fabricated.
"""

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


HISTORY_DIR = PROJECT_ROOT / "data" / "cache" / "history"
HISTORY_ROOT = PROJECT_ROOT / "data" / "history"
DAILY_FIELDS = ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg")
DAILY_BASIC_FIELDS = ("ts_code", "trade_date", "turnover_rate", "circ_mv")
INDUSTRY_FIELDS = ("ts_code", "industry")
FLOAT_SHARE_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "cache" / "tushare" / "daily_basic_latest.csv"


def _ts_code(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"ts_code": str, "trade_date": str})
    except Exception:
        return pd.DataFrame()


def _valid_file(path: Path, fields: tuple[str, ...]) -> bool:
    data = _read_csv(path)
    return not data.empty and set(fields).issubset(data.columns)


def _field(frame: pd.DataFrame, *names: str) -> pd.Series:
    name = next((item for item in names if item in frame.columns), None)
    return frame[name] if name else pd.Series(pd.NA, index=frame.index)


def _history_rows(history_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(history_dir.glob("*.csv")):
        code = path.name.split("_")[0]
        if not code.isdigit():
            continue
        raw = _read_csv(path)
        if raw.empty:
            continue
        data = pd.DataFrame(
            {
                "ts_code": _ts_code(code.zfill(6)),
                "trade_date": pd.to_datetime(_field(raw, "trade_date", "date", "日期"), errors="coerce"),
                "open": pd.to_numeric(_field(raw, "open", "开盘"), errors="coerce"),
                "high": pd.to_numeric(_field(raw, "high", "最高"), errors="coerce"),
                "low": pd.to_numeric(_field(raw, "low", "最低"), errors="coerce"),
                "close": pd.to_numeric(_field(raw, "close", "收盘"), errors="coerce"),
                "vol": pd.to_numeric(_field(raw, "vol", "volume", "成交量"), errors="coerce"),
                "amount": pd.to_numeric(_field(raw, "amount", "成交额"), errors="coerce"),
                "pct_chg": pd.to_numeric(_field(raw, "pct_chg", "涨跌幅"), errors="coerce"),
                "turnover_rate": pd.to_numeric(_field(raw, "turnover_rate", "turnover", "换手率"), errors="coerce"),
                "circ_mv": pd.to_numeric(_field(raw, "circ_mv", "流通市值"), errors="coerce"),
            }
        )
        data = data.dropna(subset=["trade_date", "close"])
        data = data[data["trade_date"].between(start, end)]
        if not data.empty:
            frames.append(data)
    if not frames:
        return pd.DataFrame(columns=[*DAILY_FIELDS, "turnover_rate", "circ_mv"])
    output = pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"])
    output["pct_chg"] = output["pct_chg"].where(
        output["pct_chg"].notna(),
        output.groupby("ts_code")["close"].pct_change(fill_method=None).mul(100),
    )
    output["trade_date"] = output["trade_date"].dt.strftime("%Y%m%d")
    return output.drop_duplicates(["ts_code", "trade_date"], keep="last").reset_index(drop=True)


def _history_codes(history_dir: Path) -> list[str]:
    return sorted({path.name.split("_")[0].zfill(6) for path in history_dir.glob("*.csv") if path.name.split("_")[0].isdigit()})


def _fetch_sina_history(code: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    """Fetch one stock from AKShare's Sina daily endpoint (not Eastmoney)."""
    try:
        import akshare as ak

        # Beijing Stock Exchange symbols include legacy 4/8 prefixes and
        # current 92xxxx prefixes. They are not Shanghai 9xxxx symbols.
        prefix = (
            "bj"
            if code.startswith(("4", "8", "92"))
            else "sh"
            if code.startswith(("6", "9"))
            else "sz"
        )
        raw = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="")
    except Exception as exc:
        return pd.DataFrame(), f"{code}: {type(exc).__name__}: {exc}"
    required = {"date", "open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"}
    if raw is None or raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame(), f"{code}: 新浪历史日线为空或字段不完整。"
    data = pd.DataFrame(
        {
            "ts_code": _ts_code(code),
            "trade_date": pd.to_datetime(raw["date"], errors="coerce"),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "vol": pd.to_numeric(raw["volume"], errors="coerce"),
            "amount": pd.to_numeric(raw["amount"], errors="coerce"),
            # Sina expresses turnover as a ratio; the scoring cache uses percent.
            "turnover_rate": pd.to_numeric(raw["turnover"], errors="coerce") * 100,
            "outstanding_share": pd.to_numeric(raw["outstanding_share"], errors="coerce"),
        }
    )
    data = data[data["trade_date"].between(start, end)].dropna(subset=["trade_date", "close"])
    data = data.sort_values("trade_date")
    data["pct_chg"] = data["close"].pct_change(fill_method=None).mul(100)
    # Sina outstanding_share is in shares; circ_mv must be in 万元.
    data["circ_mv"] = data["close"] * data["outstanding_share"] / 10_000
    data["trade_date"] = data["trade_date"].dt.strftime("%Y%m%d")
    return data.drop_duplicates(["ts_code", "trade_date"], keep="last"), None


def _sina_history_rows(
    history_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    workers: int,
    max_codes: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    codes = _history_codes(history_dir)
    if max_codes > 0:
        codes = codes[:max_codes]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    if workers <= 1:
        for code in codes:
            frame, error = _fetch_sina_history(code, start, end)
            if not frame.empty:
                frames.append(frame)
            if error:
                errors.append(error)
        if not frames:
            return pd.DataFrame(columns=[*DAILY_FIELDS, "turnover_rate", "circ_mv"]), errors
        return pd.concat(frames, ignore_index=True), errors
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_fetch_sina_history, code, start, end): code for code in codes}
        for future in as_completed(futures):
            frame, error = future.result()
            if not frame.empty:
                frames.append(frame)
            if error:
                errors.append(error)
    if not frames:
        return pd.DataFrame(columns=[*DAILY_FIELDS, "turnover_rate", "circ_mv"]), errors
    return pd.concat(frames, ignore_index=True), errors


def _load_industries() -> pd.DataFrame:
    paths = [
        HISTORY_ROOT / "stock_basic_industry.csv",
        PROJECT_ROOT / "data" / "reports" / "backtest_cache" / "stock_basic_industry.csv",
    ]
    for path in paths:
        raw = _read_csv(path)
        if raw.empty or "ts_code" not in raw.columns or "industry" not in raw.columns:
            continue
        data = raw[["ts_code", "industry"]].copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["industry"] = data["industry"].astype("string").str.strip()
        data = data[data["industry"].notna() & data["industry"].ne("")]
        if not data.empty:
            return data.drop_duplicates("ts_code")
    return pd.DataFrame(columns=list(INDUSTRY_FIELDS))


def _load_float_share_snapshot() -> tuple[pd.DataFrame, str | None]:
    """Load an existing local float-share snapshot without making any request."""
    raw = _read_csv(FLOAT_SHARE_SNAPSHOT_PATH)
    required = {"ts_code", "float_share"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame(columns=["ts_code", "float_share"]), None
    data = raw[["ts_code", "float_share"]].copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data["float_share"] = pd.to_numeric(data["float_share"], errors="coerce")
    snapshot_date = None
    if "trade_date" in raw.columns:
        dates = raw["trade_date"].astype(str).str.replace(".0", "", regex=False).str.strip()
        snapshot_date = next((value for value in dates if value.isdigit() and len(value) == 8), None)
    return data.dropna(subset=["float_share"]).query("float_share > 0").drop_duplicates("ts_code"), snapshot_date


def _merge_existing_daily(path: Path, local_rows: pd.DataFrame) -> pd.DataFrame:
    existing = _read_csv(path)
    if not existing.empty:
        for field in DAILY_FIELDS:
            if field not in existing.columns:
                existing[field] = pd.NA
        existing = existing[list(DAILY_FIELDS)]
    combined = pd.concat([existing, local_rows[list(DAILY_FIELDS)]], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=list(DAILY_FIELDS))
    return combined.drop_duplicates("ts_code", keep="last").sort_values("ts_code")


def _merge_existing_daily_basic(path: Path, local_rows: pd.DataFrame) -> pd.DataFrame:
    existing = _read_csv(path)
    if not existing.empty:
        existing = existing[[field for field in DAILY_BASIC_FIELDS if field in existing.columns]]
        for field in DAILY_BASIC_FIELDS:
            if field not in existing.columns:
                existing[field] = pd.NA
    combined = pd.concat([existing, local_rows[list(DAILY_BASIC_FIELDS)]], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=list(DAILY_BASIC_FIELDS))
    return combined.drop_duplicates("ts_code", keep="last").sort_values("ts_code")


def _completeness(directory: Path, prefix: str, fields: tuple[str, ...], dates: list[str]) -> dict[str, Any]:
    files = [directory / f"{prefix}_{date}.csv" for date in dates]
    valid = [path for path in files if _valid_file(path, fields)]
    rows = 0
    field_cells = 0
    populated_cells = 0
    for path in valid:
        data = _read_csv(path)
        rows += len(data)
        field_cells += len(data) * len(fields)
        populated_cells += int(data[list(fields)].notna().sum().sum())
    return {
        "expected_files": len(files),
        "valid_files": len(valid),
        "missing_dates": [path.stem.removeprefix(f"{prefix}_") for path in files if path not in valid],
        "rows": rows,
        "field_completeness": round(populated_cells / field_cells, 6) if field_cells else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从本地缓存构建离线 V3 历史数据目录")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-07-18")
    parser.add_argument("--source-history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--source", choices=("local", "sina"), default="local")
    parser.add_argument("--workers", type=int, default=1, help="仅新浪来源使用的并发请求数；默认1，避免接口运行时并发崩溃。")
    parser.add_argument("--max-codes", type=int, default=0, help="仅新浪来源使用；限制处理股票数以便分批断点构建。")
    parser.add_argument("--overwrite", action="store_true", help="重建已有的完整日文件。")
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    daily_dir = args.history_root / "daily"
    daily_basic_dir = args.history_root / "daily_basic"
    industry_dir = args.history_root / "industry"
    for directory in (daily_dir, daily_basic_dir, industry_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_errors: list[str] = []
    if args.source == "sina":
        local, source_errors = _sina_history_rows(
            args.source_history_dir, start, end, args.workers, args.max_codes
        )
    else:
        local = _history_rows(args.source_history_dir, start, end)
    dates = sorted(local["trade_date"].dropna().unique().tolist()) if not local.empty else []
    industries = _load_industries()
    float_shares, float_share_snapshot_date = _load_float_share_snapshot()
    daily_created = 0
    daily_basic_created = 0
    industry_created = 0
    daily_basic_missing_source_dates: list[str] = []

    for trade_date in dates:
        day_rows = local[local["trade_date"] == trade_date].copy()
        daily_path = daily_dir / f"daily_{trade_date}.csv"
        if args.source == "sina" or args.overwrite or not _valid_file(daily_path, DAILY_FIELDS):
            _merge_existing_daily(daily_path, day_rows).to_csv(daily_path, index=False, encoding="utf-8-sig")
            daily_created += 1

        industry_path = industry_dir / f"industry_{trade_date}.csv"
        if (args.overwrite or not _valid_file(industry_path, INDUSTRY_FIELDS)) and not industries.empty:
            industries.to_csv(industry_path, index=False, encoding="utf-8-sig")
            industry_created += 1

        basic_path = daily_basic_dir / f"daily_basic_{trade_date}.csv"
        basic_rows = day_rows.dropna(subset=["turnover_rate", "circ_mv"])[list(DAILY_BASIC_FIELDS)].copy()
        if basic_rows.empty and not float_shares.empty:
            calculated = day_rows.merge(float_shares, on="ts_code", how="left")
            # Local history volume is quoted in hands, while float_share is in
            # 万股.  Therefore turnover_rate (%) = volume_hand / float_share.
            calculated["turnover_rate"] = calculated["vol"] / calculated["float_share"]
            # close (yuan) × float_share (万股) gives circ_mv in 万元.
            calculated["circ_mv"] = calculated["close"] * calculated["float_share"]
            basic_rows = calculated.dropna(subset=["turnover_rate", "circ_mv"])[list(DAILY_BASIC_FIELDS)].copy()
        if args.source == "sina" or args.overwrite or not _valid_file(basic_path, DAILY_BASIC_FIELDS):
            if basic_rows.empty:
                daily_basic_missing_source_dates.append(trade_date)
            else:
                _merge_existing_daily_basic(basic_path, basic_rows).to_csv(basic_path, index=False, encoding="utf-8-sig")
                daily_basic_created += 1

    status = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0 if args.source == "local" else int(local["ts_code"].nunique()) if not local.empty else 0,
        "source": "AKShare 新浪历史日线 + 本地行业映射" if args.source == "sina" else "本地 data/cache/history K线 + 本地行业映射",
        "source_errors": source_errors[:100],
        "requested_range": [args.start_date, args.end_date],
        "available_dates": dates,
        "stock_count": int(local["ts_code"].nunique()) if not local.empty else 0,
        "industry_stock_count": int(industries["ts_code"].nunique()) if not industries.empty else 0,
        "float_share_snapshot": {
            "path": str(FLOAT_SHARE_SNAPSHOT_PATH),
            "snapshot_date": float_share_snapshot_date,
            "stock_count": int(float_shares["ts_code"].nunique()) if not float_shares.empty else 0,
            "calculation": "换手率(%) = 历史成交量(手) ÷ 流通股本(万股)；circ_mv(万元) = 收盘价(元) × 流通股本(万股)。",
            "limitation": "流通股本为本地快照，不是逐日股本；遇到拆股、增发、限售解禁等变动时，历史值为近似值。",
            "used_for_daily_basic": args.source != "sina",
        },
        "created": {"daily": daily_created, "daily_basic": daily_basic_created, "industry": industry_created},
        "daily_basic_missing_source_dates": daily_basic_missing_source_dates,
        "daily": _completeness(daily_dir, "daily", DAILY_FIELDS, dates),
        "daily_basic": _completeness(daily_basic_dir, "daily_basic", DAILY_BASIC_FIELDS, dates),
        "industry": _completeness(industry_dir, "industry", INDUSTRY_FIELDS, dates),
    }
    (args.history_root / "history_cache_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
