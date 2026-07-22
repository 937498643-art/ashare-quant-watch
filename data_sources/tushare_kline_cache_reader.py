"""Read and manually update local Tushare daily K-line caches.

The dashboard reads local files only. A Tushare request happens solely when a
user explicitly invokes the single-stock cache update command or button.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data_sources.tushare_cache_source import get_pro_api, save_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUSHARE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tushare"
STOCK_KLINE_CACHE_DIR = TUSHARE_CACHE_DIR / "kline"
RAW_KLINE_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
KLINE_COLUMNS = [*RAW_KLINE_COLUMNS, "ma5", "ma10", "ma20", "ma60"]


def normalize_code(code: str) -> str:
    digits = "".join(ch for ch in str(code or "") if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6) if digits else ""


def tushare_ts_code(code: str) -> str:
    """Convert a six-digit A-share symbol to its Tushare market code."""
    symbol = normalize_code(code)
    if symbol.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{symbol}.SH"
    if symbol.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{symbol}.SZ"
    if symbol.startswith(("4", "8", "9")):
        return f"{symbol}.BJ"
    return ""


def stock_kline_cache_path(code: str) -> Path:
    return STOCK_KLINE_CACHE_DIR / f"{normalize_code(code)}.csv"


def list_daily_cache_files(cache_dir: Path = TUSHARE_CACHE_DIR) -> list[Path]:
    """List only full-market daily files, never daily_basic caches."""
    if not cache_dir.exists():
        return []
    return sorted(
        path
        for path in cache_dir.glob("daily_*.csv")
        if re.fullmatch(r"daily_\d{8}\.csv", path.name)
    )


def load_daily_cache_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
    except Exception:
        return pd.DataFrame()


def _empty_kline() -> pd.DataFrame:
    return pd.DataFrame(columns=KLINE_COLUMNS)


def _prepare_kline_frame(data: pd.DataFrame, code: str, days: int, keep_all_gaps: bool) -> pd.DataFrame:
    """Normalize one target stock, then compute moving averages in date order."""
    target_ts_code = tushare_ts_code(code)
    if data is None or data.empty or not target_ts_code or "ts_code" not in data.columns:
        return _empty_kline()

    output = data.copy()
    output["ts_code"] = output["ts_code"].astype(str).str.upper().str.strip()
    output = output[output["ts_code"] == target_ts_code].copy()
    if output.empty or "trade_date" not in output.columns:
        return _empty_kline()

    output["trade_date"] = pd.to_datetime(output["trade_date"], errors="coerce")
    for column in RAW_KLINE_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=["trade_date", "open", "high", "low", "close"])
    output = output[(output[["open", "high", "low", "close"]] > 0).all(axis=1)]
    output = output[output["high"] >= output["low"]]
    output = output.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    output = output.sort_values("trade_date").reset_index(drop=True)
    if output.empty:
        return _empty_kline()

    if not keep_all_gaps:
        gaps = output["trade_date"].diff().dt.days
        recent_gap_indexes = output.index[gaps.gt(14)]
        if len(recent_gap_indexes):
            output = output.loc[recent_gap_indexes[-1] :].reset_index(drop=True)

    output = output.tail(max(int(days), 1)).reset_index(drop=True)
    output["ma5"] = output["close"].rolling(window=5, min_periods=5).mean()
    output["ma10"] = output["close"].rolling(window=10, min_periods=10).mean()
    output["ma20"] = output["close"].rolling(window=20, min_periods=20).mean()
    output["ma60"] = output["close"].rolling(window=60, min_periods=60).mean()
    return output[KLINE_COLUMNS]


def _load_stock_kline_cache(code: str) -> pd.DataFrame:
    path = stock_kline_cache_path(code)
    if not path.exists():
        return pd.DataFrame()
    return load_daily_cache_file(path)


def read_tushare_kline_from_cache(code: str, days: int = 120, cache_dir: Path = TUSHARE_CACHE_DIR) -> pd.DataFrame:
    """Read local single-stock history first, then fall back to daily market caches."""
    stock_history = _load_stock_kline_cache(code)
    if not stock_history.empty:
        return _prepare_kline_frame(stock_history, code, days, keep_all_gaps=True)

    target_ts_code = tushare_ts_code(code)
    if not target_ts_code:
        return _empty_kline()

    frames = []
    for path in list_daily_cache_files(cache_dir):
        data = load_daily_cache_file(path)
        if data.empty or "ts_code" not in data.columns:
            continue
        matched = data[data["ts_code"].astype(str).str.upper().str.strip() == target_ts_code].copy()
        if not matched.empty:
            frames.append(matched)
    if not frames:
        return _empty_kline()
    return _prepare_kline_frame(pd.concat(frames, ignore_index=True), code, days, keep_all_gaps=False)


def update_stock_kline_cache(code: str, days: int = 120, end_date: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Manually fill only missing ranges for one stock's local daily cache."""
    target_ts_code = tushare_ts_code(code)
    if not target_ts_code:
        return _empty_kline(), "股票代码无法识别。"

    days = max(int(days), 1)
    end = pd.to_datetime(end_date or datetime.now().strftime("%Y%m%d"), errors="coerce")
    if pd.isna(end):
        return _empty_kline(), "结束日期格式无效。"
    end = end.normalize()

    existing_raw = _load_stock_kline_cache(code)
    existing = _prepare_kline_frame(existing_raw, code, days=10_000, keep_all_gaps=True)
    target_start = end - timedelta(days=max(days * 3, 120))
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if existing.empty:
        ranges.append((target_start, end))
    else:
        first_date = existing["trade_date"].min()
        last_date = existing["trade_date"].max()
        if len(existing) < days and first_date > target_start:
            ranges.append((target_start, first_date - timedelta(days=1)))
        if last_date < end:
            ranges.append((last_date + timedelta(days=1), end))

    if not ranges:
        return existing.tail(days).reset_index(drop=True), None

    pro = get_pro_api()
    if pro is None:
        return existing.tail(days).reset_index(drop=True), "Tushare 初始化失败，未请求新数据。"

    fetched_frames = []
    fields = ",".join(RAW_KLINE_COLUMNS)
    for start, finish in ranges[:2]:
        if start > finish:
            continue
        try:
            fetched = pro.daily(
                ts_code=target_ts_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=finish.strftime("%Y%m%d"),
                fields=fields,
            )
        except Exception as exc:
            return existing.tail(days).reset_index(drop=True), f"Tushare daily 请求失败：{type(exc).__name__}: {exc}"
        if fetched is not None and not fetched.empty:
            fetched_frames.append(fetched)

    if not fetched_frames:
        return existing.tail(days).reset_index(drop=True), "Tushare daily 未返回可用历史数据。"

    merged = pd.concat([existing_raw, *fetched_frames], ignore_index=True)
    merged["ts_code"] = merged.get("ts_code", pd.Series(dtype="string")).astype(str).str.upper().str.strip()
    merged["trade_date"] = pd.to_datetime(merged.get("trade_date"), errors="coerce")
    merged = merged.dropna(subset=["trade_date"])
    merged = merged.drop_duplicates(subset=["ts_code", "trade_date"], keep="last").sort_values("trade_date")
    merged["trade_date"] = merged["trade_date"].dt.strftime("%Y%m%d")
    merged = merged[[column for column in RAW_KLINE_COLUMNS if column in merged.columns]]
    if not save_cache(merged, stock_kline_cache_path(code)):
        return _prepare_kline_frame(merged, code, days, keep_all_gaps=True), "单股 K 线缓存保存失败。"
    return _prepare_kline_frame(merged, code, days, keep_all_gaps=True), None
