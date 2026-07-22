"""Merge Sina realtime quotes with local Tushare cache as reference data.

This module is a read-only test layer. It does not create official strategy
candidates and does not contain any trading, order, broker, or account logic.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUSHARE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tushare"
STATUS_PATH = TUSHARE_CACHE_DIR / "tushare_cache_status.json"

OUTPUT_COLUMNS = [
    "code",
    "market_code",
    "name",
    "price",
    "pct_chg",
    "amount",
    "amount_display",
    "turnover_rate_ref",
    "volume_ratio_ref",
    "total_mv",
    "circ_mv",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tushare_trade_date",
    "realtime_source",
    "reference_source",
    "data_source_level",
    "is_realtime_turnover",
    "is_realtime_volume_ratio",
    "allow_official_strategy_candidate",
    "allow_reference_candidate",
    "warning_message",
]


def format_amount(value: Any) -> str:
    number = to_number(value)
    if number is None or number <= 0:
        return "--"
    if number >= 100000000:
        return f"{number / 100000000:.2f} 亿"
    if number >= 10000:
        return f"{number / 10000:.2f} 万"
    return f"{number:.2f}"


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "--", "nan", "None"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def normalize_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6) if digits else ""


def make_market_code(code: Any) -> str:
    symbol = normalize_code(code)
    if not symbol:
        return ""
    if symbol.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{symbol}.SH"
    if symbol.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{symbol}.SZ"
    if symbol.startswith(("4", "8", "9")):
        return f"{symbol}.BJ"
    return symbol


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def normalize_sina_spot(sina_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Sina spot quotes to the fields needed for reference merge."""

    if sina_df is None or sina_df.empty:
        return pd.DataFrame(columns=["code", "market_code", "name", "price", "pct_chg", "amount", "amount_display"])

    code_col = find_column(sina_df.columns, ["code", "symbol", "股票代码", "代码", "证券代码"])
    name_col = find_column(sina_df.columns, ["name", "股票名称", "名称", "证券简称"])
    price_col = find_column(sina_df.columns, ["trade", "price", "最新价", "现价", "当前价"])
    pct_col = find_column(sina_df.columns, ["changepercent", "pct_chg", "涨跌幅", "涨幅"])
    amount_col = find_column(sina_df.columns, ["amount", "成交额"])

    data = pd.DataFrame()
    data["code"] = sina_df[code_col].map(normalize_code) if code_col else ""
    data["market_code"] = data["code"].map(make_market_code)
    data["name"] = sina_df[name_col].astype(str) if name_col else ""
    data["price"] = sina_df[price_col].map(to_number) if price_col else None
    data["pct_chg"] = sina_df[pct_col].map(to_number) if pct_col else None
    data["amount"] = sina_df[amount_col].map(to_number) if amount_col else None
    data["amount_display"] = data["amount"].map(format_amount)
    data = data[data["code"].astype(str).str.len() == 6].copy()
    return data


def read_tushare_status() -> Dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status_read_error": f"{type(exc).__name__}: {exc}"}


def load_csv(path: Path, dtype: dict | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype or {})
    except Exception:
        return pd.DataFrame()


def load_tushare_caches(trade_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    stock_basic = load_csv(TUSHARE_CACHE_DIR / "stock_basic.csv", dtype={"ts_code": str, "symbol": str})
    daily_basic = load_csv(
        TUSHARE_CACHE_DIR / f"daily_basic_{trade_date}.csv",
        dtype={"ts_code": str, "trade_date": str},
    )
    daily = load_csv(
        TUSHARE_CACHE_DIR / f"daily_{trade_date}.csv",
        dtype={"ts_code": str, "trade_date": str},
    )
    status = read_tushare_status()
    return stock_basic, daily_basic, daily, status


def prepare_stock_basic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts_code" not in df.columns:
        return pd.DataFrame(columns=["code", "stock_basic_name"])
    out = df.copy()
    out["code"] = out["ts_code"].astype(str).str.split(".").str[0].map(normalize_code)
    rename = {}
    if "name" in out.columns:
        rename["name"] = "stock_basic_name"
    return out.rename(columns=rename)


def prepare_daily_basic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts_code" not in df.columns:
        return pd.DataFrame(columns=["code", "turnover_rate_ref", "volume_ratio_ref", "total_mv", "circ_mv"])
    out = df.copy()
    out["code"] = out["ts_code"].astype(str).str.split(".").str[0].map(normalize_code)
    rename = {
        "turnover_rate": "turnover_rate_ref",
        "volume_ratio": "volume_ratio_ref",
    }
    out = out.rename(columns=rename)
    keep = ["code", "turnover_rate_ref", "volume_ratio_ref", "total_mv", "circ_mv"]
    for col in keep:
        if col not in out.columns:
            out[col] = None
    return out[keep]


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts_code" not in df.columns:
        return pd.DataFrame(columns=["code", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["code"] = out["ts_code"].astype(str).str.split(".").str[0].map(normalize_code)
    if "vol" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"vol": "volume"})
    keep = ["code", "open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in out.columns:
            out[col] = None
    return out[keep]


def build_warning_message(trade_date: str, status: Dict[str, Any], has_daily_basic: bool, has_daily: bool) -> str:
    warnings = ["Tushare 指标为日级缓存参考数据，非盘中实时。"]
    today = datetime.now().strftime("%Y%m%d")
    if trade_date != today:
        warnings.append(f"Tushare 缓存日期为 {trade_date}，不是今天 {today}。")
    if not has_daily_basic:
        warnings.append("缺少 daily_basic 缓存，参考换手率和参考量比为空。")
    if not has_daily:
        warnings.append("缺少 daily 缓存，日 K 开高低收为空。")
    status_warning = status.get("warning_message")
    if status_warning:
        warnings.append(str(status_warning))
    return "；".join(warnings)


def merge_sina_with_tushare_cache(sina_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """Merge Sina realtime quotes and local Tushare caches as reference data."""

    realtime = normalize_sina_spot(sina_df)
    if realtime.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    stock_basic, daily_basic, daily, status = load_tushare_caches(trade_date)
    stock_basic_prepared = prepare_stock_basic(stock_basic)
    daily_basic_prepared = prepare_daily_basic(daily_basic)
    daily_prepared = prepare_daily(daily)

    merged = realtime.merge(daily_basic_prepared, on="code", how="left")
    merged = merged.merge(daily_prepared, on="code", how="left")
    merged = merged.merge(stock_basic_prepared, on="code", how="left")

    if "stock_basic_name" in merged.columns:
        merged["name"] = merged["name"].where(merged["name"].astype(str).str.len() > 0, merged["stock_basic_name"])

    warning_message = build_warning_message(trade_date, status, not daily_basic.empty, not daily.empty)
    merged["tushare_trade_date"] = trade_date
    merged["realtime_source"] = "新浪实时行情"
    merged["reference_source"] = "Tushare 本地缓存"
    merged["data_source_level"] = "REFERENCE"
    merged["is_realtime_turnover"] = False
    merged["is_realtime_volume_ratio"] = False
    merged["allow_official_strategy_candidate"] = False
    merged["allow_reference_candidate"] = merged["price"].notna() & merged["amount"].notna()
    merged["warning_message"] = warning_message

    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    return merged[OUTPUT_COLUMNS]


def cache_file_status(trade_date: str) -> Dict[str, Any]:
    return {
        "stock_basic_exists": (TUSHARE_CACHE_DIR / "stock_basic.csv").exists(),
        "daily_basic_exists": (TUSHARE_CACHE_DIR / f"daily_basic_{trade_date}.csv").exists(),
        "daily_exists": (TUSHARE_CACHE_DIR / f"daily_{trade_date}.csv").exists(),
        "status_exists": STATUS_PATH.exists(),
    }
