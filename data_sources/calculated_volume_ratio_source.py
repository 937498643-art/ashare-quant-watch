"""Read-only helpers for estimating intraday volume ratio from public caches.

Realtime cumulative volume comes from Sina quotes. Historical daily volume comes
only from local Tushare daily cache files. This module is intentionally separate
from the scanner until the calculation is independently verified.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from .calculated_turnover_source import get_sina_standardized_quotes, load_float_share_cache
from .tushare_cache_source import load_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUSHARE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tushare"
REALTIME_VOLUME_UNIT = "shares"
HISTORICAL_VOLUME_UNIT = "hands"
HISTORICAL_VOLUME_SHARES_UNIT = "shares"
VOLUME_RATIO_SOURCE = "sina_realtime_volume_plus_tushare_5d_volume"


def normalize_code(value: Any) -> str:
    """Return a six-digit code from a source code value."""
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def trading_minutes_at(now: datetime | None = None) -> tuple[int, str]:
    """Return A-share elapsed continuous-auction minutes without counting lunch."""
    current = now or datetime.now()
    if current.weekday() >= 5:
        return 0, "non_trading_day"

    current_time = current.time()
    morning_open = time(9, 30)
    morning_close = time(11, 30)
    afternoon_open = time(13, 0)
    afternoon_close = time(15, 0)

    if current_time < morning_open:
        return 0, "before_open"
    if current_time <= morning_close:
        minutes = int((datetime.combine(current.date(), current_time) - datetime.combine(current.date(), morning_open)).total_seconds() // 60)
        return minutes, "morning_session"
    if current_time < afternoon_open:
        return 120, "lunch_break"
    if current_time <= afternoon_close:
        minutes = 120 + int((datetime.combine(current.date(), current_time) - datetime.combine(current.date(), afternoon_open)).total_seconds() // 60)
        return minutes, "afternoon_session"
    return 240, "after_close"


def load_recent_daily_volume(cache_dir: Path = TUSHARE_CACHE_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Load all available daily caches and return their distinct trade dates."""
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("daily_????????.csv")):
        data = load_cache(path)
        required = {"ts_code", "trade_date", "vol"}
        if data.empty or not required.issubset(data.columns):
            continue
        data = data[["ts_code", "trade_date", "vol"]].copy()
        data["code"] = data["ts_code"].map(normalize_code)
        data["trade_date"] = data["trade_date"].astype("string")
        data["vol"] = pd.to_numeric(data["vol"], errors="coerce")
        data = data[(data["code"].str.len() == 6) & data["vol"].notna()]
        frames.append(data)

    if not frames:
        return pd.DataFrame(columns=["code", "trade_date", "vol", "historical_volume_shares"]), []

    history = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "trade_date"], keep="last")
    dates = sorted(history["trade_date"].dropna().unique().tolist(), reverse=True)
    history["historical_volume_shares"] = history["vol"] * 100
    return history, dates


def calculate_realtime_volume_ratio(
    quotes: pd.DataFrame,
    historical_daily: pd.DataFrame,
    historical_trade_dates: list[str],
    elapsed_minutes: int,
) -> pd.DataFrame:
    """Calculate a strictly validated intraday volume-ratio estimate.

    Tushare daily ``vol`` is expressed in hands. It is converted to shares so
    it matches the verified Sina realtime cumulative volume unit.
    """
    output = quotes.copy()
    output["code"] = output["code"].map(normalize_code)
    output["realtime_volume"] = pd.to_numeric(output.get("volume"), errors="coerce")
    output["realtime_volume_unit"] = REALTIME_VOLUME_UNIT
    output["elapsed_trading_minutes"] = elapsed_minutes
    output["avg_5d_volume"] = pd.NA
    output["avg_5d_volume_unit"] = HISTORICAL_VOLUME_SHARES_UNIT
    output["calculated_volume_ratio"] = pd.NA
    output["calculated_volume_ratio_display"] = "--"
    output["volume_ratio_source"] = VOLUME_RATIO_SOURCE
    output["volume_ratio_calculated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output["is_realtime_volume_ratio"] = False
    output["volume_ratio_validation_status"] = "invalid"

    if len(historical_trade_dates) < 5:
        output["volume_ratio_validation_status"] = "invalid_history_insufficient"
        return output
    if elapsed_minutes <= 0:
        output["volume_ratio_validation_status"] = "invalid_market_not_open"
        return output

    last_five_dates = historical_trade_dates[:5]
    history = historical_daily[historical_daily["trade_date"].isin(last_five_dates)].copy()
    day_count = history.groupby("code")["trade_date"].nunique().rename("history_day_count")
    avg_volume = history.groupby("code")["historical_volume_shares"].mean().rename("avg_5d_volume")
    output = output.drop(columns=["avg_5d_volume"], errors="ignore").merge(
        pd.concat([day_count, avg_volume], axis=1), on="code", how="left"
    )
    output["avg_5d_volume_unit"] = HISTORICAL_VOLUME_SHARES_UNIT
    output["elapsed_trading_minutes"] = elapsed_minutes
    output["realtime_volume_unit"] = REALTIME_VOLUME_UNIT
    output["volume_ratio_source"] = VOLUME_RATIO_SOURCE
    output["volume_ratio_calculated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output["is_realtime_volume_ratio"] = False
    output["volume_ratio_validation_status"] = "invalid"

    ratio = output["realtime_volume"] / (output["avg_5d_volume"] * elapsed_minutes / 240)
    valid = (
        output["realtime_volume"].notna()
        & (output["realtime_volume"] >= 0)
        & output["history_day_count"].eq(5)
        & output["avg_5d_volume"].notna()
        & (output["avg_5d_volume"] > 0)
        & ratio.notna()
        & ratio.between(0, 100)
    )
    output["calculated_volume_ratio"] = ratio.where(valid)
    output.loc[valid, "is_realtime_volume_ratio"] = True
    output.loc[valid, "volume_ratio_validation_status"] = "valid"
    output["calculated_volume_ratio_display"] = output["calculated_volume_ratio"].map(
        lambda value: f"{value:.2f}" if pd.notna(value) else "--"
    )
    return output


def apply_realtime_volume_ratio_priority(
    quotes: pd.DataFrame,
    data_source_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write validated realtime volume ratio values back to scanner quotes.

    Push2 f10 remains the direct realtime value when available. Otherwise, the
    already-validated Sina cumulative-volume calculation fills ``volume_ratio``.
    Tushare ``volume_ratio_ref`` is never copied into the realtime field.
    """
    if quotes is None or quotes.empty:
        return quotes.copy(), _volume_ratio_stats(0, 0, 0, 0, 0, 0)

    output = quotes.copy()
    output["code"] = output["code"].map(normalize_code)
    existing_ratio = pd.to_numeric(
        output.get("volume_ratio", pd.Series(pd.NA, index=output.index)), errors="coerce"
    )
    is_push2 = "东方财富 push2" in str(data_source_name)
    is_sina = "新浪" in str(data_source_name)
    eastmoney_mask = is_push2 & existing_ratio.between(0, 100)

    output["volume_ratio"] = existing_ratio.where(eastmoney_mask)
    output["volume_ratio_source"] = pd.NA
    output["is_realtime_volume_ratio"] = False
    output.loc[eastmoney_mask, "volume_ratio_source"] = "eastmoney_realtime"
    output.loc[eastmoney_mask, "is_realtime_volume_ratio"] = True

    calculated_mask = pd.Series(False, index=output.index)
    elapsed_minutes = 0
    historical_dates: list[str] = []
    if is_sina:
        historical_daily, historical_dates = load_recent_daily_volume()
        elapsed_minutes, _ = trading_minutes_at()
        calculated = calculate_realtime_volume_ratio(
            output,
            historical_daily,
            historical_dates,
            elapsed_minutes,
        )
        calculated_values = pd.to_numeric(calculated.get("calculated_volume_ratio"), errors="coerce")
        calculated_valid = calculated.get(
            "is_realtime_volume_ratio", pd.Series(False, index=calculated.index)
        ).fillna(False).astype(bool)
        calculated_mask = calculated_valid & calculated_values.between(0, 100)
        output.loc[calculated_mask, "volume_ratio"] = calculated_values[calculated_mask]
        output.loc[calculated_mask, "volume_ratio_source"] = "calculated_realtime"
        output.loc[calculated_mask, "is_realtime_volume_ratio"] = True

    reference_values = pd.to_numeric(
        output.get("volume_ratio_ref", pd.Series(pd.NA, index=output.index)), errors="coerce"
    )
    reference_count = int(reference_values.notna().sum())
    reference_mask = output["volume_ratio"].isna() & reference_values.notna()
    output.loc[reference_mask, "volume_ratio_source"] = "tushare_reference"
    return output, _volume_ratio_stats(
        total_rows=len(output),
        eastmoney_count=int(eastmoney_mask.sum()),
        calculated_count=int(calculated_mask.sum()),
        reference_count=reference_count,
        elapsed_minutes=elapsed_minutes,
        history_day_count=min(len(historical_dates), 5),
    )


def _volume_ratio_stats(
    total_rows: int,
    eastmoney_count: int,
    calculated_count: int,
    reference_count: int,
    elapsed_minutes: int,
    history_day_count: int,
) -> dict[str, Any]:
    realtime_count = eastmoney_count + calculated_count
    if eastmoney_count and calculated_count:
        source = "mixed"
    elif eastmoney_count:
        source = "eastmoney_realtime"
    elif calculated_count:
        source = "calculated_realtime"
    else:
        source = "tushare_reference" if reference_count else "unavailable"
    return {
        "realtime_volume_ratio_available": realtime_count > 0,
        "realtime_volume_ratio_source": source,
        "realtime_volume_ratio_count": realtime_count,
        "calculated_volume_ratio_count": calculated_count,
        "eastmoney_volume_ratio_count": eastmoney_count,
        "reference_volume_ratio_count": reference_count,
        "realtime_volume_ratio_coverage": round(realtime_count / total_rows, 6) if total_rows else 0.0,
        "volume_ratio_elapsed_trading_minutes": elapsed_minutes,
        "volume_ratio_history_day_count": history_day_count,
    }


def attach_reference_volume_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """Add daily Tushare reference ratio while retaining a separate field."""
    output = data.copy()
    cache = load_float_share_cache()
    if cache.empty or "volume_ratio" not in cache.columns:
        output["volume_ratio_ref"] = pd.NA
        output["reference_trade_date"] = pd.NA
        return output

    lookup = cache.drop_duplicates("code").set_index("code")
    output["volume_ratio_ref"] = output["code"].map(lookup["volume_ratio"])
    output["reference_trade_date"] = output["code"].map(lookup["trade_date"])
    return output


def fetch_sina_quotes_for_volume_ratio() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read normalized Sina quotes through the existing robust read-only parser."""
    return get_sina_standardized_quotes()
