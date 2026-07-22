"""Read-only helpers for calculating intraday turnover from public data.

The calculation combines Sina realtime cumulative volume with a cached Tushare
float-share snapshot. It is deliberately independent from the main scanner so
that units and accuracy can be validated before any later integration.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .akshare_source import AkshareSource
from .tushare_cache_source import load_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUSHARE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tushare"
FLOAT_SHARE_CACHE_PATH = TUSHARE_CACHE_DIR / "daily_basic_latest.csv"
FLOAT_SHARE_UNIT = "万股"
TURNOVER_SOURCE = "新浪实时成交量 + Tushare流通股本计算"


def normalize_code(value: Any) -> str:
    """Return a six-digit stock code from a vendor code value."""
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def load_float_share_cache(path: Path = FLOAT_SHARE_CACHE_PATH) -> pd.DataFrame:
    """Load the latest daily_basic cache with a normalized six-digit code."""
    data = load_cache(path)
    if data.empty:
        return pd.DataFrame(columns=["code", "float_share", "trade_date"])

    data = data.copy()
    data["code"] = data.get("ts_code", pd.Series(index=data.index, dtype="string")).map(normalize_code)
    for column in ["float_share", "free_share", "total_share", "turnover_rate", "volume_ratio"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data[data["code"].str.len() == 6].drop_duplicates("code").reset_index(drop=True)


def get_sina_standardized_quotes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Sina quotes through the project's fault-tolerant parser."""
    source = AkshareSource()
    raw = source._fetch_sina_spot_robust()
    standardized, _ = source._normalize_spot(raw, "新浪备用源", "D", False, False)
    return raw, standardized


def diagnose_sina_volume(raw: pd.DataFrame, standardized: pd.DataFrame) -> dict[str, Any]:
    """Infer quote-volume unit from price, amount, and volume consistency."""
    raw_volume_field = next((field for field in ["成交量", "volume"] if field in raw.columns), None)
    volume = pd.to_numeric(standardized.get("volume"), errors="coerce")
    price = pd.to_numeric(standardized.get("price"), errors="coerce")
    amount = pd.to_numeric(standardized.get("amount"), errors="coerce")
    valid = (volume > 0) & (price > 0) & (amount > 0)
    implied_ratio = (amount[valid] / (price[valid] * volume[valid])).replace([float("inf"), float("-inf")], pd.NA)
    median_ratio = float(implied_ratio.median()) if not implied_ratio.empty else None

    detected_unit = "unknown"
    if median_ratio is not None and 0.5 <= median_ratio <= 2.0:
        detected_unit = "股"
    elif median_ratio is not None and 50.0 <= median_ratio <= 200.0:
        detected_unit = "手"

    return {
        "sina_volume_field": "volume",
        "sina_raw_volume_field": raw_volume_field,
        "sina_volume_dtype": str(volume.dtype),
        "sina_volume_first20": [float(value) for value in volume.dropna().head(20).tolist()],
        "sina_volume_min": float(volume.min()) if volume.notna().any() else None,
        "sina_volume_median": float(volume.median()) if volume.notna().any() else None,
        "sina_volume_max": float(volume.max()) if volume.notna().any() else None,
        "amount_price_volume_median_ratio": median_ratio,
        "detected_volume_unit": detected_unit,
        "standardization_conversion": "numeric conversion only; no x100, /100, x10000, or other unit multiplier",
    }


def candidate_turnover(volume: pd.Series, float_share: pd.Series, volume_unit: str) -> pd.Series:
    """Calculate percent turnover for a stated realtime volume unit."""
    volume = pd.to_numeric(volume, errors="coerce")
    float_share = pd.to_numeric(float_share, errors="coerce")
    if volume_unit == "股":
        return volume / (float_share * 10_000) * 100
    if volume_unit == "手":
        return (volume * 100) / (float_share * 10_000) * 100
    return pd.Series(pd.NA, index=volume.index, dtype="Float64")


def apply_realtime_turnover_priority(
    quotes: pd.DataFrame,
    data_source_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach a realtime turnover value without ever promoting cached references.

    Eastmoney push2 f8 is used when the current source provides it. Otherwise,
    validated Sina cumulative volume and cached Tushare ``float_share`` are
    used to calculate an intraday value. Tushare ``turnover_rate`` remains in
    the separate ``turnover_rate_ref`` column in every case.
    """
    if quotes is None or quotes.empty:
        return quotes.copy(), _turnover_stats(0, 0, 0, 0)

    output = quotes.copy()
    output["code"] = output["code"].map(normalize_code)
    existing_turnover = pd.to_numeric(output.get("turnover"), errors="coerce")
    is_push2 = "东方财富 push2" in str(data_source_name)
    is_sina = "新浪" in str(data_source_name)
    eastmoney_mask = is_push2 & existing_turnover.between(0, 100)

    output["turnover"] = pd.NA
    output["turnover_source"] = pd.NA
    output["is_realtime_turnover"] = False
    output.loc[eastmoney_mask, "turnover"] = existing_turnover[eastmoney_mask]
    output.loc[eastmoney_mask, "turnover_source"] = "eastmoney_realtime"
    output.loc[eastmoney_mask, "is_realtime_turnover"] = True

    float_cache = load_float_share_cache()
    if float_cache.empty:
        output["float_share"] = pd.NA
        output["reference_trade_date"] = pd.NA
        output["turnover_rate_ref"] = pd.NA
        output["volume_ratio_ref"] = pd.NA
        return output, _turnover_stats(len(output), int(eastmoney_mask.sum()), 0, 0)

    float_cache = float_cache.set_index("code")
    cache_fields = {
        "float_share": "float_share",
        "reference_trade_date": "trade_date",
        "turnover_rate_ref": "turnover_rate",
        "volume_ratio_ref": "volume_ratio",
    }
    for target, source in cache_fields.items():
        cached = output["code"].map(float_cache[source]) if source in float_cache.columns else pd.Series(pd.NA, index=output.index)
        if target in output.columns:
            output[target] = output[target].where(output[target].notna(), cached)
        else:
            output[target] = cached

    volume = pd.to_numeric(output.get("volume"), errors="coerce")
    float_share = pd.to_numeric(output.get("float_share"), errors="coerce")
    calculated = candidate_turnover(volume, float_share, "股")
    calculated_mask = (
        is_sina
        & ~eastmoney_mask
        & volume.notna()
        & (volume >= 0)
        & float_share.notna()
        & (float_share > 0)
        & calculated.notna()
        & calculated.between(0, 100)
    )
    output.loc[calculated_mask, "turnover"] = calculated[calculated_mask]
    output.loc[calculated_mask, "turnover_source"] = "sina_volume_plus_tushare_float_share"
    output.loc[calculated_mask, "is_realtime_turnover"] = True

    reference_available_mask = pd.to_numeric(output.get("turnover_rate_ref"), errors="coerce").notna()
    reference_mask = (
        output["turnover"].isna()
        & reference_available_mask
    )
    output.loc[reference_mask, "turnover_source"] = "tushare_reference"

    return output, _turnover_stats(
        total_rows=len(output),
        eastmoney_count=int(eastmoney_mask.sum()),
        calculated_count=int(calculated_mask.sum()),
        reference_count=int(reference_available_mask.sum()),
    )


def _turnover_stats(
    total_rows: int,
    eastmoney_count: int,
    calculated_count: int,
    reference_count: int,
) -> dict[str, Any]:
    realtime_count = eastmoney_count + calculated_count
    if eastmoney_count and calculated_count:
        source = "mixed"
    elif eastmoney_count:
        source = "eastmoney_realtime"
    elif calculated_count:
        source = "sina_volume_plus_tushare_float_share"
    else:
        source = "tushare_reference" if reference_count else "unavailable"
    return {
        "realtime_turnover_available": realtime_count > 0,
        "realtime_turnover_source": source,
        "realtime_turnover_count": realtime_count,
        "calculated_turnover_count": calculated_count,
        "eastmoney_turnover_count": eastmoney_count,
        "reference_turnover_count": reference_count,
        "realtime_turnover_coverage": round(realtime_count / total_rows, 6) if total_rows else 0.0,
    }


def calculate_realtime_turnover(
    standardized: pd.DataFrame,
    float_share_cache: pd.DataFrame,
    detected_volume_unit: str,
    validation_passed: bool = False,
) -> pd.DataFrame:
    """Merge realtime volume with float shares and mark invalid observations."""
    realtime = standardized.copy()
    realtime["code"] = realtime["code"].map(normalize_code)
    cache_columns = [column for column in ["code", "float_share", "trade_date"] if column in float_share_cache.columns]
    merged = realtime.merge(float_share_cache[cache_columns], on="code", how="left")
    merged["realtime_volume"] = pd.to_numeric(merged.get("volume"), errors="coerce")
    merged["float_share"] = pd.to_numeric(merged.get("float_share"), errors="coerce")
    merged["candidate_turnover_A"] = candidate_turnover(merged["realtime_volume"], merged["float_share"], "股")
    merged["candidate_turnover_B"] = candidate_turnover(merged["realtime_volume"], merged["float_share"], "手")

    selected = "candidate_turnover_A" if detected_volume_unit == "股" else "candidate_turnover_B"
    merged["calculated_turnover"] = merged[selected] if detected_volume_unit in {"股", "手"} else pd.NA
    merged["realtime_volume_unit"] = detected_volume_unit
    merged["float_share_unit"] = FLOAT_SHARE_UNIT
    merged["turnover_source"] = TURNOVER_SOURCE
    merged["turnover_calculated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged["is_realtime_turnover"] = False
    merged["turnover_validation_status"] = "invalid"
    merged["turnover_validation_error"] = ""

    invalid_conditions = [
        (merged["float_share"].isna() | (merged["float_share"] <= 0), "float_share <= 0 or missing"),
        (merged["realtime_volume"].isna(), "realtime volume missing"),
        (merged["realtime_volume"] < 0, "volume < 0"),
        (merged["calculated_turnover"].isna(), "volume unit not verified"),
        (merged["calculated_turnover"] < 0, "calculated turnover < 0"),
        (merged["calculated_turnover"] > 100, "calculated turnover > 100"),
    ]
    for condition, message in invalid_conditions:
        mask = condition & (merged["turnover_validation_status"] == "invalid")
        merged.loc[mask, "turnover_validation_error"] = message

    valid_input = (
        merged["float_share"].notna()
        & (merged["float_share"] > 0)
        & merged["realtime_volume"].notna()
        & (merged["realtime_volume"] >= 0)
        & merged["calculated_turnover"].notna()
        & (merged["calculated_turnover"] >= 0)
        & (merged["calculated_turnover"] <= 100)
    )
    if validation_passed:
        merged.loc[valid_input, "turnover_validation_status"] = "valid"
        merged.loc[valid_input, "is_realtime_turnover"] = True
        merged.loc[valid_input, "turnover_validation_error"] = ""
    else:
        merged.loc[valid_input, "turnover_validation_status"] = "pending_validation"
        merged.loc[valid_input, "turnover_validation_error"] = "independent validation is required before use"

    merged["calculated_turnover_display"] = merged["calculated_turnover"].map(
        lambda value: f"{value:.2f}%" if pd.notna(value) else "--"
    )
    return merged


def validate_historical_turnover() -> dict[str, Any]:
    """Validate Tushare daily vol (hand) against matching daily_basic cache."""
    results: list[pd.DataFrame] = []
    for daily_path in sorted(TUSHARE_CACHE_DIR.glob("daily_????????.csv")):
        trade_date = daily_path.stem.removeprefix("daily_")
        basic_path = TUSHARE_CACHE_DIR / f"daily_basic_{trade_date}.csv"
        if not basic_path.exists():
            continue
        daily = load_cache(daily_path)
        basic = load_cache(basic_path)
        required_daily = {"ts_code", "vol"}
        required_basic = {"ts_code", "float_share", "turnover_rate"}
        if daily.empty or basic.empty or not required_daily.issubset(daily.columns) or not required_basic.issubset(basic.columns):
            continue
        merged = daily[["ts_code", "vol"]].merge(
            basic[["ts_code", "float_share", "turnover_rate"]], on="ts_code", how="inner"
        )
        merged["historical_calculated_turnover"] = candidate_turnover(merged["vol"], merged["float_share"], "手")
        merged["abs_error"] = (merged["historical_calculated_turnover"] - pd.to_numeric(merged["turnover_rate"], errors="coerce")).abs()
        results.append(merged.dropna(subset=["abs_error"]))

    if not results:
        return {
            "available": False,
            "sample_count": 0,
            "mean_abs_error": None,
            "median_abs_error": None,
            "p90_abs_error": None,
            "within_0_05_ratio": None,
            "within_0_10_ratio": None,
            "within_0_20_ratio": None,
            "passed": False,
        }

    data = pd.concat(results, ignore_index=True)
    error = data["abs_error"]
    return {
        "available": True,
        "sample_count": int(len(data)),
        "mean_abs_error": float(error.mean()),
        "median_abs_error": float(error.median()),
        "p90_abs_error": float(error.quantile(0.90)),
        "within_0_05_ratio": float((error < 0.05).mean()),
        "within_0_10_ratio": float((error < 0.10).mean()),
        "within_0_20_ratio": float((error < 0.20).mean()),
        "passed": bool(len(data) >= 100 and error.mean() <= 0.10 and (error < 0.10).mean() >= 0.90),
    }


def fetch_eastmoney_turnover_sample() -> pd.DataFrame:
    """Fetch one small push2 sample for validation only; never used as a source."""
    hosts = [
        "https://82.push2.eastmoney.com/api/qt/clist/get",
        "https://push2.eastmoney.com/api/qt/clist/get",
    ]
    params = {
        "pn": 1,
        "pz": 200,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f12,f14,f8",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    for host in hosts:
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(host, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                rows = ((response.json() or {}).get("data") or {}).get("diff") or []
                if isinstance(rows, dict):
                    rows = list(rows.values())
                if rows:
                    data = pd.DataFrame(rows)
                    data["code"] = data["f12"].map(normalize_code)
                    data["eastmoney_turnover"] = pd.to_numeric(data["f8"], errors="coerce")
                    return data[["code", "eastmoney_turnover"]]
        except Exception:
            continue
    return pd.DataFrame(columns=["code", "eastmoney_turnover"])


def validate_against_eastmoney(calculated: pd.DataFrame) -> dict[str, Any]:
    """Compare a selected calculated turnover sample with push2 f8 when reachable."""
    eastmoney = fetch_eastmoney_turnover_sample()
    if eastmoney.empty:
        return {
            "available": False,
            "matched_count": 0,
            "mean_abs_error": None,
            "median_abs_error": None,
            "p90_abs_error": None,
            "within_0_05_pct_count": 0,
            "within_0_10_pct_count": 0,
            "within_0_20_pct_count": 0,
            "within_0_05_pct": None,
            "within_0_10_pct": None,
            "within_0_20_pct": None,
        }

    merged = calculated[["code", "candidate_turnover_A", "candidate_turnover_B"]].merge(eastmoney, on="code", how="inner")
    merged = merged.dropna(subset=["eastmoney_turnover"])
    if merged.empty:
        return {
            "available": True,
            "matched_count": 0,
            "mean_abs_error": None,
            "median_abs_error": None,
            "p90_abs_error": None,
            "within_0_05_pct_count": 0,
            "within_0_10_pct_count": 0,
            "within_0_20_pct_count": 0,
            "within_0_05_pct": None,
            "within_0_10_pct": None,
            "within_0_20_pct": None,
        }

    error_a = (merged["candidate_turnover_A"] - merged["eastmoney_turnover"]).abs()
    error_b = (merged["candidate_turnover_B"] - merged["eastmoney_turnover"]).abs()
    selected_error = error_a if error_a.mean() <= error_b.mean() else error_b
    return {
        "available": True,
        "matched_count": int(len(merged)),
        "mean_abs_error": float(selected_error.mean()),
        "median_abs_error": float(selected_error.median()),
        "p90_abs_error": float(selected_error.quantile(0.90)),
        "within_0_05_pct_count": int((selected_error < 0.05).sum()),
        "within_0_10_pct_count": int((selected_error < 0.10).sum()),
        "within_0_20_pct_count": int((selected_error < 0.20).sum()),
        "within_0_05_pct": float((selected_error < 0.05).mean()),
        "within_0_10_pct": float((selected_error < 0.10).mean()),
        "within_0_20_pct": float((selected_error < 0.20).mean()),
        "best_formula": "A" if error_a.mean() <= error_b.mean() else "B",
    }
