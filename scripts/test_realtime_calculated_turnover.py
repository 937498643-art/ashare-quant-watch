"""Validate calculated intraday turnover without changing the main workflow.

The script keeps three independent values separate:
* calculated realtime turnover from Sina cumulative volume and Tushare float_share;
* Tushare daily_basic turnover_rate reference data; and
* Eastmoney f8, only when a same-run public request is reachable.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.calculated_turnover_source import (  # noqa: E402
    candidate_turnover,
    diagnose_sina_volume,
    fetch_eastmoney_turnover_sample,
    get_sina_standardized_quotes,
    load_float_share_cache,
)


LATEST_CANDIDATES_PATH = PROJECT_ROOT / "data" / "latest_candidates.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
CSV_PATH = DIAGNOSTICS_DIR / "realtime_calculated_turnover_check.csv"
JSON_PATH = DIAGNOSTICS_DIR / "realtime_calculated_turnover_check.json"
MANUAL_PATH = DIAGNOSTICS_DIR / "manual_turnover_benchmark.csv"


def main() -> int:
    check_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    watch_codes = _active_watch_codes()
    if not watch_codes:
        print("未找到活跃观察池股票，无法生成当前换手率对比表。")
        return 1

    raw, standardized = get_sina_standardized_quotes()
    volume_diagnosis = diagnose_sina_volume(raw, standardized)
    raw_volume_field = volume_diagnosis["sina_raw_volume_field"]
    raw_volume_series = raw[raw_volume_field] if raw_volume_field in raw.columns else pd.Series(dtype="object")
    raw_volume = pd.to_numeric(raw_volume_series, errors="coerce")
    standard_volume = pd.to_numeric(standardized.get("volume"), errors="coerce")
    float_cache = load_float_share_cache()
    if standardized.empty or float_cache.empty:
        print("新浪实时行情或 Tushare float_share 缓存为空。")
        return 1

    quote_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    realtime = standardized.copy()
    realtime["code"] = realtime["code"].astype("string").str.zfill(6)
    realtime = realtime[realtime["code"].isin(watch_codes)].copy()
    float_cache = float_cache.copy()
    float_cache["code"] = float_cache["code"].astype("string").str.zfill(6)

    columns = ["code", "float_share", "trade_date", "turnover_rate"]
    merged = realtime.merge(float_cache[columns].drop_duplicates("code"), on="code", how="left")
    merged["realtime_volume"] = pd.to_numeric(merged.get("volume"), errors="coerce")
    merged["float_share"] = pd.to_numeric(merged.get("float_share"), errors="coerce")
    merged["turnover_A"] = candidate_turnover(merged["realtime_volume"], merged["float_share"], "股")
    merged["turnover_B"] = candidate_turnover(merged["realtime_volume"], merged["float_share"], "手")
    merged["turnover_rate_ref"] = pd.to_numeric(merged.get("turnover_rate"), errors="coerce")

    eastmoney = fetch_eastmoney_turnover_sample()
    eastmoney_available = not eastmoney.empty
    if eastmoney_available:
        eastmoney["code"] = eastmoney["code"].astype("string").str.zfill(6)
        merged = merged.merge(eastmoney, on="code", how="left")
    else:
        merged["eastmoney_turnover"] = pd.NA
    merged = merged.rename(columns={"eastmoney_turnover": "turnover_eastmoney"})

    selected_formula, errors = _select_formula(merged, volume_diagnosis["detected_volume_unit"])
    selected_column = "turnover_A" if selected_formula == "A" else "turnover_B"
    merged["selected_realtime_turnover"] = pd.to_numeric(merged[selected_column], errors="coerce")
    merged["turnover_realtime_calculated"] = merged["selected_realtime_turnover"]
    merged["selected_formula"] = selected_formula
    merged["realtime_quote_time"] = quote_time
    merged["calculation_time"] = check_time
    merged["tushare_float_share_trade_date"] = merged["trade_date"].astype("string")
    merged["tushare_reference_trade_date"] = merged["trade_date"].astype("string")
    merged["realtime_volume_unit"] = volume_diagnosis["detected_volume_unit"]

    valid_selected = merged["selected_realtime_turnover"].between(0, 100)
    output_columns = [
        "code",
        "name",
        "realtime_volume",
        "realtime_volume_unit",
        "float_share",
        "turnover_A",
        "turnover_B",
        "turnover_rate_ref",
        "turnover_eastmoney",
        "turnover_realtime_calculated",
        "selected_realtime_turnover",
        "selected_formula",
        "realtime_quote_time",
        "calculation_time",
        "tushare_float_share_trade_date",
        "tushare_reference_trade_date",
    ]
    result = merged[[column for column in output_columns if column in merged.columns]].copy()

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    _build_manual_template(result).to_csv(MANUAL_PATH, index=False, encoding="utf-8-sig")

    report = {
        "check_time": check_time,
        "sina_rows": int(len(standardized)),
        "sina_volume_raw_field": volume_diagnosis["sina_raw_volume_field"],
        "sina_volume_standard_field": volume_diagnosis["sina_volume_field"],
        "sina_volume_raw_first20": [float(value) for value in raw_volume.dropna().head(20).tolist()],
        "sina_volume_standard_first20": [float(value) for value in standard_volume.dropna().head(20).tolist()],
        "sina_volume_raw_dtype": str(raw_volume_series.dtype) if raw_volume_field else None,
        "sina_volume_dtype": volume_diagnosis["sina_volume_dtype"],
        "sina_standardization_conversion": volume_diagnosis["standardization_conversion"],
        "detected_volume_unit": volume_diagnosis["detected_volume_unit"],
        "float_share_rows": int(len(float_cache)),
        "matched_count": int(merged["float_share"].notna().sum()),
        "formula_A_available": bool(merged["turnover_A"].notna().any()),
        "formula_B_available": bool(merged["turnover_B"].notna().any()),
        "eastmoney_validation_available": eastmoney_available,
        "eastmoney_matched_count": int(merged["turnover_eastmoney"].notna().sum()),
        "A_mean_abs_error": errors["A"]["mean"],
        "B_mean_abs_error": errors["B"]["mean"],
        "A_median_abs_error": errors["A"]["median"],
        "B_median_abs_error": errors["B"]["median"],
        "selected_formula": selected_formula,
        "realtime_turnover_calculated_count": int(valid_selected.sum()),
        "realtime_quote_time": quote_time,
        "calculation_time": check_time,
        "tushare_float_share_trade_date": _single_trade_date(merged),
        "tushare_reference_trade_date": _single_trade_date(merged),
        "validation_status": "eastmoney_cross_checked" if errors["A"]["count"] else "pending_manual_broker_benchmark",
        "recommended_next_step": "在 manual_turnover_benchmark.csv 填写同一时刻证券软件换手率，再比较 selected_realtime_turnover。",
    }
    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("盘中自主换手率验证")
    print(result.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _active_watch_codes() -> list[str]:
    if not LATEST_CANDIDATES_PATH.exists():
        return []
    data = pd.read_csv(LATEST_CANDIDATES_PATH, encoding="utf-8-sig", dtype={"code": "string"})
    active = data[data.get("source_type", pd.Series(dtype="object")).eq("active_watchlist")]
    return active["code"].astype("string").str.zfill(6).dropna().head(30).tolist()


def _select_formula(data: pd.DataFrame, detected_unit: str) -> tuple[str, dict[str, dict[str, float | int | None]]]:
    """Use same-run f8 when available; otherwise use the verified volume unit."""
    errors: dict[str, dict[str, float | int | None]] = {}
    for name, column in (("A", "turnover_A"), ("B", "turnover_B")):
        comparable = data[[column, "turnover_eastmoney"]].apply(pd.to_numeric, errors="coerce").dropna()
        error = (comparable[column] - comparable["turnover_eastmoney"]).abs()
        errors[name] = {
            "count": int(len(error)),
            "mean": float(error.mean()) if not error.empty else None,
            "median": float(error.median()) if not error.empty else None,
        }

    if errors["A"]["count"] and errors["B"]["count"]:
        return ("A" if errors["A"]["mean"] <= errors["B"]["mean"] else "B"), errors
    if detected_unit == "股":
        return "A", errors
    if detected_unit == "手":
        return "B", errors
    return "A", errors


def _build_manual_template(data: pd.DataFrame) -> pd.DataFrame:
    template = data[[
        "code",
        "name",
        "calculation_time",
        "turnover_A",
        "turnover_B",
        "selected_realtime_turnover",
    ]].copy()
    template["broker_turnover_manual"] = pd.NA
    template["abs_error"] = pd.NA
    return template


def _single_trade_date(data: pd.DataFrame) -> str | None:
    values = data["trade_date"].dropna().astype(str).unique().tolist()
    return values[0] if len(values) == 1 else None


if __name__ == "__main__":
    raise SystemExit(main())
