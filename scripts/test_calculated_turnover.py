"""Validate calculated realtime turnover without changing the main workflow."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.calculated_turnover_source import (  # noqa: E402
    FLOAT_SHARE_CACHE_PATH,
    calculate_realtime_turnover,
    diagnose_sina_volume,
    get_sina_standardized_quotes,
    load_float_share_cache,
    validate_against_eastmoney,
    validate_historical_turnover,
)


DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
SAMPLE_PATH = DIAGNOSTICS_DIR / "calculated_turnover_sample.csv"
REPORT_PATH = DIAGNOSTICS_DIR / "calculated_turnover_check.json"


def main() -> int:
    report = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sina_available": False,
        "sina_rows": 0,
        "sina_volume_field": None,
        "detected_volume_unit": "unknown",
        "tushare_float_share_rows": 0,
        "matched_float_share_count": 0,
        "turnover_calculated_count": 0,
        "turnover_coverage": 0.0,
        "historical_validation_available": False,
        "historical_validation_sample_count": 0,
        "historical_sample_count": 0,
        "historical_mean_abs_error": None,
        "historical_p90_abs_error": None,
        "eastmoney_validation_available": False,
        "eastmoney_matched_count": 0,
        "eastmoney_mean_abs_error": None,
        "selected_formula": None,
        "validation_passed": False,
        "recommended_next_step": "",
        "error_message": "",
    }

    try:
        raw, standardized = get_sina_standardized_quotes()
        report["sina_available"] = not standardized.empty
        report["sina_rows"] = int(len(standardized))
        volume_diagnosis = diagnose_sina_volume(raw, standardized)
        report.update(volume_diagnosis)
    except Exception as exc:
        report["error_message"] = f"Sina realtime quote fetch failed: {type(exc).__name__}: {exc}"
        _save_report(report)
        print(report["error_message"])
        return 1

    float_share_cache = load_float_share_cache()
    report["float_share_cache_path"] = str(FLOAT_SHARE_CACHE_PATH)
    report["tushare_float_share_rows"] = int(len(float_share_cache))
    if float_share_cache.empty:
        report["error_message"] = "Tushare float-share cache is missing or empty."
        _save_report(report)
        print(report["error_message"])
        return 1

    historical = validate_historical_turnover()
    report["historical_validation_available"] = historical["available"]
    report["historical_validation_sample_count"] = historical["sample_count"]
    report["historical_sample_count"] = historical["sample_count"]
    report["historical_mean_abs_error"] = historical["mean_abs_error"]
    report["historical_median_abs_error"] = historical["median_abs_error"]
    report["historical_p90_abs_error"] = historical["p90_abs_error"]
    report["historical_within_0_05_ratio"] = historical["within_0_05_ratio"]
    report["historical_within_0_10_ratio"] = historical["within_0_10_ratio"]
    report["historical_within_0_20_ratio"] = historical["within_0_20_ratio"]
    report["historical_validation_passed"] = historical["passed"]

    detected_unit = report["detected_volume_unit"]
    provisional_formula = "A" if detected_unit == "股" else "B" if detected_unit == "手" else None
    preliminary = calculate_realtime_turnover(
        standardized,
        float_share_cache,
        detected_unit,
        validation_passed=False,
    )
    calculated = pd.to_numeric(preliminary["calculated_turnover"], errors="coerce")
    matched_float_share = preliminary["float_share"].notna() & (preliminary["float_share"] > 0)
    valid_calculated = calculated.notna() & calculated.between(0, 100)
    report["matched_float_share_count"] = int(matched_float_share.sum())
    report["turnover_calculated_count"] = int(valid_calculated.sum())
    report["turnover_coverage"] = round(int(valid_calculated.sum()) / len(preliminary), 6) if len(preliminary) else 0.0

    eastmoney = validate_against_eastmoney(preliminary)
    report["eastmoney_validation_available"] = eastmoney["available"]
    report["eastmoney_matched_count"] = eastmoney["matched_count"]
    report["eastmoney_mean_abs_error"] = eastmoney["mean_abs_error"]
    report["eastmoney_median_abs_error"] = eastmoney["median_abs_error"]
    report["eastmoney_p90_abs_error"] = eastmoney["p90_abs_error"]
    report["within_0_05_pct_count"] = eastmoney["within_0_05_pct_count"]
    report["within_0_10_pct_count"] = eastmoney["within_0_10_pct_count"]
    report["within_0_20_pct_count"] = eastmoney["within_0_20_pct_count"]
    report["within_0_05_pct"] = eastmoney["within_0_05_pct"]
    report["within_0_10_pct"] = eastmoney["within_0_10_pct"]
    report["within_0_20_pct"] = eastmoney["within_0_20_pct"]

    if eastmoney.get("best_formula"):
        report["eastmoney_best_formula"] = eastmoney["best_formula"]
    report["selected_formula"] = provisional_formula

    eastmoney_validation_passed = bool(
        eastmoney["available"]
        and eastmoney["matched_count"] >= 100
        and eastmoney["mean_abs_error"] is not None
        and eastmoney["mean_abs_error"] <= 0.20
        and eastmoney.get("best_formula") == provisional_formula
    )
    report["eastmoney_validation_passed"] = eastmoney_validation_passed
    coverage_ok = report["turnover_coverage"] >= 0.95
    float_share_coverage = report["matched_float_share_count"] / len(preliminary) if len(preliminary) else 0.0
    report["float_share_coverage"] = round(float_share_coverage, 6)
    float_share_coverage_ok = float_share_coverage >= 0.95
    not_all_zero = bool((calculated > 0).any())
    validation_evidence_passed = bool(historical["passed"] or eastmoney_validation_passed)
    report["validation_evidence_passed"] = validation_evidence_passed
    report["validation_passed"] = bool(
        detected_unit in {"股", "手"}
        and float_share_coverage_ok
        and coverage_ok
        and not_all_zero
        and validation_evidence_passed
    )

    final = calculate_realtime_turnover(
        standardized,
        float_share_cache,
        detected_unit,
        validation_passed=report["validation_passed"],
    )
    if report["validation_passed"]:
        report["recommended_next_step"] = "Independent turnover validation passed; review before any separate main-flow integration."
    elif not historical["available"] and not eastmoney["available"]:
        report["recommended_next_step"] = "Build a matching daily and daily_basic cache, or rerun when Eastmoney push2 is reachable, then perform independent validation."
    elif not historical["passed"]:
        report["recommended_next_step"] = "Historical closed-loop validation did not meet thresholds; do not integrate calculated turnover."
    elif eastmoney["available"] and not eastmoney_validation_passed:
        report["recommended_next_step"] = "Eastmoney cross-validation disagreed with the inferred unit; do not integrate calculated turnover."
    else:
        report["recommended_next_step"] = "Investigate coverage or invalid values before any integration."

    display_columns = [
        "code",
        "name",
        "realtime_volume",
        "realtime_volume_unit",
        "float_share",
        "float_share_unit",
        "calculated_turnover",
        "calculated_turnover_display",
        "turnover_source",
        "turnover_calculated_at",
        "turnover_validation_status",
        "turnover_validation_error",
        "is_realtime_turnover",
        "candidate_turnover_A",
        "candidate_turnover_B",
    ]
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    final[[column for column in display_columns if column in final.columns]].head(500).to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    _save_report(report)

    print("Calculated turnover diagnosis")
    print(f"Sina raw volume field: {report['sina_raw_volume_field']}")
    print(f"Sina normalized volume field: {report['sina_volume_field']}")
    print(f"Sina volume dtype: {report['sina_volume_dtype']}")
    print(f"Detected volume unit: {report['detected_volume_unit']}")
    print(f"Selected formula: {report['selected_formula']}")
    print(f"Float-share matched: {report['matched_float_share_count']}")
    print(f"Calculated turnover count: {report['turnover_calculated_count']}")
    print(f"Turnover coverage: {report['turnover_coverage']:.2%}")
    print(f"Historical validation available: {report['historical_validation_available']}")
    print(f"Historical mean absolute error: {report['historical_mean_abs_error']}")
    print(f"Eastmoney validation available: {report['eastmoney_validation_available']}")
    print(f"Eastmoney mean absolute error: {report['eastmoney_mean_abs_error']}")
    print(f"Validation passed: {report['validation_passed']}")
    print(f"Sample: {SAMPLE_PATH}")
    print(f"Report: {REPORT_PATH}")
    return 0


def _save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
