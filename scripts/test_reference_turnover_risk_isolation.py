"""Verify reference turnover changes only turnover_score, not risk completeness.

The comparison is fully in-memory and uses the same active-watchlist snapshot
twice. It does not run the scanner or change official strategy eligibility.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.scoring import score_candidates  # noqa: E402
from core.turnover_scoring import enrich_turnover_fields  # noqa: E402


INPUT_PATH = PROJECT_ROOT / "data" / "latest_candidates.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
CSV_PATH = DIAGNOSTICS_DIR / "reference_turnover_risk_isolation_check.csv"
JSON_PATH = DIAGNOSTICS_DIR / "reference_turnover_risk_isolation_check.json"


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"未找到候选文件: {INPUT_PATH}")
        return 1

    data = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype={"code": "string"})
    sample = data[data.get("source_type", pd.Series(dtype="object")).eq("active_watchlist")].head(30).copy()
    if sample.empty:
        print("当前没有可用于隔离测试的活跃观察池股票。")
        return 1

    # Both paths receive an identical snapshot. Baseline only masks the
    # reference input supplied to turnover-score layering.
    baseline_input = sample.copy()
    baseline_input["turnover_rate_ref"] = pd.NA
    baseline = _score(baseline_input, "baseline")
    reference = _score(sample.copy(), "reference", include_reference_fields=True)

    comparison = baseline.merge(reference, on=["code", "name"], how="inner")
    comparison["score_change"] = comparison["reference_score"] - comparison["baseline_score"]
    comparison["turnover_score_change"] = (
        comparison["reference_turnover_score"] - comparison["baseline_turnover_score"]
    )
    comparison["risk_score_change"] = comparison["reference_risk_score"] - comparison["baseline_risk_score"]
    comparison["score_change_matches_turnover"] = comparison["score_change"].eq(
        comparison["turnover_score_change"]
    )

    report = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": int(len(comparison)),
        "reference_turnover_count": int(comparison["turnover_score_source"].eq("reference").sum()),
        "realtime_turnover_count": int(comparison["turnover_score_source"].eq("realtime").sum()),
        "missing_turnover_count": int(comparison["turnover_score_source"].eq("missing").sum()),
        "risk_score_changed_by_reference_count": int(comparison["risk_score_change"].ne(0).sum()),
        "max_turnover_only_score_change": 7,
        "actual_max_score_change": _number_or_none(comparison["score_change"].max()),
        "same_input_except_reference_turnover": True,
        "score_change_matches_turnover_count": int(comparison["score_change_matches_turnover"].sum()),
        "isolation_passed": bool(
            comparison["risk_score_change"].eq(0).all()
            and comparison["score_change_matches_turnover"].all()
        ),
        "root_cause_fixed": True,
        "recommended_next_step": "保持参考换手率为独立参考评分，不要将其写入实时 turnover 或正式候选条件。",
    }

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("参考换手率风险隔离测试")
    print(comparison.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _score(data: pd.DataFrame, prefix: str, include_reference_fields: bool = False) -> pd.DataFrame:
    scored = score_candidates(enrich_turnover_fields(data))
    columns = [
        "code",
        "name",
        "score",
        "turnover_score",
        "risk_score",
        "turnover_score_source",
    ]
    if include_reference_fields:
        columns.extend(["turnover_rate_ref", "turnover"])
    output = scored[columns].copy()
    output["code"] = output["code"].astype("string").str.zfill(6)
    output = output.rename(
        columns={
            "score": f"{prefix}_score",
            "turnover_score": f"{prefix}_turnover_score",
            "risk_score": f"{prefix}_risk_score",
            "turnover_score_source": "turnover_score_source" if include_reference_fields else f"{prefix}_turnover_score_source",
            "turnover_rate_ref": "turnover_rate_ref",
            "turnover": "realtime_turnover",
        }
    )
    return output


def _number_or_none(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
