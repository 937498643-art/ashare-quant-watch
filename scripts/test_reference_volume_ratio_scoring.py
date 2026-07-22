"""Verify isolated reference volume-ratio scoring on the active watchlist."""

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
CSV_PATH = DIAGNOSTICS_DIR / "reference_volume_ratio_scoring_check.csv"
JSON_PATH = DIAGNOSTICS_DIR / "reference_volume_ratio_scoring_check.json"


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"未找到候选文件: {INPUT_PATH}")
        return 1

    data = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype={"code": "string"})
    sample = data[data.get("source_type", pd.Series(dtype="object")).eq("active_watchlist")].head(30).copy()
    if sample.empty:
        print("当前没有可用于参考量比评分测试的活跃观察池股票。")
        return 1

    baseline_input = sample.copy()
    baseline_input["volume_ratio_ref"] = pd.NA
    baseline = _score(baseline_input, "baseline")
    reference = _score(sample.copy(), "reference", include_ratio_fields=True)

    comparison = baseline.merge(reference, on=["code", "name"], how="inner")
    comparison["score_change"] = comparison["reference_score"] - comparison["baseline_score"]
    comparison["volume_ratio_score_change"] = (
        comparison["reference_volume_ratio_score"] - comparison["baseline_volume_ratio_score"]
    )
    comparison["risk_score_change"] = comparison["reference_risk_score"] - comparison["baseline_risk_score"]
    comparison["score_change_matches_volume_ratio"] = comparison["score_change"].eq(
        comparison["volume_ratio_score_change"]
    )

    report = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": int(len(comparison)),
        "realtime_volume_ratio_count": int(comparison["volume_ratio_score_source"].eq("realtime").sum()),
        "reference_volume_ratio_count": int(comparison["volume_ratio_score_source"].eq("reference").sum()),
        "missing_volume_ratio_count": int(comparison["volume_ratio_score_source"].eq("missing").sum()),
        "score_changed_count": int(comparison["score_change"].ne(0).sum()),
        "risk_score_changed_by_reference_count": int(comparison["risk_score_change"].ne(0).sum()),
        "max_reference_volume_ratio_score": _number_or_none(comparison["reference_volume_ratio_score"].max()),
        "baseline_score_min": _number_or_none(comparison["baseline_score"].min()),
        "baseline_score_max": _number_or_none(comparison["baseline_score"].max()),
        "reference_score_min": _number_or_none(comparison["reference_score"].min()),
        "reference_score_max": _number_or_none(comparison["reference_score"].max()),
        "reference_score_mean": _number_or_none(comparison["reference_score"].mean()),
        "reference_score_median": _number_or_none(comparison["reference_score"].median()),
        "scoring_passed": bool(
            comparison["risk_score_change"].eq(0).all()
            and comparison["score_change_matches_volume_ratio"].all()
            and comparison["reference_volume_ratio_score"].le(3).all()
            and comparison["reference_volume_ratio_score"].ge(-2).all()
        ),
        "recommended_next_step": "参考量比仅用于参考评分；不要写入实时 volume_ratio 或正式候选条件。",
    }

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("参考量比评分对比")
    print(comparison.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _score(data: pd.DataFrame, prefix: str, include_ratio_fields: bool = False) -> pd.DataFrame:
    scored = score_candidates(enrich_turnover_fields(data))
    columns = ["code", "name", "score", "volume_ratio_score", "risk_score", "volume_ratio_score_source"]
    if include_ratio_fields:
        columns.extend(["volume_ratio_ref", "volume_ratio"])
    output = scored[columns].copy()
    output["code"] = output["code"].astype("string").str.zfill(6)
    return output.rename(
        columns={
            "score": f"{prefix}_score",
            "volume_ratio_score": f"{prefix}_volume_ratio_score",
            "risk_score": f"{prefix}_risk_score",
            "volume_ratio_score_source": "volume_ratio_score_source" if include_ratio_fields else f"{prefix}_volume_ratio_score_source",
            "volume_ratio_ref": "volume_ratio_ref",
            "volume_ratio": "realtime_volume_ratio",
        }
    )


def _number_or_none(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
