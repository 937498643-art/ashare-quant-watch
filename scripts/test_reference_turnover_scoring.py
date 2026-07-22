"""Compare stored scores with reference-turnover scoring for active watchlist rows.

This diagnostic is read-only: it neither runs the main scanner nor changes any
strategy eligibility. Tushare turnover_rate_ref remains explicitly daily-cache
reference data rather than a realtime turnover field.
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
CSV_PATH = DIAGNOSTICS_DIR / "reference_turnover_scoring_check.csv"
JSON_PATH = DIAGNOSTICS_DIR / "reference_turnover_scoring_check.json"


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"未找到候选文件: {INPUT_PATH}")
        return 1

    data = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype={"code": "string"})
    active = data[data.get("source_type", pd.Series(dtype="object")).eq("active_watchlist")].head(30).copy()
    if active.empty:
        print("当前没有可用于测试的活跃观察池股票。")
        return 1

    old_score = pd.to_numeric(active.get("score"), errors="coerce")
    old_turnover_score = pd.to_numeric(active.get("turnover_score"), errors="coerce")
    rescored = score_candidates(enrich_turnover_fields(active))

    comparison = pd.DataFrame(
        {
            "code": active["code"].astype("string").str.zfill(6),
            "name": active.get("name", ""),
            "old_score": old_score,
            "realtime_turnover": pd.to_numeric(active.get("turnover"), errors="coerce"),
            "turnover_rate_ref": pd.to_numeric(active.get("turnover_rate_ref"), errors="coerce"),
            "old_turnover_score": old_turnover_score,
        }
    )
    new_scores = rescored[["code", "score", "turnover_score", "turnover_score_source"]].copy()
    new_scores["code"] = new_scores["code"].astype("string").str.zfill(6)
    new_scores = new_scores.rename(
        columns={"score": "new_score", "turnover_score": "new_turnover_score"}
    )
    comparison = comparison.merge(new_scores, on="code", how="left")
    comparison["score_change"] = comparison["new_score"] - comparison["old_score"]

    summary = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(comparison)),
        "realtime_turnover_count": int(comparison["turnover_score_source"].eq("realtime").sum()),
        "reference_turnover_count": int(comparison["turnover_score_source"].eq("reference").sum()),
        "missing_turnover_count": int(comparison["turnover_score_source"].eq("missing").sum()),
        "old_score_min": _number_or_none(comparison["old_score"].min()),
        "old_score_max": _number_or_none(comparison["old_score"].max()),
        "new_score_min": _number_or_none(comparison["new_score"].min()),
        "new_score_max": _number_or_none(comparison["new_score"].max()),
        "old_score_30_count": int(comparison["old_score"].eq(30).sum()),
        "new_score_30_count": int(comparison["new_score"].eq(30).sum()),
        "score_changed_count": int(comparison["score_change"].ne(0).sum()),
        "reference_turnover_is_realtime": False,
        "warning_message": "turnover_rate_ref 为 Tushare 日级缓存参考数据，不是盘中实时换手率。",
    }

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("参考换手率评分对比")
    print(comparison.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _number_or_none(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
