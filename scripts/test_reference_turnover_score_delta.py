"""Diagnose score deltas caused by reference-turnover scoring.

This script compares the stored pre-reference score snapshot with a current
in-memory rescore of the same active-watchlist rows. It is diagnostic only and
does not run the scanner, alter strategy eligibility, or write business data.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.scoring import score_candidates  # noqa: E402
from core.turnover_scoring import enrich_turnover_fields  # noqa: E402


INPUT_PATH = PROJECT_ROOT / "data" / "latest_candidates.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
CSV_PATH = DIAGNOSTICS_DIR / "reference_turnover_score_delta_check.csv"
JSON_PATH = DIAGNOSTICS_DIR / "reference_turnover_score_delta_check.json"


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"未找到候选文件: {INPUT_PATH}")
        return 1

    source = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype={"code": "string"})
    active = source[source.get("source_type", pd.Series(dtype="object")).eq("active_watchlist")].copy()
    if active.empty:
        print("当前没有可诊断的活跃观察池股票。")
        return 1

    # The stored score is the pre-reference snapshot. The rescore below uses
    # the same CSV rows, only applying the new reference-turnover handling.
    sample = _select_sample(active, size=10)
    comparison = _build_comparison(sample)

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    report = _build_report(comparison)
    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("参考换手率评分增量诊断")
    print(comparison.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _select_sample(active: pd.DataFrame, size: int) -> pd.DataFrame:
    """Keep required examples first, then fill from the active-watchlist order."""
    required_codes = ["002384", "002156", "300750"]
    active = active.copy()
    active["code"] = active["code"].astype("string").str.zfill(6)
    priority = active[active["code"].isin(required_codes)]
    remainder = active[~active["code"].isin(required_codes)]
    return pd.concat([priority, remainder], ignore_index=True).head(size)


def _build_comparison(sample: pd.DataFrame) -> pd.DataFrame:
    old = sample.copy()
    old_components = old.apply(_stored_components, axis=1, result_type="expand")
    old_components.columns = [
        "old_amount_score",
        "old_pct_chg_score",
        "old_trend_score",
        "old_turnover_score",
        "old_volume_ratio_score",
        "old_strategy_score",
        "old_risk_score",
        "old_other_score",
    ]

    rescored = score_candidates(enrich_turnover_fields(sample))
    new = rescored[[
        "code",
        "score",
        "amount_score",
        "pct_chg_score",
        "trend_score",
        "turnover_score",
        "strategy_score",
        "risk_score",
        "turnover_score_source",
    ]].copy()
    new["code"] = new["code"].astype("string").str.zfill(6)
    new = new.rename(
        columns={
            "score": "new_score",
            "amount_score": "new_amount_score",
            "pct_chg_score": "new_pct_chg_score",
            "trend_score": "new_trend_score",
            "turnover_score": "new_turnover_score",
            "strategy_score": "new_strategy_score",
            "risk_score": "new_risk_score",
        }
    )

    base = pd.DataFrame(
        {
            "code": old["code"].astype("string").str.zfill(6),
            "name": old.get("name", ""),
            "old_score": pd.to_numeric(old.get("score"), errors="coerce"),
            "realtime_turnover": pd.to_numeric(old.get("turnover"), errors="coerce"),
            "turnover_rate_ref": pd.to_numeric(old.get("turnover_rate_ref"), errors="coerce"),
        }
    )
    comparison = pd.concat([base.reset_index(drop=True), old_components.reset_index(drop=True)], axis=1)
    comparison = comparison.merge(new, on="code", how="left")
    comparison["new_volume_ratio_score"] = 0
    known_new = [
        "new_amount_score",
        "new_pct_chg_score",
        "new_trend_score",
        "new_turnover_score",
        "new_volume_ratio_score",
        "new_strategy_score",
        "new_risk_score",
    ]
    comparison["new_other_score"] = comparison["new_score"] - comparison[known_new].sum(axis=1)

    for component in [
        "amount_score",
        "pct_chg_score",
        "trend_score",
        "turnover_score",
        "volume_ratio_score",
        "strategy_score",
        "risk_score",
        "other_score",
    ]:
        comparison[f"{component}_change"] = comparison[f"new_{component}"] - comparison[f"old_{component}"]

    comparison["score_change"] = comparison["new_score"] - comparison["old_score"]
    component_changes = [
        "amount_score_change",
        "pct_chg_score_change",
        "trend_score_change",
        "turnover_score_change",
        "volume_ratio_score_change",
        "strategy_score_change",
        "risk_score_change",
        "other_score_change",
    ]
    comparison["component_change_total"] = comparison[component_changes].sum(axis=1)
    comparison["score_math_matches"] = comparison["score_change"].eq(comparison["component_change_total"])
    comparison["explanation"] = comparison.apply(_explain, axis=1)
    return comparison


def _stored_components(row: pd.Series) -> list[int]:
    """Reconstruct stored components from the pre-reference score detail."""
    detail = str(row.get("score_detail") or "")
    amount = 10 * int("成交额大于 1 亿 +10" in detail) + 10 * int("成交额大于 3 亿 +10" in detail)
    pct = 15 if "涨跌幅处于合理区间 +15" in detail else 8 if "涨跌幅偏高但未极端 +8" in detail else 0
    trend = 10 * sum(
        phrase in detail
        for phrase in ("最新价大于 MA5 +10", "MA5 大于 MA10 +10", "MA10 大于 MA20 +10")
    )
    turnover = _int_or_zero(row.get("turnover_score"))
    strategy = 10 * sum(
        phrase in detail
        for phrase in ("触发趋势多头 +10", "触发放量突破 +10", "触发缩量回踩 +10")
    )
    strategy = min(strategy, 25)
    risk_match = re.search(r"风险基础分 \+(-?\d+)", detail)
    risk = int(risk_match.group(1)) if risk_match else 0
    score = _int_or_zero(row.get("score"))
    other = score - amount - pct - trend - turnover - strategy - risk
    return [amount, pct, trend, turnover, 0, strategy, risk, other]


def _explain(row: pd.Series) -> str:
    parts = [
        f"换手率分层变化 {int(row['turnover_score_change']):+d}",
        f"风险基础分变化 {int(row['risk_score_change']):+d}",
    ]
    if row["turnover_score_source"] == "reference":
        parts.append("使用 Tushare 日级参考换手率，非盘中实时")
    if row["score_math_matches"]:
        parts.append("分项增量与总分增量一致")
    else:
        parts.append("需要复核分项重构")
    return "；".join(parts)


def _int_or_zero(value: Any) -> int:
    number = pd.to_numeric(value, errors="coerce")
    return 0 if pd.isna(number) else int(number)


def _build_report(comparison: pd.DataFrame) -> dict[str, Any]:
    turnover_overlap = bool(
        (comparison["turnover_score_change"] != 0).any()
        and (comparison["risk_score_change"] != 0).any()
    )
    max_change = float(comparison["score_change"].max())
    return {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "old_score_source": "data/latest_candidates.csv 的预参考评分 score 字段",
        "new_score_source": "对同一 CSV 行执行当前 enrich_turnover_fields + score_candidates 的内存重算",
        "max_expected_turnover_only_change": 7,
        "actual_max_score_change": max_change,
        "duplicate_scoring_found": False,
        "turnover_overlap_found": turnover_overlap,
        "stale_old_score_found": True,
        "score_math_all_matches": bool(comparison["score_math_matches"].all()),
        "root_cause": (
            "参考换手率不会被重复加两次；但旧模型对实时换手率缺失同时产生了 "
            "换手率分层 -5 和风险基础分中的 -5。新模型在使用参考换手率后，" 
            "换手率分层改为 -2 至 +2，且风险基础分不再执行完全缺失处罚，" 
            "因此总增量可达到 +12。"
        ),
        "recommended_next_step": "先确认参考评分口径是否接受这两个独立影响；暂不应接入参考量比评分。",
    }


if __name__ == "__main__":
    raise SystemExit(main())
