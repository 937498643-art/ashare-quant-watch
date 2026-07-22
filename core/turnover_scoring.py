"""Turnover layer analysis for read-only watch results."""

from __future__ import annotations

from typing import Any

import pandas as pd


def enrich_turnover_fields(stocks: pd.DataFrame) -> pd.DataFrame:
    """Add turnover level, score, and summary fields."""
    if stocks.empty:
        return stocks.copy()
    rows = []
    for _, row in stocks.iterrows():
        result = analyze_turnover(
            row.get("turnover"),
            row.get("board_type"),
            row.get("turnover_rate_ref"),
        )
        output = row.to_dict()
        output.update(result)
        rows.append(output)
    return pd.DataFrame(rows)


def analyze_turnover(
    turnover: Any,
    board_type: Any = None,
    turnover_rate_ref: Any = None,
) -> dict[str, Any]:
    """Score realtime turnover first, then a lower-weight daily reference value."""
    value = pd.to_numeric(turnover, errors="coerce")
    reference_value = pd.to_numeric(turnover_rate_ref, errors="coerce")
    board = str(board_type or "")
    if pd.isna(value) and pd.isna(reference_value):
        return {
            "turnover_level": "missing",
            "turnover_level_display": "缺失",
            "turnover_score": -5,
            "turnover_score_source": "missing",
            "turnover_summary": "换手率缺失，不能进入正式策略候选",
        }
    source = "realtime" if pd.notna(value) else "reference"
    score_value = value if source == "realtime" else reference_value
    if board in {"chi_next", "star_market"}:
        result = _layer(score_value, [2, 5, 15, 25])
    else:
        result = _layer(score_value, [1, 3, 10, 20])

    if source == "reference":
        # The daily cached indicator is informative but must stay below the
        # realtime turnover contribution and never claim intraday freshness.
        result["turnover_score"] = min(int(result["turnover_score"]), 2)
        result["turnover_summary"] = f"参考换手率：{result['turnover_summary']}"

    result["turnover_score_source"] = source
    if board == "bj":
        result["turnover_summary"] += "；北交所波动较大"
        if result["turnover_level"] == "too_high":
            result["turnover_summary"] += "，高换手风险更强"
    return result


def _layer(value: float, limits: list[float]) -> dict[str, Any]:
    low, normal, active, high = limits
    if value < low:
        return {"turnover_level": "low", "turnover_level_display": "活跃度不足", "turnover_score": -2, "turnover_summary": "换手率偏低，活跃度不足"}
    if value < normal:
        return {"turnover_level": "normal_low", "turnover_level_display": "正常偏低", "turnover_score": 0, "turnover_summary": "换手率正常偏低"}
    if value < active:
        return {"turnover_level": "active", "turnover_level_display": "较活跃", "turnover_score": 3, "turnover_summary": "换手率较活跃，对短线观察有支持"}
    if value <= high:
        return {"turnover_level": "high_active", "turnover_level_display": "高度活跃", "turnover_score": 1, "turnover_summary": "换手率高度活跃，波动风险增加"}
    return {"turnover_level": "too_high", "turnover_level_display": "过高", "turnover_score": -3, "turnover_summary": "换手率过高，短线高波动风险"}
