"""Neutral action-state labels for watch results."""

from __future__ import annotations

from typing import Any

import pandas as pd


ACTION_DISPLAY = {
    "observe": "观察",
    "wait_pullback": "等回踩",
    "no_chase": "不追高",
    "high_risk": "风险偏高",
    "weak_ignore": "暂不关注",
    "data_insufficient": "数据不足",
}


def enrich_action_state(candidates: pd.DataFrame, market_environment: str = "") -> pd.DataFrame:
    """Add neutral observation state and summary."""
    if candidates.empty:
        return candidates.copy()
    rows = []
    for _, row in candidates.iterrows():
        result = decide_action_state(row, market_environment)
        output = row.to_dict()
        output.update(result)
        rows.append(output)
    return pd.DataFrame(rows)


def decide_action_state(row: pd.Series | dict[str, Any], market_environment: str = "") -> dict[str, str]:
    turnover_level = str(row.get("turnover_level") if hasattr(row, "get") else "")
    strategy = str(row.get("strategy_names") if hasattr(row, "get") else "")
    pct = pd.to_numeric(row.get("pct_chg") if hasattr(row, "get") else None, errors="coerce")
    position = str(row.get("position_risk_level") if hasattr(row, "get") else "")
    sector = str(row.get("sector_strength_level") if hasattr(row, "get") else "")
    if turnover_level == "missing":
        state = "data_insufficient"
    elif market_environment == "极弱":
        state = "weak_ignore"
    elif position == "高" and turnover_level in {"high_active", "too_high"}:
        state = "high_risk"
    elif "volume_breakout" in strategy and pd.notna(pct) and pct > 7:
        state = "no_chase"
    elif "pullback_low_volume" in strategy:
        state = "wait_pullback"
    elif "trend_bullish" in strategy and turnover_level in {"active", "normal_low", "high_active"} and sector in {"强", "中", "暂无"}:
        state = "observe"
    elif position == "高":
        state = "no_chase"
    else:
        state = "observe"
    return {
        "action_state": state,
        "action_state_display": ACTION_DISPLAY[state],
        "action_summary": f"{ACTION_DISPLAY[state]}；仅为辅助观察，不构成交易建议",
    }
