"""A-share T+1 overnight risk reminders."""

from __future__ import annotations

from typing import Any

import pandas as pd


def enrich_t1_risk(candidates: pd.DataFrame, market_environment: str = "") -> pd.DataFrame:
    """Add T+1 risk level and summary."""
    if candidates.empty:
        return candidates.copy()
    rows = []
    for _, row in candidates.iterrows():
        output = row.to_dict()
        output.update(analyze_t1_risk(row, market_environment))
        rows.append(output)
    return pd.DataFrame(rows)


def analyze_t1_risk(row: pd.Series | dict[str, Any], market_environment: str = "") -> dict[str, str]:
    pct = pd.to_numeric(row.get("pct_chg") if hasattr(row, "get") else None, errors="coerce")
    turnover_level = str(row.get("turnover_level") if hasattr(row, "get") else "")
    position = str(row.get("position_risk_level") if hasattr(row, "get") else "")
    sector = str(row.get("sector_strength_level") if hasattr(row, "get") else "")
    points = 0
    reasons = ["T+1 风险：今日买入无法当日卖出，若次日低开可能被动"]
    if pd.notna(pct) and pct > 7:
        points += 1
        reasons.append("当前涨幅较高")
    if turnover_level in {"high_active", "too_high"}:
        points += 1
        reasons.append("换手率偏高")
    if position == "高":
        points += 1
        reasons.append("高位风险高")
    if market_environment in {"偏弱", "极弱"}:
        points += 1
        reasons.append("市场环境偏弱")
    if sector == "弱":
        points += 1
        reasons.append("板块强度弱")
    level = "低" if points == 0 else "中" if points <= 2 else "高"
    return {"t1_risk_level": level, "t1_risk_summary": "；".join(reasons)}
