"""High-position and chase-risk analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


def enrich_position_risk(stocks: pd.DataFrame) -> pd.DataFrame:
    """Add high-position and chase-risk fields."""
    if stocks.empty:
        return stocks.copy()
    rows = []
    for _, row in stocks.iterrows():
        output = row.to_dict()
        output.update(analyze_position_risk(row))
        rows.append(output)
    return pd.DataFrame(rows)


def analyze_position_risk(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    price = _num(row, "price")
    ma5 = _num(row, "ma5")
    ma10 = _num(row, "ma10")
    ma20 = _num(row, "ma20")
    high_20d = _num(row, "high_20d")
    low_20d = _num(row, "low_20d")
    pct5 = _num(row, "pct_chg_5d")
    pct10 = _num(row, "pct_chg_10d")
    pct20 = _num(row, "pct_chg_20d")
    dev5 = _pct_dev(price, ma5)
    dev10 = _pct_dev(price, ma10)
    dev20 = _pct_dev(price, ma20)
    high_gap = ((high_20d - price) / high_20d * 100) if _valid(price, high_20d) and high_20d else pd.NA
    low_gain = ((price - low_20d) / low_20d * 100) if _valid(price, low_20d) and low_20d else pd.NA
    reasons = []
    risk_score = 0
    if pd.notna(pct5) and pct5 > 20:
        reasons.append("5 日涨幅偏高")
        risk_score += 1
    if pd.notna(pct10) and pct10 > 35:
        reasons.append("10 日涨幅偏高")
        risk_score += 2
    if pd.notna(pct20) and pct20 > 50:
        reasons.append("20 日涨幅偏高")
        risk_score += 2
    if pd.notna(dev5) and dev5 > 8:
        reasons.append("偏离 MA5 超过 8%")
        risk_score += 1
    if pd.notna(dev10) and dev10 > 15:
        reasons.append("偏离 MA10 超过 15%")
        risk_score += 2
    level = "低" if risk_score == 0 else "中" if risk_score <= 2 else "高"
    chase_level = "高" if any("偏离" in reason for reason in reasons) or risk_score >= 3 else "中" if risk_score else "低"
    summary = "；".join(reasons) if reasons else "位置风险暂未发现明显异常"
    return {
        "ma5_deviation": dev5,
        "ma10_deviation": dev10,
        "ma20_deviation": dev20,
        "high_20d_gap": high_gap,
        "low_20d_gain": low_gain,
        "position_risk_level": level,
        "position_risk_summary": summary,
        "chase_risk_level": chase_level,
        "chase_risk_summary": "追高风险较高" if chase_level == "高" else "追高风险中等" if chase_level == "中" else "追高风险较低",
    }


def _pct_dev(price: float, ma: float) -> float:
    if not _valid(price, ma) or ma == 0:
        return float("nan")
    return float((price / ma - 1) * 100)


def _num(row: pd.Series | dict[str, Any], key: str) -> float:
    return float(pd.to_numeric(row.get(key) if hasattr(row, "get") else None, errors="coerce"))


def _valid(*values: Any) -> bool:
    return all(pd.notna(pd.to_numeric(value, errors="coerce")) for value in values)
