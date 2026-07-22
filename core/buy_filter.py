"""Optional V5.6 entry-admission filter, independent from core scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BuyFilterConfig:
    enabled: bool = True
    signal_change_enabled: bool = True
    position_risk_enabled: bool = True
    volume_anomaly_enabled: bool = True
    intraday_reversal_enabled: bool = True
    max_soft_risk: int = 6


def _number(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value


def evaluate_buy_filter(
    row: dict[str, Any], config: BuyFilterConfig | None = None
) -> dict[str, Any]:
    """Evaluate independent entry-risk conditions without changing base scores."""
    config = config or BuyFilterConfig()
    if not config.enabled:
        return {
            "allowed": True,
            "hard_reject": False,
            "soft_risk": 0,
            "advantage": 0,
            "signal_day_label": "模块关闭",
            "detail": "买入过滤模块关闭",
        }

    signal_change = _number(row, "signal_day_change")
    distance_high = _number(row, "distance_to_20d_high_pct")
    volume_ratio = _number(row, "volume_to_ma20_ratio")
    reversal = _number(row, "intraday_reversal_pct")
    hard_reject = False
    soft_risk = 0
    advantage = 0
    labels: list[str] = []
    detail: list[str] = []

    signal_label = "其他"
    if config.signal_change_enabled and pd.notna(signal_change):
        if signal_change > 9:
            hard_reject = True
            signal_label = "高风险"
            detail.append("当日涨幅>9%，不进入核心交易池")
        elif 5 <= signal_change <= 9:
            signal_label = "正常"
            detail.append("当日涨幅5%–9%，正常")
        elif 0 <= signal_change < 5:
            signal_label = "优先"
            detail.append("当日涨幅0%–5%，优先")

    if config.position_risk_enabled and pd.notna(distance_high):
        if distance_high < 5:
            soft_risk += 3
            labels.append("接近20日高点")
            detail.append("距20日最高价<5%，风险+3")
        elif distance_high > 20:
            advantage += 2
            labels.append("距高点回调充分")
            detail.append("距20日最高价>20%，优势+2")

    if config.volume_anomaly_enabled and pd.notna(volume_ratio):
        if 1.5 <= volume_ratio <= 3:
            labels.append("放量健康")
            detail.append("量能1.5–3倍，放量健康")
        elif volume_ratio > 3:
            soft_risk += 3
            labels.append("巨量风险")
            detail.append("量能>3倍，巨量风险+3")

    if config.intraday_reversal_enabled and pd.notna(reversal) and reversal > 5:
        soft_risk += 3
        labels.append("冲高回落")
        detail.append("最高涨幅-收盘涨幅>5%，风险+3")

    allowed = not hard_reject and soft_risk < config.max_soft_risk
    if not allowed and not hard_reject:
        detail.append(f"软风险{soft_risk}达到准入阈值{config.max_soft_risk}，不进入核心交易池")
    return {
        "allowed": allowed,
        "hard_reject": hard_reject,
        "soft_risk": soft_risk,
        "advantage": advantage,
        "signal_day_label": signal_label,
        "labels": "；".join(labels),
        "detail": "；".join(detail) or "数据不足，不施加准入过滤",
    }
