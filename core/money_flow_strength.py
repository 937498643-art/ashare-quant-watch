"""Money and volume strength helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def enrich_money_strength(stocks: pd.DataFrame) -> pd.DataFrame:
    """Add relative amount/volume strength fields when history indicators exist."""
    if stocks.empty:
        return stocks.copy()
    output = stocks.copy()
    amount = _numeric_series(output, "amount")
    volume = _numeric_series(output, "volume")
    avg_amount_5d = _numeric_series(output, "avg_amount_5d")
    avg_amount_20d = _numeric_series(output, "avg_amount_20d")
    avg_volume_5d = _numeric_series(output, "avg_volume_5d")
    avg_volume_20d = _numeric_series(output, "avg_volume_20d")
    output["amount_ratio_5d"] = _safe_ratio(amount, avg_amount_5d)
    output["amount_ratio_20d"] = _safe_ratio(amount, avg_amount_20d)
    output["volume_ratio_5d"] = _safe_ratio(volume, avg_volume_5d)
    output["volume_ratio_20d"] = _safe_ratio(volume, avg_volume_20d)
    levels = output["amount_ratio_5d"].map(classify_money_strength)
    output["money_strength_level"] = levels.map(lambda item: item[0])
    output["money_strength_summary"] = levels.map(lambda item: item[1])
    return output


def classify_money_strength(value: Any) -> tuple[str, str]:
    ratio = pd.to_numeric(value, errors="coerce")
    if pd.isna(ratio):
        return "暂无", "资金强度数据不足"
    if ratio < 0.7:
        return "缩量", "成交额明显低于 5 日均值"
    if ratio < 1.2:
        return "正常", "成交额接近 5 日均值"
    if ratio < 1.8:
        return "温和放量", "成交额温和放大"
    if ratio < 2.5:
        return "放量", "成交额明显放大"
    return "爆量", "成交额爆量，需结合上影线和位置风险"


def _safe_ratio(left: pd.Series, right: pd.Series) -> pd.Series:
    return left / right.replace(0, pd.NA)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")
