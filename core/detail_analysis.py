"""Detail analysis helpers for read-only stock K-line panels."""

from __future__ import annotations

from typing import Any

import pandas as pd


def prepare_history_detail(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize history and add MA/volume-average columns for detail charts."""
    if history.empty:
        return history.copy()

    data = history.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
        if column not in data.columns:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["date", "open", "close", "high", "low"]).sort_values("date")
    data = data.tail(60).reset_index(drop=True)
    data["ma5"] = data["close"].rolling(5).mean()
    data["ma10"] = data["close"].rolling(10).mean()
    data["ma20"] = data["close"].rolling(20).mean()
    data["avg_volume_5d"] = data["volume"].rolling(5).mean()
    data["avg_volume_20d"] = data["volume"].rolling(20).mean()
    return data


def build_detail_analysis(stock: pd.Series | dict[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    """Build text summary, risks, and helper labels for one stock."""
    data = prepare_history_detail(history)
    if data.empty:
        return {
            "summary": ["历史 K 线数据不足，暂无法生成技术摘要。"],
            "risks": ["该股票历史 K 线数据获取失败，请稍后重试。"],
            "labels": _empty_labels(),
        }

    latest = data.iloc[-1]
    price = _num(stock, "price")
    if pd.isna(price):
        price = _num(latest, "close")

    ma5 = _num(latest, "ma5")
    ma10 = _num(latest, "ma10")
    ma20 = _num(latest, "ma20")
    high_20d = data["high"].tail(20).max()
    low_20d = data["low"].tail(20).min()
    avg_amount_5d = data["amount"].tail(5).mean()
    current_amount = _num(stock, "amount")
    current_turnover = _num(stock, "turnover")
    pct_5d = _recent_return(data["close"], 5)
    pct_10d = _recent_return(data["close"], 10)
    pct_20d = _recent_return(data["close"], 20)
    high_gap = ((high_20d - price) / high_20d * 100) if _valid(price, high_20d) and high_20d else pd.NA
    amount_ratio = current_amount / avg_amount_5d if _valid(current_amount, avg_amount_5d) and avg_amount_5d else pd.NA

    summary = [
        f"当前价格{_above_text(price, ma5, 'MA5')}，{_above_text(price, ma10, 'MA10')}，{_above_text(price, ma20, 'MA20')}。",
        f"均线排列：{_ma_alignment(ma5, ma10, ma20)}。",
        f"最近 20 日高点 {format_number(high_20d)}，低点 {format_number(low_20d)}。",
        f"当前价格距离 20 日高点约 {format_percent(high_gap)}。",
        f"当前成交额与近 5 日平均成交额对比：{format_ratio(amount_ratio)}。",
        f"当前换手率水平：{_turnover_level(current_turnover)}。",
        f"最近 5 日涨幅 {format_percent(pct_5d)}，10 日涨幅 {format_percent(pct_10d)}，20 日涨幅 {format_percent(pct_20d)}。",
    ]

    labels = {
        "趋势状态": _trend_status(price, ma5, ma10, ma20),
        "量能状态": _volume_status(amount_ratio),
        "位置风险": _position_risk(price, ma5, high_20d),
        "换手状态": _turnover_status(current_turnover),
    }
    labels["操作状态"] = _action_status(labels, pct_5d, price, ma5)

    return {
        "summary": summary,
        "risks": _risk_tips(pct_5d, current_turnover, price, ma5, ma10),
        "labels": labels,
    }


def _risk_tips(
    pct_5d: float,
    turnover: float,
    price: float,
    ma5: float,
    ma10: float,
) -> list[str]:
    """Generate neutral technical risk tips."""
    risks = []
    if pd.notna(pct_5d) and pct_5d > 12:
        risks.append("近 5 日涨幅偏高，注意追高风险。")
    if pd.notna(turnover) and turnover > 20:
        risks.append("换手率过高，短线波动可能加大。")
    if _valid(price, ma5) and ma5 and (price / ma5 - 1) * 100 > 8:
        risks.append("当前价格偏离 MA5 较大，不宜无脑追高。")
    if _valid(price, ma10) and price < ma10:
        risks.append("若跌破 MA10，短线趋势可能转弱。")
    risks.append("当前仅基于个股技术面判断。")
    return risks


def _empty_labels() -> dict[str, str]:
    """Return default neutral labels when data is unavailable."""
    return {
        "趋势状态": "--",
        "量能状态": "--",
        "位置风险": "--",
        "换手状态": "--",
        "操作状态": "观察",
    }


def _trend_status(price: float, ma5: float, ma10: float, ma20: float) -> str:
    if _valid(price, ma5, ma10, ma20) and price > ma5 > ma10 > ma20:
        return "多头"
    if _valid(price, ma20) and price < ma20:
        return "转弱"
    return "震荡"


def _volume_status(amount_ratio: float) -> str:
    if pd.isna(amount_ratio):
        return "--"
    if amount_ratio >= 2.5:
        return "爆量"
    if amount_ratio >= 1.5:
        return "放量"
    if amount_ratio >= 0.8:
        return "温和放量"
    return "缩量"


def _position_risk(price: float, ma5: float, high_20d: float) -> str:
    if not _valid(price, ma5, high_20d) or not ma5 or not high_20d:
        return "--"
    ma5_gap = (price / ma5 - 1) * 100
    high_gap = (high_20d - price) / high_20d * 100
    if ma5_gap > 8 or high_gap < 3:
        return "高"
    if ma5_gap > 4 or high_gap < 8:
        return "中"
    return "低"


def _turnover_status(turnover: float) -> str:
    if pd.isna(turnover):
        return "--"
    if turnover < 1:
        return "偏低"
    if turnover < 3:
        return "正常"
    if turnover <= 10:
        return "较活跃"
    return "过高"


def _turnover_level(turnover: float) -> str:
    """Return a readable turnover level for the technical summary."""
    status = _turnover_status(turnover)
    if status == "--":
        return "缺失"
    return status


def _action_status(labels: dict[str, str], pct_5d: float, price: float, ma5: float) -> str:
    if labels["趋势状态"] == "转弱":
        return "暂不关注"
    if labels["位置风险"] == "高" or (pd.notna(pct_5d) and pct_5d > 12):
        return "不追高"
    if labels["换手状态"] == "过高":
        return "风险偏高"
    if labels["趋势状态"] == "多头" and _valid(price, ma5) and price > ma5:
        return "等回踩"
    return "观察"


def _ma_alignment(ma5: float, ma10: float, ma20: float) -> str:
    if _valid(ma5, ma10, ma20) and ma5 > ma10 > ma20:
        return "MA5 > MA10 > MA20，多头排列"
    if _valid(ma5, ma10, ma20) and ma5 < ma10 < ma20:
        return "MA5 < MA10 < MA20，偏弱排列"
    return "均线交织，偏震荡"


def _above_text(price: float, ma_value: float, ma_name: str) -> str:
    if not _valid(price, ma_value):
        return f"{ma_name} 数据不足"
    return f"站上 {ma_name}" if price > ma_value else f"未站上 {ma_name}"


def _recent_return(values: pd.Series, window: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < window:
        return float("nan")
    base = clean.iloc[-window]
    latest = clean.iloc[-1]
    if pd.isna(base) or base == 0:
        return float("nan")
    return float((latest / base - 1) * 100)


def format_number(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.2f}"


def format_percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.2f}%"


def format_ratio(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.2f} 倍"


def _num(row: pd.Series | dict[str, Any], key: str) -> float:
    value = row.get(key) if hasattr(row, "get") else None
    return float(pd.to_numeric(value, errors="coerce"))


def _valid(*values: Any) -> bool:
    return all(pd.notna(pd.to_numeric(value, errors="coerce")) for value in values)
