"""Read-only stock screening strategies.

Each strategy returns an informational result with a strategy name, trigger
status, reason, and risk note. These results are watchlist signals only and do
not contain any trading execution logic.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd


StrategyResult = dict[str, Any]

DEFAULT_MIN_AMOUNT = 100_000_000
DEFAULT_MAX_TREND_PCT_CHG = 7.0
DEFAULT_LIMIT_UP_RISK_PCT = 9.5
DEFAULT_BREAKOUT_PROXIMITY_PCT = 2.0
DEFAULT_VOLUME_MULTIPLIER = 1.5
DEFAULT_AMOUNT_MULTIPLIER = 1.3
DEFAULT_PULLBACK_MIN_PCT_CHG = -5.0
DEFAULT_PULLBACK_MAX_PCT_CHG = 2.0


def evaluate_trend_bullish(row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> StrategyResult:
    """Evaluate the trend bullish strategy for one stock."""
    params = _strategy_params(config, "trend_bullish")
    price = _num(row, "price", "最新价")
    ma5 = _num(row, "ma5", "MA5")
    ma10 = _num(row, "ma10", "MA10")
    ma20 = _num(row, "ma20", "MA20")
    amount = _num(row, "amount", "成交额")
    pct_chg = _num(row, "pct_chg", "涨跌幅")

    min_amount = _min_amount(config, params)
    max_pct_chg = float(params.get("max_pct_chg", DEFAULT_MAX_TREND_PCT_CHG))

    checks = [
        ("最新价 > MA5", _gt(price, ma5)),
        ("MA5 > MA10", _gt(ma5, ma10)),
        ("MA10 > MA20", _gt(ma10, ma20)),
        ("成交额充足", _gte(amount, min_amount)),
        ("涨跌幅未过高", _lte(pct_chg, max_pct_chg)),
    ]

    return _build_result(
        strategy_name="trend_bullish",
        checks=checks,
        success_reason="趋势多头：价格站上 MA5，MA5 > MA10 > MA20，成交额充足，涨跌幅未过高。",
        risk="趋势延续不代表确定上涨，需留意高位回落、板块退潮和放量滞涨风险。",
    )


def evaluate_volume_breakout(row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> StrategyResult:
    """Evaluate the volume breakout strategy for one stock."""
    params = _strategy_params(config, "volume_breakout")
    price = _num(row, "price", "最新价")
    high_20d = _num(row, "high_20d", "recent_high_20d", "20日高点")
    volume = _num(row, "volume", "成交量")
    amount = _num(row, "amount", "成交额")
    avg_volume_5d = _num(row, "avg_volume_5d", "5日均量")
    avg_volume_20d = _num(row, "avg_volume_20d", "20日均量")
    pct_chg = _num(row, "pct_chg", "涨跌幅")
    volume_ratio = _num(row, "volume_ratio", "量比")

    min_amount = _min_amount(config, params)
    proximity_pct = float(params.get("breakout_proximity_pct", DEFAULT_BREAKOUT_PROXIMITY_PCT))
    volume_multiplier = float(params.get("volume_multiplier", DEFAULT_VOLUME_MULTIPLIER))
    amount_multiplier = float(params.get("amount_multiplier", DEFAULT_AMOUNT_MULTIPLIER))
    limit_up_risk_pct = float(params.get("limit_up_risk_pct", DEFAULT_LIMIT_UP_RISK_PCT))

    near_breakout = _valid(price, high_20d) and price >= high_20d * (1 - proximity_pct / 100)
    volume_expanded = (
        (_valid(volume, avg_volume_5d) and volume >= avg_volume_5d * volume_multiplier)
        or (_valid(volume, avg_volume_20d) and volume >= avg_volume_20d * volume_multiplier)
        or (_valid(volume_ratio) and volume_ratio >= volume_multiplier)
        or (_valid(amount) and amount >= min_amount * amount_multiplier)
    )

    checks = [
        ("最新价接近或突破最近20日高点", near_breakout),
        ("成交量或成交额明显放大", volume_expanded),
        ("涨跌幅为正", _gt(pct_chg, 0)),
        ("不是一字涨停风险", _lt(pct_chg, limit_up_risk_pct)),
        ("成交额充足", _gte(amount, min_amount)),
    ]

    return _build_result(
        strategy_name="volume_breakout",
        checks=checks,
        success_reason="放量突破：价格接近或突破 20 日高点，量能放大，涨跌幅为正且未进入一字涨停风险区。",
        risk="突破后可能冲高回落，需关注假突破、涨停附近流动性下降和板块联动减弱风险。",
    )


def evaluate_pullback_low_volume(row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> StrategyResult:
    """Evaluate the low-volume pullback strategy for one stock."""
    params = _strategy_params(config, "pullback_low_volume")
    price = _num(row, "price", "最新价")
    ma5 = _num(row, "ma5", "MA5")
    ma10 = _num(row, "ma10", "MA10")
    ma20 = _num(row, "ma20", "MA20")
    pct_chg = _num(row, "pct_chg", "涨跌幅")
    volume = _num(row, "volume", "成交量")
    avg_volume_5d = _num(row, "avg_volume_5d", "5日均量")
    amount = _num(row, "amount", "成交额")

    min_amount = _min_amount(config, params)
    min_pct_chg = float(params.get("min_pct_chg", DEFAULT_PULLBACK_MIN_PCT_CHG))
    max_pct_chg = float(params.get("max_pct_chg", DEFAULT_PULLBACK_MAX_PCT_CHG))

    bullish_alignment = _gt(ma5, ma10) and _gt(ma10, ma20)
    price_above_ma20 = _gt(price, ma20)
    mild_pullback = _gte(pct_chg, min_pct_chg) and _lte(pct_chg, max_pct_chg)

    checks = [
        ("均线多头或价格仍在 MA20 上方", bullish_alignment or price_above_ma20),
        ("今日下跌或小幅震荡", mild_pullback),
        ("成交量低于5日均量", _lt(volume, avg_volume_5d)),
        ("没有跌破 MA10", _gte(price, ma10)),
        ("没有跌破 MA20", _gte(price, ma20)),
        ("成交额不能太低", _gte(amount, min_amount)),
    ]

    return _build_result(
        strategy_name="pullback_low_volume",
        checks=checks,
        success_reason="缩量回踩：趋势结构仍在，价格未跌破 MA10/MA20，成交量低于 5 日均量，成交额达标。",
        risk="缩量回踩可能继续转弱，需关注跌破均线、量能突然放大下跌和市场整体走弱风险。",
    )


def evaluate_strategies(row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> list[StrategyResult]:
    """Evaluate all V1 strategies for one stock."""
    return [
        evaluate_trend_bullish(row, config),
        evaluate_volume_breakout(row, config),
        evaluate_pullback_low_volume(row, config),
    ]


def apply_strategies(
    stocks: pd.DataFrame,
    config: dict[str, Any] | None = None,
    only_triggered: bool = False,
) -> pd.DataFrame:
    """Evaluate all strategies for a DataFrame and return strategy result rows."""
    if stocks.empty:
        return pd.DataFrame(columns=["code", "name", "strategy_name", "triggered", "reason", "risk"])

    result_rows: list[dict[str, Any]] = []

    for _, row in stocks.iterrows():
        code = _value(row, "code", "代码")
        name = _value(row, "name", "名称")
        for result in evaluate_strategies(row, config):
            if only_triggered and not result["triggered"]:
                continue
            result_rows.append(
                {
                    "code": code,
                    "name": name,
                    **result,
                }
            )

    return pd.DataFrame(result_rows)


def apply_strategy_tags(stocks: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Attach triggered strategy names, reasons, and risks to stock rows."""
    if stocks.empty:
        return stocks.copy()

    tagged = stocks.copy()
    triggered_names: list[str] = []
    reasons: list[str] = []
    risks: list[str] = []

    for _, row in tagged.iterrows():
        allowed, block_reason, block_risk = _strategy_candidate_gate(row)
        if not allowed:
            triggered_names.append("")
            reasons.append(block_reason)
            risks.append(block_risk)
            continue

        triggered = [result for result in evaluate_strategies(row, config) if result["triggered"]]
        triggered_names.append(",".join(result["strategy_name"] for result in triggered))
        reasons.append("；".join(result["reason"] for result in triggered))
        risks.append("；".join(result["risk"] for result in triggered))

    tagged["strategy_name"] = triggered_names
    tagged["triggered"] = [bool(value) for value in triggered_names]
    tagged["reason"] = reasons
    tagged["risk"] = risks
    return tagged


def _strategy_candidate_gate(row: pd.Series | dict[str, Any]) -> tuple[bool, str, str]:
    """Apply hard data-quality gates before any strategy may trigger."""
    turnover = _num(row, "turnover", "换手率")
    price = _num(row, "price", "最新价")
    amount = _num(row, "amount", "成交额")
    allow_strategy = _to_bool(_value(row, "allow_strategy_candidate"))

    if not allow_strategy:
        return (
            False,
            "未触发：最终评分字段不完整，不能生成正式策略候选股",
            "关键评分字段缺失，仅展示活跃观察池",
        )
    if pd.isna(turnover):
        return (
            False,
            "未触发：换手率缺失，换手率是硬性指标，本轮不生成策略候选股",
            "换手率缺失：本轮不生成策略候选股",
        )
    if pd.isna(price) or pd.isna(amount):
        return (
            False,
            "未触发：价格或成交额无效",
            "关键行情字段缺失，不能进入正式策略候选",
        )
    return True, "", ""


def _build_result(
    *,
    strategy_name: str,
    checks: list[tuple[str, bool]],
    success_reason: str,
    risk: str,
) -> StrategyResult:
    """Build a strategy result from named boolean checks."""
    failed = [name for name, passed in checks if not passed]
    triggered = not failed
    return {
        "strategy_name": strategy_name,
        "triggered": triggered,
        "reason": success_reason if triggered else "未触发：" + "、".join(failed),
        "risk": risk,
    }


def _strategy_params(config: dict[str, Any] | None, strategy_name: str) -> dict[str, Any]:
    """Read strategy-specific params from either full config or strategy config."""
    if not config:
        return {}

    strategies = config.get("strategies", config)
    return strategies.get(strategy_name, strategies.get(strategy_name.replace("_", "-"), {}))


def _min_amount(config: dict[str, Any] | None, params: dict[str, Any]) -> float:
    """Read the minimum amount threshold from strategy, universe, or filters config."""
    if "min_amount" in params:
        return float(params["min_amount"])

    if not config:
        return float(DEFAULT_MIN_AMOUNT)

    for section_name in ("universe", "filters"):
        section = config.get(section_name, {})
        if "min_amount" in section:
            return float(section["min_amount"])
        if "min_turnover_amount" in section:
            return float(section["min_turnover_amount"])

    if "min_amount" in config:
        return float(config["min_amount"])
    if "min_turnover_amount" in config:
        return float(config["min_turnover_amount"])

    return float(DEFAULT_MIN_AMOUNT)


def _value(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    """Get the first non-null value from a row-like object."""
    for key in keys:
        value = row.get(key) if hasattr(row, "get") else None
        if value is not None and not pd.isna(value):
            return value
    return None


def _to_bool(value: Any) -> bool:
    """Parse bool-like values from pandas rows."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _num(row: pd.Series | dict[str, Any], *keys: str) -> float:
    """Get a numeric row value, returning NaN when unavailable."""
    value = _value(row, *keys)
    if value is None:
        return float("nan")
    return float(pd.to_numeric(value, errors="coerce"))


def _valid(*values: float) -> bool:
    """Return True when all values are valid numbers."""
    return all(pd.notna(value) for value in values)


def _gt(left: float, right: float) -> bool:
    return _valid(left, right) and left > right


def _gte(left: float, right: float) -> bool:
    return _valid(left, right) and left >= right


def _lt(left: float, right: float) -> bool:
    return _valid(left, right) and left < right


def _lte(left: float, right: float) -> bool:
    return _valid(left, right) and left <= right
