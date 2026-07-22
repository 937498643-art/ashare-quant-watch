"""Read-only stock risk checks based on market fields.

V1 only uses quote fields and stock names. It does not fetch announcements,
connect to broker accounts, or perform any trading operation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_MIN_PRICE = 3.0
DEFAULT_MIN_AMOUNT = 100_000_000
DEFAULT_HIGH_PCT_CHG = 8.0
DEFAULT_LOW_PCT_CHG = -8.0
DEFAULT_HIGH_TURNOVER = 20.0

RISK_TEXT = {
    "st": "ST 或 *ST 股票，存在特别处理风险",
    "delisting": "名称含“退”，存在退市风险",
    "suspended": "停牌或价格为空，流动性不可确认",
    "low_price": "价格过低，波动和流动性风险较高",
    "low_amount": "成交额过低，流动性不足",
    "high_pct_chg": "涨幅过高，存在追高风险",
    "low_pct_chg": "跌幅过大，短线弱势风险较高",
    "missing_turnover": "换手率缺失：本轮不生成策略候选股",
    "fallback_source": "关键评分字段不完整，仅展示活跃观察池",
    "low_turnover": "换手率低于 1%，活跃度不足",
    "normal_low_turnover": "换手率 1% 到 3%，活跃度正常偏低",
    "active_turnover": "换手率 3% 到 10%，交投较活跃",
    "high_active_turnover": "换手率 10% 到 20%，高活跃并存在短线波动风险",
    "high_turnover": "换手率过高，短线波动风险较高",
}


def check_stock_risk(row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check one stock and return risk flags plus a Chinese summary."""
    thresholds = _resolve_thresholds(config or {})
    name = str(_value(row, "name", "名称") or "")
    price = _num(row, "price", "最新价")
    amount = _num(row, "amount", "成交额")
    pct_chg = _num(row, "pct_chg", "涨跌幅")
    turnover = _num(row, "turnover", "换手率")
    allow_strategy = _to_bool(_value(row, "allow_strategy_candidate"))
    raw_quality_level = _value(row, "data_quality_level")
    if raw_quality_level is None or pd.isna(raw_quality_level):
        data_quality_level = "A" if allow_strategy else "C"
    else:
        data_quality_level = str(raw_quality_level).strip().upper()

    flags: list[str] = []

    upper_name = name.upper()
    if "ST" in upper_name:
        flags.append("st")

    if "退" in name:
        flags.append("delisting")

    if pd.isna(price):
        flags.append("suspended")
    elif price < thresholds["min_price"]:
        flags.append("low_price")

    if pd.isna(amount) or amount < thresholds["min_amount"]:
        flags.append("low_amount")

    if pd.notna(pct_chg) and pct_chg > thresholds["high_pct_chg"]:
        flags.append("high_pct_chg")

    if pd.notna(pct_chg) and pct_chg < thresholds["low_pct_chg"]:
        flags.append("low_pct_chg")

    if not allow_strategy or data_quality_level == "C":
        flags.append("fallback_source")

    if pd.isna(turnover):
        flags.append("missing_turnover")
    elif turnover < 1:
        flags.append("low_turnover")
    elif turnover < 3:
        flags.append("normal_low_turnover")
    elif turnover < 10:
        flags.append("active_turnover")
    elif turnover <= thresholds["high_turnover"]:
        flags.append("high_active_turnover")
    else:
        flags.append("high_turnover")

    flags = list(dict.fromkeys(flags))
    extra_summary = _extra_risk_summary(row)
    return {
        "risk_flags": flags,
        "risk_summary": _join_summary(build_risk_summary(flags), extra_summary),
    }


def check_risks(stocks: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Append risk flags and summaries to a stock DataFrame."""
    if stocks.empty:
        checked = stocks.copy()
        checked["risk_flags"] = pd.Series(dtype="object")
        checked["risk_summary"] = pd.Series(dtype="object")
        return checked

    rows: list[dict[str, Any]] = []
    for _, row in stocks.iterrows():
        risk_result = check_stock_risk(row, config)
        output_row = row.to_dict()
        output_row.update(risk_result)
        rows.append(output_row)

    return pd.DataFrame(rows)


def build_risk_summary(flags: list[str]) -> str:
    """Build a compact Chinese risk summary from flag codes."""
    if not flags:
        return "未发现明显行情风险"
    return "；".join(RISK_TEXT.get(flag, flag) for flag in flags)


def _extra_risk_summary(row: pd.Series | dict[str, Any]) -> str:
    extras: list[str] = []
    for key in [
        "sector_summary",
        "position_risk_summary",
        "chase_risk_summary",
        "money_strength_summary",
        "announcement_risk_summary",
        "t1_risk_summary",
    ]:
        value = _value(row, key)
        if value and str(value) not in {"--", "nan"}:
            extras.append(str(value))
    return "；".join(dict.fromkeys(extras))


def _join_summary(base: str, extra: str) -> str:
    if not extra:
        return base
    if not base or base == "未发现明显行情风险":
        return extra
    return f"{base}；{extra}"


def assert_no_trading_capability(config: dict[str, Any] | None = None) -> None:
    """Validate the project remains alert-only and read-only."""
    if not config:
        return

    risk_boundary = config.get("risk_boundary", config.get("risk_control", {}))
    if risk_boundary.get("allow_trading") or risk_boundary.get("trading_enabled"):
        raise ValueError("quant_stock_watch is alert-only; trading capability is forbidden.")


def _resolve_thresholds(config: dict[str, Any]) -> dict[str, float]:
    """Read risk thresholds from risk, universe, filters, or flat config."""
    risk_config = config.get("risk_check", config.get("risk_control", {}))
    universe_config = config.get("universe", {})
    filters_config = config.get("filters", {})

    return {
        "min_price": float(
            _first_config_value("min_price", risk_config, universe_config, filters_config, config)
            or DEFAULT_MIN_PRICE
        ),
        "min_amount": float(
            _first_config_value(
                "min_amount",
                risk_config,
                universe_config,
                filters_config,
                config,
                fallback_key="min_turnover_amount",
            )
            or DEFAULT_MIN_AMOUNT
        ),
        "high_pct_chg": float(
            _first_config_value("high_pct_chg", risk_config, config)
            or _first_config_value("max_pct_chg", universe_config, filters_config, config)
            or DEFAULT_HIGH_PCT_CHG
        ),
        "low_pct_chg": float(
            _first_config_value("low_pct_chg", risk_config, config)
            or _first_config_value("min_pct_chg", universe_config, filters_config, config)
            or DEFAULT_LOW_PCT_CHG
        ),
        "high_turnover": float(
            _first_config_value("high_turnover", risk_config, config)
            or DEFAULT_HIGH_TURNOVER
        ),
    }


def _first_config_value(
    key: str,
    *sections: dict[str, Any],
    fallback_key: str | None = None,
) -> Any:
    """Return the first configured value for key or fallback key."""
    for section in sections:
        if key in section:
            return section[key]
        if fallback_key and fallback_key in section:
            return section[fallback_key]
    return None


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
