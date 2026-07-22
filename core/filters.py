"""Filtering helpers for the A-share watch universe.

The functions in this module only filter market data. They do not create,
submit, cancel, or manage any trading order.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)

NAME_COLUMNS = ("name", "名称")
PRICE_COLUMNS = ("price", "最新价")
AMOUNT_COLUMNS = ("amount", "成交额")
PCT_CHG_COLUMNS = ("pct_chg", "涨跌幅")
TURNOVER_COLUMNS = ("turnover", "换手率")

DEFAULT_MIN_PRICE = 3.0
DEFAULT_MIN_AMOUNT = 100_000_000
DEFAULT_MAX_PCT_CHG = 9.5
DEFAULT_MIN_PCT_CHG = -9.5


def filter_stock_pool(quotes: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a filtered stock pool and print exclusion statistics.

    Supported input columns are the normalized schema from ``AkshareSource``
    (``name``, ``price``, ``amount``, ``pct_chg``) and common AKShare Chinese
    columns (``名称``, ``最新价``, ``成交额``, ``涨跌幅``).
    """
    config = _resolve_filter_config(config or {})
    original_count = len(quotes)

    if quotes.empty:
        _print_filter_summary(original_count, 0, Counter())
        return quotes.copy()

    data = quotes.copy()
    name = _get_text_series(data, NAME_COLUMNS)
    price = _get_numeric_series(data, PRICE_COLUMNS)
    amount = _get_numeric_series(data, AMOUNT_COLUMNS)
    pct_chg = _get_numeric_series(data, PCT_CHG_COLUMNS)

    exclusion_reasons: list[str | None] = []
    keep_mask: list[bool] = []

    for index in data.index:
        reason = _get_exclusion_reason(
            name=name.loc[index],
            price=price.loc[index],
            amount=amount.loc[index],
            pct_chg=pct_chg.loc[index],
            config=config,
        )
        exclusion_reasons.append(reason)
        keep_mask.append(reason is None)

    reason_stats = Counter(reason for reason in exclusion_reasons if reason is not None)
    filtered = data.loc[keep_mask].copy()

    _print_filter_summary(original_count, len(filtered), reason_stats)
    return filtered.reset_index(drop=True)


def apply_basic_filters(quotes: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Backward-compatible alias used by older scanner code."""
    return filter_stock_pool(quotes, config)


def filter_strategy_eligible(stocks: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Keep only rows that may become formal strategy candidates.

    This is stricter than the active watchlist pool. Real-time turnover is a
    hard condition for strategy candidates, so fallback sources without
    turnover can still display active stocks but cannot trigger strategies.
    """
    if stocks.empty:
        return stocks.copy()

    data = stocks.copy()
    config_values = _resolve_filter_config(config or {})
    name = _get_text_series(data, NAME_COLUMNS)
    price = _get_numeric_series(data, PRICE_COLUMNS)
    amount = _get_numeric_series(data, AMOUNT_COLUMNS)
    turnover = _get_numeric_series(data, TURNOVER_COLUMNS)
    allow_series = data.get("allow_strategy_candidate", pd.Series(True, index=data.index))

    eligible = []
    blocked_reasons: Counter[str] = Counter()
    for index in data.index:
        stock_name = str(name.loc[index] or "")
        allow_strategy = _to_bool(allow_series.loc[index])
        reason = None
        if not allow_strategy:
            reason = "数据源不允许策略候选"
        elif config_values["exclude_st"] and ("ST" in stock_name.upper() or "*ST" in stock_name.upper()):
            reason = "ST或*ST"
        elif config_values["exclude_delisting_risk"] and "退" in stock_name:
            reason = "名称含退"
        elif pd.isna(price.loc[index]) or price.loc[index] < config_values["min_price"]:
            reason = "价格无效或低于阈值"
        elif pd.isna(amount.loc[index]) or amount.loc[index] < config_values["min_amount"]:
            reason = "成交额无效或低于阈值"
        elif pd.isna(turnover.loc[index]):
            reason = "换手率缺失"

        if reason:
            blocked_reasons[reason] += 1
            eligible.append(False)
        else:
            eligible.append(True)

    if blocked_reasons:
        summary = "策略候选资格过滤: " + "，".join(
            f"{reason} {count} 只" for reason, count in blocked_reasons.items()
        )
        print(summary)
        LOGGER.info(summary)

    return data.loc[eligible].reset_index(drop=True)


def _get_exclusion_reason(
    *,
    name: str,
    price: float,
    amount: float,
    pct_chg: float,
    config: dict[str, Any],
) -> str | None:
    """Return the first exclusion reason for one stock, or ``None`` to keep it."""
    stock_name = str(name or "")

    if config["exclude_st"] and ("ST" in stock_name.upper() or "*ST" in stock_name.upper()):
        return "ST或*ST"

    if config["exclude_delisting_risk"] and "退" in stock_name:
        return "名称含退"

    if config["exclude_suspended"] and (pd.isna(price) or price <= 0):
        return "停牌或价格为空"

    if pd.isna(price) or price < config["min_price"]:
        return "价格低于阈值"

    if pd.isna(amount) or amount < config["min_amount"]:
        return "成交额低于阈值"

    if pd.notna(pct_chg) and pct_chg > config["max_pct_chg"]:
        return "涨幅过于极端"

    if pd.notna(pct_chg) and pct_chg < config["min_pct_chg"]:
        return "跌幅过于极端"

    return None


def _resolve_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Support both direct filter config and full project config."""
    source = config.get("universe") or config.get("filters") or config

    return {
        "exclude_st": bool(source.get("exclude_st", True)),
        "exclude_delisting_risk": bool(source.get("exclude_delisting_risk", True)),
        "exclude_suspended": bool(source.get("exclude_suspended", True)),
        "min_price": float(source.get("min_price", DEFAULT_MIN_PRICE)),
        "min_amount": float(
            source.get(
                "min_amount",
                source.get("min_turnover_amount", DEFAULT_MIN_AMOUNT),
            )
        ),
        "max_pct_chg": float(
            source.get(
                "max_pct_chg",
                source.get("limit_up_risk_pct", DEFAULT_MAX_PCT_CHG),
            )
        ),
        "min_pct_chg": float(
            source.get(
                "min_pct_chg",
                source.get("limit_down_risk_pct", DEFAULT_MIN_PCT_CHG),
            )
        ),
    }


def _get_text_series(data: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Return the first matching text column, or an empty string series."""
    column = _first_existing_column(data, columns)
    if column is None:
        LOGGER.warning("Missing text columns: %s", columns)
        return pd.Series("", index=data.index, dtype="string")
    return data[column].astype("string").fillna("")


def _get_numeric_series(data: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Return the first matching numeric column, or an NA numeric series."""
    column = _first_existing_column(data, columns)
    if column is None:
        LOGGER.warning("Missing numeric columns: %s", columns)
        return pd.Series(pd.NA, index=data.index, dtype="Float64")
    text = data[column].astype("string").str.strip().str.replace(",", "", regex=False)
    missing = {"", "--", "-", "None", "none", "null", "NULL", "nan", "NaN"}
    return pd.to_numeric(text.mask(text.isin(missing)), errors="coerce")


def _first_existing_column(data: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    """Find the first available column from a list of aliases."""
    for column in columns:
        if column in data.columns:
            return column
    return None


def _to_bool(value: Any) -> bool:
    """Parse bool-like values from pandas rows."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _print_filter_summary(
    original_count: int,
    filtered_count: int,
    reason_stats: Counter[str],
) -> None:
    """Print and log a compact filter summary after each scan."""
    lines = [
        f"原始股票数量: {original_count}",
        f"过滤后股票数量: {filtered_count}",
        "被排除原因统计:",
    ]

    if reason_stats:
        lines.extend(f"- {reason}: {count}" for reason, count in reason_stats.items())
    else:
        lines.append("- 无: 0")

    summary = "\n".join(lines)
    print(summary)
    LOGGER.info(summary)
