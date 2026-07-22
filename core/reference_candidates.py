"""Reference candidate builder for Sina realtime + Tushare local cache data.

Reference candidates are not official strategy candidates. The turnover and
volume-ratio fields used here are daily cached reference data, not intraday
realtime fields.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


REFERENCE_WARNING = (
    "参考候选股使用新浪实时行情 + Tushare 日级缓存指标，"
    "换手率和量比不是盘中实时数据。"
)


def build_reference_candidates(
    merged_quotes: pd.DataFrame,
    config: dict[str, Any] | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Build reference candidates from merged realtime/reference quotes."""

    if merged_quotes is None or merged_quotes.empty:
        return _empty_reference_frame()

    filters = (config or {}).get("filters", {})
    min_price = float(filters.get("min_price") or 3.0)
    min_amount = float(filters.get("min_amount") or 100_000_000)

    data = merged_quotes.copy()
    for column in ["price", "pct_chg", "amount", "turnover_rate_ref", "volume_ratio_ref"]:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")

    name_series = data.get("name", pd.Series("", index=data.index)).fillna("").astype(str)
    valid_name = ~name_series.str.contains("ST", case=False, na=False)
    valid_name &= ~name_series.str.contains("退", na=False)

    mask = (
        valid_name
        & data["price"].notna()
        & (data["price"] > min_price)
        & data["amount"].notna()
        & (data["amount"] >= min_amount)
        & data["pct_chg"].notna()
        & (data["pct_chg"] > 0)
        & data["turnover_rate_ref"].notna()
        & data["volume_ratio_ref"].notna()
    )

    candidates = data[mask].copy()
    if candidates.empty:
        return _empty_reference_frame(data.columns)

    candidates = candidates.sort_values("amount", ascending=False).head(limit).reset_index(drop=True)
    candidates["source_type"] = "reference_candidate"
    candidates["source_type_display"] = "参考候选股"
    candidates["reference_reason"] = (
        "新浪实时成交额较高、涨跌幅为正，且本地 Tushare 缓存包含参考换手率和参考量比。"
    )
    candidates["reference_warning"] = REFERENCE_WARNING
    candidates["allow_official_strategy_candidate"] = False
    candidates["allow_reference_candidate"] = True
    return candidates


def _empty_reference_frame(extra_columns=None) -> pd.DataFrame:
    columns = list(extra_columns) if extra_columns is not None else []
    for column in [
        "source_type",
        "source_type_display",
        "reference_reason",
        "reference_warning",
        "allow_official_strategy_candidate",
        "allow_reference_candidate",
    ]:
        if column not in columns:
            columns.append(column)
    return pd.DataFrame(columns=columns)
