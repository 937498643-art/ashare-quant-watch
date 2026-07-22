"""Indicator calculation and merge helpers.

This module calculates read-only technical indicators from historical daily
bars and merges them into realtime quote rows. It does not contain any trading
execution, order, account, or credential logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


LOGGER = logging.getLogger(__name__)

CODE_COLUMNS = ("code", "代码")
DATE_COLUMNS = ("date", "日期")
CLOSE_COLUMNS = ("close", "收盘")
HIGH_COLUMNS = ("high", "最高")
LOW_COLUMNS = ("low", "最低")
VOLUME_COLUMNS = ("volume", "成交量")
AMOUNT_COLUMNS = ("amount", "成交额")

REALTIME_NUMERIC_COLUMNS = (
    "price",
    "pct_chg",
    "change",
    "volume",
    "amount",
    "amplitude",
    "high",
    "low",
    "open",
    "prev_close",
    "turnover",
    "turnover_rate",
    "换手率",
    "换手",
    "最新价",
    "涨跌幅",
    "涨跌额",
    "成交量",
    "成交额",
    "振幅",
    "最高",
    "最低",
    "今开",
    "开盘",
    "昨收",
    "换手率",
)

HISTORY_INDICATOR_COLUMNS = (
    "ma5",
    "ma10",
    "ma20",
    "pct_chg_5d",
    "pct_chg_10d",
    "pct_chg_20d",
    # Point-in-time helpers used only by the V5.8 quality layer.  They keep
    # the existing V5.3 base-score inputs unchanged.
    "prior_price",
    "prior_ma20",
    "close_10d_ago",
    "close_20d_ago",
    "up_days_4d",
    "quality_pct_chg_10d",
    "quality_pct_chg_20d",
    "quality_up_days_5d",
    "avg_volume_5d",
    "avg_volume_10d",
    "avg_volume_20d",
    "avg_amount_5d",
    "avg_amount_20d",
    "high_20d",
    "low_20d",
    # Keep realtime scoring inputs aligned with core.backtest.build_daily_score_row.
    "prior_20d_high",
    "limit_up_count_20d",
    "consecutive_limit_up_days",
    "recent_low_volume_pullback",
    # These two fields are diagnostics for the same buy-point inputs; they do
    # not introduce a new scoring rule.
    "distance_to_20d_high_pct",
    "volume_breakout",
)


class HistoricalDataSource(Protocol):
    """Minimal read-only interface needed for historical indicator merging."""

    def fetch_history(self, code: str, days: int = 60) -> pd.DataFrame:
        """Fetch historical daily bars for one stock."""


@dataclass
class HistoryIndicatorCache:
    """Simple in-memory cache reserved for later performance optimization.

    V1 uses this lightweight dict cache only within the current process. Later
    versions can replace it with a TTL cache, SQLite table, or local parquet
    files without changing the indicator merge interface.
    """

    values: dict[tuple[str, int], dict[str, float | pd._libs.missing.NAType]] = field(
        default_factory=dict
    )

    def get(self, code: str, days: int) -> dict[str, Any] | None:
        """Return cached indicator values for one stock and lookback window."""
        return self.values.get((code, days))

    def set(self, code: str, days: int, indicators: dict[str, Any]) -> None:
        """Store calculated indicator values."""
        self.values[(code, days)] = indicators


def enrich_realtime_indicators(quotes: pd.DataFrame) -> pd.DataFrame:
    """Normalize realtime numeric fields without fetching historical data."""
    if quotes.empty:
        return quotes.copy()

    enriched = quotes.copy()
    for column in REALTIME_NUMERIC_COLUMNS:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    return enriched


def calculate_history_indicators(history: pd.DataFrame, code: str = "") -> dict[str, Any]:
    """Calculate the realtime counterpart of backtest score-input fields."""
    if history.empty:
        raise ValueError("history data is empty")

    normalized = _normalize_history_frame(history)
    close = normalized["close"].dropna()
    high = normalized["high"].dropna()
    low = normalized["low"].dropna()
    volume = normalized["volume"].dropna()
    amount = normalized["amount"].dropna()

    if close.empty:
        raise ValueError("history data has no valid close prices")

    daily_returns = close.pct_change(fill_method=None).mul(100)
    prior_20d_high = _rolling_high(high, 20)
    limit_threshold = _limit_up_threshold(code)
    recent_returns = daily_returns.tail(20).dropna()
    limit_up_count = int((recent_returns >= limit_threshold).sum()) if len(recent_returns) >= 20 else pd.NA
    consecutive_limit_up_days = _consecutive_limit_up_days(daily_returns, limit_threshold)
    up_days_4d = int((daily_returns.tail(4) > 0).sum()) if len(daily_returns.dropna()) >= 4 else pd.NA

    return {
        "ma5": _moving_average(close, 5),
        "ma10": _moving_average(close, 10),
        "ma20": _moving_average(close, 20),
        "pct_chg_5d": _recent_return(close, 5),
        "pct_chg_10d": _recent_return(close, 10),
        "pct_chg_20d": _recent_return(close, 20),
        "prior_price": float(close.iloc[-1]),
        "prior_ma20": _moving_average(close, 20),
        "close_10d_ago": _value_from_end(close, 10),
        "close_20d_ago": _value_from_end(close, 20),
        "up_days_4d": up_days_4d,
        "avg_volume_5d": _average_volume(volume, 5),
        "avg_volume_10d": _average_volume(volume, 10),
        "avg_volume_20d": _average_volume(volume, 20),
        "avg_amount_5d": _average_volume(amount, 5),
        "avg_amount_20d": _average_volume(amount, 20),
        "high_20d": prior_20d_high,
        "low_20d": _rolling_low(low, 20),
        "prior_20d_high": prior_20d_high,
        "limit_up_count_20d": limit_up_count,
        "consecutive_limit_up_days": consecutive_limit_up_days,
        "recent_low_volume_pullback": _has_recent_low_volume_pullback(close, volume),
    }


def merge_realtime_with_history_indicators(
    realtime_quotes: pd.DataFrame,
    data_source: HistoricalDataSource,
    days: int = 60,
    cache: HistoryIndicatorCache | None = None,
) -> pd.DataFrame:
    """Fetch history for active stocks and merge calculated indicators.

    The caller should pass the filtered active stock pool, not the whole
    market, to avoid slow full-market per-stock historical requests.
    """
    if realtime_quotes.empty:
        return _with_indicator_columns(realtime_quotes.copy())

    code_column = _first_existing_column(realtime_quotes, CODE_COLUMNS)
    if code_column is None:
        LOGGER.warning("Cannot merge historical indicators: missing stock code column.")
        return _with_indicator_columns(realtime_quotes.copy())

    active_quotes = enrich_realtime_indicators(realtime_quotes)
    indicator_cache = cache or HistoryIndicatorCache()
    merged_rows: list[dict[str, Any]] = []
    skipped_count = 0

    for _, row in active_quotes.iterrows():
        code = _normalize_code(row.get(code_column))
        if not code:
            skipped_count += 1
            LOGGER.warning("Skip row with invalid stock code: %s", row.get(code_column))
            continue

        try:
            indicators = indicator_cache.get(code, days)
            if indicators is None:
                history = data_source.fetch_history(code, days=days)
                if history is None or history.empty:
                    raise ValueError("history data is empty")
                indicators = calculate_history_indicators(history, code=code)
                indicator_cache.set(code, days, indicators)

            merged_row = row.to_dict()
            merged_row.update(indicators)
            merged_rows.append(merged_row)
        except Exception as exc:
            skipped_count += 1
            LOGGER.warning("Skip %s because historical indicators failed: %s", code, exc)

    if skipped_count:
        LOGGER.info("Skipped %s stocks due to historical indicator failures.", skipped_count)

    if not merged_rows:
        return _with_indicator_columns(active_quotes.head(0).copy())

    return _with_indicator_columns(pd.DataFrame(merged_rows))


def enrich_indicators_from_local_daily_cache(
    realtime_quotes: pd.DataFrame,
    daily_dir: str | Path,
    days: int = 60,
) -> pd.DataFrame:
    """Attach score inputs from local ``data/history/daily`` cache only.

    The live scan still supplies the current price, volume and amount.  The
    local daily cache supplies the strictly historical observations needed for
    20-day limit-up activity, moving averages and buy-point context.  Missing
    cache rows remain null rather than being replaced with synthetic values.
    """
    active = enrich_realtime_indicators(realtime_quotes)
    if active.empty:
        return _with_indicator_columns(active)

    code_column = _first_existing_column(active, CODE_COLUMNS)
    if code_column is None:
        LOGGER.warning("Cannot load local score inputs: missing stock code column.")
        return _with_indicator_columns(active)

    active = active.copy()
    active["_indicator_code"] = active[code_column].map(_normalize_code)
    codes = {code for code in active["_indicator_code"] if code}
    history = _load_local_daily_history(Path(daily_dir), codes, days)
    if history.empty:
        LOGGER.warning("Local daily cache has no usable score inputs: %s", daily_dir)
        return _with_indicator_columns(active.drop(columns=["_indicator_code"]))

    indicator_rows: list[dict[str, Any]] = []
    for code, group in history.groupby("_indicator_code", sort=False):
        values = calculate_history_indicators(group, code=code)
        indicator_rows.append({"_indicator_code": code, **values})

    indicators = pd.DataFrame(indicator_rows)
    output = active.merge(indicators, on="_indicator_code", how="left", suffixes=("", "_local"))
    for column in HISTORY_INDICATOR_COLUMNS:
        local_column = f"{column}_local"
        if local_column in output.columns:
            output[column] = output[local_column].combine_first(output.get(column, pd.Series(index=output.index, dtype="object")))
            output = output.drop(columns=[local_column])
    output = _attach_realtime_buy_point_fields(output)
    return _with_indicator_columns(output.drop(columns=["_indicator_code"]))


def enrich_indicators(
    realtime_quotes: pd.DataFrame,
    data_source: HistoricalDataSource | None = None,
    days: int = 60,
    cache: HistoryIndicatorCache | None = None,
) -> pd.DataFrame:
    """Enrich realtime quotes with numeric fields and optional history indicators."""
    if data_source is None:
        return _with_indicator_columns(enrich_realtime_indicators(realtime_quotes))

    return merge_realtime_with_history_indicators(
        realtime_quotes=realtime_quotes,
        data_source=data_source,
        days=days,
        cache=cache,
    )


def _normalize_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize historical K-line fields used by indicator calculations."""
    normalized = history.copy()

    date_column = _first_existing_column(normalized, DATE_COLUMNS)
    if date_column is not None:
        normalized = normalized.sort_values(date_column)

    close_column = _first_existing_column(normalized, CLOSE_COLUMNS)
    high_column = _first_existing_column(normalized, HIGH_COLUMNS)
    low_column = _first_existing_column(normalized, LOW_COLUMNS)
    volume_column = _first_existing_column(normalized, VOLUME_COLUMNS)
    amount_column = _first_existing_column(normalized, AMOUNT_COLUMNS)

    if close_column is None:
        raise ValueError("missing close column")
    if volume_column is None:
        raise ValueError("missing volume column")

    return pd.DataFrame(
        {
            "close": pd.to_numeric(normalized[close_column], errors="coerce"),
            "high": pd.to_numeric(
                normalized[high_column if high_column is not None else close_column],
                errors="coerce",
            ),
            "low": pd.to_numeric(
                normalized[low_column if low_column is not None else close_column],
                errors="coerce",
            ),
            "volume": pd.to_numeric(normalized[volume_column], errors="coerce"),
            "amount": pd.to_numeric(
                normalized[amount_column if amount_column is not None else volume_column],
                errors="coerce",
            ),
        }
    )


def _moving_average(values: pd.Series, window: int) -> float | pd._libs.missing.NAType:
    """Calculate the latest moving average for a window."""
    if len(values) < window:
        return pd.NA
    return float(values.tail(window).mean())


def _recent_return(values: pd.Series, window: int) -> float | pd._libs.missing.NAType:
    """Calculate percentage return over the latest window of trading days."""
    if len(values) < window:
        return pd.NA

    base = values.iloc[-window]
    latest = values.iloc[-1]
    if pd.isna(base) or base == 0:
        return pd.NA

    return float((latest / base - 1) * 100)


def _average_volume(values: pd.Series, window: int) -> float | pd._libs.missing.NAType:
    """Calculate latest average volume for a window."""
    if len(values) < window:
        return pd.NA
    return float(values.tail(window).mean())


def _rolling_high(values: pd.Series, window: int) -> float | pd._libs.missing.NAType:
    """Calculate latest rolling high for a window."""
    if len(values) < window:
        return pd.NA
    return float(values.tail(window).max())


def _rolling_low(values: pd.Series, window: int) -> float | pd._libs.missing.NAType:
    """Calculate latest rolling low for a window."""
    if len(values) < window:
        return pd.NA
    return float(values.tail(window).min())


def _load_local_daily_history(daily_dir: Path, codes: set[str], days: int) -> pd.DataFrame:
    """Read only the most recent local daily-cache files for requested codes."""
    if not codes or not daily_dir.exists():
        return pd.DataFrame()

    paths = sorted(daily_dir.glob("daily_*.csv"))[-max(int(days), 21) :]
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Skip unreadable local daily cache %s: %s", path.name, exc)
            continue
        code_column = _first_existing_column(raw, ("ts_code", "code", "股票代码"))
        date_column = _first_existing_column(raw, ("trade_date", "date", "日期"))
        close_column = _first_existing_column(raw, CLOSE_COLUMNS)
        high_column = _first_existing_column(raw, HIGH_COLUMNS)
        low_column = _first_existing_column(raw, LOW_COLUMNS)
        volume_column = _first_existing_column(raw, ("vol", *VOLUME_COLUMNS))
        amount_column = _first_existing_column(raw, AMOUNT_COLUMNS)
        if not all((code_column, date_column, close_column, volume_column)):
            continue
        data = pd.DataFrame(
            {
                "_indicator_code": raw[code_column].map(_normalize_code),
                "date": pd.to_datetime(raw[date_column], errors="coerce"),
                "close": pd.to_numeric(raw[close_column], errors="coerce"),
                "high": pd.to_numeric(raw[high_column if high_column else close_column], errors="coerce"),
                "low": pd.to_numeric(raw[low_column if low_column else close_column], errors="coerce"),
                "volume": pd.to_numeric(raw[volume_column], errors="coerce"),
                "amount": pd.to_numeric(raw[amount_column if amount_column else volume_column], errors="coerce"),
            }
        )
        data = data[data["_indicator_code"].isin(codes)]
        if not data.empty:
            frames.append(data)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["_indicator_code", "date", "close", "volume"])
        .drop_duplicates(["_indicator_code", "date"], keep="last")
        .sort_values(["_indicator_code", "date"])
        .reset_index(drop=True)
    )


def _has_recent_low_volume_pullback(close: pd.Series, volume: pd.Series) -> bool:
    """Mirror the backtest's prior-three-session low-volume pullback check."""
    start = max(1, len(close) - 3)
    for index in range(start, len(close)):
        prior_close = pd.to_numeric(close.iloc[index - 1], errors="coerce")
        current_close = pd.to_numeric(close.iloc[index], errors="coerce")
        current_volume = pd.to_numeric(volume.iloc[index], errors="coerce")
        average_volume = pd.to_numeric(volume.iloc[max(0, index - 5) : index], errors="coerce").dropna()
        if (
            len(average_volume) >= 5
            and pd.notna(prior_close)
            and pd.notna(current_close)
            and pd.notna(current_volume)
            and current_close < prior_close
            and current_volume < average_volume.mean()
        ):
            return True
    return False


def _consecutive_limit_up_days(daily_returns: pd.Series, threshold: float) -> int | pd._libs.missing.NAType:
    values = pd.to_numeric(daily_returns, errors="coerce").dropna().tolist()
    if not values:
        return pd.NA
    count = 0
    for value in reversed(values):
        if value >= threshold:
            count += 1
        else:
            break
    return count


def _limit_up_threshold(code: str) -> float:
    if str(code).startswith(("300", "301", "688", "689")):
        return 19.5
    if str(code).startswith(("4", "8", "9")):
        return 29.5
    return 9.5


def _attach_realtime_buy_point_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Add diagnostic buy-point fields using current quotes and prior cache data."""
    output = data.copy()
    price = _numeric_column(output, "price")
    prior_high = _numeric_column(output, "prior_20d_high")
    volume = _numeric_column(output, "volume")
    average_volume = _numeric_column(output, "avg_volume_5d")
    valid_high = prior_high.notna() & prior_high.gt(0) & price.notna()
    output["distance_to_20d_high_pct"] = pd.NA
    output.loc[valid_high, "distance_to_20d_high_pct"] = (
        (prior_high.loc[valid_high] - price.loc[valid_high]).clip(lower=0)
        / prior_high.loc[valid_high]
        * 100
    )
    output["volume_breakout"] = (
        price.gt(prior_high)
        & volume.gt(average_volume * 1.3)
        & valid_high
    )
    current_pct_chg = _numeric_column(output, "pct_chg")
    close_10d_ago = _numeric_column(output, "close_10d_ago")
    close_20d_ago = _numeric_column(output, "close_20d_ago")
    up_days_4d = _numeric_column(output, "up_days_4d")
    output["quality_pct_chg_10d"] = pd.NA
    output["quality_pct_chg_20d"] = pd.NA
    valid_10d = price.notna() & close_10d_ago.gt(0)
    valid_20d = price.notna() & close_20d_ago.gt(0)
    output.loc[valid_10d, "quality_pct_chg_10d"] = (
        (price.loc[valid_10d] / close_10d_ago.loc[valid_10d] - 1) * 100
    )
    output.loc[valid_20d, "quality_pct_chg_20d"] = (
        (price.loc[valid_20d] / close_20d_ago.loc[valid_20d] - 1) * 100
    )
    output["quality_up_days_5d"] = pd.NA
    valid_up_days = up_days_4d.notna() & current_pct_chg.notna()
    output.loc[valid_up_days, "quality_up_days_5d"] = (
        up_days_4d.loc[valid_up_days] + (current_pct_chg.loc[valid_up_days] > 0).astype(int)
    )
    return output


def _numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column aligned to ``data`` even when it is absent."""
    values = data[column] if column in data.columns else pd.Series(pd.NA, index=data.index)
    return pd.to_numeric(values, errors="coerce")


def _value_from_end(values: pd.Series, offset: int) -> float | pd._libs.missing.NAType:
    """Return a close ``offset`` sessions before the latest cached close."""
    if len(values) < offset:
        return pd.NA
    value = pd.to_numeric(values.iloc[-offset], errors="coerce")
    return float(value) if pd.notna(value) else pd.NA


def _with_indicator_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure the standard history indicator columns exist."""
    enriched = data.copy()
    for column in HISTORY_INDICATOR_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.NA
    return enriched


def _first_existing_column(data: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    """Find the first available column from a list of aliases."""
    for column in columns:
        if column in data.columns:
            return column
    return None


def _normalize_code(code: Any) -> str:
    """Normalize common A-share code inputs to six digits."""
    digits = re.sub(r"\D", "", str(code))
    if not digits:
        return ""
    return digits[-6:].zfill(6)
