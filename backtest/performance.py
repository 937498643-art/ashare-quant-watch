"""Performance aggregation for V5.10 candidate-pool backtests."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


SCORE_INTERVALS = (
    ("90-100", 90.0, 100.0, True),
    ("80-90", 80.0, 90.0, False),
    ("70-80", 70.0, 80.0, False),
)


def build_performance_summary(
    trades: pd.DataFrame,
    coverage: dict[str, Any],
    holding_days: Iterable[int] = (3, 5, 10),
) -> dict[str, Any]:
    """Build requested score-band metrics independently for every horizon."""
    horizons = tuple(sorted({int(day) for day in holding_days if int(day) > 0}))
    rows: list[dict[str, Any]] = []
    scores = pd.to_numeric(trades.get("buy_score", pd.Series(dtype="float64")), errors="coerce")
    for label, lower, upper, include_upper in SCORE_INTERVALS:
        score_mask = (scores >= lower) & (scores <= upper if include_upper else scores < upper)
        for day in horizons:
            returns = pd.to_numeric(
                trades.loc[score_mask, f"return_{day}d"] if f"return_{day}d" in trades.columns else pd.Series(dtype="float64"),
                errors="coerce",
            ).dropna()
            rows.append(_statistics_row(label, day, returns))
    return {
        "method": "信号日收盘买入，分别在第3/5/10个后续交易日收盘卖出；未计交易成本和滑点。",
        "coverage": coverage,
        "trade_rows": int(len(trades)),
        "score_intervals": rows,
    }


def _statistics_row(score_interval: str, holding_days: int, returns: pd.Series) -> dict[str, Any]:
    count = int(len(returns))
    up_count = int((returns > 0).sum())
    return {
        "score_interval": score_interval,
        "holding_days": int(holding_days),
        "sample_count": count,
        "up_count": up_count,
        "win_rate_pct": round(up_count / count * 100, 4) if count else None,
        "average_return_pct": round(float(returns.mean()), 6) if count else None,
        "max_return_pct": round(float(returns.max()), 6) if count else None,
        "max_loss_pct": round(float(returns.min()), 6) if count else None,
    }
