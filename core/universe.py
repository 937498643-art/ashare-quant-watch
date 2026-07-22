"""Build the filtered A-share stock universe.

This module receives full-market realtime quotes and returns a stock pool that
is suitable for read-only strategy screening. It contains no trading logic.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .filters import filter_stock_pool


def build_universe(quotes: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build the filtered stock universe from full-market realtime quotes."""
    return filter_stock_pool(quotes, config)


def apply_universe_rules(
    quotes: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for older code paths."""
    return build_universe(quotes, config)
