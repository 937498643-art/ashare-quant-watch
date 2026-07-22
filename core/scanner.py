"""Candidate scanner skeleton."""

from __future__ import annotations

import pandas as pd

from core.filters import apply_basic_filters
from core.indicators import enrich_realtime_indicators


class StockScanner:
    """Run filters and lightweight candidate checks."""

    def __init__(self, filter_config: dict, strategy_config: dict) -> None:
        self.filter_config = filter_config
        self.strategy_config = strategy_config

    def scan(self, quotes: pd.DataFrame) -> pd.DataFrame:
        if quotes.empty:
            return quotes

        filtered = apply_basic_filters(quotes, self.filter_config)
        enriched = enrich_realtime_indicators(filtered)

        # First skeleton version: keep strategy judgement conservative.
        # Detailed breakout, pullback, and MA alignment checks will be added
        # after historical quote fetching is introduced.
        if "量比" in enriched.columns:
            candidates = enriched[enriched["量比"] >= self.filter_config.get("min_volume_ratio", 1.0)].copy()
        else:
            candidates = enriched.head(0).copy()

        if not candidates.empty:
            candidates["触发原因"] = "基础过滤通过，量比达到配置阈值；待后续加入 MA 与形态复核"

        return candidates
