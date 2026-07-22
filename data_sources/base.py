"""Common interfaces for market data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataSource(ABC):
    """Read-only market data source interface."""

    @abstractmethod
    def fetch_realtime_quotes(self) -> pd.DataFrame:
        """Fetch the latest A-share market quotes."""
