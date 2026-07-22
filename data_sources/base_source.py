"""Base interface for read-only market data sources.

All data source adapters must only read market data. They must not implement
trading, account login, credential storage, fund transfer, or broker actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDataSource(ABC):
    """Abstract read-only quote data source."""

    @abstractmethod
    def fetch_spot(self) -> pd.DataFrame:
        """Fetch the latest full-market spot quotes."""

    @abstractmethod
    def fetch_history(self, code: str, days: int = 60) -> pd.DataFrame:
        """Fetch daily historical quotes for one stock code."""
