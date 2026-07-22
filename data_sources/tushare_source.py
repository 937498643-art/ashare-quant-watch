"""Tushare read-only data source placeholder."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base_source import BaseDataSource


class TushareSource(BaseDataSource):
    """Reserved adapter for future Tushare quote access."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fetch_spot(self) -> pd.DataFrame:
        """Tushare spot quotes are not enabled in V1."""
        raise NotImplementedError("TushareSource is reserved for a future version.")

    def fetch_history(self, code: str, days: int = 60) -> pd.DataFrame:
        """Tushare history quotes are not enabled in V1."""
        raise NotImplementedError("TushareSource is reserved for a future version.")
