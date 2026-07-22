"""iFinD read-only data source placeholder."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base_source import BaseDataSource


class IfindSource(BaseDataSource):
    """Reserved adapter for future iFinD quote access."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fetch_spot(self) -> pd.DataFrame:
        """iFinD spot quotes are not enabled in V1."""
        raise NotImplementedError("IfindSource is reserved for a future version.")

    def fetch_history(self, code: str, days: int = 60) -> pd.DataFrame:
        """iFinD history quotes are not enabled in V1."""
        raise NotImplementedError("IfindSource is reserved for a future version.")
