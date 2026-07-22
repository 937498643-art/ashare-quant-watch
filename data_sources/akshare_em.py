"""AKShare / East Money market data source."""

from __future__ import annotations

import logging

import pandas as pd

from data_sources.base import MarketDataSource


class AkshareEastMoneySource(MarketDataSource):
    """Read A-share realtime quotes from AKShare's East Money endpoint."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_realtime_quotes(self) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError(
                "AKShare is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.logger.info("Fetching realtime A-share quotes from AKShare.")
        return ak.stock_zh_a_spot_em()
