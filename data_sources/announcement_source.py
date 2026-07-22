"""Read-only public announcement title source."""

from __future__ import annotations

import logging
from typing import Any


class AnnouncementSource:
    """Best-effort announcement title fetcher; failures return empty data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_recent_titles(self, code: str, limit: int = 5) -> list[str]:
        """Fetch recent announcement titles for one stock when AKShare supports it."""
        try:
            import akshare as ak
            # AKShare announcement APIs vary by version; keep V1 best-effort.
            func = getattr(ak, "stock_notice_report", None)
            if not callable(func):
                return []
            data = func(symbol=code)
            if data is None or data.empty:
                return []
            title_col = "公告标题" if "公告标题" in data.columns else "title" if "title" in data.columns else None
            if not title_col:
                return []
            return data[title_col].dropna().astype(str).head(limit).tolist()
        except Exception as exc:
            self.logger.warning("Announcement fetch failed for %s: %s", code, exc)
            return []
