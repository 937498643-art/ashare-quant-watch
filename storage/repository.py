"""Local CSV storage for scan results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


class LocalCsvRepository:
    """Persist candidate snapshots to local CSV files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_candidates(self, candidates: pd.DataFrame) -> Path | None:
        if candidates.empty:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"candidates_{timestamp}.csv"
        candidates.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
