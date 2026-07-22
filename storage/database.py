"""SQLite persistence for candidate scan results.

The database stores read-only watchlist output only. Write failures are logged
and never stop the main scan loop.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "quant_watch.db"

CANDIDATE_COLUMNS = [
    "scan_time",
    "trade_date",
    "selection_trade_date",
    "source_type",
    "source_type_display",
    "board_type",
    "board_type_display",
    "code",
    "market_code",
    "name",
    "price",
    "pct_chg",
    "amount",
    "turnover",
    "volume_ratio",
    "strategy_names",
    "score",
    "level",
    "level_display",
    "reason",
    "risk_summary",
    "data_source",
    "data_source_level",
    "turnover_level_display",
    "turnover_score",
    "turnover_summary",
    "money_strength_level",
    "money_strength_summary",
    "sector_name",
    "sector_strength_level",
    "theme_tags",
    "theme_heat_level",
    "action_state_display",
    "position_risk_level",
    "position_risk_summary",
    "t1_risk_level",
    "t1_risk_summary",
    "first_seen_today",
    "consecutive_count",
    "selection_days_total",
    "consecutive_selection_days",
    "score_trend",
    "tracking_summary",
]

SELECTION_DAY_COLUMNS = [
    "code",
    "selection_trade_date",
    "first_scan_time",
    "latest_scan_time",
    "name",
    "source_type",
    "latest_price",
    "latest_score",
    "latest_pct_chg",
    "latest_amount",
    "latest_turnover",
    "latest_volume_ratio",
    "level",
    "risk_summary",
    "selection_days_total",
    "consecutive_selection_days",
]


class CandidateDatabase:
    """Persist candidate scan snapshots to SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Create the candidates table if it does not exist."""
        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candidates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_time TEXT NOT NULL,
                        trade_date TEXT,
                        source_type TEXT,
                        source_type_display TEXT,
                        board_type TEXT,
                        board_type_display TEXT,
                        code TEXT,
                        market_code TEXT,
                        name TEXT,
                        price REAL,
                        pct_chg REAL,
                        amount REAL,
                        turnover REAL,
                        volume_ratio REAL,
                        strategy_names TEXT,
                        score REAL,
                        level TEXT,
                        level_display TEXT,
                        reason TEXT,
                        risk_summary TEXT,
                        data_source TEXT,
                        data_source_level TEXT
                    )
                    """
                )
                for column in CANDIDATE_COLUMNS:
                    if column != "scan_time":
                        _ensure_column(connection, "candidates", column, _sqlite_type(column))
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candidate_selection_days (
                        code TEXT NOT NULL,
                        selection_trade_date TEXT NOT NULL,
                        first_scan_time TEXT NOT NULL,
                        latest_scan_time TEXT NOT NULL,
                        name TEXT,
                        source_type TEXT,
                        latest_price REAL,
                        latest_score REAL,
                        latest_pct_chg REAL,
                        latest_amount REAL,
                        latest_turnover REAL,
                        latest_volume_ratio REAL,
                        level TEXT,
                        risk_summary TEXT,
                        selection_days_total REAL,
                        consecutive_selection_days REAL,
                        PRIMARY KEY (code, selection_trade_date)
                    )
                    """
                )
                connection.commit()
        except Exception:
            self.logger.exception("Failed to initialize SQLite database: %s", self.db_path)

    def save_candidates(
        self,
        candidates: pd.DataFrame,
        scan_time: datetime | str | None = None,
    ) -> None:
        """Persist one scan while upserting one effective record per trade day.

        The legacy ``candidates`` table remains an append-only scan snapshot.
        ``candidate_selection_days`` is the deduplicated day-level source used
        for selection-day and consecutive-day statistics.
        """
        try:
            self.initialize()
            if candidates.empty:
                self.logger.info("No candidates to save into SQLite.")
                return

            rows = _prepare_candidate_rows(candidates, scan_time)
            with sqlite3.connect(self.db_path) as connection:
                rows.to_sql("candidates", connection, if_exists="append", index=False)
                _upsert_selection_days(connection, rows)
                connection.commit()

            self.logger.info("Saved %s candidates into SQLite: %s", len(rows), self.db_path)
        except Exception:
            self.logger.exception("Failed to save candidates into SQLite: %s", self.db_path)


def save_candidates_to_db(
    candidates: pd.DataFrame,
    scan_time: datetime | str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Convenience function for saving candidates to the default database."""
    CandidateDatabase(db_path).save_candidates(candidates, scan_time)


def _prepare_candidate_rows(
    candidates: pd.DataFrame,
    scan_time: datetime | str | None,
) -> pd.DataFrame:
    """Map scanner result columns to the candidates table schema."""
    timestamp = _format_scan_time(scan_time)
    default_trade_date = timestamp[:10]
    selection_dates = _selection_dates(candidates, default_trade_date)
    rows = pd.DataFrame()
    rows["scan_time"] = [timestamp] * len(candidates)
    rows["trade_date"] = selection_dates
    rows["selection_trade_date"] = selection_dates
    for column in CANDIDATE_COLUMNS:
        if column in {"scan_time", "trade_date", "selection_trade_date"}:
            continue
        if _sqlite_type(column) == "REAL":
            rows[column] = pd.to_numeric(_series(candidates, column), errors="coerce")
        else:
            rows[column] = _series(candidates, column)
    return rows[CANDIDATE_COLUMNS]


def _selection_dates(candidates: pd.DataFrame, default_trade_date: str) -> pd.Series:
    """Use an existing resolved trading date, otherwise use the scan date."""
    if "selection_trade_date" not in candidates.columns:
        return pd.Series([default_trade_date] * len(candidates), index=candidates.index)
    values = candidates["selection_trade_date"].fillna("").astype(str).str[:10]
    return values.where(values.str.match(r"^\d{4}-\d{2}-\d{2}$"), default_trade_date)


def _upsert_selection_days(connection: sqlite3.Connection, rows: pd.DataFrame) -> None:
    """Insert or refresh the one effective selection snapshot for each day."""
    selections = pd.DataFrame({
        "code": rows["code"].fillna("").astype(str).str.zfill(6),
        "selection_trade_date": rows["selection_trade_date"].fillna("").astype(str),
        "first_scan_time": rows["scan_time"].astype(str),
        "latest_scan_time": rows["scan_time"].astype(str),
        "name": rows.get("name", ""),
        "source_type": rows.get("source_type", ""),
        "latest_price": rows.get("price"),
        "latest_score": rows.get("score"),
        "latest_pct_chg": rows.get("pct_chg"),
        "latest_amount": rows.get("amount"),
        "latest_turnover": rows.get("turnover"),
        "latest_volume_ratio": rows.get("volume_ratio"),
        "level": rows.get("level", ""),
        "risk_summary": rows.get("risk_summary", ""),
        "selection_days_total": rows.get("selection_days_total"),
        "consecutive_selection_days": rows.get("consecutive_selection_days"),
    })
    selections = selections[
        selections["code"].str.match(r"^\d{6}$")
        & selections["selection_trade_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")
    ]
    if selections.empty:
        return

    placeholders = ", ".join(f":{column}" for column in SELECTION_DAY_COLUMNS)
    update_columns = [column for column in SELECTION_DAY_COLUMNS if column not in {"code", "selection_trade_date", "first_scan_time"}]
    updates = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
    statement = (
        f"INSERT INTO candidate_selection_days ({', '.join(SELECTION_DAY_COLUMNS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(code, selection_trade_date) DO UPDATE SET {updates}"
    )
    records = selections[SELECTION_DAY_COLUMNS].where(pd.notna(selections), None).to_dict("records")
    connection.executemany(statement, records)


def _series(candidates: pd.DataFrame, column: str) -> pd.Series:
    """Return a string-safe column or an empty series."""
    if column not in candidates.columns:
        return pd.Series([""] * len(candidates))

    values = candidates[column]
    if _sqlite_type(column) == "REAL":
        return values
    return values.fillna("").astype(str)


def _sqlite_type(column: str) -> str:
    """Return SQLite type for candidate columns."""
    numeric = {
        "price",
        "pct_chg",
        "amount",
        "turnover",
        "volume_ratio",
        "score",
        "turnover_score",
        "consecutive_count",
        "selection_days_total",
        "consecutive_selection_days",
    }
    return "REAL" if column in numeric else "TEXT"


def _format_scan_time(scan_time: datetime | str | None) -> str:
    """Format scan time for SQLite storage."""
    if isinstance(scan_time, datetime):
        return scan_time.strftime("%Y-%m-%d %H:%M:%S")
    if scan_time:
        return str(scan_time)
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    """Add a missing column to an existing SQLite table."""
    existing_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in existing_columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
