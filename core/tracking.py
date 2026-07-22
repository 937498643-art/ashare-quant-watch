"""Candidate appearance tracking based on deduplicated selection days."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADE_CALENDAR_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "tushare" / "trade_cal.csv"


def enrich_tracking(
    candidates: pd.DataFrame,
    db_path: Path,
    selection_trade_date: str | None = None,
) -> pd.DataFrame:
    """Add tracking fields using unique code plus selection-trade-date history."""
    if candidates.empty:
        return candidates.copy()
    history = _read_history(db_path)
    trade_calendar = _load_cached_trade_dates()
    resolved_date = _resolve_selection_trade_date(history, selection_trade_date, trade_calendar)
    rows = []
    for _, row in candidates.iterrows():
        code = str(row.get("code") or "").zfill(6)
        hist = history[history["code"].astype(str).str.zfill(6) == code].copy()
        result = _tracking_for_row(row, hist, resolved_date, trade_calendar)
        output = row.to_dict()
        output["selection_trade_date"] = resolved_date
        output.update(result)
        rows.append(output)
    return pd.DataFrame(rows)


def _tracking_for_row(
    row: pd.Series,
    hist: pd.DataFrame,
    selection_date: str,
    trade_calendar: list[str],
) -> dict[str, Any]:
    """Calculate day-level totals, ignoring repeated scans of the same day."""
    hist = _deduplicate_history(hist)
    same_day_exists = bool(hist["selection_trade_date"].eq(selection_date).any()) if not hist.empty else False
    prior = hist[hist["selection_trade_date"] < selection_date].sort_values("latest_scan_time")
    recent = prior.tail(3)
    current_score = pd.to_numeric(row.get("score"), errors="coerce")
    previous_score = pd.to_numeric(recent.iloc[-1].get("score"), errors="coerce") if not recent.empty else pd.NA
    score_trend = "持平"
    if pd.notna(current_score) and pd.notna(previous_score):
        if current_score > previous_score:
            score_trend = "上升"
        elif current_score < previous_score:
            score_trend = "下降"
    previous_type = str(recent.iloc[-1].get("source_type") or "") if not recent.empty else ""
    current_type = str(row.get("source_type") or "")
    upgraded = previous_type == "active_watchlist" and current_type == "strategy_candidate"
    downgraded = previous_type == "strategy_candidate" and current_type == "active_watchlist"
    all_dates = set(prior["selection_trade_date"].tolist()) | {selection_date}
    selection_days_total = len(all_dates)
    consecutive = _consecutive_selection_days(all_dates, selection_date, trade_calendar)
    parts = [f"连续入选 {consecutive} 个交易日", f"累计入选 {selection_days_total} 个交易日", f"分数{score_trend}"]
    if upgraded:
        parts.append("今日升级")
    if downgraded:
        parts.append("今日降级")
    return {
        "first_seen_today": not same_day_exists,
        "consecutive_count": consecutive,
        "selection_days_total": selection_days_total,
        "consecutive_selection_days": consecutive,
        "score_trend": score_trend,
        "level_trend": "持平",
        "upgraded_today": upgraded,
        "downgraded_today": downgraded,
        "tracking_summary": "；".join(parts),
    }


def _consecutive_selection_days(
    selection_dates: set[str],
    current_date: str,
    trade_calendar: list[str],
) -> int:
    """Count adjacent trading days from cached calendar or weekday fallback."""
    normalized = {_normalize_date(value) for value in selection_dates}
    normalized.discard("")
    if current_date not in normalized:
        return 0

    count = 1
    cursor = datetime.strptime(current_date, "%Y-%m-%d").date()
    while True:
        previous = _previous_trade_day(cursor.isoformat(), trade_calendar)
        if not previous or previous not in normalized:
            return count
        cursor = datetime.strptime(previous, "%Y-%m-%d").date()
        count += 1


def _previous_weekday(current: date) -> date:
    """Return the previous weekday; this avoids false breaks across weekends."""
    previous = current - timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= timedelta(days=1)
    return previous


def _previous_trade_day(current_date: str, trade_calendar: list[str]) -> str | None:
    """Use a cached A-share calendar when available, otherwise skip weekends."""
    if trade_calendar:
        calendar_last_day = trade_calendar[-1]
        # A stale cache cannot determine the prior trading day for a newer
        # requested date. Fall back to weekdays instead of reusing its last day.
        if current_date <= calendar_last_day:
            earlier = [day for day in trade_calendar if day < current_date]
            if earlier:
                return earlier[-1]
    try:
        return _previous_weekday(datetime.strptime(current_date, "%Y-%m-%d").date()).isoformat()
    except ValueError:
        return None


def _read_history(db_path: Path) -> pd.DataFrame:
    """Read day records plus legacy snapshots, then deduplicate by code/date."""
    if not db_path.exists():
        return pd.DataFrame(columns=["code", "selection_trade_date", "latest_scan_time"])
    try:
        with sqlite3.connect(db_path) as connection:
            table_names = {
                row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            frames: list[pd.DataFrame] = []
            if "candidates" in table_names:
                legacy = pd.read_sql_query("SELECT * FROM candidates", connection)
                if not legacy.empty:
                    legacy["selection_trade_date"] = _selection_date_series(legacy)
                    legacy["latest_scan_time"] = legacy.get("scan_time", "")
                    legacy["history_priority"] = 0
                    frames.append(legacy)
            if "candidate_selection_days" in table_names:
                daily = pd.read_sql_query("SELECT * FROM candidate_selection_days", connection)
                if not daily.empty:
                    daily["selection_trade_date"] = daily["selection_trade_date"].map(_normalize_date)
                    daily["score"] = daily.get("latest_score", pd.NA)
                    daily["history_priority"] = 1
                    frames.append(daily)
            if not frames:
                return pd.DataFrame(columns=["code", "selection_trade_date", "latest_scan_time"])
            return _deduplicate_history(pd.concat(frames, ignore_index=True, sort=False))
    except Exception:
        return pd.DataFrame(columns=["code", "selection_trade_date", "latest_scan_time"])


def _deduplicate_history(history: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest snapshot for each effective code and trading date."""
    if history.empty:
        return history.copy()
    output = history.copy()
    output["code"] = output.get("code", pd.Series(dtype="string")).fillna("").astype(str).str.zfill(6)
    if "selection_trade_date" not in output.columns:
        output["selection_trade_date"] = _selection_date_series(output)
    output["selection_trade_date"] = output["selection_trade_date"].map(_normalize_date)
    output["latest_scan_time"] = output.get("latest_scan_time", output.get("scan_time", "")).fillna("").astype(str)
    priority = output.get("history_priority")
    if priority is None:
        priority = pd.Series(0, index=output.index)
    output["history_priority"] = pd.to_numeric(priority, errors="coerce").fillna(0)
    output = output[
        output["code"].str.match(r"^\d{6}$")
        & output["selection_trade_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")
    ]
    return output.sort_values(["code", "selection_trade_date", "history_priority", "latest_scan_time"]).drop_duplicates(
        ["code", "selection_trade_date"], keep="last"
    )


def _selection_date_series(history: pd.DataFrame) -> pd.Series:
    values = history.get("selection_trade_date", history.get("trade_date", pd.Series(index=history.index, dtype="object")))
    values = values.fillna("").astype(str)
    fallback = history.get("scan_time", pd.Series(index=history.index, dtype="object")).fillna("").astype(str).str[:10]
    return values.where(values.str.match(r"^\d{4}-\d{2}-\d{2}$"), fallback).map(_normalize_date)


def _normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4:5] == "-":
        return text[:10]
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return ""


def _resolve_selection_trade_date(
    history: pd.DataFrame,
    requested: str | None,
    trade_calendar: list[str],
) -> str:
    """Resolve an effective market date without creating a weekend selection day."""
    proposed = _normalize_date(requested) if requested else datetime.now().date().isoformat()
    try:
        current = datetime.strptime(proposed, "%Y-%m-%d").date()
    except ValueError:
        current = datetime.now().date()
    if trade_calendar:
        if current.isoformat() in trade_calendar:
            return current.isoformat()
        if current.isoformat() <= trade_calendar[-1]:
            earlier = [day for day in trade_calendar if day <= current.isoformat()]
            if earlier:
                return earlier[-1]
    if current.weekday() < 5:
        return current.isoformat()
    known_dates = sorted(history.get("selection_trade_date", pd.Series(dtype="object")).dropna().astype(str).tolist())
    return known_dates[-1] if known_dates else _previous_weekday(current).isoformat()


def _load_cached_trade_dates() -> list[str]:
    """Read an optional local Tushare trade-calendar cache without requesting data."""
    if not TRADE_CALENDAR_CACHE_PATH.exists():
        return []
    try:
        calendar = pd.read_csv(TRADE_CALENDAR_CACHE_PATH, dtype=str)
        date_column = "cal_date" if "cal_date" in calendar.columns else "trade_date"
        if date_column not in calendar.columns:
            return []
        if "is_open" in calendar.columns:
            calendar = calendar[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)]
        return sorted({_normalize_date(value) for value in calendar[date_column].tolist() if _normalize_date(value)})
    except Exception:
        return []
