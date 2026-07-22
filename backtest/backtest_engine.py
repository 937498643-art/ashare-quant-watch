"""V5.10 offline backtest engine for already-selected daily candidate pools.

The engine deliberately does not call or alter the scoring, Top50, or today's
trade-pool logic.  It consumes point-in-time daily candidate-pool CSV files and
uses only local historical daily bars to simulate close-to-close holding returns.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDING_DAYS = (3, 5, 10)
TRADE_COLUMNS = [
    "code",
    "buy_date",
    "buy_price",
    "buy_score",
    "turnover_rate",
    "volume_ratio",
    "return_3d",
    "return_5d",
    "return_10d",
    "candidate_source",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Local paths and holding horizons used by one deterministic run."""

    candidate_dir: Path
    daily_dir: Path = PROJECT_ROOT / "data" / "history" / "daily"
    holding_days: tuple[int, ...] = DEFAULT_HOLDING_DAYS


def load_daily_candidate_pools(candidate_dir: str | Path) -> dict[pd.Timestamp, pd.DataFrame]:
    """Load point-in-time candidate snapshots from ``*.csv`` files.

    A snapshot must provide a stock code and a score.  Its date may be stored in
    ``trade_date``/``signal_date``/``date`` or in the filename, for example
    ``v5_9_top50_20260721.csv``.  The rows are not rescored or filtered here.
    """
    root = Path(candidate_dir)
    pools: dict[pd.Timestamp, list[pd.DataFrame]] = {}
    for file_path in sorted(root.glob("*.csv")):
        raw = _read_csv(file_path)
        if raw.empty:
            continue
        normalized = _normalize_candidate_snapshot(raw, file_path)
        if normalized.empty:
            continue
        for signal_date, day_data in normalized.groupby("buy_date", sort=True):
            pools.setdefault(pd.Timestamp(signal_date), []).append(day_data)

    merged: dict[pd.Timestamp, pd.DataFrame] = {}
    for signal_date, frames in pools.items():
        data = pd.concat(frames, ignore_index=True)
        data = data.sort_values(["buy_score", "code"], ascending=[False, True], kind="stable")
        merged[signal_date] = data.drop_duplicates("code", keep="first").reset_index(drop=True)
    return merged


def load_daily_close_history(daily_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Build code-indexed close histories from local ``daily_YYYYMMDD.csv`` files."""
    frames: list[pd.DataFrame] = []
    for file_path in sorted(Path(daily_dir).glob("daily_*.csv")):
        raw = _read_csv(file_path)
        if raw.empty:
            continue
        code = _series_from_columns(raw, ("ts_code", "code", "股票代码"), text=True)
        trade_date = _series_from_columns(raw, ("trade_date", "date", "日期"))
        close = _series_from_columns(raw, ("close", "收盘"))
        frame = pd.DataFrame(
            {
                "code": code.map(_normalize_code),
                "date": pd.to_datetime(trade_date, errors="coerce"),
                "close": pd.to_numeric(close, errors="coerce"),
            }
        )
        frame = frame[(frame["code"] != "") & frame["date"].notna() & (frame["close"] > 0)]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return {}
    all_bars = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    return {
        str(code): group.sort_values("date").reset_index(drop=True)
        for code, group in all_bars.groupby("code", sort=True)
    }


def run_candidate_pool_backtest(config: BacktestConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate candidate-pool entries at close and exits after each horizon.

    Missing history is reported in coverage rather than replaced by an estimate.
    A horizon means the Nth subsequent cached trading day, not calendar days.
    """
    horizons = _normalize_horizons(config.holding_days)
    pools = load_daily_candidate_pools(config.candidate_dir)
    histories = load_daily_close_history(config.daily_dir)
    coverage: dict[str, Any] = {
        "candidate_pool_dates": len(pools),
        "candidate_rows": 0,
        "simulated_rows": 0,
        "missing_price_history": 0,
        "missing_signal_date": 0,
        "insufficient_future_history": {str(day): 0 for day in horizons},
        "network_requests": 0,
    }
    rows: list[dict[str, Any]] = []

    for buy_date, candidates in sorted(pools.items()):
        for _, candidate in candidates.iterrows():
            coverage["candidate_rows"] += 1
            code = _normalize_code(candidate.get("code"))
            history = histories.get(code)
            if history is None or history.empty:
                coverage["missing_price_history"] += 1
                continue

            entry_index = _date_index(history, buy_date)
            if entry_index is None:
                coverage["missing_signal_date"] += 1
                continue

            entry_price = _finite_number(history.at[entry_index, "close"])
            if entry_price is None or entry_price <= 0:
                coverage["missing_signal_date"] += 1
                continue

            trade: dict[str, Any] = {
                "code": code,
                "buy_date": pd.Timestamp(buy_date).strftime("%Y-%m-%d"),
                "buy_price": round(entry_price, 6),
                "buy_score": _finite_number(candidate.get("buy_score")),
                "turnover_rate": _finite_number(candidate.get("turnover_rate")),
                "volume_ratio": _finite_number(candidate.get("volume_ratio")),
                "candidate_source": str(candidate.get("candidate_source") or ""),
            }
            has_future_return = False
            for day in DEFAULT_HOLDING_DAYS:
                column = f"return_{day}d"
                if day not in horizons or entry_index + day >= len(history):
                    trade[column] = pd.NA
                    if day in horizons:
                        coverage["insufficient_future_history"][str(day)] += 1
                    continue
                exit_price = _finite_number(history.at[entry_index + day, "close"])
                trade[column] = (
                    round((exit_price / entry_price - 1) * 100, 6)
                    if exit_price is not None and exit_price > 0
                    else pd.NA
                )
                has_future_return = has_future_return or pd.notna(trade[column])
            if has_future_return:
                rows.append(trade)
                coverage["simulated_rows"] += 1

    trades = pd.DataFrame(rows)
    for column in TRADE_COLUMNS:
        if column not in trades.columns:
            trades[column] = pd.NA
    return trades[TRADE_COLUMNS].sort_values(["buy_date", "buy_score", "code"], ascending=[True, False, True]).reset_index(drop=True), coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5.10 offline candidate-pool backtest")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Directory containing daily candidate-pool CSV snapshots")
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "daily")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_10")
    parser.add_argument("--holding-days", default="3,5,10", help="Comma-separated trading-day holding horizons")
    return parser.parse_args()


def main() -> int:
    """Run the V5.10 report pipeline when invoked as a module."""
    args = parse_args()
    try:
        from .performance import build_performance_summary
        from .report import write_backtest_report
    except ImportError:  # Supports ``python backtest/backtest_engine.py`` too.
        from performance import build_performance_summary
        from report import write_backtest_report

    config = BacktestConfig(
        candidate_dir=args.candidate_dir,
        daily_dir=args.daily_dir,
        holding_days=_normalize_horizons(args.holding_days.split(",")),
    )
    trades, coverage = run_candidate_pool_backtest(config)
    summary = build_performance_summary(trades, coverage, config.holding_days)
    paths = write_backtest_report(args.output_dir, trades, summary)
    print(f"V5.10 回测完成：交易样本 {len(trades)}，报告 {paths['markdown']}")
    return 0


def _normalize_candidate_snapshot(raw: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    code = _series_from_columns(raw, ("code", "ts_code", "股票代码"), text=True).map(_normalize_code)
    raw_score = _first_numeric_series(raw, ("final_trade_score", "score", "base_score", "买入评分", "评分"))
    buy_date = _snapshot_dates(raw, file_path)
    output = pd.DataFrame(
        {
            "code": code,
            "buy_date": buy_date,
            "buy_score": raw_score,
            "turnover_rate": _first_numeric_series(raw, ("realtime_turnover_value", "turnover_rate", "turnover", "换手率")),
            "volume_ratio": _first_numeric_series(raw, ("volume_ratio", "量比")),
            "candidate_source": file_path.name,
        }
    )
    output = output[(output["code"] != "") & output["buy_date"].notna() & output["buy_score"].notna()]
    return output


def _snapshot_dates(raw: pd.DataFrame, file_path: Path) -> pd.Series:
    values = _series_from_columns(raw, ("trade_date", "signal_date", "buy_date", "date", "日期"))
    dates = pd.to_datetime(values, errors="coerce")
    if dates.notna().any():
        return dates
    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", file_path.stem)
    fallback = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce") if match else pd.NaT
    return pd.Series(fallback, index=raw.index)


def _series_from_columns(raw: pd.DataFrame, names: Iterable[str], *, text: bool = False) -> pd.Series:
    for name in names:
        if name in raw.columns:
            values = raw[name]
            return values.astype("string") if text else values
    return pd.Series(pd.NA, index=raw.index, dtype="string" if text else "object")


def _first_numeric_series(raw: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    values = [pd.to_numeric(raw[name], errors="coerce") for name in names if name in raw.columns]
    if not values:
        return pd.Series(float("nan"), index=raw.index)
    return pd.concat(values, axis=1).bfill(axis=1).iloc[:, 0]


def _date_index(history: pd.DataFrame, date: pd.Timestamp) -> int | None:
    matches = history.index[history["date"].eq(pd.Timestamp(date))]
    return int(matches[0]) if len(matches) else None


def _normalize_horizons(values: Iterable[int | str]) -> tuple[int, ...]:
    horizons = tuple(sorted({int(value) for value in values if str(value).strip() and int(value) > 0}))
    if not horizons:
        raise ValueError("holding_days must contain at least one positive integer")
    unsupported = sorted(set(horizons).difference(DEFAULT_HOLDING_DAYS))
    if unsupported:
        raise ValueError(f"V5.10 仅支持持有期 {DEFAULT_HOLDING_DAYS}，不支持 {unsupported}")
    return horizons


def _normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text in {"", "<NA>", "NAN", "NONE"}:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6) if text.isdigit() else text


def _finite_number(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"code": "string", "ts_code": "string"})
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


if __name__ == "__main__":
    raise SystemExit(main())
