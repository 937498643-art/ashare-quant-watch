"""CSV persistence for the local strategy-tracking pool.

The tracking pool is an observation record only.  It does not connect to a
broker, submit orders, or change any scoring or selection result.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_TRACKING_POOL_PATH = PROJECT_ROOT / "data" / "user" / "strategy_tracking_pool.csv"
STRATEGY_TRACKING_HISTORY_PATH = PROJECT_ROOT / "data" / "user" / "strategy_tracking_history.csv"
LOCAL_DAILY_HISTORY_DIR = PROJECT_ROOT / "data" / "history" / "daily"
STRATEGY_TRACKING_COLUMNS = [
    "tracking_id",
    "code",
    "name",
    "source_page",
    "strategy_type",
    "entry_reason",
    "entry_snapshot",
    "added_at",
    "added_date",
    "entry_price",
    "entry_final_trade_score",
    "entry_turnover",
    "entry_volume_ratio",
    "last_update_at",
    "current_price",
    "current_return_pct",
    "holding_days",
    "highest_price",
    "highest_return_pct",
    "max_drawdown_pct",
    "current_final_trade_score",
    "exit_date",
    "exit_price",
    "exit_reason",
    "profit_loss",
    "status",
    "data_status",
]
STRATEGY_TRACKING_HISTORY_COLUMNS = [
    "tracking_id",
    "code",
    "snapshot_date",
    "snapshot_time",
    "name",
    "entry_price",
    "price",
    "pct_chg",
    "cumulative_return_pct",
    "entry_final_trade_score",
    "highest_return_pct",
    "max_drawdown_pct",
    "current_final_trade_score",
    "status",
]
STRATEGY_TRACKING_TEXT_COLUMNS = {
    "tracking_id",
    "code",
    "name",
    "source_page",
    "strategy_type",
    "entry_reason",
    "entry_snapshot",
    "added_at",
    "added_date",
    "last_update_at",
    "exit_date",
    "exit_reason",
    "status",
    "data_status",
}


def ensure_strategy_tracking_file(path: Path = STRATEGY_TRACKING_POOL_PATH) -> Path:
    """Create the independent tracking CSV with its stable schema when absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_pool(pd.DataFrame(columns=STRATEGY_TRACKING_COLUMNS), path)
    return path


def ensure_strategy_tracking_history_file(path: Path = STRATEGY_TRACKING_HISTORY_PATH) -> Path:
    """Create the independent append/upsert daily snapshot CSV when absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_csv(pd.DataFrame(columns=STRATEGY_TRACKING_HISTORY_COLUMNS), path)
    return path


def load_strategy_tracking_pool(path: Path = STRATEGY_TRACKING_POOL_PATH) -> pd.DataFrame:
    """Load the local strategy-tracking pool without touching other user CSVs."""
    ensure_strategy_tracking_file(path)
    try:
        data = pd.read_csv(path, dtype={"code": "string"})
    except Exception:
        data = pd.DataFrame(columns=STRATEGY_TRACKING_COLUMNS)
    for column in STRATEGY_TRACKING_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    for column in STRATEGY_TRACKING_TEXT_COLUMNS:
        data[column] = data[column].astype("object")
    data["code"] = data["code"].map(_normalize_code).astype("string")
    data["tracking_id"] = data.apply(_ensure_tracking_id, axis=1).astype("string")
    return data[STRATEGY_TRACKING_COLUMNS]


def add_strategy_tracking_stock(
    row: Mapping[str, Any] | pd.Series,
    source_page: str,
    now: datetime | None = None,
    path: Path = STRATEGY_TRACKING_POOL_PATH,
) -> tuple[bool, str]:
    """Persist the exact score and quote values visible when the user adds a stock."""
    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    code = _normalize_code(values.get("code"))
    if not code or code == "000000":
        return False, "股票代码无效"

    pool = load_strategy_tracking_pool(path)
    active_codes = set(
        pool.loc[pool["status"].fillna("").astype(str) != "已结束", "code"].astype(str).str.zfill(6)
    )
    if code in active_codes:
        return False, "该股票已在策略跟踪池"

    timestamp = now or datetime.now()
    entry_price = _number(values, "price", "最新价")
    entry_final_score = _number(values, "final_trade_score")
    entry_turnover = _number(values, "realtime_turnover_value", "turnover", "换手率")
    entry_volume_ratio = _number(values, "volume_ratio", "量比")
    entry_snapshot = _build_entry_snapshot(
        values,
        code=code,
        name=_text(values.get("name")),
        entry_price=entry_price,
        turnover=entry_turnover,
        volume_ratio=entry_volume_ratio,
        final_trade_score=entry_final_score,
    )
    has_price = pd.notna(entry_price) and entry_price > 0
    row_to_add = {
        "tracking_id": _new_tracking_id(code, timestamp),
        "code": code,
        "name": _text(values.get("name")),
        "source_page": str(source_page or "").strip(),
        "strategy_type": _first_text(values, "strategy_names_display", "strategy_names", "strategy_name") or str(source_page or "").strip(),
        "entry_reason": _build_entry_reason(entry_snapshot),
        "entry_snapshot": json.dumps(entry_snapshot, ensure_ascii=False, separators=(",", ":")),
        "added_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "added_date": timestamp.date().isoformat(),
        "entry_price": entry_price if has_price else pd.NA,
        "entry_final_trade_score": entry_final_score,
        "entry_turnover": entry_turnover,
        "entry_volume_ratio": entry_volume_ratio,
        "last_update_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": entry_price if has_price else pd.NA,
        "current_return_pct": 0.0 if has_price else pd.NA,
        "holding_days": 1,
        "highest_price": entry_price if has_price else pd.NA,
        "highest_return_pct": 0.0 if has_price else pd.NA,
        "max_drawdown_pct": 0.0 if has_price else pd.NA,
        "current_final_trade_score": entry_final_score,
        "exit_date": pd.NA,
        "exit_price": pd.NA,
        "exit_reason": pd.NA,
        "profit_loss": pd.NA,
        "status": "持有观察" if has_price else "待行情更新",
        "data_status": "加入时数据完整" if has_price else "加入时缺少价格",
    }
    output = pd.concat([pool, pd.DataFrame([row_to_add])], ignore_index=True)
    _write_pool(output[STRATEGY_TRACKING_COLUMNS], path)
    return True, "已加入策略跟踪池"


def update_strategy_tracking_pool(
    score_snapshot: pd.DataFrame,
    updated_at: datetime | None = None,
    path: Path = STRATEGY_TRACKING_POOL_PATH,
    history_path: Path = STRATEGY_TRACKING_HISTORY_PATH,
) -> dict[str, int]:
    """Update tracked rows from the current scan's already-generated snapshot.

    This function performs no network access and deliberately does not invoke
    scoring functions.  It only records quote and score values that the scan
    has already calculated.
    """
    pool = load_strategy_tracking_pool(path)
    ensure_strategy_tracking_history_file(history_path)
    result = {"tracked": int(len(pool)), "updated": 0, "unmatched": 0, "price_missing": 0}
    if pool.empty:
        return result

    timestamp = updated_at or datetime.now()
    latest = _latest_rows_by_code(score_snapshot)
    history_rows: list[dict[str, Any]] = []
    for index, tracked in pool.iterrows():
        if _is_closed(tracked):
            # A completed record remains part of the long-term observation
            # ledger.  Preserve its manually-recorded close data in the
            # daily snapshot rather than altering it with live quotes.
            history_rows.append(_history_row(tracked, timestamp, _number(tracked, "current_price")))
            continue
        code = _normalize_code(tracked.get("code"))
        matched = latest.get(code)
        pool.at[index, "last_update_at"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        pool.at[index, "holding_days"] = _holding_days(tracked.get("added_date"), timestamp)
        if matched is None:
            pool.at[index, "data_status"] = "当前扫描未包含该股票"
            result["unmatched"] += 1
            history_rows.append(_history_row(pool.loc[index], timestamp, float("nan")))
            continue

        current_price = _number(matched, "price", "最新价")
        current_final_score = _number(matched, "final_trade_score")
        pool.at[index, "current_final_trade_score"] = current_final_score
        if pd.isna(current_price) or current_price <= 0:
            pool.at[index, "data_status"] = "当前快照缺少有效价格"
            result["price_missing"] += 1
            history_rows.append(_history_row(pool.loc[index], timestamp, float("nan")))
            continue

        entry_price = _number(tracked, "entry_price")
        old_high = _number(tracked, "highest_price")
        observed_high = _number(matched, "high")
        high_candidates = [value for value in (entry_price, old_high, current_price, observed_high) if pd.notna(value) and value > 0]
        highest_price = max(high_candidates) if high_candidates else current_price

        pool.at[index, "current_price"] = current_price
        pool.at[index, "highest_price"] = highest_price
        current_return = float("nan")
        if pd.notna(entry_price) and entry_price > 0:
            current_return = (current_price / entry_price - 1) * 100
            highest_return = (highest_price / entry_price - 1) * 100
            current_drawdown = (current_price / highest_price - 1) * 100 if highest_price > 0 else pd.NA
            old_drawdown = _number(tracked, "max_drawdown_pct")
            max_drawdown = min(
                [value for value in (old_drawdown, current_drawdown) if pd.notna(value)] or [0.0]
            )
            pool.at[index, "current_return_pct"] = current_return
            pool.at[index, "highest_return_pct"] = highest_return
            pool.at[index, "max_drawdown_pct"] = max_drawdown
        pool.at[index, "status"] = _automatic_tracking_status(current_return, pool.at[index, "holding_days"])
        pool.at[index, "data_status"] = "已按本轮扫描更新"
        result["updated"] += 1
        history_rows.append(
            _history_row(
                pool.loc[index],
                timestamp,
                current_price,
                _number(matched, "pct_chg", "涨跌幅"),
            )
        )

    _write_pool(pool[STRATEGY_TRACKING_COLUMNS], path)
    _upsert_history_rows(history_rows, history_path)
    return result


def end_strategy_tracking_stock(
    tracking_id: str,
    exit_price: float,
    exit_date: Any,
    exit_reason: str,
    ended_at: datetime | None = None,
    path: Path = STRATEGY_TRACKING_POOL_PATH,
    history_path: Path = STRATEGY_TRACKING_HISTORY_PATH,
) -> tuple[bool, str]:
    """Manually close one observation record and retain its realized return.

    ``profit_loss`` is a percentage because strategy tracking records do not
    contain position size or cash amounts.
    """
    pool = load_strategy_tracking_pool(path)
    normalized_id = _text(tracking_id)
    matched = pool[pool["tracking_id"].astype(str) == normalized_id]
    if matched.empty:
        return False, "策略跟踪记录不存在"

    price = pd.to_numeric(exit_price, errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        return False, "卖出价格必须大于0"
    date_value = pd.to_datetime(exit_date, errors="coerce")
    if pd.isna(date_value):
        return False, "卖出日期无效"

    index = matched.index[0]
    tracked = pool.loc[index]
    entry_price = _number(tracked, "entry_price")
    realized_return = (float(price) / entry_price - 1) * 100 if pd.notna(entry_price) and entry_price > 0 else float("nan")
    old_high = _number(tracked, "highest_price")
    highest_price = max(
        [value for value in (entry_price, old_high, float(price)) if pd.notna(value) and value > 0] or [float(price)]
    )
    highest_return = (highest_price / entry_price - 1) * 100 if pd.notna(entry_price) and entry_price > 0 else float("nan")
    drawdown = (float(price) / highest_price - 1) * 100 if highest_price > 0 else float("nan")
    old_drawdown = _number(tracked, "max_drawdown_pct")
    max_drawdown = min([value for value in (old_drawdown, drawdown) if pd.notna(value)] or [0.0])
    timestamp = ended_at or datetime.now()
    exit_snapshot_time = datetime.combine(date_value.date(), timestamp.time())

    pool.at[index, "exit_date"] = date_value.date().isoformat()
    pool.at[index, "exit_price"] = float(price)
    pool.at[index, "exit_reason"] = str(exit_reason or "手动结束跟踪").strip() or "手动结束跟踪"
    pool.at[index, "profit_loss"] = realized_return
    pool.at[index, "current_price"] = float(price)
    pool.at[index, "current_return_pct"] = realized_return
    pool.at[index, "highest_price"] = highest_price
    pool.at[index, "highest_return_pct"] = highest_return
    pool.at[index, "max_drawdown_pct"] = max_drawdown
    pool.at[index, "holding_days"] = _holding_days(tracked.get("added_date"), exit_snapshot_time)
    pool.at[index, "last_update_at"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    pool.at[index, "status"] = "已结束"
    pool.at[index, "data_status"] = "手动结束跟踪"
    _write_pool(pool[STRATEGY_TRACKING_COLUMNS], path)

    ensure_strategy_tracking_history_file(history_path)
    _upsert_history_rows([_history_row(pool.loc[index], exit_snapshot_time, float(price))], history_path)
    return True, "已结束策略跟踪"


def build_strategy_tracking_statistics(tracking_pool: pd.DataFrame) -> dict[str, Any]:
    """Summarize saved strategy-tracking outcomes without recalculating scores."""
    data = tracking_pool.copy()
    if data.empty:
        return {
            "overview": _empty_tracking_overview(),
            "return_distribution": _return_distribution(data),
            "score_groups": _group_statistics(data, _empty_series(data), _score_group_rules()),
            "turnover_groups": _group_statistics(data, _empty_series(data), _turnover_group_rules()),
            "volume_ratio_groups": _group_statistics(data, _empty_series(data), _volume_ratio_group_rules()),
            "reason_groups": _reason_group_statistics(data),
        }

    status = data.get("status", pd.Series("", index=data.index)).fillna("").astype(str)
    closed = status.eq("已结束")
    current_return = pd.to_numeric(data.get("current_return_pct", _empty_series(data)), errors="coerce")
    realized_return = pd.to_numeric(data.get("profit_loss", _empty_series(data)), errors="coerce")
    data["_return_pct"] = realized_return.where(closed & realized_return.notna(), current_return)
    returns = data["_return_pct"].dropna()
    holding_days = pd.to_numeric(data.get("holding_days", _empty_series(data)), errors="coerce").dropna()
    highest_returns = pd.to_numeric(data.get("highest_return_pct", _empty_series(data)), errors="coerce").dropna()
    drawdowns = pd.to_numeric(data.get("max_drawdown_pct", _empty_series(data)), errors="coerce").dropna()

    overview = {
        "current_tracking_count": int((~closed).sum()),
        "completed_tracking_count": int(closed.sum()),
        "profit_count": int((returns > 0).sum()),
        "loss_count": int((returns < 0).sum()),
        "win_rate": float((returns > 0).mean() * 100) if not returns.empty else None,
        "average_return_pct": float(returns.mean()) if not returns.empty else None,
        "average_holding_days": float(holding_days.mean()) if not holding_days.empty else None,
        "max_return_pct": float(highest_returns.max()) if not highest_returns.empty else None,
        "max_drawdown_pct": float(drawdowns.min()) if not drawdowns.empty else None,
        "max_profit_pct": float(returns.max()) if not returns.empty else None,
        "max_loss_pct": float(returns.min()) if not returns.empty else None,
    }
    score = pd.to_numeric(data.get("entry_final_trade_score", _empty_series(data)), errors="coerce")
    turnover = pd.to_numeric(data.get("entry_turnover", _empty_series(data)), errors="coerce")
    volume_ratio = pd.to_numeric(data.get("entry_volume_ratio", _empty_series(data)), errors="coerce")
    return {
        "overview": overview,
        "return_distribution": _return_distribution(data),
        "score_groups": _group_statistics(data, score, _score_group_rules()),
        "turnover_groups": _group_statistics(data, turnover, _turnover_group_rules()),
        "volume_ratio_groups": _group_statistics(data, volume_ratio, _volume_ratio_group_rules()),
        "reason_groups": _reason_group_statistics(data),
    }


def _empty_tracking_overview() -> dict[str, Any]:
    return {
        "current_tracking_count": 0,
        "completed_tracking_count": 0,
        "profit_count": 0,
        "loss_count": 0,
        "win_rate": None,
        "average_return_pct": None,
        "average_holding_days": None,
        "max_return_pct": None,
        "max_drawdown_pct": None,
        "max_profit_pct": None,
        "max_loss_pct": None,
    }


def _return_distribution(data: pd.DataFrame) -> list[dict[str, Any]]:
    returns = pd.to_numeric(data.get("_return_pct", _empty_series(data)), errors="coerce")
    rules = [
        ("盈利 0–5%", (returns > 0) & (returns <= 5)),
        ("盈利 5–10%", (returns > 5) & (returns < 10)),
        ("盈利 10%以上", returns >= 10),
        ("亏损 0~-5%", (returns < 0) & (returns >= -5)),
        ("亏损 -5%~-10%", (returns < -5) & (returns >= -10)),
        ("亏损 <-10%", returns < -10),
    ]
    return [{"收益区间": label, "股票数量": int(mask.sum())} for label, mask in rules]


def _score_group_rules() -> list[tuple[str, Any]]:
    return [
        ("90分以上", lambda value: value >= 90),
        ("85–90分", lambda value: (value >= 85) & (value < 90)),
        ("80–85分", lambda value: (value >= 80) & (value < 85)),
    ]


def _turnover_group_rules() -> list[tuple[str, Any]]:
    return [
        ("5%–10%", lambda value: (value >= 5) & (value < 10)),
        ("10%–20%", lambda value: (value >= 10) & (value < 20)),
        ("20%以上", lambda value: value >= 20),
    ]


def _volume_ratio_group_rules() -> list[tuple[str, Any]]:
    return [
        ("<1", lambda value: value < 1),
        ("1–2", lambda value: (value >= 1) & (value <= 2)),
        (">2", lambda value: value > 2),
    ]


def _group_statistics(
    data: pd.DataFrame,
    values: pd.Series,
    rules: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    returns = pd.to_numeric(data.get("_return_pct", _empty_series(data)), errors="coerce")
    rows: list[dict[str, Any]] = []
    for label, rule in rules:
        mask = rule(values).fillna(False)
        subset_returns = returns[mask].dropna()
        rows.append(
            {
                "分组": label,
                "股票数量": int(mask.sum()),
                "胜率": float((subset_returns > 0).mean() * 100) if not subset_returns.empty else None,
                "平均收益率": float(subset_returns.mean()) if not subset_returns.empty else None,
            }
        )
    return rows


def _reason_group_statistics(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize immutable entry labels without deriving a new score."""
    entry_reason = data.get("entry_reason", pd.Series("", index=data.index)).fillna("").astype(str)
    snapshots = data.get("entry_snapshot", pd.Series("", index=data.index)).map(_parse_entry_snapshot)
    money_strength = snapshots.map(lambda value: _text(value.get("money_strength")))
    sector_rank = pd.to_numeric(snapshots.map(lambda value: value.get("sector_rank")), errors="coerce")
    sector_strength = snapshots.map(lambda value: _text(value.get("sector_strength")))
    turnover = pd.to_numeric(data.get("entry_turnover", _empty_series(data)), errors="coerce")
    volume_ratio = pd.to_numeric(data.get("entry_volume_ratio", _empty_series(data)), errors="coerce")
    rules = [
        ("放量", entry_reason.str.contains("放量", na=False) | money_strength.str.contains("放量", na=False)),
        (
            "热点板块",
            entry_reason.str.contains("热点", na=False)
            | (sector_rank <= 20)
            | sector_strength.str.contains("热点|强势", na=False, regex=True),
        ),
        ("高换手", turnover >= 20),
        ("高量比", volume_ratio > 2),
    ]
    returns = pd.to_numeric(data.get("_return_pct", _empty_series(data)), errors="coerce")
    rows: list[dict[str, Any]] = []
    for label, mask in rules:
        selected = returns[mask.fillna(False)].dropna()
        rows.append(
            {
                "标签": label,
                "股票数量": int(mask.fillna(False).sum()),
                "胜率": float((selected > 0).mean() * 100) if not selected.empty else None,
                "平均收益率": float(selected.mean()) if not selected.empty else None,
            }
        )
    return rows


def _empty_series(data: pd.DataFrame) -> pd.Series:
    return pd.Series(index=data.index, dtype="float64")


def _latest_rows_by_code(snapshot: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if snapshot is None or snapshot.empty or "code" not in snapshot.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for _, row in snapshot.iterrows():
        code = _normalize_code(row.get("code"))
        if code and code not in rows:
            rows[code] = row.to_dict()
    return rows


def _history_row(
    tracked: pd.Series,
    timestamp: datetime,
    price: float,
    pct_chg: float = float("nan"),
) -> dict[str, Any]:
    return {
        "tracking_id": _text(tracked.get("tracking_id")),
        "code": _normalize_code(tracked.get("code")),
        "snapshot_date": timestamp.date().isoformat(),
        "snapshot_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "name": _text(tracked.get("name")),
        "entry_price": _number(tracked, "entry_price"),
        "price": price if pd.notna(price) and price > 0 else pd.NA,
        "pct_chg": pct_chg if pd.notna(pct_chg) else pd.NA,
        "cumulative_return_pct": _number(tracked, "current_return_pct"),
        "entry_final_trade_score": _number(tracked, "entry_final_trade_score"),
        "highest_return_pct": _number(tracked, "highest_return_pct"),
        "max_drawdown_pct": _number(tracked, "max_drawdown_pct"),
        "current_final_trade_score": _number(tracked, "current_final_trade_score"),
        "status": _text(tracked.get("status")),
    }


def _upsert_history_rows(rows: list[dict[str, Any]], path: Path = STRATEGY_TRACKING_HISTORY_PATH) -> None:
    if not rows:
        return
    history = _load_strategy_tracking_history(path)
    additions = pd.DataFrame(rows)
    for column in STRATEGY_TRACKING_HISTORY_COLUMNS:
        if column not in additions.columns:
            additions[column] = pd.NA
    combined = pd.concat([history, additions[STRATEGY_TRACKING_HISTORY_COLUMNS]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["tracking_id", "snapshot_date"], keep="last")
    combined = combined.sort_values(["snapshot_date", "tracking_id"], kind="stable").reset_index(drop=True)
    _write_csv(combined[STRATEGY_TRACKING_HISTORY_COLUMNS], path)


def _load_strategy_tracking_history(path: Path) -> pd.DataFrame:
    ensure_strategy_tracking_history_file(path)
    try:
        history = pd.read_csv(path, dtype={"tracking_id": "string", "code": "string"})
    except Exception:
        history = pd.DataFrame(columns=STRATEGY_TRACKING_HISTORY_COLUMNS)
    for column in STRATEGY_TRACKING_HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA
    history["code"] = history["code"].map(_normalize_code).astype("string")
    history["tracking_id"] = history["tracking_id"].astype("string")
    return history[STRATEGY_TRACKING_HISTORY_COLUMNS]


def _holding_days(added_date: Any, timestamp: datetime) -> int:
    start = pd.to_datetime(added_date, errors="coerce")
    if pd.isna(start):
        return 1
    start_date = start.date()
    end_date = timestamp.date()
    if end_date < start_date:
        return 1
    local_trade_dates = _local_trade_dates()
    if not local_trade_dates:
        return max(1, len(pd.bdate_range(start_date, end_date)))

    cache_start, cache_end = local_trade_dates[0], local_trade_dates[-1]
    count = sum(start_date <= value <= end_date for value in local_trade_dates)
    # The local cache does not include a still-open session.  Only the
    # uncovered edge is approximated with business days; cached dates retain
    # the project's actual A-share trading calendar.
    if start_date < cache_start:
        count += len(pd.bdate_range(start_date, min(end_date, cache_start - timedelta(days=1))))
    if end_date > cache_end:
        count += len(pd.bdate_range(max(start_date, cache_end + timedelta(days=1)), end_date))
    return max(1, count)


@lru_cache(maxsize=1)
def _local_trade_dates() -> tuple[Any, ...]:
    if not LOCAL_DAILY_HISTORY_DIR.exists():
        return ()
    dates = []
    for path in LOCAL_DAILY_HISTORY_DIR.glob("daily_*.csv"):
        stamp = path.stem.removeprefix("daily_")
        parsed = pd.to_datetime(stamp, format="%Y%m%d", errors="coerce")
        if pd.notna(parsed):
            dates.append(parsed.date())
    return tuple(sorted(set(dates)))


def _automatic_tracking_status(current_return_pct: float, holding_days: Any) -> str:
    """Return a display-only monitoring status; it never closes a record."""
    return_pct = pd.to_numeric(current_return_pct, errors="coerce")
    days = pd.to_numeric(holding_days, errors="coerce")
    if pd.notna(return_pct) and return_pct >= 25:
        return "趋势强势"
    if pd.notna(return_pct) and return_pct >= 15:
        return "大幅盈利关注"
    if pd.notna(return_pct) and return_pct <= -12:
        return "严重风险"
    if pd.notna(return_pct) and return_pct <= -8:
        return "风险警告"
    if pd.notna(days) and days >= 10:
        return "时间观察"
    return "持有观察"


def _build_entry_snapshot(
    values: Mapping[str, Any],
    *,
    code: str,
    name: str,
    entry_price: float,
    turnover: float,
    volume_ratio: float,
    final_trade_score: float,
) -> dict[str, Any]:
    """Capture immutable, visible input values when a stock enters tracking."""
    money_strength = _first_text(
        values,
        "money_strength",
        "money_strength_display",
        "money_strength_level",
        "资金强度",
        "资金强度显示",
    )
    if not money_strength and pd.notna(volume_ratio) and volume_ratio >= 1.5:
        money_strength = "放量"
    return {
        "code": code,
        "name": name,
        "price": _json_number(entry_price),
        "pct_chg": _json_number(_number(values, "pct_chg", "涨跌幅")),
        "amount": _json_number(_number(values, "amount", "成交额")),
        "turnover": _json_number(turnover),
        "volume_ratio": _json_number(volume_ratio),
        "base_score": _json_number(_number(values, "base_score")),
        "trading_quality_score": _json_number(_number(values, "trading_quality_score")),
        "final_trade_score": _json_number(final_trade_score),
        "risk_level": _first_text(
            values,
            "trade_risk_level",
            "position_risk_level",
            "risk_summary",
            "t1_risk_level",
            "risk_level",
            "风险等级",
        ),
        "buy_status": _first_text(
            values,
            "buy_status_display",
            "buy_point_status_display",
            "action_state_display",
            "buy_status",
            "买入状态",
        ),
        "sector": _first_text(values, "sector_name", "sector", "industry", "industry_name", "行业"),
        "sector_rank": _json_number(_number(values, "sector_rank")),
        "sector_strength": _first_text(
            values,
            "sector_strength_level",
            "sector_strength",
            "sector_strength_display",
        ),
        "money_strength": money_strength,
    }


def _build_entry_reason(snapshot: Mapping[str, Any]) -> str:
    """Provide a concise, stable explanation derived from the saved entry snapshot."""
    score = snapshot.get("final_trade_score")
    turnover = snapshot.get("turnover")
    volume_ratio = snapshot.get("volume_ratio")
    score_text = "--" if score is None else f"{float(score):.0f}" if float(score).is_integer() else f"{float(score):.2f}"
    turnover_text = "--" if turnover is None else f"{float(turnover):.2f}%"
    volume_text = "--" if volume_ratio is None else f"{float(volume_ratio):.2f}"
    sector = _text(snapshot.get("sector")) or "板块未知"
    money_strength = _text(snapshot.get("money_strength")) or "资金强度未知"
    return f"评分{score_text} | 换手{turnover_text} | 量比{volume_text} | {sector} | {money_strength}"


def _parse_entry_snapshot(value: Any) -> dict[str, Any]:
    text = _text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_number(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


def _ensure_tracking_id(row: pd.Series) -> str:
    existing = _text(row.get("tracking_id"))
    if existing:
        return existing
    code = _normalize_code(row.get("code")) or "unknown"
    added_at = _text(row.get("added_at")).replace("-", "").replace(":", "").replace(" ", "")
    return f"legacy-{code}-{added_at or row.name}"


def _new_tracking_id(code: str, timestamp: datetime) -> str:
    return f"{code}-{timestamp.strftime('%Y%m%d%H%M%S%f')}"


def _normalize_code(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6)


def _number(values: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = values.get(key)
        if value is None or pd.isna(value):
            continue
        number = pd.to_numeric(value, errors="coerce")
        if pd.notna(number):
            return float(number)
    return float("nan")


def _first_text(values: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        text = _text(values.get(key))
        if text:
            return text
    return ""


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _is_closed(values: Mapping[str, Any]) -> bool:
    return _text(values.get("status")) == "已结束"


def _write_pool(data: pd.DataFrame, path: Path) -> None:
    _write_csv(data, path)


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    data.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)
