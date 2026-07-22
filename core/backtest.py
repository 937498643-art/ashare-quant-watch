"""Historical, read-only V5.3 score backtesting helpers.

Scores are calculated from information available at each signal-day close.
Rows missing required historical scoring fields are explicitly classified as data
insufficient and are never assigned a partial score.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.scoring import score_stock


HOLDING_DAYS = (3, 5, 10)
SCORE_BINS = (
    ("score_ge_90", "评分 >= 90", 90, None),
    ("score_85_90", "评分 85–90", 85, 90),
    ("score_80_85", "评分 80–85", 80, 85),
)
MIN_LOOKBACK_DAYS = 20
# The extended cache is intentionally built for the 300-stock backtest
# universe.  Requiring a full-market file here would discard the entire
# extended period; 250 keeps a small tolerance for listings/suspensions while
# still rejecting a materially incomplete daily cache.  The report labels the
# resulting sector ranking as a cached-universe approximation.
MIN_SECTOR_UNIVERSE_SIZE = 250
REQUIRED_HISTORICAL_FIELDS = ("turnover", "circ_mv", "sector_rank")


def load_local_historical_enrichment(
    history_dir: str | Path,
    daily_basic_dir: str | Path,
    daily_cache_dir: str | Path,
    industry_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load V4 historical inputs exclusively from local cache files.

    The backtest must be reproducible and must never fetch Tushare data.  The
    separate ``download_daily_basic_history.py`` script is responsible for
    preparing daily_basic files in advance.
    """
    dates = _history_dates(history_dir)
    basic_root = Path(daily_basic_dir)
    daily_root = Path(daily_cache_dir)
    industry_root = Path(industry_dir)
    status: dict[str, Any] = {
        "requested_trade_dates": len(dates),
        "daily_basic_cache_hits": 0,
        "daily_basic_missing_dates": 0,
        "daily_cache_hits": 0,
        "daily_missing_dates": 0,
        "daily_partial_dates": 0,
        "industry_cache_hits": 0,
        "industry_missing_dates": 0,
        "network_requests": 0,
        "errors": [],
    }
    if not dates:
        status["errors"].append("本地历史K线中没有可用交易日期。")
        return pd.DataFrame(), status

    missing_basic_dates: list[str] = []
    missing_daily_dates: list[str] = []
    partial_daily_dates: list[str] = []
    missing_industry_dates: list[str] = []
    rows: list[pd.DataFrame] = []
    for trade_date in dates:
        basic = _normalize_daily_basic(_read_csv(basic_root / f"daily_basic_{trade_date}.csv"))
        daily = _normalize_daily(_read_csv(daily_root / f"daily_{trade_date}.csv"))
        industries = _normalize_stock_basic(_read_csv(industry_root / f"industry_{trade_date}.csv"))
        if basic.empty:
            missing_basic_dates.append(trade_date)
        if daily.empty:
            missing_daily_dates.append(trade_date)
        elif len(daily) < MIN_SECTOR_UNIVERSE_SIZE:
            partial_daily_dates.append(trade_date)
        if industries.empty:
            missing_industry_dates.append(trade_date)
        if basic.empty:
            continue
        status["daily_basic_cache_hits"] += 1
        if not daily.empty:
            status["daily_cache_hits"] += 1
        if not industries.empty:
            status["industry_cache_hits"] += 1
        if daily.empty:
            merged = basic.assign(amount=pd.NA, volume=pd.NA, sector_rank=pd.NA)
        else:
            merged = basic.merge(daily[["code", "date", "amount", "volume", "pct_chg"]], on=["code", "date"], how="left")
            if industries.empty or len(daily) < MIN_SECTOR_UNIVERSE_SIZE:
                merged["sector_rank"] = pd.NA
            else:
                sector_input = daily.merge(industries, on="code", how="left")
                sector_rank = _build_sector_rank(sector_input)
                merged = merged.merge(sector_rank, on=["code", "date"], how="left")
        rows.append(merged[["code", "date", "amount", "volume", "turnover", "circ_mv", "sector_rank"]])

    status["daily_basic_missing_dates"] = len(missing_basic_dates)
    status["daily_missing_dates"] = len(missing_daily_dates)
    status["daily_partial_dates"] = len(partial_daily_dates)
    status["industry_missing_dates"] = len(missing_industry_dates)
    if missing_basic_dates:
        status["errors"].append(_missing_cache_message("daily_basic", basic_root, missing_basic_dates))
    if missing_daily_dates:
        status["errors"].append(_missing_cache_message("daily", daily_root, missing_daily_dates))
    if partial_daily_dates:
        status["errors"].append(
            _missing_cache_message("daily（行业排名样本不足）", daily_root, partial_daily_dates)
        )
    if missing_industry_dates:
        status["errors"].append(_missing_cache_message("industry", industry_root, missing_industry_dates))

    if not rows:
        return pd.DataFrame(), status
    data = pd.concat(rows, ignore_index=True)
    data = data.drop_duplicates(["code", "date"], keep="last")
    data = data.sort_values(["code", "date"]).reset_index(drop=True)
    return data, status


def load_local_historical_market(market_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load point-in-time market cache fields without performing network I/O."""
    root = Path(market_dir)
    rows: list[pd.DataFrame] = []
    status: dict[str, Any] = {
        "market_cache_hits": 0,
        "market_missing_dates": 0,
        "rows_with_index": 0,
        "rows_with_amount_ma20": 0,
        "rows_with_breadth": 0,
        "network_requests": 0,
        "errors": [],
    }
    for path in sorted(root.glob("market_*.csv")):
        raw = _read_csv(path)
        if raw.empty:
            continue
        data = pd.DataFrame({
            "date": _parse_trade_dates(raw.get("trade_date")),
            "sh_close": pd.to_numeric(raw.get("sh_close"), errors="coerce"),
            "sh_pct_chg": pd.to_numeric(raw.get("sh_pct_chg"), errors="coerce"),
            "sh_ma5": pd.to_numeric(raw.get("sh_ma5"), errors="coerce"),
            "sh_ma10": pd.to_numeric(raw.get("sh_ma10"), errors="coerce"),
            "up_count": pd.to_numeric(raw.get("up_count"), errors="coerce"),
            "down_count": pd.to_numeric(raw.get("down_count"), errors="coerce"),
            "market_amount": pd.to_numeric(raw.get("market_amount"), errors="coerce"),
        }).dropna(subset=["date"]).drop_duplicates("date", keep="last")
        if not data.empty:
            rows.append(data)
            status["market_cache_hits"] += 1
    if not rows:
        status["errors"].append(f"本地市场环境缓存为空：{root}")
        return pd.DataFrame(columns=["date"]), status
    market = pd.concat(rows, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    # Current-day turnover is compared with the strictly prior 20 complete
    # market-amount observations, so partial-cache days neither leak future
    # data nor break the rolling reference for later complete days.
    market["market_amount_ma20"] = pd.NA
    valid_amount = market["market_amount"].dropna()
    market.loc[valid_amount.index, "market_amount_ma20"] = (
        valid_amount.shift(1).rolling(20, min_periods=20).mean()
    )
    status["rows_with_index"] = int(market[["sh_close", "sh_pct_chg", "sh_ma5", "sh_ma10"]].notna().all(axis=1).sum())
    status["rows_with_amount_ma20"] = int(market["market_amount_ma20"].notna().sum())
    status["rows_with_breadth"] = int((market["up_count"].notna() & market["down_count"].notna()).sum())
    return market, status


def run_score_backtest(
    history_dir: str | Path,
    enrichment: pd.DataFrame | None,
    holding_days: Iterable[int] = HOLDING_DAYS,
    min_lookback_days: int = MIN_LOOKBACK_DAYS,
    market: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run point-in-time V5.3 score backtest using complete historical fields.

    Signals are entered at the signal-day close and exited at the close of the
    3rd, 5th, or 10th subsequent trading day. Incomplete rows are excluded
    before score calculation instead of being assigned a partial score.
    """
    root = Path(history_dir)
    horizons = tuple(sorted({int(day) for day in holding_days if int(day) > 0}))
    if not horizons:
        raise ValueError("holding_days must contain at least one positive integer")
    lookback = max(int(min_lookback_days), MIN_LOOKBACK_DAYS)
    max_horizon = max(horizons)
    enrichment_index = _enrichment_index(enrichment)
    market_index = _market_index(market)
    available_codes = set(enrichment_index["code"].tolist()) if not enrichment_index.empty else None
    history_frames = _load_history_frames_from_daily_cache(root, available_codes)

    trades: list[dict[str, Any]] = []
    coverage = {
        "history_files": len(history_frames),
        "usable_files": 0,
        "skipped_files": 0,
        "candidate_daily_rows": 0,
        "scored_daily_rows": 0,
        "insufficient_daily_rows": 0,
        "rows_with_turnover": 0,
        "rows_with_float_market_cap": 0,
        "rows_with_sector_rank": 0,
        "rows_with_consecutive_count": 0,
        "rows_with_limit_up_count": 0,
        "insufficient_reasons": {},
    }
    all_scores: list[int] = []
    date_values: list[pd.Timestamp] = []

    for code, history in history_frames.items():
        history = _merge_history_enrichment(history, code, enrichment_index)
        if history.empty or len(history) <= lookback + max_horizon:
            coverage["skipped_files"] += 1
            continue
        coverage["usable_files"] += 1
        date_values.extend(history["date"].dropna().tolist())

        for index in range(lookback, len(history) - max_horizon):
            coverage["candidate_daily_rows"] += 1
            score_row = build_daily_score_row(history, index, code)
            score_row.update(market_index.get(pd.Timestamp(score_row["date"]), {}))
            _count_available_v3_fields(score_row, coverage)
            missing = _required_missing_fields(score_row)
            if missing:
                coverage["insufficient_daily_rows"] += 1
                for field in missing:
                    coverage["insufficient_reasons"][field] = coverage["insufficient_reasons"].get(field, 0) + 1
                continue

            coverage["scored_daily_rows"] += 1
            score_result = score_stock(score_row)
            score = int(score_result["score"])
            all_scores.append(score)
            bin_key = score_bin_key(score)
            if bin_key is None:
                continue

            entry_price = float(score_row["price"])
            for days in horizons:
                exit_row = history.iloc[index + days]
                exit_price = float(exit_row["close"])
                trades.append(
                    {
                        "code": code,
                        "signal_date": score_row["date"].strftime("%Y-%m-%d"),
                        "entry_close": entry_price,
                        "holding_days": days,
                        "exit_date": exit_row["date"].strftime("%Y-%m-%d"),
                        "exit_close": exit_price,
                        "return_pct": round((exit_price / entry_price - 1) * 100, 6),
                        "score": score,
                        "score_bin": bin_key,
                        "score_breakdown": score_result["score_breakdown"],
                    }
                )

    trade_frame = pd.DataFrame(trades, columns=_trade_columns())
    report = build_backtest_report(trade_frame, coverage, all_scores, date_values, horizons, lookback)
    return trade_frame, report


def _load_history_frames_from_daily_cache(
    daily_dir: Path,
    allowed_codes: set[str] | None,
) -> dict[str, pd.DataFrame]:
    """Build per-stock history frames solely from data/history/daily files."""
    parts: list[pd.DataFrame] = []
    for path in sorted(daily_dir.glob("daily_*.csv")):
        daily = _normalize_daily(_read_csv(path))
        if daily.empty:
            continue
        if allowed_codes is not None:
            daily = daily[daily["code"].isin(allowed_codes)]
        if not daily.empty:
            parts.append(daily[["code", "date", "high", "close", "volume", "amount"]])
    if not parts:
        return {}
    merged = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    return {
        str(code): group.drop(columns="code").sort_values("date").reset_index(drop=True)
        for code, group in merged.groupby("code", sort=True)
    }


def load_history_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize one locally cached daily K-line file."""
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    aliases = {
        "date": ("date", "日期", "trade_date"),
        "open": ("open", "开盘"),
        "close": ("close", "收盘"),
        "high": ("high", "最高"),
        "low": ("low", "最低"),
        "volume": ("volume", "vol", "成交量"),
        "amount": ("amount", "成交额"),
        "turnover": ("turnover", "换手率"),
        "float_market_cap": ("float_market_cap", "流通市值", "circulating_market_cap"),
        "circ_mv": ("circ_mv",),
        "sector_rank": ("sector_rank", "板块涨幅排名"),
        "consecutive_count": ("consecutive_count", "consecutive_selection_days", "连续入选天数"),
    }
    normalized = pd.DataFrame(index=raw.index)
    for target, names in aliases.items():
        source = _first_column(raw, names)
        normalized[target] = raw[source] if source else pd.NA
    normalized["date"] = _parse_trade_dates(normalized["date"])
    for column in normalized.columns:
        if column != "date":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "close"])
    normalized = normalized[normalized["close"] > 0]
    return normalized.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def build_daily_score_row(history: pd.DataFrame, index: int, code: str) -> dict[str, Any]:
    """Build one point-in-time V4 input row from daily bars through index."""
    current = history.iloc[index]
    closes = history.loc[:index, "close"]
    highs = history.loc[:index, "high"]
    volumes = history.loc[:index, "volume"]
    amounts = history.loc[:index, "amount"]
    prior_close = _number(history.loc[index - 1, "close"]) if index > 0 else float("nan")
    close = _number(current["close"])
    pct_chg = (close / prior_close - 1) * 100 if pd.notna(prior_close) and prior_close > 0 else float("nan")
    daily_returns = closes.pct_change(fill_method=None).mul(100)
    up_days_5d = int((daily_returns.tail(5) > 0).sum())
    limit_count = int((daily_returns.tail(20) >= _limit_up_threshold(code)).sum())
    base_close = _number(closes.iloc[-21]) if len(closes) >= 21 else float("nan")
    pct_chg_20d = (close / base_close - 1) * 100 if pd.notna(base_close) and base_close > 0 else float("nan")
    base_close_5d = _number(closes.iloc[-6]) if len(closes) >= 6 else float("nan")
    pct_chg_5d = (close / base_close_5d - 1) * 100 if pd.notna(base_close_5d) and base_close_5d > 0 else float("nan")
    high_20d = _number(highs.tail(20).max())
    prior_20d_high = _number(highs.iloc[-21:-1].max()) if len(highs) >= 21 else float("nan")
    recent_low_volume_pullback = _has_recent_low_volume_pullback(history, index)
    consecutive_up_days = _consecutive_positive_days(daily_returns)
    return {
        "code": code,
        "date": current["date"],
        "price": close,
        "pct_chg": pct_chg,
        "volume": _number(current["volume"]),
        "amount": _number(current["amount"]),
        "turnover": _number(current.get("turnover", pd.NA)),
        "float_market_cap": _number(current.get("float_market_cap", pd.NA)),
        "circ_mv": _number(current.get("circ_mv", pd.NA)),
        "ma5": _mean_tail(closes, 5),
        "ma10": _mean_tail(closes, 10),
        "ma20": _mean_tail(closes, 20),
        "avg_volume_5d": _mean_before(volumes, index, 5),
        "avg_volume_10d": _mean_before(volumes, index, 10),
        "avg_amount_5d": _mean_before(amounts, index, 5),
        "pct_chg_20d": pct_chg_20d,
        "pct_chg_5d": pct_chg_5d,
        "up_days_5d": up_days_5d,
        "high_20d": high_20d,
        "prior_20d_high": prior_20d_high,
        "recent_low_volume_pullback": recent_low_volume_pullback,
        "consecutive_up_days": consecutive_up_days,
        "limit_up_count_20d": limit_count,
        "sector_rank": _number(current.get("sector_rank", pd.NA)),
        # The local historical source has no watchlist history. Zero means no
        # additional continuity bonus; it is not used as a risk penalty.
        "consecutive_count": _number(current.get("consecutive_count", pd.NA)) if pd.notna(_number(current.get("consecutive_count", pd.NA))) else 0.0,
    }


def _has_recent_low_volume_pullback(history: pd.DataFrame, index: int) -> bool:
    """Whether one of the prior three sessions was a lower-volume pullback."""
    start = max(1, index - 3)
    for day in range(start, index):
        volume = _number(history.loc[day, "volume"])
        avg_volume = _mean_before(history.loc[:day, "volume"], day, 5)
        close = _number(history.loc[day, "close"])
        prior_close = _number(history.loc[day - 1, "close"])
        if (
            pd.notna(volume)
            and pd.notna(avg_volume)
            and volume < avg_volume
            and pd.notna(close)
            and pd.notna(prior_close)
            and close < prior_close
        ):
            return True
    return False


def _consecutive_positive_days(daily_returns: pd.Series) -> int:
    """Count consecutive positive close-to-close sessions ending today."""
    count = 0
    for value in reversed(daily_returns.dropna().tolist()):
        if value > 0:
            count += 1
        else:
            break
    return count


def score_bin_key(score: int | float) -> str | None:
    """Return the requested high-score interval key for one score."""
    value = float(score)
    if value >= 90:
        return "score_ge_90"
    if value >= 85:
        return "score_85_90"
    if value >= 80:
        return "score_80_85"
    return None


def build_backtest_report(
    trades: pd.DataFrame,
    coverage: dict[str, Any],
    all_scores: list[int],
    date_values: list[pd.Timestamp],
    horizons: tuple[int, ...],
    lookback: int,
) -> dict[str, Any]:
    """Aggregate return statistics for every requested score bin and horizon."""
    score_distribution = {
        "score_ge_90": int(sum(score >= 90 for score in all_scores)),
        "score_85_90": int(sum(85 <= score < 90 for score in all_scores)),
        "score_80_85": int(sum(80 <= score < 85 for score in all_scores)),
        "score_below_80": int(sum(score < 80 for score in all_scores)),
    }
    buckets: list[dict[str, Any]] = []
    for key, label, _lower, _upper in SCORE_BINS:
        for days in horizons:
            subset = trades[(trades["score_bin"] == key) & (trades["holding_days"] == days)] if not trades.empty else trades
            buckets.append(_return_statistics(subset, key, label, days))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "当日收盘买入，分别于第3、第5、第10个后续交易日收盘卖出；未计交易成本和滑点。",
        "lookback_days": lookback,
        "holding_days": list(horizons),
        "date_range": {
            "start": min(date_values).strftime("%Y-%m-%d") if date_values else None,
            "end": max(date_values).strftime("%Y-%m-%d") if date_values else None,
        },
        "coverage": coverage,
        "score_distribution": score_distribution,
        "trade_count": int(len(trades)),
        "statistics": buckets,
        "limitations": [
            "评分输入只使用信号日及此前数据；后续价格仅用于计算持有期收益。",
            "缺少历史换手率、流通市值或当天行业排名的行会标记为数据不足并跳过，不参与评分。",
            "涨停活跃度由截至信号日的20根日K线计算；创业板、科创板按20%阈值，北交所按30%阈值。",
            "行业热度为同一行业股票当天涨跌幅均值的排名；行业映射只读取 data/history/industry 日文件。",
            "扩展区间的行业排名基于每日缓存的回测股票池（约300只），不是全市场行业排名；板块热度结果应按样本池近似解读。",
            "市场环境只读取 data/history/market 本地缓存；缺失市场字段不会扣分，也不会计入归一化基数。",
            "历史连续观察记录目前不可得，按0分处理，不构成风险扣分。",
            "回测结果不构成投资建议，未计交易成本、滑点、涨跌停无法成交或复权影响。",
        ],
    }


def write_backtest_report(trades: pd.DataFrame, report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write Markdown, JSON summary, and detailed simulated trades."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "score_v5_3_backtest_report.md"
    json_path = directory / "score_v5_3_backtest_summary.json"
    trades_path = directory / "score_v5_3_backtest_trades.csv"
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    return {"markdown": markdown_path, "json": json_path, "trades": trades_path}


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a compact, inspectable Markdown report."""
    coverage = report["coverage"]
    insufficient = coverage.get("insufficient_reasons", {})
    lines = [
        "# V5.3 历史评分回测报告", "",
        f"生成时间：{report['generated_at']}", "",
        f"回测口径：{report['method']}", "",
        f"K线日期范围：{report['date_range']['start'] or '--'} 至 {report['date_range']['end'] or '--'}", "",
        "## 数据覆盖", "",
        f"- 历史文件：{coverage['history_files']}；可用文件：{coverage['usable_files']}；跳过文件：{coverage['skipped_files']}。",
        f"- 候选日评分行：{coverage['candidate_daily_rows']}；完整字段评分行：{coverage['scored_daily_rows']}；数据不足跳过：{coverage['insufficient_daily_rows']}。",
        f"- 含历史换手率：{coverage['rows_with_turnover']}；含历史流通市值：{coverage['rows_with_float_market_cap']}；含行业排名：{coverage['rows_with_sector_rank']}；含20日涨停统计：{coverage['rows_with_limit_up_count']}。", "",
        "### 数据不足原因", "",
    ]
    lines.extend(f"- {key}：{value}" for key, value in sorted(insufficient.items()))
    if not insufficient:
        lines.append("- 无")
    lines.extend([
        "", "## 评分分布", "",
        f"- >=90：{report['score_distribution']['score_ge_90']}",
        f"- 85–90：{report['score_distribution']['score_85_90']}",
        f"- 80–85：{report['score_distribution']['score_80_85']}",
        f"- <80：{report['score_distribution']['score_below_80']}", "",
        "## 收益统计", "",
        "| 评分区间 | 持有天数 | 样本数 | 胜率 | 平均收益率 | 最大盈利 | 最大亏损 | 盈亏比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in report["statistics"]:
        lines.append(
            f"| {item['score_bin_label']} | {item['holding_days']} | {item['sample_count']} | "
            f"{_format_pct(item['win_rate'])} | {_format_pct(item['average_return'])} | "
            f"{_format_pct(item['max_return'])} | {_format_pct(item['max_loss'])} | {_format_ratio(item['profit_loss_ratio'])} |"
        )
    enrichment = report.get("historical_enrichment") or {}
    enrichment_errors = enrichment.get("errors") or []
    if enrichment_errors:
        lines.extend(["", "## 历史字段补齐状态", ""])
        lines.extend(f"- {error}" for error in enrichment_errors)
    lines.extend(["", "## 口径与限制", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _normalize_stock_basic(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "ts_code" not in data.columns:
        return pd.DataFrame(columns=["code", "industry"])
    output = pd.DataFrame({
        "code": data["ts_code"].map(_normalize_code),
        "industry": data.get("industry", pd.Series(pd.NA, index=data.index)).astype("string").str.strip(),
    })
    return output[output["industry"].notna() & output["industry"].ne("")].drop_duplicates("code")


def _normalize_daily_basic(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "ts_code" not in data.columns:
        return pd.DataFrame(columns=["code", "date", "turnover", "circ_mv"])
    output = pd.DataFrame({
        "code": data["ts_code"].map(_normalize_code),
        "date": _parse_trade_dates(data.get("trade_date")),
        "turnover": pd.to_numeric(data.get("turnover_rate"), errors="coerce"),
        "circ_mv": pd.to_numeric(data.get("circ_mv"), errors="coerce"),
    })
    return output.dropna(subset=["code", "date"]).drop_duplicates(["code", "date"], keep="last")


def _normalize_daily(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "ts_code" not in data.columns:
        return pd.DataFrame(columns=["code", "date", "high", "close", "pct_chg", "volume", "amount"])
    output = pd.DataFrame({
        "code": data["ts_code"].map(_normalize_code),
        "date": _parse_trade_dates(data.get("trade_date")),
        "high": pd.to_numeric(data.get("high"), errors="coerce"),
        "close": pd.to_numeric(data.get("close"), errors="coerce"),
        "pct_chg": pd.to_numeric(data.get("pct_chg"), errors="coerce"),
        "volume": pd.to_numeric(data.get("vol"), errors="coerce"),
        # Tushare daily.amount is in 千元; scoring expects yuan.
        "amount": pd.to_numeric(data.get("amount"), errors="coerce") * 1000,
    })
    return output.dropna(subset=["code", "date"]).drop_duplicates(["code", "date"], keep="last")


def _build_sector_rank(daily: pd.DataFrame) -> pd.DataFrame:
    usable = daily.dropna(subset=["date", "industry", "pct_chg"]).copy()
    usable = usable[usable["industry"].astype(str).str.strip().ne("")]
    if usable.empty:
        return pd.DataFrame(columns=["code", "date", "sector_rank"])
    sector = usable.groupby(["date", "industry"], as_index=False)["pct_chg"].mean().rename(columns={"pct_chg": "sector_pct_chg"})
    sector["sector_rank"] = sector.groupby("date")["sector_pct_chg"].rank(method="min", ascending=False)
    return usable[["code", "date", "industry"]].merge(sector[["date", "industry", "sector_rank"]], on=["date", "industry"], how="left")[["code", "date", "sector_rank"]]


def _enrichment_index(enrichment: pd.DataFrame | None) -> pd.DataFrame:
    if enrichment is None or enrichment.empty:
        return pd.DataFrame(columns=["code", "date"])
    data = enrichment.copy()
    data["code"] = data["code"].map(_normalize_code)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data.dropna(subset=["code", "date"]).drop_duplicates(["code", "date"], keep="last")


def _market_index(market: pd.DataFrame | None) -> dict[pd.Timestamp, dict[str, Any]]:
    """Map each signal date to only same-day cached market fields."""
    if market is None or market.empty or "date" not in market.columns:
        return {}
    data = market.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    indexed: dict[pd.Timestamp, dict[str, Any]] = {}
    for row in data.to_dict("records"):
        date = pd.Timestamp(row.pop("date"))
        indexed[date] = row
    return indexed


def _merge_history_enrichment(history: pd.DataFrame, code: str, enrichment: pd.DataFrame) -> pd.DataFrame:
    if history.empty or enrichment.empty:
        return history
    selected = enrichment[enrichment["code"] == _normalize_code(code)].copy()
    if selected.empty:
        return history
    result = history.merge(selected, on="date", how="left", suffixes=("", "_enriched"))
    for column in ("volume", "amount", "turnover", "circ_mv", "sector_rank"):
        if column not in result.columns:
            result[column] = pd.NA
        enriched_column = f"{column}_enriched"
        if enriched_column in result.columns:
            result[column] = pd.to_numeric(result[enriched_column], errors="coerce").combine_first(
                pd.to_numeric(result.get(column), errors="coerce")
            )
            result = result.drop(columns=[enriched_column])
    return result


def _required_missing_fields(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if pd.isna(row["turnover"]):
        missing.append("历史换手率")
    if pd.isna(row["circ_mv"]) and pd.isna(row["float_market_cap"]):
        missing.append("历史流通市值")
    if pd.isna(row["sector_rank"]):
        missing.append("当天行业涨幅排名")
    return missing


def _history_dates(history_dir: str | Path) -> list[str]:
    dates: set[pd.Timestamp] = set()
    for path in Path(history_dir).glob("*.csv"):
        frame = load_history_csv(path)
        dates.update(frame.get("date", pd.Series(dtype="datetime64[ns]")).dropna().tolist())
    return [date.strftime("%Y%m%d") for date in sorted(dates)]


def _missing_cache_message(kind: str, directory: Path, dates: list[str]) -> str:
    preview = ", ".join(dates[:5])
    suffix = " …" if len(dates) > 5 else ""
    return f"本地 {kind} 缓存缺少 {len(dates)} 个交易日：{preview}{suffix}（目录：{directory}）。"


def _return_statistics(data: pd.DataFrame, key: str, label: str, days: int) -> dict[str, Any]:
    if data.empty:
        return {"score_bin": key, "score_bin_label": label, "holding_days": days, "sample_count": 0, "win_rate": None, "average_return": None, "max_return": None, "max_loss": None, "profit_loss_ratio": None}
    returns = pd.to_numeric(data["return_pct"], errors="coerce").dropna()
    wins, losses = returns[returns > 0], returns[returns < 0]
    ratio = float(wins.mean() / abs(losses.mean())) if not wins.empty and not losses.empty else None
    return {"score_bin": key, "score_bin_label": label, "holding_days": days, "sample_count": int(len(returns)), "win_rate": float((returns > 0).mean() * 100), "average_return": float(returns.mean()), "max_return": float(returns.max()), "max_loss": float(returns.min()), "profit_loss_ratio": ratio}


def _count_available_v3_fields(row: dict[str, Any], coverage: dict[str, Any]) -> None:
    if pd.notna(row["turnover"]): coverage["rows_with_turnover"] += 1
    if pd.notna(row["float_market_cap"]) or pd.notna(row["circ_mv"]): coverage["rows_with_float_market_cap"] += 1
    if pd.notna(row["sector_rank"]): coverage["rows_with_sector_rank"] += 1
    if pd.notna(row["consecutive_count"]): coverage["rows_with_consecutive_count"] += 1
    if pd.notna(row["limit_up_count_20d"]): coverage["rows_with_limit_up_count"] += 1


def _trade_columns() -> list[str]:
    return ["code", "signal_date", "entry_close", "holding_days", "exit_date", "exit_close", "return_pct", "score", "score_bin", "score_breakdown"]


def _code_from_path(path: Path) -> str:
    return path.name.split("_")[0].zfill(6)


def _normalize_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:6].zfill(6) if digits else ""


def _limit_up_threshold(code: str) -> float:
    if str(code).startswith(("300", "301", "688", "689")): return 19.5
    if str(code).startswith(("4", "8", "9")): return 29.5
    return 9.5


def _mean_tail(values: pd.Series, size: int) -> float:
    valid = pd.to_numeric(values.tail(size), errors="coerce").dropna()
    return float(valid.mean()) if len(valid) >= size else float("nan")


def _mean_before(values: pd.Series, index: int, size: int) -> float:
    valid = pd.to_numeric(values.iloc[max(0, index - size):index], errors="coerce").dropna()
    return float(valid.mean()) if len(valid) >= size else float("nan")


def _number(value: Any) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


def _parse_trade_dates(values: Any) -> pd.Series:
    series = pd.Series(values).astype("string").str.strip().str.replace(".0", "", regex=False)
    compact = series.str.fullmatch(r"\d{8}", na=False)
    output = pd.to_datetime(series.where(~compact), errors="coerce")
    output.loc[compact] = pd.to_datetime(series.loc[compact], format="%Y%m%d", errors="coerce")
    return output


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"ts_code": str, "trade_date": str})
    except Exception:
        return pd.DataFrame()


def _format_pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}%"


def _format_ratio(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"
