#!/usr/bin/env python3
"""Read-only V5.5 full-market backtest with optional trading-quality modules."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest import (  # noqa: E402
    HOLDING_DAYS,
    MIN_LOOKBACK_DAYS,
    _enrichment_index,
    _market_index,
    _normalize_daily,
    _normalize_stock_basic,
    _read_csv,
    _required_missing_fields,
    build_daily_score_row,
    load_local_historical_enrichment,
    load_local_historical_market,
)
from core.scoring import score_stock  # noqa: E402
from core.trading_quality import TradingQualityConfig, evaluate_trading_quality  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5.5 trading-quality backtest from local history only"
    )
    parser.add_argument("--history-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "daily")
    parser.add_argument("--daily-basic-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "daily_basic")
    parser.add_argument("--industry-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "industry")
    parser.add_argument("--market-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "market")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_5")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2026-07-17")
    parser.add_argument("--disable-market-score", action="store_true")
    parser.add_argument("--disable-trend-stage", action="store_true")
    parser.add_argument("--disable-buy-point", action="store_true")
    parser.add_argument("--disable-sector-linkage", action="store_true")
    return parser.parse_args()


def _limit_threshold(code: str) -> float:
    if code.startswith(("4", "8", "92")):
        return 29.5
    if code.startswith(("300", "301", "688", "689")):
        return 19.5
    return 9.5


def _load_histories_with_low(daily_dir: Path, allowed_codes: set[str]) -> dict[str, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for path in sorted(daily_dir.glob("daily_*.csv")):
        raw = _read_csv(path)
        daily = _normalize_daily(raw)
        daily["low"] = pd.to_numeric(raw.get("low"), errors="coerce")
        daily = daily[daily["code"].isin(allowed_codes)]
        if not daily.empty:
            parts.append(daily[["code", "date", "high", "low", "close", "volume", "amount"]])
    if not parts:
        return {}
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    return {
        str(code): group.drop(columns="code").sort_values("date").reset_index(drop=True)
        for code, group in data.groupby("code", sort=True)
    }


def _load_market_quality(market_dir: Path) -> dict[pd.Timestamp, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    for path in sorted(market_dir.glob("market_*.csv")):
        raw = _read_csv(path)
        if raw.empty:
            continue
        date = pd.to_datetime(raw.get("trade_date"), format="%Y%m%d", errors="coerce")
        frame = pd.DataFrame(
            {
                "date": date,
                "sh_close": pd.to_numeric(raw.get("sh_close"), errors="coerce"),
                "up_ratio": pd.to_numeric(raw.get("up_ratio"), errors="coerce"),
                "limit_up_count": pd.to_numeric(raw.get("limit_up_count"), errors="coerce"),
            }
        ).dropna(subset=["date"])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return {}
    market = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    market["sh_ma20"] = market["sh_close"].rolling(20, min_periods=20).mean()
    return {
        pd.Timestamp(row.date): {
            "sh_close": row.sh_close,
            "sh_ma20": row.sh_ma20,
            "up_ratio": row.up_ratio,
            "limit_up_count": row.limit_up_count,
        }
        for row in market.itertuples(index=False)
    }


def _build_industry_metrics(daily_dir: Path, industry_dir: Path) -> dict[str, pd.DataFrame]:
    """Derive full-market industry linkage metrics from local daily files."""
    rows: list[pd.DataFrame] = []
    for daily_path in sorted(daily_dir.glob("daily_*.csv")):
        trade_date = daily_path.stem.removeprefix("daily_")
        industry_path = industry_dir / f"industry_{trade_date}.csv"
        daily = _normalize_daily(_read_csv(daily_path))
        industry = _normalize_stock_basic(_read_csv(industry_path))
        if daily.empty or industry.empty:
            continue
        data = daily.merge(industry[["code", "industry"]], on="code", how="left")
        data = data.dropna(subset=["date", "industry", "pct_chg"]).copy()
        data = data[data["industry"].astype(str).str.strip().ne("")]
        if data.empty:
            continue
        data["is_up"] = data["pct_chg"] > 0
        data["is_down"] = data["pct_chg"] < 0
        data["is_limit_up"] = [
            pct >= _limit_threshold(str(code))
            for code, pct in zip(data["code"], data["pct_chg"], strict=False)
        ]
        stats = (
            data.groupby(["date", "industry"], as_index=False)
            .agg(
                industry_pct_chg=("pct_chg", "mean"),
                industry_up_count=("is_up", "sum"),
                industry_down_count=("is_down", "sum"),
                industry_limit_up_count=("is_limit_up", "sum"),
            )
        )
        stats["industry_count"] = stats.groupby("date")["industry"].transform("count")
        stats["industry_rank"] = stats.groupby("date")["industry_pct_chg"].rank(
            method="min", ascending=False
        )
        directional = stats["industry_up_count"] + stats["industry_down_count"]
        stats["industry_up_ratio"] = stats["industry_up_count"] / directional.where(directional > 0)
        rows.append(
            data[["code", "date", "industry"]].merge(
                stats[
                    [
                        "date",
                        "industry",
                        "industry_rank",
                        "industry_count",
                        "industry_up_ratio",
                        "industry_limit_up_count",
                    ]
                ],
                on=["date", "industry"],
                how="left",
            )
        )
    if not rows:
        return {}
    result = pd.concat(rows, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    return {
        str(code): group.drop(columns="code").sort_values("date").reset_index(drop=True)
        for code, group in result.groupby("code", sort=True)
    }


def _merge_history(
    history: pd.DataFrame,
    enrichment: pd.DataFrame,
    industry_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    merged = history.merge(enrichment, on="date", how="left", suffixes=("", "_enriched"))
    for column in ("volume", "amount"):
        source = f"{column}_enriched"
        if source in merged.columns:
            merged[column] = merged[source].combine_first(merged[column])
            merged = merged.drop(columns=source)
    if industry_metrics is not None and not industry_metrics.empty:
        columns = [
            "date",
            "industry_rank",
            "industry_count",
            "industry_up_ratio",
            "industry_limit_up_count",
        ]
        merged = merged.merge(industry_metrics[columns], on="date", how="left")
    else:
        for column in (
            "industry_rank",
            "industry_count",
            "industry_up_ratio",
            "industry_limit_up_count",
        ):
            merged[column] = pd.NA
    return merged


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stats(data: pd.DataFrame, return_column: str, days: int) -> dict[str, Any]:
    subset = data[data["holding_days"] == days] if not data.empty else data
    return_values = (
        subset[return_column]
        if return_column in subset.columns
        else pd.Series(dtype="float64")
    )
    drawdown_values = (
        subset["max_drawdown_pct"]
        if "max_drawdown_pct" in subset.columns
        else pd.Series(dtype="float64")
    )
    returns = pd.to_numeric(return_values, errors="coerce").dropna()
    drawdowns = pd.to_numeric(drawdown_values, errors="coerce").dropna()
    if returns.empty:
        return {
            "sample_count": 0,
            "average_return_pct": None,
            "win_rate_pct": None,
            "max_profit_pct": None,
            "max_loss_pct": None,
            "profit_loss_ratio": None,
            "max_drawdown_pct": None,
        }
    wins, losses = returns[returns > 0], returns[returns < 0]
    ratio = float(wins.mean() / abs(losses.mean())) if not wins.empty and not losses.empty else None
    return {
        "sample_count": int(len(returns)),
        "average_return_pct": float(returns.mean()),
        "win_rate_pct": float((returns > 0).mean() * 100),
        "max_profit_pct": float(returns.max()),
        "max_loss_pct": float(returns.min()),
        "profit_loss_ratio": ratio,
        "max_drawdown_pct": float(drawdowns.min()) if not drawdowns.empty else None,
    }


def _pct(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}%"


def _ratio(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _write_report(output_dir: Path, report: dict[str, Any], trades: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score_v5_5_backtest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trades.to_csv(output_dir / "score_v5_5_core_pool_trades.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# V5.5 交易质量优化全市场回测报告",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 方法",
        "",
        "- 基础评分：完全沿用 core/scoring.py 的 V5.3 原始分数，未修改任何基础权重。",
        "- V5.5 最终交易分 = 基础分 + 可独立开关的交易质量调整，结果限制在 0–100。",
        "- 质量层只评估基础分>=85的股票；保留 V5.4 核心池条件：最终分>=85、换手率5%–30%、资金攻击全市场前20%、原买点质量>=10。",
        "- 止损模拟：买入后的任一交易日最低价触及入场价 -8% 时，以 -8% 退出；这是日线级近似。",
        "",
        "## 模块开关与数据覆盖",
        "",
        f"- 模块配置：{report['module_config']}",
        f"- 基础完整评分行：{report['coverage']['scored_rows']}；资金攻击排名日期：{report['coverage']['attack_rank_dates']}；network_requests：0。",
        f"- 实际信号日期：{report['signal_range']['start'] or '--'} 至 {report['signal_range']['end'] or '--'}。",
        "",
        "## 85分以上及核心交易池统计",
        "",
        f"- 最终分>=85信号：{report['final_score_ge_85_signals']}。",
        f"- V5.5核心交易池信号：{report['core_pool_signal_count']}。",
        f"- 每日平均信号数：{report['signal_statistics']['daily_average']:.2f}；最多：{report['signal_statistics']['daily_max']}；最少：{report['signal_statistics']['daily_min']}。",
        "",
        "## 收益与止损效果（核心交易池）",
        "",
        "| 持有期 | 无止损平均收益 | 止损后平均收益 | 无止损胜率 | 止损后胜率 | 无止损盈亏比 | 止损后盈亏比 | 无止损最大亏损 | 止损后最大亏损 | 最大回撤 | 止损数/比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for days, item in report["statistics"].items():
        raw = item["raw"]
        stopped = item["stop_loss"]
        lines.append(
            f"| {days}日 | {_pct(raw['average_return_pct'])} | {_pct(stopped['average_return_pct'])} | "
            f"{_pct(raw['win_rate_pct'])} | {_pct(stopped['win_rate_pct'])} | "
            f"{_ratio(raw['profit_loss_ratio'])} | {_ratio(stopped['profit_loss_ratio'])} | "
            f"{_pct(raw['max_loss_pct'])} | {_pct(stopped['max_loss_pct'])} | "
            f"{_pct(raw['max_drawdown_pct'])} | {item['stop_loss_count']}/{_pct(item['stop_loss_rate_pct'])} |"
        )
    comparison = report["comparison"]
    lines.extend([
        "",
        "## V5.4 vs V5.5",
        "",
        "| 模型与口径 | 85分以上/核心池样本 | 10日平均收益 | 胜率 | 盈亏比 | 最大亏损 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| V5.4核心池（无止损） | {comparison['v5_4']['sample_count']} | {_pct(comparison['v5_4']['average_return_pct'])} | {_pct(comparison['v5_4']['win_rate_pct'])} | {_ratio(comparison['v5_4']['profit_loss_ratio'])} | {_pct(comparison['v5_4']['max_loss_pct'])} |",
        f"| V5.5核心池（无止损） | {comparison['v5_5_raw']['sample_count']} | {_pct(comparison['v5_5_raw']['average_return_pct'])} | {_pct(comparison['v5_5_raw']['win_rate_pct'])} | {_ratio(comparison['v5_5_raw']['profit_loss_ratio'])} | {_pct(comparison['v5_5_raw']['max_loss_pct'])} |",
        f"| V5.5核心池（-8%止损） | {comparison['v5_5_stop']['sample_count']} | {_pct(comparison['v5_5_stop']['average_return_pct'])} | {_pct(comparison['v5_5_stop']['win_rate_pct'])} | {_ratio(comparison['v5_5_stop']['profit_loss_ratio'])} | {_pct(comparison['v5_5_stop']['max_loss_pct'])} |",
        "",
        f"- 最大亏损是否较 V5.4 下降（以V5.5止损口径）：{'是' if comparison['max_loss_improved'] else '否'}。",
        "- 未构建组合净值曲线；报告的“最大回撤”为单笔持有期内最低价相对入场价的最大不利波动。",
        "",
    ])
    (output_dir / "score_v5_5_backtest_report.md").write_text("\n".join(lines), encoding="utf-8")
    comparison_lines = [
        "# V5.4 vs V5.5 交易质量优化对比",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "| 模型与口径 | 样本数 | 10日平均收益 | 胜率 | 盈亏比 | 最大亏损 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| V5.4核心池 | {comparison['v5_4']['sample_count']} | {_pct(comparison['v5_4']['average_return_pct'])} | {_pct(comparison['v5_4']['win_rate_pct'])} | {_ratio(comparison['v5_4']['profit_loss_ratio'])} | {_pct(comparison['v5_4']['max_loss_pct'])} |",
        f"| V5.5无止损 | {comparison['v5_5_raw']['sample_count']} | {_pct(comparison['v5_5_raw']['average_return_pct'])} | {_pct(comparison['v5_5_raw']['win_rate_pct'])} | {_ratio(comparison['v5_5_raw']['profit_loss_ratio'])} | {_pct(comparison['v5_5_raw']['max_loss_pct'])} |",
        f"| V5.5 -8%止损 | {comparison['v5_5_stop']['sample_count']} | {_pct(comparison['v5_5_stop']['average_return_pct'])} | {_pct(comparison['v5_5_stop']['win_rate_pct'])} | {_ratio(comparison['v5_5_stop']['profit_loss_ratio'])} | {_pct(comparison['v5_5_stop']['max_loss_pct'])} |",
        "",
        f"结论：V5.5止损口径最大亏损{'下降' if comparison['max_loss_improved'] else '未下降'}。若未下降，不应继续升级交易质量模块。",
        "",
    ]
    (output_dir / "V5.4_vs_V5.5_comparison.md").write_text("\n".join(comparison_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start, end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    config = TradingQualityConfig(
        market_enabled=not args.disable_market_score,
        trend_stage_enabled=not args.disable_trend_stage,
        buy_point_enabled=not args.disable_buy_point,
        sector_linkage_enabled=not args.disable_sector_linkage,
    )
    enrichment, enrichment_status = load_local_historical_enrichment(
        args.history_dir, args.daily_basic_dir, args.history_dir, args.industry_dir
    )
    base_market, market_status = load_local_historical_market(args.market_dir)
    base_market_index = _market_index(base_market)
    enrichment_index = _enrichment_index(enrichment)
    histories = _load_histories_with_low(args.history_dir, set(enrichment_index["code"]))
    market = _load_market_quality(args.market_dir)
    industry_metrics = _build_industry_metrics(args.history_dir, args.industry_dir)
    attack_values: dict[str, list[float]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    coverage = {"scored_rows": 0, "insufficient_rows": 0, "attack_rank_dates": 0, "network_requests": 0}
    signal_dates: list[pd.Timestamp] = []

    for stock_index, (code, history) in enumerate(histories.items(), start=1):
        selected = enrichment_index[enrichment_index["code"] == code].drop(columns="code", errors="ignore")
        merged = _merge_history(history, selected, industry_metrics.get(code))
        if len(merged) <= MIN_LOOKBACK_DAYS + max(HOLDING_DAYS):
            continue
        for index in range(MIN_LOOKBACK_DAYS, len(merged) - max(HOLDING_DAYS)):
            row = build_daily_score_row(merged, index, code)
            signal_date = pd.Timestamp(row["date"])
            if signal_date < start or signal_date > end:
                continue
            row.update(base_market_index.get(signal_date, {}))
            missing = _required_missing_fields(row)
            if missing:
                coverage["insufficient_rows"] += 1
                continue
            result = score_stock(row)
            ratio = _finite(result["capital_attack_ratio"])
            if ratio is None:
                continue
            date_key = signal_date.strftime("%Y-%m-%d")
            attack_values[date_key].append(ratio)
            coverage["scored_rows"] += 1
            signal_dates.append(signal_date)
            # V5.5 is a quality-confirmation layer, not a score-expansion
            # mechanism: only stocks already at the V5.4 >=85 threshold can
            # enter the adjusted trading-pool evaluation.
            if int(result["score"]) < 85:
                continue
            closes = merged.loc[:index, "close"]
            volumes = merged.loc[:index, "volume"]
            row["prior_price"] = float(merged.loc[index - 1, "close"])
            row["prior_ma20"] = float(closes.iloc[:-1].tail(20).mean())
            row["pct_chg_10d"] = float((row["price"] / closes.iloc[-11] - 1) * 100) if len(closes) >= 11 else float("nan")
            row["avg_volume_20d"] = float(volumes.iloc[:-1].tail(20).mean())
            industry_row = {
                key: merged.loc[index, key]
                for key in ("industry_rank", "industry_count", "industry_up_ratio", "industry_limit_up_count")
            }
            quality = evaluate_trading_quality(
                row, market.get(signal_date, {}), industry_row, config
            )
            final_score = max(0, min(100, int(round(int(result["score"]) + quality["adjustment"]))))
            if final_score < 85:
                continue
            candidate = {
                "code": code,
                "signal_date": date_key,
                "base_score": int(result["score"]),
                "quality_adjustment": int(quality["adjustment"]),
                "final_score": final_score,
                "market_score": quality["market_score"],
                "trend_stage": quality["trend_stage"],
                "trend_stage_score": quality["trend_stage_score"],
                "buy_point_score": quality["buy_point_score"],
                "sector_linkage_score": quality["sector_linkage_score"],
                "turnover_rate": float(row["turnover"]),
                "capital_attack_ratio": ratio,
                "base_buy_point_quality_score": int(result["buy_point_quality_score"]),
                "entry_close": float(row["price"]),
                "quality_detail": quality["detail"],
            }
            for days in HOLDING_DAYS:
                path = merged.iloc[index + 1 : index + days + 1]
                exit_close = float(path.iloc[-1]["close"])
                lows = pd.to_numeric(path["low"], errors="coerce").dropna()
                minimum_low = float(lows.min()) if not lows.empty else exit_close
                max_drawdown = (minimum_low / candidate["entry_close"] - 1) * 100
                stopped = minimum_low <= candidate["entry_close"] * 0.92
                candidate[f"raw_return_{days}d"] = round((exit_close / candidate["entry_close"] - 1) * 100, 6)
                candidate[f"stop_return_{days}d"] = -8.0 if stopped else candidate[f"raw_return_{days}d"]
                candidate[f"max_drawdown_{days}d"] = round(max_drawdown, 6)
                candidate[f"stop_triggered_{days}d"] = bool(stopped)
            candidates.append(candidate)
        if stock_index % 250 == 0:
            print(f"评分进度：{stock_index}/{len(histories)}，完整评分行={coverage['scored_rows']}，85分候选={len(candidates)}", flush=True)

    thresholds = {
        date: sorted(values, reverse=True)[max(0, math.ceil(len(values) * 0.20) - 1)]
        for date, values in attack_values.items()
        if values
    }
    coverage["attack_rank_dates"] = len(thresholds)
    trade_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        top20 = candidate["capital_attack_ratio"] >= thresholds.get(candidate["signal_date"], float("inf"))
        core = (
            5.0 <= candidate["turnover_rate"] <= 30.0
            and top20
            and candidate["base_buy_point_quality_score"] >= 10
        )
        if not core:
            continue
        signal_rows.append(candidate)
        for days in HOLDING_DAYS:
            trade_rows.append(
                {
                    **{key: value for key, value in candidate.items() if not key.endswith(("3d", "5d", "10d"))},
                    "holding_days": days,
                    "raw_return_pct": candidate[f"raw_return_{days}d"],
                    "stop_return_pct": candidate[f"stop_return_{days}d"],
                    "max_drawdown_pct": candidate[f"max_drawdown_{days}d"],
                    "stop_triggered": candidate[f"stop_triggered_{days}d"],
                }
            )
    trades = pd.DataFrame(trade_rows)
    core_signals = pd.DataFrame(signal_rows)
    signal_counts = core_signals.groupby("signal_date").size() if not core_signals.empty else pd.Series(dtype=int)
    statistics: dict[str, Any] = {}
    for days in HOLDING_DAYS:
        subset = trades[trades["holding_days"] == days] if not trades.empty else trades
        stop_count = int(subset["stop_triggered"].sum()) if not subset.empty else 0
        statistics[str(days)] = {
            "raw": _stats(trades, "raw_return_pct", days),
            "stop_loss": _stats(trades, "stop_return_pct", days),
            "stop_loss_count": stop_count,
            "stop_loss_rate_pct": float(stop_count / len(subset) * 100) if len(subset) else 0.0,
        }
    v54_path = PROJECT_ROOT / "data" / "reports" / "v5_4" / "score_v5_4_core_pool_trades.csv"
    v54 = pd.read_csv(v54_path, encoding="utf-8-sig") if v54_path.exists() else pd.DataFrame()
    v54_stats = _stats(v54.rename(columns={"return_pct": "raw_return_pct"}), "raw_return_pct", 10)
    v55_raw = statistics["10"]["raw"]
    v55_stop = statistics["10"]["stop_loss"]
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module_config": {
            "market_score": config.market_enabled,
            "trend_stage": config.trend_stage_enabled,
            "buy_point": config.buy_point_enabled,
            "sector_linkage": config.sector_linkage_enabled,
        },
        "coverage": coverage,
        "historical_enrichment": enrichment_status,
        "historical_market": market_status,
        "signal_range": {
            "start": min(signal_dates).strftime("%Y-%m-%d") if signal_dates else None,
            "end": max(signal_dates).strftime("%Y-%m-%d") if signal_dates else None,
        },
        "final_score_ge_85_signals": len(candidates),
        "core_pool_signal_count": len(core_signals),
        "signal_statistics": {
            "daily_average": float(signal_counts.mean()) if not signal_counts.empty else 0.0,
            "daily_max": int(signal_counts.max()) if not signal_counts.empty else 0,
            "daily_min": int(signal_counts.min()) if not signal_counts.empty else 0,
        },
        "statistics": statistics,
        "comparison": {
            "v5_4": v54_stats,
            "v5_5_raw": v55_raw,
            "v5_5_stop": v55_stop,
            "max_loss_improved": (
                v55_stop["max_loss_pct"] is not None
                and v54_stats["max_loss_pct"] is not None
                and v55_stop["max_loss_pct"] > v54_stats["max_loss_pct"]
            ),
        },
    }
    _write_report(args.output_dir, report, trades)
    print(f"回测完成：最终85分信号={len(candidates)}，核心池={len(core_signals)}。")
    print(f"报告：{args.output_dir / 'score_v5_5_backtest_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
