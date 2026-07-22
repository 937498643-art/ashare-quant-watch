#!/usr/bin/env python3
"""Run the V5.4 selection-layer backtest using only local historical caches.

This script deliberately leaves core.scoring unchanged.  It evaluates the
current V5.3 score for every eligible stock-day, then applies V5.4's
cross-sectional core-pool screen: score >= 85, healthy turnover, top-20%
capital-attack ratio, and buy-point quality >= 10.
"""

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
    _load_history_frames_from_daily_cache,
    _market_index,
    _merge_history_enrichment,
    _required_missing_fields,
    build_daily_score_row,
    load_local_historical_enrichment,
    load_local_historical_market,
)
from core.scoring import score_stock  # noqa: E402


SCORE_BINS = (
    ("score_ge_90", "评分 >= 90", 90, None),
    ("score_85_90", "评分 85–90", 85, 90),
    ("score_80_85", "评分 80–85", 80, 85),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run read-only V5.4 full-market selection backtest from local cache"
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "history" / "daily",
    )
    parser.add_argument(
        "--daily-basic-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "history" / "daily_basic",
    )
    parser.add_argument(
        "--industry-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "history" / "industry",
    )
    parser.add_argument(
        "--market-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "history" / "market",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "v5_4",
    )
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default="2026-07-17")
    parser.add_argument(
        "--holding-days", default=",".join(str(day) for day in HOLDING_DAYS)
    )
    return parser.parse_args()


def _within(date: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return start <= date <= end


def _score_bin(score: int) -> str | None:
    if score >= 90:
        return "score_ge_90"
    if score >= 85:
        return "score_85_90"
    if score >= 80:
        return "score_80_85"
    return None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _return_statistics(data: pd.DataFrame, days: int) -> dict[str, Any]:
    subset = data[data["holding_days"] == days] if not data.empty else data
    returns = pd.to_numeric(subset.get("return_pct"), errors="coerce").dropna()
    if returns.empty:
        return {
            "sample_count": 0,
            "average_return_pct": None,
            "win_rate_pct": None,
            "max_profit_pct": None,
            "max_loss_pct": None,
            "profit_loss_ratio": None,
        }
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    ratio = (
        float(wins.mean() / abs(losses.mean()))
        if not wins.empty and not losses.empty
        else None
    )
    return {
        "sample_count": int(len(returns)),
        "average_return_pct": float(returns.mean()),
        "win_rate_pct": float((returns > 0).mean() * 100),
        "max_profit_pct": float(returns.max()),
        "max_loss_pct": float(returns.min()),
        "profit_loss_ratio": ratio,
    }


def _pct(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}%"


def _ratio(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _legacy_statistics(
    label: str,
    trades_path: Path,
    lower: int,
    upper: int | None,
) -> tuple[str, dict[int, dict[str, Any]]]:
    if not trades_path.exists():
        return label, {days: _return_statistics(pd.DataFrame(), days) for days in HOLDING_DAYS}
    frame = pd.read_csv(trades_path, encoding="utf-8-sig")
    score = pd.to_numeric(frame.get("score"), errors="coerce")
    selected = frame[score >= lower]
    if upper is not None:
        selected = selected[score.loc[selected.index] < upper]
    return label, {days: _return_statistics(selected, days) for days in HOLDING_DAYS}


def _write_report(
    output_dir: Path,
    report: dict[str, Any],
    all_high_trades: pd.DataFrame,
    core_trades: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score_v5_4_backtest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_high_trades.to_csv(
        output_dir / "score_v5_4_all_high_score_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    core_trades.to_csv(
        output_dir / "score_v5_4_core_pool_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )

    distribution = report["score_distribution"]
    lines = [
        "# V5.4 全市场真实历史回测报告",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 回测口径",
        "",
        "- 评分：完全沿用当前 core/scoring.py 的 V5.3 评分，不修改任何评分权重或分数映射。",
        "- V5.4 仅作为交易池筛选层：评分 >=85、换手率 5%–30%、当日资金攻击强度位列全市场前20%、买点质量 >=10。",
        "- 买入：信号日收盘；卖出：第3、第5、第10个后续交易日收盘；未计交易成本、滑点及涨跌停无法成交。",
        "- 数据：仅读取 data/history/daily、daily_basic、industry、market；network_requests=0。",
        "",
        "## 数据覆盖",
        "",
        f"- 缓存请求范围：{report['requested_range']['start']} 至 {report['requested_range']['end']}。",
        f"- 实际可评分信号日期：{report['signal_range']['start'] or '--'} 至 {report['signal_range']['end'] or '--'}。",
        f"- 全市场完整字段评分行：{report['coverage']['scored_rows']}；数据不足跳过：{report['coverage']['insufficient_rows']}。",
        f"- 参与资金攻击横截面排名的有效行：{report['coverage']['attack_rank_rows']}；排名日期：{report['coverage']['attack_rank_dates']}。",
        f"- network_requests：{report['coverage']['network_requests']}。",
        "",
        "## 评分分布（未筛选，V5.3 原始评分）",
        "",
        f"- >=90：{distribution['score_ge_90']}",
        f"- 85–90：{distribution['score_85_90']}",
        f"- 80–85：{distribution['score_80_85']}",
        f"- <80：{distribution['score_below_80']}",
        "",
        "## V5.4 核心交易池",
        "",
        f"- 核心交易池信号数：{report['core_pool']['signal_count']}。",
        f"- 85分以上原始评分信号数：{report['core_pool']['score_ge_85_count']}。",
        f"- 通过资金攻击前20%筛选的85分以上信号数：{report['core_pool']['attack_top_20_score_ge_85_count']}。",
        "",
        "### 核心交易池收益",
        "",
        "| 持有天数 | 样本数 | 平均收益 | 胜率 | 最大盈利 | 最大亏损 | 盈亏比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for days in HOLDING_DAYS:
        stats = report["core_pool"]["statistics"][str(days)]
        lines.append(
            f"| {days}日 | {stats['sample_count']} | {_pct(stats['average_return_pct'])} | "
            f"{_pct(stats['win_rate_pct'])} | {_pct(stats['max_profit_pct'])} | "
            f"{_pct(stats['max_loss_pct'])} | {_ratio(stats['profit_loss_ratio'])} |"
        )

    lines.extend([
        "",
        "## 评分区间收益（未应用核心池筛选）",
        "",
        "| 区间 | 持有天数 | 样本数 | 平均收益 | 胜率 | 最大亏损 | 盈亏比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label, _lower, _upper in SCORE_BINS:
        for days in HOLDING_DAYS:
            stats = report["score_bin_statistics"][key][str(days)]
            lines.append(
                f"| {label} | {days}日 | {stats['sample_count']} | "
                f"{_pct(stats['average_return_pct'])} | {_pct(stats['win_rate_pct'])} | "
                f"{_pct(stats['max_loss_pct'])} | {_ratio(stats['profit_loss_ratio'])} |"
            )

    lines.extend([
        "",
        "## 与 V5.1 / V5.2 / V5.3 的已有报告对比",
        "",
        "| 版本与目标池 | 持有天数 | 样本数 | 平均收益 | 胜率 | 最大亏损 | 盈亏比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, statistics in report["legacy_comparison"].items():
        for days in HOLDING_DAYS:
            stats = statistics[str(days)]
            lines.append(
                f"| {label} | {days}日 | {stats['sample_count']} | "
                f"{_pct(stats['average_return_pct'])} | {_pct(stats['win_rate_pct'])} | "
                f"{_pct(stats['max_loss_pct'])} | {_ratio(stats['profit_loss_ratio'])} |"
            )
    lines.extend([
        "",
        "## 解释与限制",
        "",
        "- 既有 V5.1/V5.2/V5.3 报告使用此前较小历史样本，故仅作历史基线参考，不能与本次全市场结果作严格同口径优劣结论。",
        "- 当前固定持有期模型没有构建组合净值曲线，因此报告最大亏损为单笔最大亏损，不将其误称为组合最大回撤。",
        "- 2025-01-02 是缓存首日，市场宽度缺少前一交易日比较基础；评分模块按既有逻辑将可选市场字段视为不评分，不扣分。",
        "- 回测不构成投资建议。",
        "",
    ])
    (output_dir / "score_v5_4_backtest_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    horizons = tuple(
        sorted(
            {
                int(value.strip())
                for value in args.holding_days.split(",")
                if value.strip() and int(value.strip()) > 0
            }
        )
    )
    if not horizons:
        raise ValueError("至少需要一个正数持有天数。")
    max_horizon = max(horizons)

    enrichment, enrichment_status = load_local_historical_enrichment(
        args.history_dir,
        args.daily_basic_dir,
        args.history_dir,
        args.industry_dir,
    )
    market, market_status = load_local_historical_market(args.market_dir)
    enrichment_index = _enrichment_index(enrichment)
    market_index = _market_index(market)
    allowed_codes = set(enrichment_index["code"].tolist())
    history_frames = _load_history_frames_from_daily_cache(
        args.history_dir, allowed_codes
    )

    coverage = {
        "history_stock_count": len(history_frames),
        "scored_rows": 0,
        "insufficient_rows": 0,
        "attack_rank_rows": 0,
        "attack_rank_dates": 0,
        "network_requests": 0,
        "insufficient_reasons": defaultdict(int),
    }
    distribution = {
        "score_ge_90": 0,
        "score_85_90": 0,
        "score_80_85": 0,
        "score_below_80": 0,
    }
    attack_values: dict[str, list[float]] = defaultdict(list)
    high_candidates: list[dict[str, Any]] = []
    signal_dates: list[pd.Timestamp] = []

    for stock_index, (code, history) in enumerate(history_frames.items(), start=1):
        history = _merge_history_enrichment(history, code, enrichment_index)
        if len(history) <= MIN_LOOKBACK_DAYS + max_horizon:
            continue
        for index in range(MIN_LOOKBACK_DAYS, len(history) - max_horizon):
            score_row = build_daily_score_row(history, index, code)
            signal_date = pd.Timestamp(score_row["date"])
            if not _within(signal_date, start, end):
                continue
            score_row.update(market_index.get(signal_date, {}))
            missing = _required_missing_fields(score_row)
            if missing:
                coverage["insufficient_rows"] += 1
                for field in missing:
                    coverage["insufficient_reasons"][field] += 1
                continue

            result = score_stock(score_row)
            score = int(result["score"])
            coverage["scored_rows"] += 1
            signal_dates.append(signal_date)
            if score >= 90:
                distribution["score_ge_90"] += 1
            elif score >= 85:
                distribution["score_85_90"] += 1
            elif score >= 80:
                distribution["score_80_85"] += 1
            else:
                distribution["score_below_80"] += 1

            ratio = _finite_number(result["capital_attack_ratio"])
            date_key = signal_date.strftime("%Y-%m-%d")
            if ratio is not None:
                attack_values[date_key].append(ratio)
                coverage["attack_rank_rows"] += 1

            if score < 80 or ratio is None:
                continue
            candidate = {
                "code": code,
                "signal_date": date_key,
                "score": score,
                "score_bin": _score_bin(score),
                "turnover_rate": float(score_row["turnover"]),
                "capital_attack_ratio": ratio,
                "buy_point_quality_score": int(result["buy_point_quality_score"]),
                "entry_close": float(score_row["price"]),
                "score_breakdown": result["score_breakdown"],
            }
            for days in horizons:
                exit_row = history.iloc[index + days]
                exit_close = float(exit_row["close"])
                candidate[f"exit_date_{days}d"] = pd.Timestamp(
                    exit_row["date"]
                ).strftime("%Y-%m-%d")
                candidate[f"return_{days}d_pct"] = round(
                    (exit_close / candidate["entry_close"] - 1) * 100, 6
                )
            high_candidates.append(candidate)
        if stock_index % 250 == 0:
            print(
                f"评分进度：{stock_index}/{len(history_frames)}，"
                f"完整评分行={coverage['scored_rows']}，高分候选={len(high_candidates)}",
                flush=True,
            )

    attack_thresholds: dict[str, float] = {}
    for date_key, values in attack_values.items():
        if not values:
            continue
        ordered = sorted(values, reverse=True)
        threshold_index = max(0, math.ceil(len(ordered) * 0.20) - 1)
        attack_thresholds[date_key] = ordered[threshold_index]
    coverage["attack_rank_dates"] = len(attack_thresholds)

    all_high_rows: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []
    core_pool_signals: list[dict[str, Any]] = []
    score_ge_85_count = 0
    attack_top_20_score_ge_85_count = 0
    for candidate in high_candidates:
        threshold = attack_thresholds.get(candidate["signal_date"])
        top_20 = threshold is not None and candidate["capital_attack_ratio"] >= threshold
        candidate["capital_attack_top_20pct"] = bool(top_20)
        for days in horizons:
            row = {
                "code": candidate["code"],
                "signal_date": candidate["signal_date"],
                "score": candidate["score"],
                "score_bin": candidate["score_bin"],
                "turnover_rate": candidate["turnover_rate"],
                "capital_attack_ratio": candidate["capital_attack_ratio"],
                "capital_attack_top_20pct": candidate["capital_attack_top_20pct"],
                "buy_point_quality_score": candidate["buy_point_quality_score"],
                "entry_close": candidate["entry_close"],
                "holding_days": days,
                "exit_date": candidate[f"exit_date_{days}d"],
                "return_pct": candidate[f"return_{days}d_pct"],
            }
            all_high_rows.append(row)

        if candidate["score"] >= 85:
            score_ge_85_count += 1
            if top_20:
                attack_top_20_score_ge_85_count += 1
        core = (
            candidate["score"] >= 85
            and 5.0 <= candidate["turnover_rate"] <= 30.0
            and top_20
            and candidate["buy_point_quality_score"] >= 10
        )
        if not core:
            continue
        core_pool_signals.append(candidate)
        for days in horizons:
            core_rows.append(
                {
                    "code": candidate["code"],
                    "signal_date": candidate["signal_date"],
                    "score": candidate["score"],
                    "turnover_rate": candidate["turnover_rate"],
                    "capital_attack_ratio": candidate["capital_attack_ratio"],
                    "buy_point_quality_score": candidate[
                        "buy_point_quality_score"
                    ],
                    "holding_days": days,
                    "exit_date": candidate[f"exit_date_{days}d"],
                    "return_pct": candidate[f"return_{days}d_pct"],
                }
            )

    all_high_trades = pd.DataFrame(all_high_rows)
    core_trades = pd.DataFrame(core_rows)
    bin_statistics: dict[str, dict[str, dict[str, Any]]] = {}
    for key, _label, _lower, _upper in SCORE_BINS:
        subset = (
            all_high_trades[all_high_trades["score_bin"] == key]
            if not all_high_trades.empty
            else all_high_trades
        )
        bin_statistics[key] = {
            str(days): _return_statistics(subset, days) for days in horizons
        }

    legacy_specs = (
        (
            "V5.1 >=88（既有小样本）",
            PROJECT_ROOT / "data" / "reports" / "v5_1" / "score_v5_1_backtest_trades.csv",
            88,
            None,
        ),
        (
            "V5.2 85–90（既有小样本）",
            PROJECT_ROOT / "data" / "reports" / "v5_2" / "score_v5_2_backtest_trades.csv",
            85,
            90,
        ),
        (
            "V5.3 85–90（既有小样本）",
            PROJECT_ROOT / "data" / "reports" / "v5_3" / "score_v5_3_backtest_trades.csv",
            85,
            90,
        ),
    )
    legacy_comparison = {
        label: {str(days): stats for days, stats in statistics.items()}
        for label, statistics in (
            _legacy_statistics(label, path, lower, upper)
            for label, path, lower, upper in legacy_specs
        )
    }
    coverage["insufficient_reasons"] = dict(coverage["insufficient_reasons"])
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_range": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        },
        "signal_range": {
            "start": min(signal_dates).strftime("%Y-%m-%d") if signal_dates else None,
            "end": max(signal_dates).strftime("%Y-%m-%d") if signal_dates else None,
        },
        "coverage": coverage,
        "historical_enrichment": enrichment_status,
        "historical_market": market_status,
        "score_distribution": distribution,
        "core_pool": {
            "signal_count": len(core_pool_signals),
            "score_ge_85_count": score_ge_85_count,
            "attack_top_20_score_ge_85_count": attack_top_20_score_ge_85_count,
            "statistics": {
                str(days): _return_statistics(core_trades, days) for days in horizons
            },
        },
        "score_bin_statistics": bin_statistics,
        "legacy_comparison": legacy_comparison,
    }
    _write_report(args.output_dir, report, all_high_trades, core_trades)
    print(
        f"回测完成：评分行={coverage['scored_rows']}，"
        f"高分信号={len(high_candidates)}，核心池信号={len(core_pool_signals)}。"
    )
    print(f"报告：{args.output_dir / 'score_v5_4_backtest_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
