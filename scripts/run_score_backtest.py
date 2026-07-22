"""Run the read-only V5.3 historical score backtest from local caches."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest import (  # noqa: E402
    HOLDING_DAYS,
    MIN_LOOKBACK_DAYS,
    load_local_historical_enrichment,
    load_local_historical_market,
    run_score_backtest,
    write_backtest_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="V5.3 historical score backtest using local caches")
    parser.add_argument("--history-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "daily")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_3")
    parser.add_argument("--daily-basic-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "daily_basic")
    parser.add_argument("--daily-cache-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "daily")
    parser.add_argument("--industry-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "industry")
    parser.add_argument("--market-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "market")
    parser.add_argument("--v5-summary", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5" / "score_v5_backtest_summary.json")
    parser.add_argument("--v5-trades", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5" / "score_v5_backtest_trades.csv")
    parser.add_argument("--v5-2-summary", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_2" / "score_v5_2_backtest_summary.json")
    parser.add_argument("--v5-2-trades", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_2" / "score_v5_2_backtest_trades.csv")
    parser.add_argument("--comparison-report", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_v5_2_v5_3_comparison.md")
    parser.add_argument("--lookback-days", type=int, default=MIN_LOOKBACK_DAYS)
    parser.add_argument("--holding-days", default=",".join(map(str, HOLDING_DAYS)))
    args = parser.parse_args()

    try:
        horizons = tuple(int(value.strip()) for value in args.holding_days.split(",") if value.strip())
        enrichment, enrichment_status = load_local_historical_enrichment(
            args.history_dir,
            args.daily_basic_dir,
            args.daily_cache_dir,
            args.industry_dir,
        )
        market, market_status = load_local_historical_market(args.market_dir)
        trades, report = run_score_backtest(args.history_dir, enrichment, horizons, args.lookback_days, market)
        report["historical_enrichment"] = enrichment_status
        report["historical_market"] = market_status
    except Exception as exc:
        print(f"回测失败：{type(exc).__name__}: {exc}")
        return 1

    paths = write_backtest_report(trades, report, args.output_dir)
    _write_comparison(
        args.v5_summary,
        args.v5_trades,
        args.v5_2_summary,
        args.v5_2_trades,
        paths["json"],
        paths["trades"],
        args.comparison_report,
    )
    print(
        f"回测完成：完整字段评分行 {report['coverage']['scored_daily_rows']}，"
        f"数据不足跳过 {report['coverage']['insufficient_daily_rows']}，模拟交易 {report['trade_count']}。"
    )
    print(f"Markdown 报告：{paths['markdown']}")
    print(f"JSON 摘要：{paths['json']}")
    print(f"交易明细：{paths['trades']}")
    print(f"V5/V5.2/V5.3 对比：{args.comparison_report}")
    return 0


def _write_comparison(
    v5_summary_path: Path,
    v5_trades_path: Path,
    v51_summary_path: Path,
    v51_trades_path: Path,
    v52_summary_path: Path,
    v52_trades_path: Path,
    output_path: Path,
) -> None:
    """Compare high-score pools from V5, V5.2, and V5.3."""
    required = (v5_summary_path, v5_trades_path, v51_summary_path, v51_trades_path)
    if not all(path.exists() for path in required):
        output_path.write_text("# V5 / V5.2 / V5.3 回测对比\n\n缺少历史基线报告或交易明细。\n", encoding="utf-8")
        return
    v5 = json.loads(v5_summary_path.read_text(encoding="utf-8"))
    v51 = json.loads(v51_summary_path.read_text(encoding="utf-8"))
    v52 = json.loads(v52_summary_path.read_text(encoding="utf-8"))
    v5_trades = pd.read_csv(v5_trades_path, encoding="utf-8-sig")
    v51_trades = pd.read_csv(v51_trades_path, encoding="utf-8-sig")
    v52_trades = pd.read_csv(v52_trades_path, encoding="utf-8-sig")
    lines = [
        "# V5 / V5.2 / V5.3 历史回测对比", "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "## 评分分布", "",
        "| 模型 | 核心区间 | 次高区间 | 第三档区间 |",
        "|---|---:|---:|---:|",
        f"| V5 | >=90：{v5['score_distribution']['score_ge_90']} | 85–90：{v5['score_distribution']['score_85_90']} | 75–85：{v5['score_distribution']['score_75_85']} |",
        f"| V5.2 | >=90：{v51['score_distribution']['score_ge_90']} | 85–90：{v51['score_distribution']['score_85_90']} | 80–85：{v51['score_distribution']['score_80_85']} |",
        f"| V5.3 | >=90：{v52['score_distribution']['score_ge_90']} | 85–90：{v52['score_distribution']['score_85_90']} | 80–85：{v52['score_distribution']['score_80_85']} |",
        "", "## 高分目标池收益对比", "",
        "| 目标池 | 天数 | 样本数 | 平均收益 | 胜率 | 最大亏损 | 盈亏比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, frame, threshold in (
        ("V5 >=85", v5_trades, 85),
        ("V5.2 85–90", v51_trades, 85),
        ("V5.3 85–90", v52_trades, 85),
    ):
        for days in (3, 5, 10):
            stats = _threshold_statistics(
                frame, threshold, days,
                upper=90 if label in {"V5.2 85–90", "V5.3 85–90"} else None,
            )
            lines.append(
                f"| {label} | {days} | {stats['sample_count']} | {_pct(stats['average_return'])} | "
                f"{_pct(stats['win_rate'])} | {_pct(stats['max_loss'])} | {_ratio(stats['profit_loss_ratio'])} |"
            )
    lines.extend([
        "", "## 口径提示", "",
        "- 三个版本的等级阈值不同；目标池结果用于观察风险收益取舍，不是同一分数区间的严格同比。",
        "- V5.3 缺失市场字段时不扣分，且不把缺失模块计入归一化基数。",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _threshold_statistics(
    data: pd.DataFrame, threshold: int, days: int, upper: int | None = None
) -> dict[str, float | int | None]:
    scores = pd.to_numeric(data["score"], errors="coerce")
    subset = data[(scores >= threshold) & (data["holding_days"] == days)]
    if upper is not None:
        subset = subset[pd.to_numeric(subset["score"], errors="coerce") < upper]
    returns = pd.to_numeric(subset["return_pct"], errors="coerce").dropna()
    if returns.empty:
        return {"sample_count": 0, "average_return": None, "win_rate": None, "max_loss": None, "profit_loss_ratio": None}
    wins, losses = returns[returns > 0], returns[returns < 0]
    ratio = float(wins.mean() / abs(losses.mean())) if not wins.empty and not losses.empty else None
    return {
        "sample_count": int(len(returns)),
        "average_return": float(returns.mean()),
        "win_rate": float((returns > 0).mean() * 100),
        "max_loss": float(returns.min()),
        "profit_loss_ratio": ratio,
    }


def _pct(value: object) -> str:
    return "--" if value is None else f"{float(value):.2f}%"


def _ratio(value: object) -> str:
    return "--" if value is None else f"{float(value):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
