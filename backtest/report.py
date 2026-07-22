"""Artifact writers for the V5.10 candidate-pool backtest."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def write_backtest_report(
    output_dir: str | Path,
    trades: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Path]:
    """Write a trade ledger, machine-readable summary, and Markdown report."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trades": root / "v5_10_candidate_pool_trades.csv",
        "summary": root / "v5_10_backtest_summary.json",
        "markdown": root / "v5_10_backtest_report.md",
    }
    trades.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **summary}
    paths["summary"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(build_markdown_report(payload), encoding="utf-8")
    return paths


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Render requested score-band statistics without recalculating values."""
    lines = [
        "# V5.10 回测验证报告",
        "",
        f"生成时间：{summary.get('generated_at', '--')}",
        "",
        "## 模拟规则",
        "",
        str(summary.get("method", "--")),
        "",
        "## 覆盖情况",
        "",
        f"- 候选池交易日数：{summary.get('coverage', {}).get('candidate_pool_dates', 0)}",
        f"- 候选池原始记录：{summary.get('coverage', {}).get('candidate_rows', 0)}",
        f"- 可模拟交易记录：{summary.get('trade_rows', 0)}",
        f"- 缺少价格历史：{summary.get('coverage', {}).get('missing_price_history', 0)}",
        f"- 信号日未在历史行情中找到：{summary.get('coverage', {}).get('missing_signal_date', 0)}",
        "",
        "## 评分区间表现",
        "",
        "| 评分区间 | 持有天数 | 样本数量 | 上涨数量 | 胜率 | 平均收益 | 最大收益 | 最大亏损 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("score_intervals", []):
        lines.append(
            "| {score_interval} | {holding_days} | {sample_count} | {up_count} | {win_rate} | {average} | {maximum} | {minimum} |".format(
                score_interval=row["score_interval"],
                holding_days=row["holding_days"],
                sample_count=row["sample_count"],
                up_count=row["up_count"],
                win_rate=_percent(row.get("win_rate_pct")),
                average=_percent(row.get("average_return_pct")),
                maximum=_percent(row.get("max_return_pct")),
                minimum=_percent(row.get("max_loss_pct")),
            )
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 评分和交易候选池由对应日期的快照提供；本模块不会重算评分、调整 Top50 或改变今日候选池逻辑。",
            "- 仅使用本地历史日线收盘价计算收益，不访问网络。",
            "- 缺少后续交易日价格的样本在对应持有期统计中不计入。",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "--" if pd.isna(number) else f"{float(number):.2f}%"
