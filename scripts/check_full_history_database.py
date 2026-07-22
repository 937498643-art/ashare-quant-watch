#!/usr/bin/env python3
"""Check completeness of the local offline A-share history database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HISTORY_ROOT = PROJECT_ROOT / "data" / "history"
REPORT_PATH = PROJECT_ROOT / "history_database_check.md"
DAILY_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
]
BASIC_FIELDS = ["ts_code", "trade_date", "turnover_rate", "circ_mv"]
MARKET_FIELDS = [
    "trade_date",
    "sh_close",
    "sh_ma5",
    "sh_ma10",
    "sh_pct_chg",
    "up_count",
    "down_count",
    "up_ratio",
]


def _files(directory: Path, prefix: str) -> list[Path]:
    return sorted(directory.glob(f"{prefix}_*.csv")) if directory.exists() else []


def _field_rate(frame: pd.DataFrame, fields: list[str]) -> float:
    if frame.empty or any(field not in frame.columns for field in fields):
        return 0.0
    return float(frame[fields].notna().all(axis=1).mean() * 100)


def _per_stock_summary(
    directory: Path,
    prefix: str,
    fields: list[str],
    min_universe: int,
) -> dict[str, object]:
    files = _files(directory, prefix)
    counts: list[int] = []
    field_rates: list[float] = []
    dates: list[str] = []
    undercovered: list[str] = []
    for path in files:
        frame = pd.read_csv(path)
        date = path.stem.removeprefix(f"{prefix}_")
        dates.append(date)
        counts.append(len(frame))
        field_rates.append(_field_rate(frame, fields))
        if len(frame) < min_universe:
            undercovered.append(date)
    return {
        "files": len(files),
        "dates": dates,
        "counts": counts,
        "field_rates": field_rates,
        "undercovered": undercovered,
    }


def _market_summary(directory: Path) -> dict[str, object]:
    files = _files(directory, "market")
    rates: list[float] = []
    dates: list[str] = []
    breadth_complete = 0
    for path in files:
        frame = pd.read_csv(path)
        dates.append(path.stem.removeprefix("market_"))
        rates.append(_field_rate(frame, MARKET_FIELDS))
        if not frame.empty and bool(frame.get("breadth_complete", pd.Series([False])).iloc[0]):
            breadth_complete += 1
    return {
        "files": len(files),
        "dates": dates,
        "field_rates": rates,
        "breadth_complete": breadth_complete,
    }


def _range(dates: list[str]) -> str:
    return f"{min(dates)} 至 {max(dates)}" if dates else "无文件"


def _number(values: list[int]) -> str:
    if not values:
        return "0 / 0 / 0"
    series = pd.Series(values)
    return f"{int(series.min())} / {int(series.median())} / {int(series.max())}"


def main() -> int:
    parser = argparse.ArgumentParser(description="检查离线历史数据库完整性。")
    parser.add_argument("--min-universe", type=int, default=4000)
    args = parser.parse_args()

    daily = _per_stock_summary(
        HISTORY_ROOT / "daily", "daily", DAILY_FIELDS, args.min_universe
    )
    basic = _per_stock_summary(
        HISTORY_ROOT / "daily_basic",
        "daily_basic",
        BASIC_FIELDS,
        args.min_universe,
    )
    market = _market_summary(HISTORY_ROOT / "market")

    lines = [
        "# A股离线历史数据库完整性检查",
        "",
        f"- 检查时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 全市场完整日阈值：每个股票文件至少 {args.min_universe} 行",
        "",
        "## 交易日与每日股票数量",
        "",
        f"- daily 交易日文件数：{daily['files']}；覆盖：{_range(daily['dates'])}",
        f"- daily 每日股票数（最小 / 中位数 / 最大）：{_number(daily['counts'])}",
        f"- daily_basic 交易日文件数：{basic['files']}；覆盖：{_range(basic['dates'])}",
        f"- daily_basic 每日股票数（最小 / 中位数 / 最大）：{_number(basic['counts'])}",
        "",
        "## 完整率",
        "",
        f"- daily 字段完整率（逐行）：{pd.Series(daily['field_rates']).mean() if daily['field_rates'] else 0:.2f}%",
        f"- daily 全市场完整日：{daily['files'] - len(daily['undercovered'])}/{daily['files']}；未达阈值日期：{len(daily['undercovered'])}",
        f"- daily_basic 字段完整率（逐行）：{pd.Series(basic['field_rates']).mean() if basic['field_rates'] else 0:.2f}%",
        f"- daily_basic 全市场完整日：{basic['files'] - len(basic['undercovered'])}/{basic['files']}；未达阈值日期：{len(basic['undercovered'])}",
        f"- market 字段完整率（逐文件）：{pd.Series(market['field_rates']).mean() if market['field_rates'] else 0:.2f}%",
        f"- market 市场宽度完整日：{market['breadth_complete']}/{market['files']}；覆盖：{_range(market['dates'])}",
        "",
        "## 缺失记录",
        "",
        "- 下载失败代码及未达覆盖阈值日期：data/history/full_a_share_history_failures.json",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"报告已写入：{REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
