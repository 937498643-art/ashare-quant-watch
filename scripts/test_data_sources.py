"""Probe public A-share quote endpoints used by quant_stock_watch.

Run from the project root:
    .\\.venv\\Scripts\\python.exe scripts\\test_data_sources.py

This script only reads public market data. It does not provide any trading
operation, account access, or broker connection.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.akshare_source import AkshareSource  # noqa: E402


pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)
pd.set_option("display.unicode.east_asian_width", True)

TURNOVER_ALIASES = ["换手率", "turnover", "turnover_rate", "换手", "f8"]
AMOUNT_ALIASES = ["成交额", "amount", "f6"]


@dataclass(frozen=True)
class SourceProbe:
    """One public quote endpoint to probe."""

    name: str
    description: str
    fetcher: Callable[[], pd.DataFrame]


def configure_stdout() -> None:
    """Prefer UTF-8 output on Windows terminals when available."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def print_line(title: str = "") -> None:
    """Print a readable section separator."""
    print("\n" + "=" * 88)
    if title:
        print(title)
        print("=" * 88)


def call_source(fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Call one endpoint and ensure a DataFrame object is returned."""
    data = fetcher()
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def print_success_result(data: pd.DataFrame) -> None:
    """Print row count, fields, field coverage, and a preview."""
    columns = [str(column) for column in data.columns]
    turnover_field = find_alias(columns, TURNOVER_ALIASES)
    amount_field = find_alias(columns, AMOUNT_ALIASES)

    print("是否成功: 是")
    print(f"返回行数: {len(data)}")
    print(f"原始字段列表: {columns}")
    print(f"是否包含换手率字段: {'是，字段=' + turnover_field if turnover_field else '否'}")
    print(f"是否包含成交额字段: {'是，字段=' + amount_field if amount_field else '否'}")
    print("前 5 行样例:")
    if data.empty:
        print("(接口返回空 DataFrame)")
    else:
        print(data.head(5).to_string(index=False))
    print("失败原因: 无")


def print_failure_result(reason: str) -> None:
    """Print a normalized failure report for one endpoint."""
    print("是否成功: 否")
    print("返回行数: 0")
    print("原始字段列表: []")
    print("是否包含换手率字段: 否")
    print("是否包含成交额字段: 否")
    print("前 5 行样例: 无")
    print(f"失败原因: {reason}")


def probe_single_source(source: SourceProbe) -> pd.DataFrame:
    """Probe one quote endpoint and print the complete result."""
    print_line(source.description)
    try:
        data = call_source(source.fetcher)
        if data.empty:
            print_failure_result("接口返回空 DataFrame")
            return data
        print_success_result(data)
        return data
    except Exception as exc:
        print_failure_result(f"{type(exc).__name__}: {exc}")
        return pd.DataFrame()


def probe_merged_eastmoney_parts(results: dict[str, pd.DataFrame]) -> None:
    """Try merging Shanghai, Shenzhen, and Beijing Eastmoney part endpoints."""
    print_line("东方财富沪 A / 深 A / 京 A 分市场合并测试")
    part_names = ["eastmoney_sh_a", "eastmoney_sz_a", "eastmoney_bj_a"]
    frames = [results[name] for name in part_names if not results.get(name, pd.DataFrame()).empty]

    if not frames:
        print_failure_result("沪 A、深 A、京 A 分市场接口均未返回可合并数据")
        return

    try:
        merged = pd.concat(frames, ignore_index=True)
        print_success_result(merged)
    except Exception as exc:
        print_failure_result(f"{type(exc).__name__}: {exc}")


def find_alias(columns: list[str], aliases: list[str]) -> str | None:
    """Find a field by exact or normalized alias matching."""
    for alias in aliases:
        if alias in columns:
            return alias
    normalized_aliases = {normalize_column_name(alias) for alias in aliases}
    for column in columns:
        if normalize_column_name(column) in normalized_aliases:
            return column
    return None


def normalize_column_name(value: str) -> str:
    """Normalize column labels for tolerant matching."""
    return str(value).replace(" ", "").replace("_", "").replace("%", "").replace("％", "").lower()


def main() -> int:
    """Run all public quote endpoint probes."""
    configure_stdout()
    source = AkshareSource({"retry_count": 1, "timeout": 8})
    probes = [
        SourceProbe("eastmoney_full_market", "ak.stock_zh_a_spot_em() 东方财富全市场", ak.stock_zh_a_spot_em),
        SourceProbe("eastmoney_sh_a", "ak.stock_sh_a_spot_em() 东方财富沪 A", ak.stock_sh_a_spot_em),
        SourceProbe("eastmoney_sz_a", "ak.stock_sz_a_spot_em() 东方财富深 A", ak.stock_sz_a_spot_em),
        SourceProbe("eastmoney_bj_a", "ak.stock_bj_a_spot_em() 东方财富京 A", ak.stock_bj_a_spot_em),
        SourceProbe("eastmoney_push2", "direct_eastmoney_push2_fetch() 东方财富 push2 原始接口", source.direct_eastmoney_push2_fetch),
        SourceProbe("sina_full_market", "ak.stock_zh_a_spot() 新浪备用源", ak.stock_zh_a_spot),
    ]

    print_line("A 股公开行情源字段自检")
    print("说明: 本脚本只测试公开行情数据读取，不做任何交易侧操作。")
    print(f"Python: {sys.version.split()[0]}")
    print(f"AKShare: {getattr(ak, '__version__', 'unknown')}")

    results: dict[str, pd.DataFrame] = {}
    for probe in probes:
        results[probe.name] = probe_single_source(probe)

    probe_merged_eastmoney_parts(results)

    print_line("测试完成")
    successful = [name for name, data in results.items() if not data.empty]
    failed = [name for name, data in results.items() if data.empty]
    print(f"成功接口数量: {len(successful)}")
    print(f"失败或空数据接口数量: {len(failed)}")
    print(f"成功接口: {successful if successful else '无'}")
    print(f"失败或空数据接口: {failed if failed else '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
