"""Manually build or incrementally extend one local Tushare K-line cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.tushare_kline_cache_reader import stock_kline_cache_path, tushare_ts_code, update_stock_kline_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one local Tushare daily K-line cache.")
    parser.add_argument("--code", required=True, help="Six-digit stock code, for example 002384.")
    parser.add_argument("--days", type=int, default=120, help="Target number of recent trading days.")
    args = parser.parse_args()

    ts_code = tushare_ts_code(args.code)
    if not ts_code:
        print("无法识别股票代码。")
        return 1

    data, error = update_stock_kline_cache(args.code, days=max(args.days, 1))
    if error:
        print(f"单股K线缓存更新失败：{error}")
        return 1

    print(f"股票代码：{ts_code}")
    print(f"缓存路径：{stock_kline_cache_path(args.code)}")
    print(f"有效交易日：{len(data)}")
    if not data.empty:
        print(f"首个交易日：{data['trade_date'].iloc[0].strftime('%Y-%m-%d')}")
        print(f"最新交易日：{data['trade_date'].iloc[-1].strftime('%Y-%m-%d')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
