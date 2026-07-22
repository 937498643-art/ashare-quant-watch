"""Test Sina realtime quotes merged with local Tushare cache.

This script is a read-only diagnostic. It does not modify the main workflow,
dashboard, strategies, or any trading-related system.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.merged_reference_source import (  # noqa: E402
    cache_file_status,
    merge_sina_with_tushare_cache,
)

DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics"
CHECK_JSON = DIAG_DIR / "merged_reference_source_check.json"
SAMPLE_CSV = DIAG_DIR / "merged_reference_sample.csv"


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def fetch_sina_spot() -> tuple[pd.DataFrame, str | None]:
    try:
        import akshare as ak
    except Exception as exc:
        return pd.DataFrame(), f"akshare import failed: {type(exc).__name__}: {exc}"

    try:
        df = ak.stock_zh_a_spot()
        if df is None or df.empty:
            return pd.DataFrame(), "ak.stock_zh_a_spot() returned empty data"
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"ak.stock_zh_a_spot() failed: {type(exc).__name__}: {exc}"


def has_non_empty_column(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def main() -> int:
    parser = argparse.ArgumentParser(description="Test merged Sina realtime + Tushare cache reference source.")
    parser.add_argument("--trade-date", default="20260330", help="Tushare cache trade date in YYYYMMDD format.")
    args = parser.parse_args()

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    cache_status = cache_file_status(args.trade_date)
    sina_df, sina_error = fetch_sina_spot()
    merged = pd.DataFrame()
    merge_error = None

    if sina_error is None:
        try:
            merged = merge_sina_with_tushare_cache(sina_df, args.trade_date)
        except Exception as exc:
            merge_error = f"{type(exc).__name__}: {exc}"

    if not merged.empty:
        merged.head(200).to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")

    result: Dict[str, Any] = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": args.trade_date,
        "cache_status": cache_status,
        "sina_available": sina_error is None,
        "sina_rows": int(len(sina_df)),
        "sina_error": sina_error,
        "merge_success": merge_error is None and not merged.empty,
        "merge_error": merge_error,
        "merged_rows": int(len(merged)),
        "merged_columns": list(merged.columns) if not merged.empty else [],
        "has_realtime_price": has_non_empty_column(merged, "price"),
        "has_realtime_pct_chg": has_non_empty_column(merged, "pct_chg"),
        "has_realtime_amount": has_non_empty_column(merged, "amount"),
        "has_turnover_rate_ref": has_non_empty_column(merged, "turnover_rate_ref"),
        "has_volume_ratio_ref": has_non_empty_column(merged, "volume_ratio_ref"),
        "has_daily_ohlc": all(has_non_empty_column(merged, col) for col in ["open", "high", "low", "close"]),
        "sample_saved": SAMPLE_CSV.exists(),
        "is_realtime_turnover": bool(merged["is_realtime_turnover"].any()) if "is_realtime_turnover" in merged.columns else False,
        "is_realtime_volume_ratio": bool(merged["is_realtime_volume_ratio"].any()) if "is_realtime_volume_ratio" in merged.columns else False,
        "allow_official_strategy_candidate": bool(merged["allow_official_strategy_candidate"].any())
        if "allow_official_strategy_candidate" in merged.columns
        else False,
        "allow_reference_candidate": bool(merged["allow_reference_candidate"].any())
        if "allow_reference_candidate" in merged.columns
        else False,
        "warning_message": merged["warning_message"].iloc[0] if not merged.empty and "warning_message" in merged.columns else "",
    }

    CHECK_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    print("新浪实时行情 + Tushare 本地缓存融合测试")
    print(f"Tushare 缓存状态: {cache_status}")
    print(f"新浪实时行情是否可用: {'是' if result['sina_available'] else '否'}")
    print(f"新浪返回行数: {result['sina_rows']}")
    if sina_error:
        print(f"新浪失败原因: {sina_error}")
    print(f"是否合并成功: {'是' if result['merge_success'] else '否'}")
    if merge_error:
        print(f"合并失败原因: {merge_error}")
    print(f"合并后行数: {result['merged_rows']}")
    print(f"字段列表: {result['merged_columns']}")
    print(f"包含实时价格: {'是' if result['has_realtime_price'] else '否'}")
    print(f"包含实时涨跌幅: {'是' if result['has_realtime_pct_chg'] else '否'}")
    print(f"包含实时成交额: {'是' if result['has_realtime_amount'] else '否'}")
    print(f"包含参考换手率: {'是' if result['has_turnover_rate_ref'] else '否'}")
    print(f"包含参考量比: {'是' if result['has_volume_ratio_ref'] else '否'}")
    print(f"包含日 K 开高低收: {'是' if result['has_daily_ohlc'] else '否'}")
    print(f"换手率是否实时: {'是' if result['is_realtime_turnover'] else '否'}")
    print(f"量比是否实时: {'是' if result['is_realtime_volume_ratio'] else '否'}")
    print(f"是否允许正式策略候选: {'是' if result['allow_official_strategy_candidate'] else '否'}")
    print(f"是否允许参考候选: {'是' if result['allow_reference_candidate'] else '否'}")
    if not merged.empty:
        print("前 10 行样例:")
        print(merged.head(10).to_string(index=False))
    print(f"诊断结果已保存: {CHECK_JSON}")
    print(f"样例已保存: {SAMPLE_CSV if SAMPLE_CSV.exists() else '未保存'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
