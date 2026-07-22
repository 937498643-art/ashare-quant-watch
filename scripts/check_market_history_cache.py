"""Validate historical market-environment cache coverage and write a Markdown report."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = PROJECT_ROOT / "data" / "history" / "market"
REPORT_PATH = PROJECT_ROOT / "market_history_check.md"
REQUIRED_FIELDS = (
    "trade_date", "sh_close", "sh_pct_chg", "sh_ma5", "sh_ma10",
    "up_count", "down_count", "limit_up_count", "limit_down_count", "market_amount",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local historical market cache")
    parser.add_argument("--market-dir", type=Path, default=MARKET_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    files = sorted(args.market_dir.glob("market_*.csv"))
    records: list[dict[str, object]] = []
    for path in files:
        try:
            data = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            records.append({"file": path.name, "readable": False, "error": type(exc).__name__})
            continue
        has_schema = len(data) == 1 and set(REQUIRED_FIELDS).issubset(data.columns)
        row = data.iloc[0] if len(data) else pd.Series(dtype=object)
        records.append({
            "file": path.name,
            "readable": True,
            "schema_complete": has_schema,
            "trade_date": str(row.get("trade_date", "")),
            "index_complete": bool(has_schema and row[["sh_close", "sh_pct_chg", "sh_ma5", "sh_ma10"]].notna().all()),
            "breadth_complete": bool(row.get("breadth_complete", False)),
            "field_complete": int(row[list(REQUIRED_FIELDS)].notna().sum()) if has_schema else 0,
        })

    frame = pd.DataFrame(records)
    readable = int(frame.get("readable", pd.Series(dtype=bool)).sum()) if not frame.empty else 0
    schema_ok = int(frame.get("schema_complete", pd.Series(dtype=bool)).sum()) if not frame.empty else 0
    index_ok = int(frame.get("index_complete", pd.Series(dtype=bool)).sum()) if not frame.empty else 0
    breadth_ok = int(frame.get("breadth_complete", pd.Series(dtype=bool)).sum()) if not frame.empty else 0
    all_field_total = int(frame.get("field_complete", pd.Series(dtype=int)).sum()) if not frame.empty else 0
    field_total = len(files) * len(REQUIRED_FIELDS)
    field_rate = all_field_total / field_total if field_total else 0.0
    dates = sorted(frame.get("trade_date", pd.Series(dtype=str)).dropna().astype(str).tolist()) if not frame.empty else []

    lines = [
        "# 历史市场环境缓存检查", "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "## 覆盖范围", "",
        f"- 缓存目录：`{args.market_dir}`",
        f"- 文件数量：{len(files)}",
        f"- 日期范围：{dates[0] if dates else '--'} 至 {dates[-1] if dates else '--'}", "",
        "## 完整性", "",
        f"- 可读取文件：{readable}/{len(files)}",
        f"- 字段结构完整：{schema_ok}/{len(files)}",
        f"- 上证指数字段完整：{index_ok}/{len(files)}",
        f"- 全市场宽度完整：{breadth_ok}/{len(files)}",
        f"- 必需字段完整率：{field_rate:.2%}", "",
        "## 说明", "",
        "- 市场宽度仅在本地日线覆盖达到构建门槛时写入。覆盖不足的日期保持缺失，不使用回测股票子集伪造全市场上涨/下跌与涨跌停家数。",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
