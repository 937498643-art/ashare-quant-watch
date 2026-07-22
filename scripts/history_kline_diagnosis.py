"""Diagnose public historical K-line data sources.

This script only probes public historical market data APIs. It does not modify
the dashboard, main process, strategy logic, broker software, accounts, or any
trading-related system.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIAGNOSTIC_DIR = PROJECT_ROOT / "data" / "diagnostics"
REPORT_PATH = DIAGNOSTIC_DIR / "history_kline_diagnosis.json"
SAMPLE_PATH = DIAGNOSTIC_DIR / "kline_sample.csv"

TEST_SYMBOLS = [
    {"code": "600584", "name": "长电科技"},
    {"code": "603019", "name": "中科曙光"},
    {"code": "300059", "name": "东方财富"},
    {"code": "000725", "name": "京东方A"},
    {"code": "002281", "name": "光迅科技"},
]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def normalize_column(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def has_any_column(columns: list[Any], aliases: list[str]) -> bool:
    normalized = {normalize_column(column) for column in columns}
    return any(normalize_column(alias) in normalized for alias in aliases)


def detect_kline_fields(frame: pd.DataFrame) -> dict[str, bool]:
    columns = list(frame.columns) if isinstance(frame, pd.DataFrame) else []
    return {
        "has_date": has_any_column(columns, ["date", "日期", "trade_date"]),
        "has_open": has_any_column(columns, ["open", "开盘", "开盘价"]),
        "has_high": has_any_column(columns, ["high", "最高", "最高价"]),
        "has_low": has_any_column(columns, ["low", "最低", "最低价"]),
        "has_close": has_any_column(columns, ["close", "收盘", "收盘价"]),
        "has_volume": has_any_column(columns, ["volume", "成交量"]),
        "has_amount": has_any_column(columns, ["amount", "成交额"]),
        "has_turnover": has_any_column(columns, ["turnover", "换手率", "换手"]),
    }


def has_ohlc(fields: dict[str, bool]) -> bool:
    return all(fields.get(key) for key in ["has_date", "has_open", "has_high", "has_low", "has_close"])


def test_source(source_name: str, fetcher: Callable[[str], pd.DataFrame]) -> dict[str, Any]:
    source_result = {
        "source": source_name,
        "success": False,
        "success_symbols": [],
        "tests": [],
    }
    for item in TEST_SYMBOLS:
        code = item["code"]
        result = {
            "code": code,
            "name": item["name"],
            "success": False,
            "rows": 0,
            "columns": [],
            "has_date": False,
            "has_open": False,
            "has_high": False,
            "has_low": False,
            "has_close": False,
            "has_volume": False,
            "has_amount": False,
            "has_turnover": False,
            "error": None,
        }
        try:
            frame = fetcher(code)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"返回类型不是 DataFrame: {type(frame).__name__}")
            fields = detect_kline_fields(frame)
            result.update(
                {
                    "success": not frame.empty,
                    "rows": int(len(frame)),
                    "columns": list(frame.columns),
                    **fields,
                }
            )
            if not frame.empty:
                source_result["success"] = True
                source_result["success_symbols"].append(code)
                if not SAMPLE_PATH.exists():
                    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
                    frame.head(80).to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        source_result["tests"].append(result)
    return source_result


def fetch_project_history(code: str) -> pd.DataFrame:
    from data_sources.akshare_source import AkshareSource

    return AkshareSource({"retry_count": 1, "request_interval_seconds": 0.2, "timeout": 8}).fetch_history(code, days=60)


def fetch_ak_stock_zh_a_hist(code: str) -> pd.DataFrame:
    import akshare as ak

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
    return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="")


def fetch_ak_stock_zh_a_daily(code: str) -> pd.DataFrame:
    import akshare as ak

    market_code = f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}"
    return ak.stock_zh_a_daily(symbol=market_code, adjust="")


def choose_best_source(results: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    best_score = -1
    for source in results:
        if not source["success"]:
            continue
        tests = source.get("tests") or []
        success_tests = [item for item in tests if item.get("success")]
        if not success_tests:
            continue
        field_score = 0
        first = success_tests[0]
        if has_ohlc(first):
            field_score += 5
        if first.get("has_volume"):
            field_score += 1
        if first.get("has_amount"):
            field_score += 1
        if first.get("has_turnover"):
            field_score += 1
        score = len(success_tests) * 10 + field_score
        if score > best_score:
            best_score = score
            best = source
    if best is None:
        return {
            "best_source": None,
            "best_source_has_ohlc": False,
            "best_source_has_amount": False,
            "best_source_has_turnover": False,
        }
    first_success = next(item for item in best["tests"] if item.get("success"))
    return {
        "best_source": best["source"],
        "best_source_has_ohlc": has_ohlc(first_success),
        "best_source_has_amount": bool(first_success.get("has_amount")),
        "best_source_has_turnover": bool(first_success.get("has_turnover")),
    }


def main() -> int:
    configure_stdout()
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_PATH.exists():
        SAMPLE_PATH.unlink()

    print("历史 K 线公开数据源诊断开始")
    sources = [
        ("项目已有 AkshareSource.fetch_history", fetch_project_history),
        ("AKShare stock_zh_a_hist", fetch_ak_stock_zh_a_hist),
        ("AKShare stock_zh_a_daily", fetch_ak_stock_zh_a_daily),
    ]
    results = []
    for source_name, fetcher in sources:
        print(f"正在测试: {source_name}")
        result = test_source(source_name, fetcher)
        results.append(result)
        print(f"- 是否成功: {'是' if result['success'] else '否'}")

    available_sources = [item["source"] for item in results if item["success"]]
    best = choose_best_source(results)
    recommended = (
        f"建议优先使用 {best['best_source']} 作为个股详情 K 线源。"
        if best["best_source"]
        else "当前未找到可用公开历史 K 线源，建议优先排查网络/SSL 或增加其它公开历史源。"
    )
    report = {
        "diagnose_time": datetime.now().isoformat(timespec="seconds"),
        "tested_symbols": TEST_SYMBOLS,
        "available_sources": available_sources,
        **best,
        "recommended_next_step": recommended,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n诊断总结")
    print(f"可用历史 K 线源: {' / '.join(available_sources) if available_sources else '暂无'}")
    print(f"最佳数据源: {best['best_source'] or '暂无'}")
    print(f"包含开高低收: {'是' if best['best_source_has_ohlc'] else '否'}")
    print(f"包含成交额: {'是' if best['best_source_has_amount'] else '否'}")
    print(f"包含换手率: {'是' if best['best_source_has_turnover'] else '否'}")
    print(f"是否保存样例: {'是' if SAMPLE_PATH.exists() else '否'}")
    print(f"诊断报告: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
