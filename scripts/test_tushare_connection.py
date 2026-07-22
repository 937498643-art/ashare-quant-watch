"""Read-only Tushare Pro connection diagnosis.

This script only tests whether Tushare Pro can be imported, initialized with a
token from .env, and used to read market/basic data. It does not print the full
token, store credentials in code, connect to trading systems, or perform any
trading-related action.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_DIR = PROJECT_ROOT / "data" / "diagnostics"
REPORT_PATH = DIAGNOSTIC_DIR / "tushare_connection_check.json"
DAILY_SAMPLE_PATH = DIAGNOSTIC_DIR / "tushare_daily_sample.csv"
DAILY_BASIC_SAMPLE_PATH = DIAGNOSTIC_DIR / "tushare_daily_basic_sample.csv"
STOCK_BASIC_SAMPLE_PATH = DIAGNOSTIC_DIR / "tushare_stock_basic_sample.csv"

TEST_STOCKS = ["000001.SZ", "600519.SH", "300059.SZ"]
TEST_INDEXES = ["000001.SH", "399001.SZ"]
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,vol,amount"
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,volume_ratio,total_mv,circ_mv"
STOCK_BASIC_FIELDS = "ts_code,symbol,name,area,industry,market,list_date"


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def base_report() -> dict[str, Any]:
    return {
        "check_time": datetime.now().isoformat(timespec="seconds"),
        "has_tushare": False,
        "has_env": False,
        "has_token": False,
        "pro_api_success": False,
        "stock_basic_success": False,
        "daily_success": False,
        "daily_basic_success": False,
        "index_daily_success": False,
        "daily_rows": 0,
        "daily_basic_rows": 0,
        "stock_basic_rows": 0,
        "index_daily_rows": 0,
        "daily_columns": [],
        "daily_basic_columns": [],
        "stock_basic_columns": [],
        "index_daily_columns": [],
        "has_ohlc": False,
        "has_volume": False,
        "has_amount": False,
        "has_turnover_rate": False,
        "has_volume_ratio": False,
        "error_message": "",
        "recommended_next_step": "",
        "interface_results": {},
    }


def append_error(report: dict[str, Any], message: str) -> None:
    if not message:
        return
    current = report.get("error_message") or ""
    report["error_message"] = f"{current}; {message}" if current else message


def save_report(report: dict[str, Any]) -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_token(report: dict[str, Any]) -> str:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=True, encoding="utf-8-sig")
        except Exception as exc:
            append_error(report, f"读取 .env 失败: {type(exc).__name__}: {exc}")
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    report["has_env"] = env_path.exists()
    report["has_token"] = bool(token)
    return token


def safe_token_label(token: str) -> str:
    """Return a masked token label for diagnostics."""
    if not token:
        return ""
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}***{token[-4:]}"


def detect_fields(frame: pd.DataFrame) -> dict[str, bool]:
    columns = set(map(str, frame.columns)) if isinstance(frame, pd.DataFrame) else set()
    return {
        "has_ohlc": all(field in columns for field in ["open", "high", "low", "close"]),
        "has_volume": "vol" in columns or "volume" in columns,
        "has_amount": "amount" in columns or "money" in columns,
        "has_turnover_rate": "turnover_rate" in columns,
        "has_volume_ratio": "volume_ratio" in columns,
    }


def summarize_frame(name: str, frame: pd.DataFrame, error: str | None = None) -> dict[str, Any]:
    fields = detect_fields(frame) if isinstance(frame, pd.DataFrame) else detect_fields(pd.DataFrame())
    return {
        "interface": name,
        "success": isinstance(frame, pd.DataFrame) and not frame.empty,
        "rows": int(len(frame)) if isinstance(frame, pd.DataFrame) else 0,
        "columns": list(map(str, frame.columns)) if isinstance(frame, pd.DataFrame) else [],
        **fields,
        "error": error,
    }


def fetch_daily(pro: Any, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    return pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=DAILY_FIELDS)


def fetch_daily_basic(pro: Any, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    return pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=DAILY_BASIC_FIELDS)


def concat_interface_results(fetcher: Any, pro: Any, codes: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for code in codes:
        try:
            frame = fetcher(pro, code, start_date, end_date)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{code}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame(), errors
    return pd.concat(frames, ignore_index=True), errors


def main() -> int:
    configure_stdout()
    report = base_report()
    print("Tushare Pro 本地连接诊断开始")

    try:
        import tushare as ts

        report["has_tushare"] = True
        print("是否安装 tushare: 是")
    except Exception as exc:
        append_error(report, f"tushare 未安装或导入失败: {type(exc).__name__}: {exc}")
        report["recommended_next_step"] = r"请运行：.\.venv\Scripts\python.exe -m pip install tushare python-dotenv"
        save_report(report)
        print("是否安装 tushare: 否")
        print(report["recommended_next_step"])
        print(f"诊断报告已保存: {REPORT_PATH}")
        return 0

    token = load_token(report)
    print(f"是否读取到 .env: {'是' if report['has_env'] else '否'}")
    print(f"是否读取到 TUSHARE_TOKEN: {'是' if report['has_token'] else '否'}")
    if token:
        print(f"Token 标识: {safe_token_label(token)}")
    if not token:
        report["recommended_next_step"] = "请在项目根目录创建 .env，并填写 TUSHARE_TOKEN"
        save_report(report)
        print(report["recommended_next_step"])
        print(f"诊断报告已保存: {REPORT_PATH}")
        return 0

    try:
        pro = ts.pro_api(token)
        report["pro_api_success"] = True
        print("是否成功初始化 pro_api: 是")
    except Exception as exc:
        append_error(report, f"初始化 pro_api 失败: {type(exc).__name__}: {exc}")
        report["recommended_next_step"] = "请检查 TUSHARE_TOKEN 是否正确或账号权限是否可用。"
        save_report(report)
        print("是否成功初始化 pro_api: 否")
        print(f"失败原因: {report['error_message']}")
        print(f"诊断报告已保存: {REPORT_PATH}")
        return 0

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    try:
        stock_basic = pro.stock_basic(exchange="", list_status="L", fields=STOCK_BASIC_FIELDS)
        stock_result = summarize_frame("stock_basic", stock_basic)
        report["stock_basic_success"] = stock_result["success"]
        report["stock_basic_rows"] = stock_result["rows"]
        report["stock_basic_columns"] = stock_result["columns"]
        report["interface_results"]["stock_basic"] = stock_result
        if stock_result["success"]:
            stock_basic.head(80).to_csv(STOCK_BASIC_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        append_error(report, f"stock_basic 失败: {error}")
        report["interface_results"]["stock_basic"] = summarize_frame("stock_basic", pd.DataFrame(), error)

    daily, daily_errors = concat_interface_results(fetch_daily, pro, TEST_STOCKS, start_date, end_date)
    daily_error = "; ".join(daily_errors) if daily_errors else None
    daily_result = summarize_frame("daily", daily, daily_error)
    report["daily_success"] = daily_result["success"]
    report["daily_rows"] = daily_result["rows"]
    report["daily_columns"] = daily_result["columns"]
    report["has_ohlc"] = daily_result["has_ohlc"]
    report["has_volume"] = daily_result["has_volume"]
    report["has_amount"] = daily_result["has_amount"]
    report["interface_results"]["daily"] = daily_result
    if daily_error:
        append_error(report, f"daily 部分或全部失败: {daily_error}")
    if daily_result["success"]:
        daily.head(120).to_csv(DAILY_SAMPLE_PATH, index=False, encoding="utf-8-sig")

    daily_basic, daily_basic_errors = concat_interface_results(fetch_daily_basic, pro, TEST_STOCKS, start_date, end_date)
    daily_basic_error = "; ".join(daily_basic_errors) if daily_basic_errors else None
    daily_basic_result = summarize_frame("daily_basic", daily_basic, daily_basic_error)
    report["daily_basic_success"] = daily_basic_result["success"]
    report["daily_basic_rows"] = daily_basic_result["rows"]
    report["daily_basic_columns"] = daily_basic_result["columns"]
    report["has_turnover_rate"] = daily_basic_result["has_turnover_rate"]
    report["has_volume_ratio"] = daily_basic_result["has_volume_ratio"]
    report["interface_results"]["daily_basic"] = daily_basic_result
    if daily_basic_error:
        append_error(report, f"daily_basic 部分或全部失败: {daily_basic_error}")
    if daily_basic_result["success"]:
        daily_basic.head(120).to_csv(DAILY_BASIC_SAMPLE_PATH, index=False, encoding="utf-8-sig")

    index_frames: list[pd.DataFrame] = []
    index_errors: list[str] = []
    for code in TEST_INDEXES:
        try:
            frame = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                index_frames.append(frame)
        except Exception as exc:
            index_errors.append(f"{code}: {type(exc).__name__}: {exc}")
    index_daily = pd.concat(index_frames, ignore_index=True) if index_frames else pd.DataFrame()
    index_error = "; ".join(index_errors) if index_errors else None
    index_result = summarize_frame("index_daily", index_daily, index_error)
    report["index_daily_success"] = index_result["success"]
    report["index_daily_rows"] = index_result["rows"]
    report["index_daily_columns"] = index_result["columns"]
    report["interface_results"]["index_daily"] = index_result
    if index_error:
        append_error(report, f"index_daily 部分或全部失败: {index_error}")

    report["recommended_next_step"] = (
        "Tushare Pro 日线和每日指标可用，可进一步评估作为历史 K 线、换手率和量比补充源。"
        if report["daily_success"] and report["daily_basic_success"]
        else "Tushare Pro 已初始化，但部分接口不可用；请检查 token 权限、积分或接口调用限制。"
    )
    save_report(report)

    print("\n接口结果")
    for key in ["stock_basic", "daily", "daily_basic", "index_daily"]:
        item = report["interface_results"].get(key, {})
        print(f"{key}: 成功={'是' if item.get('success') else '否'} 行数={item.get('rows', 0)} 字段={item.get('columns', [])}")
        print(f"  开高低收={'是' if item.get('has_ohlc') else '否'} 成交量={'是' if item.get('has_volume') else '否'} 成交额={'是' if item.get('has_amount') else '否'} 换手率={'是' if item.get('has_turnover_rate') else '否'} 量比={'是' if item.get('has_volume_ratio') else '否'}")
        if item.get("error"):
            print(f"  失败原因: {item.get('error')}")

    print("\n诊断总结")
    print(f"stock_basic 是否成功: {'是' if report['stock_basic_success'] else '否'}")
    print(f"daily 是否成功: {'是' if report['daily_success'] else '否'}")
    print(f"daily_basic 是否成功: {'是' if report['daily_basic_success'] else '否'}")
    print(f"daily_basic 是否包含换手率 turnover_rate: {'是' if report['has_turnover_rate'] else '否'}")
    print(f"daily_basic 是否包含量比 volume_ratio: {'是' if report['has_volume_ratio'] else '否'}")
    print(f"是否保存 daily 样例: {'是' if DAILY_SAMPLE_PATH.exists() else '否'}")
    print(f"是否保存 daily_basic 样例: {'是' if DAILY_BASIC_SAMPLE_PATH.exists() else '否'}")
    print(f"是否保存 stock_basic 样例: {'是' if STOCK_BASIC_SAMPLE_PATH.exists() else '否'}")
    print(f"诊断报告已保存: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
