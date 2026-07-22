"""Read-only JQData connection diagnosis.

This script only tests whether JQData can be imported, authenticated, and used
to read market/basic data. It does not save real credentials, print passwords,
place orders, or connect to any trading system.
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
REPORT_PATH = DIAGNOSTIC_DIR / "jqdata_connection_check.json"
SAMPLE_PATH = DIAGNOSTIC_DIR / "jqdata_kline_sample.csv"

TEST_SINGLE_SECURITY = "000001.XSHE"
TEST_SECURITIES = ["000001.XSHE", "600519.XSHG", "300059.XSHE"]
KLINE_FIELDS = ["open", "close", "high", "low", "volume", "money"]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def base_report() -> dict[str, Any]:
    return {
        "check_time": datetime.now().isoformat(timespec="seconds"),
        "has_jqdatasdk": False,
        "has_env": False,
        "auth_success": False,
        "query_count_success": False,
        "query_count": None,
        "daily_kline_success": False,
        "daily_kline_rows": 0,
        "daily_kline_columns": [],
        "multi_kline_success": False,
        "multi_kline_rows": 0,
        "has_ohlc": False,
        "has_volume": False,
        "has_amount": False,
        "security_info_success": False,
        "error_message": "",
        "recommended_next_step": "",
    }


def save_report(report: dict[str, Any]) -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_error(report: dict[str, Any], message: str) -> None:
    if not message:
        return
    current = report.get("error_message") or ""
    report["error_message"] = f"{current}; {message}" if current else message


def load_env(report: dict[str, Any]) -> tuple[str, str]:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except Exception as exc:
            append_error(report, f"读取 .env 失败: {type(exc).__name__}: {exc}")
    user = os.getenv("JQDATA_USER", "").strip()
    password = os.getenv("JQDATA_PASSWORD", "").strip()
    report["has_env"] = bool(env_path.exists() and user and password)
    return user, password


def detect_fields(frame: pd.DataFrame) -> dict[str, bool]:
    columns = set(map(str, frame.columns)) if isinstance(frame, pd.DataFrame) else set()
    return {
        "has_ohlc": all(field in columns for field in ["open", "close", "high", "low"]),
        "has_volume": "volume" in columns,
        "has_amount": "money" in columns or "amount" in columns,
    }


def main() -> int:
    configure_stdout()
    report = base_report()
    print("JQData 本地连接诊断开始")

    try:
        import jqdatasdk as jq

        report["has_jqdatasdk"] = True
        print("是否安装 jqdatasdk: 是")
    except Exception as exc:
        append_error(report, f"jqdatasdk 未安装或导入失败: {type(exc).__name__}: {exc}")
        report["recommended_next_step"] = r"请运行：.\.venv\Scripts\python.exe -m pip install jqdatasdk python-dotenv"
        save_report(report)
        print("是否安装 jqdatasdk: 否")
        print(report["recommended_next_step"])
        print(f"诊断报告已保存: {REPORT_PATH}")
        return 0

    user, password = load_env(report)
    print(f"是否读取到 .env: {'是' if report['has_env'] else '否'}")
    if not user or not password:
        report["recommended_next_step"] = "请在项目根目录创建 .env，并填写 JQDATA_USER 和 JQDATA_PASSWORD"
        save_report(report)
        print(report["recommended_next_step"])
        print(f"诊断报告已保存: {REPORT_PATH}")
        return 0

    try:
        jq.auth(user, password)
        report["auth_success"] = True
        print("是否认证成功: 是")
    except Exception as exc:
        append_error(report, f"JQData 认证失败: {type(exc).__name__}: {exc}")
        report["recommended_next_step"] = "请检查 .env 中的 JQDATA_USER / JQDATA_PASSWORD，或确认账号权限。"
        save_report(report)
        print("是否认证成功: 否")
        print(f"失败原因: {report['error_message']}")
        print(f"诊断报告已保存: {REPORT_PATH}")
        return 0

    try:
        if hasattr(jq, "get_query_count"):
            report["query_count"] = jq.get_query_count()
            report["query_count_success"] = True
            print(f"账号调用次数信息: {report['query_count']}")
        else:
            print("账号调用次数信息: 当前 jqdatasdk 未提供 get_query_count，已跳过")
    except Exception as exc:
        append_error(report, f"查询调用次数失败: {type(exc).__name__}: {exc}")
        print("账号调用次数信息: 查询失败，已跳过")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)
    try:
        kline = jq.get_price(
            TEST_SINGLE_SECURITY,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=KLINE_FIELDS,
            skip_paused=True,
        )
        if isinstance(kline, pd.DataFrame) and not kline.empty:
            fields = detect_fields(kline)
            report.update(
                {
                    "daily_kline_success": True,
                    "daily_kline_rows": int(len(kline)),
                    "daily_kline_columns": list(map(str, kline.columns)),
                    **fields,
                }
            )
            DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
            kline.head(80).to_csv(SAMPLE_PATH, encoding="utf-8-sig")
        print(f"是否能获取日 K: {'是' if report['daily_kline_success'] else '否'}")
    except Exception as exc:
        append_error(report, f"获取单只股票日 K 失败: {type(exc).__name__}: {exc}")
        print("是否能获取日 K: 否")

    try:
        multi = jq.get_price(
            TEST_SECURITIES,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=KLINE_FIELDS,
            skip_paused=True,
            panel=False,
        )
        report["multi_kline_success"] = isinstance(multi, pd.DataFrame) and not multi.empty
        report["multi_kline_rows"] = int(len(multi)) if isinstance(multi, pd.DataFrame) else 0
        print(f"是否能获取多只股票日 K: {'是' if report['multi_kline_success'] else '否'}")
    except Exception as exc:
        append_error(report, f"获取多只股票日 K 失败: {type(exc).__name__}: {exc}")
        print("是否能获取多只股票日 K: 否")

    try:
        if hasattr(jq, "get_security_info"):
            info = jq.get_security_info(TEST_SINGLE_SECURITY)
            report["security_info_success"] = info is not None
        elif hasattr(jq, "get_all_securities"):
            securities = jq.get_all_securities(types=["stock"])
            report["security_info_success"] = isinstance(securities, pd.DataFrame) and not securities.empty
        print(f"是否能获取股票基础信息: {'是' if report['security_info_success'] else '否'}")
    except Exception as exc:
        append_error(report, f"获取股票基础信息失败: {type(exc).__name__}: {exc}")
        print("是否能获取股票基础信息: 否")

    report["recommended_next_step"] = (
        "JQData 日 K 可用，可进一步评估作为历史 K 线和基础数据源。"
        if report["daily_kline_success"]
        else "JQData 已认证但日 K 暂不可用，请检查账号权限、网络或接口参数。"
    )
    save_report(report)

    print("\n诊断总结")
    print(f"日 K 返回行数: {report['daily_kline_rows']}")
    print(f"日 K 字段列表: {report['daily_kline_columns']}")
    print(f"是否包含开高低收: {'是' if report['has_ohlc'] else '否'}")
    print(f"是否包含成交量: {'是' if report['has_volume'] else '否'}")
    print(f"是否包含成交额: {'是' if report['has_amount'] else '否'}")
    print(f"是否保存样例: {'是' if SAMPLE_PATH.exists() else '否'}")
    print(f"诊断报告已保存: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
