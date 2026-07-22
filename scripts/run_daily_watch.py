"""Daily read-only watch runner and diagnostics.

This script checks public data/cache readiness, runs ``main.py --once``, and
writes a local runtime report. It does not connect to broker software, click
external applications, or provide any execution capability.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
REPORT_PATH = RUNTIME_DIR / "daily_watch_run_report.json"
ENV_PATH = PROJECT_ROOT / ".env"
TUSHARE_CACHE_DIR = DATA_DIR / "cache" / "tushare"
TUSHARE_STATUS_PATH = TUSHARE_CACHE_DIR / "tushare_cache_status.json"
LATEST_STATUS_PATH = DATA_DIR / "latest_market_status.json"
LATEST_CANDIDATES_PATH = DATA_DIR / "latest_candidates.csv"
REFERENCE_CANDIDATES_PATH = DATA_DIR / "output" / "reference_candidates.csv"
USER_WATCHLIST_PATH = DATA_DIR / "user" / "watchlist.csv"
USER_HOLDINGS_PATH = DATA_DIR / "user" / "holdings.csv"


def read_env_value(key: str) -> str:
    if not ENV_PATH.exists():
        return ""
    try:
        for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == key:
                return value.strip()
    except Exception:
        return ""
    return os.getenv(key, "")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def read_csv_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def latest_file(pattern: str, directory: Path) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def check_sina_source() -> dict[str, Any]:
    result = {"available": False, "rows": 0, "error": ""}
    try:
        import akshare as ak

        data = ak.stock_zh_a_spot()
        result["available"] = data is not None and not data.empty
        result["rows"] = int(len(data)) if data is not None else 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_main_once() -> dict[str, Any]:
    command = [sys.executable, "main.py", "--once"]
    result = {
        "command": " ".join(command),
        "success": False,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "error": "",
    }
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=360,
        )
        result["returncode"] = completed.returncode
        result["success"] = completed.returncode == 0
        result["stdout_tail"] = completed.stdout[-4000:]
        result["stderr_tail"] = completed.stderr[-4000:]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_report(main_result: dict[str, Any], sina_result: dict[str, Any]) -> dict[str, Any]:
    latest_status = load_json(LATEST_STATUS_PATH)
    tushare_status = load_json(TUSHARE_STATUS_PATH)
    latest_daily = latest_file("daily_*.csv", TUSHARE_CACHE_DIR)
    latest_daily_basic = latest_file("daily_basic_*.csv", TUSHARE_CACHE_DIR)
    candidates = load_candidates_summary()

    has_token = bool(read_env_value("TUSHARE_TOKEN"))
    issues = []
    if not ENV_PATH.exists():
        issues.append(".env 不存在，Tushare 缓存脚本无法读取 token。")
    elif not has_token:
        issues.append(".env 中未配置 TUSHARE_TOKEN。")
    if not sina_result.get("available"):
        issues.append("新浪实时行情不可用。")
    if latest_daily is None:
        issues.append("未找到 Tushare daily 缓存。")
    if latest_daily_basic is None:
        issues.append("未找到 Tushare daily_basic 缓存，参考换手率和参考量比会缺失。")
    if not main_result.get("success"):
        issues.append("main.py --once 运行失败，请查看 stdout_tail / stderr_tail。")
    if int(candidates.get("reference_candidate_count") or 0) == 0:
        issues.append("当前参考候选股数量为 0。")

    return {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "env_exists": ENV_PATH.exists(),
        "has_tushare_token": has_token,
        "tushare_token_hint": "已配置" if has_token else "未配置",
        "tushare_cache_dir_exists": TUSHARE_CACHE_DIR.exists(),
        "tushare_cache_status": tushare_status,
        "latest_tushare_daily_cache": str(latest_daily) if latest_daily else "",
        "latest_tushare_daily_basic_cache": str(latest_daily_basic) if latest_daily_basic else "",
        "has_tushare_daily_cache": latest_daily is not None,
        "has_tushare_daily_basic_cache": latest_daily_basic is not None,
        "sina_source": sina_result,
        "main_once": main_result,
        "latest_market_status": latest_status,
        "current_data_source": latest_status.get("data_source", "--"),
        "strategy_candidate_count": int(latest_status.get("strategy_candidate_count") or 0),
        "active_watchlist_count": int(latest_status.get("active_watchlist_count") or 0),
        "reference_candidate_count": int(candidates.get("reference_candidate_count") or 0),
        "latest_candidates_count": int(candidates.get("latest_candidates_count") or 0),
        "watchlist_count": read_csv_count(USER_WATCHLIST_PATH),
        "holdings_count": read_csv_count(USER_HOLDINGS_PATH),
        "issues": issues,
    }


def load_candidates_summary() -> dict[str, int]:
    latest_count = read_csv_count(LATEST_CANDIDATES_PATH)
    reference_count = read_csv_count(REFERENCE_CANDIDATES_PATH)
    return {
        "latest_candidates_count": latest_count,
        "reference_candidate_count": reference_count,
    }


def save_report(report: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print("\n日常盯盘运行总结")
    print("=" * 60)
    print(f"项目路径: {report['project_root']}")
    print(f".env: {'存在' if report['env_exists'] else '不存在'}")
    print(f"TUSHARE_TOKEN: {report['tushare_token_hint']}")
    print(f"当前数据源: {report['current_data_source']}")
    print(f"新浪实时行情: {'可用' if report['sina_source'].get('available') else '不可用'}，行数: {report['sina_source'].get('rows', 0)}")
    print(f"Tushare daily 缓存: {'有' if report['has_tushare_daily_cache'] else '无'}")
    print(f"Tushare daily_basic 缓存: {'有' if report['has_tushare_daily_basic_cache'] else '无'}")
    print(f"main.py --once: {'成功' if report['main_once'].get('success') else '失败'}")
    print(f"正式策略候选股数量: {report['strategy_candidate_count']}")
    print(f"参考候选股数量: {report['reference_candidate_count']}")
    print(f"活跃观察池数量: {report['active_watchlist_count']}")
    print(f"自选股数量: {report['watchlist_count']}")
    print(f"持仓股数量: {report['holdings_count']}")
    if report["issues"]:
        print("\n需要注意:")
        for item in report["issues"]:
            print(f"- {item}")
    else:
        print("\n当前未发现明显阻塞问题。")
    print(f"\n运行报告已保存: {REPORT_PATH}")


def main() -> int:
    print("开始日常盯盘检查...")
    print(f"项目路径: {PROJECT_ROOT}")
    print("检查新浪实时行情...")
    sina_result = check_sina_source()
    print("运行 main.py --once...")
    main_result = run_main_once()
    report = build_report(main_result, sina_result)
    save_report(report)
    print_summary(report)
    return 0 if main_result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
