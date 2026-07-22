"""Manual Eastmoney push2 connectivity probe.

This script only reads public quote data from Eastmoney push2. It does not
modify project data, place orders, access accounts, or connect to brokers.

Run from project root:
    .\.venv\Scripts\python.exe scripts\manual_push2_test.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pandas as pd
import requests


HOSTS = [
    "https://push2.eastmoney.com/api/qt/clist/get",
    "https://82.push2.eastmoney.com/api/qt/clist/get",
]

MODES = {
    "direct": False,
    "system": True,
}

PARAMS = {
    "pn": 1,
    "pz": 50,
    "po": 1,
    "np": 1,
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": 2,
    "invt": 2,
    "fid": "f3",
    "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81",
    "fields": "f12,f14,f2,f3,f4,f5,f6,f7,f8,f10,f15,f16,f17,f18,f20,f21",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json,text/plain,*/*",
}


def configure_stdout() -> None:
    """Prefer UTF-8 output on Windows terminals."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") or {}
    rows = data.get("diff") or []
    if isinstance(rows, dict):
        return list(rows.values())
    if isinstance(rows, list):
        return rows
    return []


def run_probe(mode: str, trust_env: bool, host: str) -> dict[str, Any]:
    print_section(f"测试模式: {mode} | 测试地址: {host}")
    result: dict[str, Any] = {
        "mode": mode,
        "host": host,
        "success": False,
        "status_code": None,
        "preview": "",
        "rows": 0,
        "columns": [],
        "has_f8": False,
        "has_f10": False,
        "error": None,
    }
    try:
        with requests.Session() as session:
            session.trust_env = trust_env
            response = session.get(host, params=PARAMS, headers=HEADERS, timeout=15)
            result["status_code"] = response.status_code
            result["preview"] = response.text[:100]
            print(f"HTTP 状态码: {response.status_code}")
            print(f"返回前 100 个字符: {response.text[:100]}")
            response.raise_for_status()
            payload = response.json()

        rows = extract_rows(payload)
        frame = pd.DataFrame(rows)
        result["rows"] = len(frame)
        result["columns"] = [str(column) for column in frame.columns]
        result["has_f8"] = "f8" in frame.columns
        result["has_f10"] = "f10" in frame.columns
        result["success"] = not frame.empty

        print(f"返回行数: {len(frame)}")
        print(f"字段列表: {result['columns']}")
        print(f"是否包含 f8 换手率: {'是' if result['has_f8'] else '否'}")
        print(f"是否包含 f10 量比: {'是' if result['has_f10'] else '否'}")
        print("前 3 行样例:")
        if frame.empty:
            print("无")
        else:
            print(frame.head(3).to_string(index=False))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"HTTP 状态码: {result['status_code']}")
        print(f"返回前 100 个字符: {result['preview']}")
        print("返回行数: 0")
        print("字段列表: []")
        print("是否包含 f8 换手率: 否")
        print("是否包含 f10 量比: 否")
        print("前 3 行样例: 无")
        print(f"完整失败原因: {result['error']}")
    return result


def main() -> int:
    configure_stdout()
    print_section("东方财富 push2 手动连通性测试")
    print("说明：本脚本只测试公开行情接口，不做任何交易操作。")

    results = []
    for mode, trust_env in MODES.items():
        for host in HOSTS:
            results.append(run_probe(mode, trust_env, host))

    print_section("汇总")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
