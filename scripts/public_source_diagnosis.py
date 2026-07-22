"""Diagnose public quote data sources for quant_stock_watch.

This script only probes public market data APIs and writes a diagnostic report.
It does not modify the main program, dashboard, strategies, broker software, or
any trading-related system.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_DIR = PROJECT_ROOT / "data" / "diagnostics"
REPORT_PATH = DIAGNOSTIC_DIR / "public_source_diagnosis.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PUSH2_HOSTS = [
    "https://push2.eastmoney.com/api/qt/clist/get",
    "https://82.push2.eastmoney.com/api/qt/clist/get",
]
PUSH2_FIELDS = "f12,f14,f2,f3,f6,f8,f10,f20,f21"
PUSH2_PARAMS = {
    "pn": 1,
    "pz": 50,
    "po": 1,
    "np": 1,
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": 2,
    "invt": 2,
    "fid": "f3",
    "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81",
    "fields": PUSH2_FIELDS,
}
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json,text/plain,*/*",
}

AKSHARE_EM_FUNCTIONS = [
    ("stock_zh_a_spot_em", "东方财富全市场"),
    ("stock_sh_a_spot_em", "东方财富沪A"),
    ("stock_sz_a_spot_em", "东方财富深A"),
    ("stock_bj_a_spot_em", "东方财富京A"),
]
HISTORY_TEST_CODES = ["600000", "000001", "300750", "688981", "300059"]


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


def detect_common_fields(frame: pd.DataFrame) -> dict[str, bool]:
    columns = list(frame.columns) if isinstance(frame, pd.DataFrame) else []
    return {
        "has_code": has_any_column(columns, ["代码", "股票代码", "证券代码", "code", "f12", "symbol"]),
        "has_name": has_any_column(columns, ["名称", "股票名称", "证券简称", "name", "f14"]),
        "has_price": has_any_column(columns, ["最新价", "现价", "当前价", "price", "f2"]),
        "has_pct_chg": has_any_column(columns, ["涨跌幅", "涨幅", "pct_chg", "f3"]),
        "has_amount": has_any_column(columns, ["成交额", "amount", "f6"]),
        "has_turnover": has_any_column(columns, ["换手率", "换手", "turnover", "turnover_rate", "f8"]),
        "has_volume_ratio": has_any_column(columns, ["量比", "volume_ratio", "f10"]),
        "has_volume": has_any_column(columns, ["成交量", "volume", "f5"]),
        "has_market_cap": has_any_column(columns, ["总市值", "market_cap", "f20"]),
        "has_float_market_cap": has_any_column(columns, ["流通市值", "float_market_cap", "f21"]),
    }


def diagnose_push2() -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for host in PUSH2_HOSTS:
        for mode, trust_env in [("direct", False), ("system", True)]:
            started = time.time()
            attempt = {
                "source": "东方财富 push2 原始接口",
                "host": host,
                "mode": mode,
                "success": False,
                "rows": 0,
                "columns": [],
                "has_f12_code": False,
                "has_f14_name": False,
                "has_f2_price": False,
                "has_f3_pct_chg": False,
                "has_f6_amount": False,
                "has_f8_turnover": False,
                "has_f10_volume_ratio": False,
                "has_f20_market_cap": False,
                "has_f21_float_market_cap": False,
                "elapsed_seconds": 0.0,
                "error": None,
            }
            try:
                session = requests.Session()
                session.trust_env = trust_env
                response = session.get(host, params=PUSH2_PARAMS, headers=HEADERS, timeout=8)
                response.raise_for_status()
                payload = response.json()
                diff = (payload.get("data") or {}).get("diff") or []
                columns = list(diff[0].keys()) if diff else []
                attempt.update(
                    {
                        "success": bool(diff),
                        "rows": len(diff),
                        "columns": columns,
                        "has_f12_code": "f12" in columns,
                        "has_f14_name": "f14" in columns,
                        "has_f2_price": "f2" in columns,
                        "has_f3_pct_chg": "f3" in columns,
                        "has_f6_amount": "f6" in columns,
                        "has_f8_turnover": "f8" in columns,
                        "has_f10_volume_ratio": "f10" in columns,
                        "has_f20_market_cap": "f20" in columns,
                        "has_f21_float_market_cap": "f21" in columns,
                        "sample": diff[:3],
                    }
                )
                if attempt["success"] and best is None:
                    best = attempt
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                attempt["elapsed_seconds"] = round(time.time() - started, 3)
                attempts.append(attempt)

    available = best is not None
    return {
        "available": available,
        "has_turnover": bool(best and best.get("has_f8_turnover")),
        "has_volume_ratio": bool(best and best.get("has_f10_volume_ratio")),
        "best": best,
        "attempts": attempts,
    }


def diagnose_akshare_em() -> dict[str, Any]:
    try:
        import akshare as ak
    except Exception as exc:
        return {"available": False, "has_turnover": False, "has_volume_ratio": False, "attempts": [], "error": str(exc)}

    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for function_name, label in AKSHARE_EM_FUNCTIONS:
        attempt = {
            "source": label,
            "function": function_name,
            "success": False,
            "rows": 0,
            "columns": [],
            "has_turnover": False,
            "has_volume_ratio": False,
            "error": None,
        }
        try:
            func = getattr(ak, function_name)
            frame = func()
            fields = detect_common_fields(frame)
            attempt.update(
                {
                    "success": isinstance(frame, pd.DataFrame) and not frame.empty,
                    "rows": int(len(frame)) if isinstance(frame, pd.DataFrame) else 0,
                    "columns": list(frame.columns) if isinstance(frame, pd.DataFrame) else [],
                    "has_turnover": fields["has_turnover"],
                    "has_volume_ratio": fields["has_volume_ratio"],
                    "field_flags": fields,
                    "sample": frame.head(3).to_dict("records") if isinstance(frame, pd.DataFrame) else [],
                }
            )
            if attempt["success"] and best is None:
                best = attempt
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
        attempts.append(attempt)

    return {
        "available": best is not None,
        "has_turnover": any(item["success"] and item["has_turnover"] for item in attempts),
        "has_volume_ratio": any(item["success"] and item["has_volume_ratio"] for item in attempts),
        "best": best,
        "attempts": attempts,
    }


def diagnose_sina() -> dict[str, Any]:
    try:
        import akshare as ak
    except Exception as exc:
        return {"available": False, "has_turnover": False, "has_volume_ratio": False, "error": str(exc)}

    try:
        frame = ak.stock_zh_a_spot()
        fields = detect_common_fields(frame)
        return {
            "available": isinstance(frame, pd.DataFrame) and not frame.empty,
            "rows": int(len(frame)) if isinstance(frame, pd.DataFrame) else 0,
            "columns": list(frame.columns) if isinstance(frame, pd.DataFrame) else [],
            "has_code": fields["has_code"],
            "has_name": fields["has_name"],
            "has_price": fields["has_price"],
            "has_pct_chg": fields["has_pct_chg"],
            "has_amount": fields["has_amount"],
            "has_turnover": fields["has_turnover"],
            "has_volume_ratio": fields["has_volume_ratio"],
            "can_generate_official_strategy_candidate": False,
            "usage_note": "新浪源没有实时换手率，只能用于活跃观察池。",
            "sample": frame.head(3).to_dict("records") if isinstance(frame, pd.DataFrame) else [],
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "rows": 0,
            "columns": [],
            "has_turnover": False,
            "has_volume_ratio": False,
            "can_generate_official_strategy_candidate": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def diagnose_history() -> dict[str, Any]:
    default_result = {
        "available": False,
        "tested_codes": HISTORY_TEST_CODES,
        "success_count": 0,
        "has_turnover": False,
        "has_amount": False,
        "has_volume": False,
        "has_ma5_ma10_ma20": False,
        "note": "历史换手率只能作为参考，不能当作实时换手率。",
        "samples": [],
        "error": None,
    }
    try:
        from data_sources.akshare_source import AkshareSource
    except Exception as exc:
        default_result["error"] = str(exc)
        return default_result

    source = AkshareSource({"retry_count": 1, "request_interval_seconds": 0.2})
    samples = []
    success_count = 0
    turnover_count = 0
    amount_count = 0
    volume_count = 0
    ma_count = 0
    for code in HISTORY_TEST_CODES:
        item = {"code": code, "success": False, "rows": 0, "columns": [], "error": None}
        try:
            frame = source.fetch_history(code, days=60)
            fields = detect_common_fields(frame)
            close_col = "close" if "close" in frame.columns else ("收盘" if "收盘" in frame.columns else None)
            has_ma = False
            if close_col and len(frame) >= 20:
                close = pd.to_numeric(frame[close_col], errors="coerce")
                has_ma = all(close.rolling(window).mean().notna().any() for window in [5, 10, 20])
            item.update(
                {
                    "success": isinstance(frame, pd.DataFrame) and not frame.empty,
                    "rows": int(len(frame)) if isinstance(frame, pd.DataFrame) else 0,
                    "columns": list(frame.columns) if isinstance(frame, pd.DataFrame) else [],
                    "has_amount": fields["has_amount"],
                    "has_turnover": fields["has_turnover"],
                    "has_volume": fields["has_volume"],
                    "has_ma5_ma10_ma20": has_ma,
                }
            )
            if item["success"]:
                success_count += 1
                if fields["has_turnover"]:
                    turnover_count += 1
                if fields["has_amount"]:
                    amount_count += 1
                if fields["has_volume"]:
                    volume_count += 1
                if has_ma:
                    ma_count += 1
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        samples.append(item)

    return {
        "available": success_count > 0,
        "tested_codes": HISTORY_TEST_CODES,
        "success_count": success_count,
        "has_turnover": turnover_count > 0,
        "has_amount": amount_count > 0,
        "has_volume": volume_count > 0,
        "has_ma5_ma10_ma20": ma_count > 0,
        "note": "历史换手率只能作为参考，不能当作实时换手率。",
        "samples": samples,
    }


def choose_recommendation(push2: dict[str, Any], ak_em: dict[str, Any], sina: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    if push2["available"] and push2["has_turnover"] and push2["has_volume_ratio"]:
        primary = "东方财富 push2 原始接口"
        official = True
        warning = "公开实时完整源可用，可以生成正式策略候选股。"
    elif ak_em["available"] and ak_em["has_turnover"] and ak_em["has_volume_ratio"]:
        primary = "AKShare 东方财富实时接口"
        official = True
        warning = "AKShare 东方财富实时接口可用，可以生成正式策略候选股。"
    else:
        primary = "暂无可用实时完整源"
        official = False
        warning = "当前缺少实时换手率或量比，不能生成正式策略候选股。"

    fallback = "新浪公开行情源" if sina.get("available") else "历史 K 线参考源" if history.get("available") else "无"
    reference = bool(sina.get("available") or history.get("available"))
    return {
        "recommended_primary_source": primary,
        "recommended_fallback_source": fallback,
        "can_generate_official_strategy_candidate": official,
        "can_generate_reference_candidate": reference,
        "warning_message": warning,
    }


def main() -> int:
    configure_stdout()
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    print("公开数据源健康诊断开始")
    push2 = diagnose_push2()
    ak_em = diagnose_akshare_em()
    sina = diagnose_sina()
    history = diagnose_history()
    recommendation = choose_recommendation(push2, ak_em, sina, history)

    report = {
        "diagnose_time": datetime.now().isoformat(timespec="seconds"),
        "eastmoney_push2_available": push2["available"],
        "eastmoney_push2_has_turnover": push2["has_turnover"],
        "eastmoney_push2_has_volume_ratio": push2["has_volume_ratio"],
        "akshare_em_available": ak_em["available"],
        "akshare_em_has_turnover": ak_em["has_turnover"],
        "akshare_em_has_volume_ratio": ak_em["has_volume_ratio"],
        "sina_available": sina["available"],
        "sina_has_turnover": sina["has_turnover"],
        "sina_has_volume_ratio": sina["has_volume_ratio"],
        "history_available": history["available"],
        "history_has_turnover": history["has_turnover"],
        "history_has_amount": history["has_amount"],
        "history_has_volume": history["has_volume"],
        "history_has_ma5_ma10_ma20": history["has_ma5_ma10_ma20"],
        **recommendation,
        "details": {
            "eastmoney_push2": push2,
            "akshare_eastmoney": ak_em,
            "sina": sina,
            "history": history,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    stable_source = "新浪公开行情源" if sina["available"] else recommendation["recommended_primary_source"]
    turnover_sources = []
    volume_ratio_sources = []
    if push2["has_turnover"]:
        turnover_sources.append("东方财富 push2")
    if ak_em["has_turnover"]:
        turnover_sources.append("AKShare 东方财富")
    if push2["has_volume_ratio"]:
        volume_ratio_sources.append("东方财富 push2")
    if ak_em["has_volume_ratio"]:
        volume_ratio_sources.append("AKShare 东方财富")

    print("\n诊断总结")
    print(f"1. 当前较稳定公开数据源: {stable_source}")
    print(f"2. 有实时换手率的数据源: {' / '.join(turnover_sources) if turnover_sources else '暂无'}")
    print(f"3. 有量比的数据源: {' / '.join(volume_ratio_sources) if volume_ratio_sources else '暂无'}")
    print(f"4. 新浪是否只能作为观察池: {'是' if sina['available'] else '新浪当前不可用'}")
    print("5. 历史换手率是否只能作为参考: 是")
    print(f"6. 是否可以生成正式策略候选股: {'是' if recommendation['can_generate_official_strategy_candidate'] else '否'}")
    print(f"7. 是否可以生成参考候选股: {'是' if recommendation['can_generate_reference_candidate'] else '否'}")
    print("8. 下一步建议: 优先修公开数据源稳定性和缓存，再继续调策略。")
    print(f"\n诊断报告已保存: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
