"""Read-only robustness test for the Sina fallback quote parser."""

from __future__ import annotations

import inspect
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.akshare_source import AkshareSource  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "sina_parser_robustness_check.json"


def legacy_index_diagnosis() -> dict:
    """Inspect the known AKShare page-count index operation without changing it."""
    import akshare as ak
    from akshare.stock.cons import zh_sina_a_stock_count_url

    source_file = inspect.getsourcefile(ak.stock_zh_a_spot)
    source_lines, source_start_line = inspect.getsourcelines(ak.stock_zh_a_spot)
    try:
        response = requests.get(zh_sina_a_stock_count_url, timeout=12)
        text = response.text or ""
        matches = re.findall(r"\d+", text)
        return {
            "file": source_file,
            "function": "_get_zh_a_page_count",
            "line": 36,
            "legacy_spot_function_start_line": source_start_line,
            "list_expression": "re.findall(re.compile(r'\\d+'), res.text)",
            "attempted_index": 0,
            "actual_list_length": len(matches),
            "count_response_length": len(text),
            "would_raise_index_error": len(matches) == 0,
            "traceback": "",
        }
    except Exception:
        return {
            "file": source_file,
            "function": "_get_zh_a_page_count",
            "line": 36,
            "list_expression": "re.findall(re.compile(r'\\d+'), res.text)",
            "attempted_index": 0,
            "actual_list_length": None,
            "count_response_length": None,
            "would_raise_index_error": None,
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    report = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_response_available": False,
        "total_batches": 0,
        "successful_batches": 0,
        "failed_batches": [],
        "raw_record_count": 0,
        "parsed_rows": 0,
        "skipped_bad_rows": 0,
        "valid_price_count": 0,
        "valid_volume_count": 0,
        "final_dataframe_rows": 0,
        "error_message": "",
        "legacy_index_diagnosis": legacy_index_diagnosis(),
    }
    source = AkshareSource()
    try:
        raw = source._fetch_sina_spot_robust()
        normalized, _ = source._normalize_spot(raw, "新浪备用源", "D", False, False)
        parser = source.sina_last_diagnostics.copy()
        report.update(parser)
        report["final_dataframe_rows"] = int(len(normalized))
        report["valid_price_count"] = int((pd.to_numeric(normalized["price"], errors="coerce") > 0).sum())
        report["valid_volume_count"] = int((pd.to_numeric(normalized["volume"], errors="coerce") > 0).sum())
        report["final_columns"] = [str(column) for column in normalized.columns]
        report["sample"] = normalized[["code", "name", "price", "pct_chg", "volume", "amount"]].head(10).to_dict("records")
    except Exception as exc:
        report["error_message"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report.update(source.sina_last_diagnostics)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Sina parser robustness diagnosis")
    print(f"raw_response_available: {report['raw_response_available']}")
    print(f"total_batches: {report['total_batches']}")
    print(f"successful_batches: {report['successful_batches']}")
    print(f"failed_batches: {len(report['failed_batches'])}")
    print(f"raw_record_count: {report['raw_record_count']}")
    print(f"parsed_rows: {report['parsed_rows']}")
    print(f"skipped_bad_rows: {report['skipped_bad_rows']}")
    print(f"valid_price_count: {report['valid_price_count']}")
    print(f"valid_volume_count: {report['valid_volume_count']}")
    print(f"final_dataframe_rows: {report['final_dataframe_rows']}")
    print(f"legacy_index_error_location: {report['legacy_index_diagnosis']}")
    print(f"error_message: {report['error_message']}")
    print(f"report: {REPORT_PATH}")
    return 0 if report["final_dataframe_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
