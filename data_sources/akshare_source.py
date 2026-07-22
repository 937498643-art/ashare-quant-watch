"""Read-only AKShare/Eastmoney market data source.

This module only reads public quote data. It does not contain account,
broker, transfer, or execution capabilities.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .base_source import BaseDataSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPOT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "spot"
SPOT_CACHE_CSV = SPOT_CACHE_DIR / "eastmoney_push2_latest.csv"
SPOT_CACHE_JSON = SPOT_CACHE_DIR / "eastmoney_push2_latest.json"

SPOT_COLUMNS = [
    "code",
    "name",
    "price",
    "pct_chg",
    "change",
    "volume",
    "amount",
    "amount_display",
    "amplitude",
    "high",
    "low",
    "open",
    "prev_close",
    "turnover",
    "turnover_display",
    "volume_ratio",
    "market_cap",
    "float_market_cap",
    "data_source",
    "data_source_level",
    "is_full_featured",
    "allow_strategy_candidate",
]

HISTORY_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount", "turnover"]

SINA_RAW_COLUMNS = [
    "代码", "_", "名称", "最新价", "涨跌额", "涨跌幅", "买入", "卖出", "昨收", "今开",
    "最高", "最低", "成交量", "成交额", "时间戳", "_", "_", "_", "_", "_",
]

PUSH2_FIELDS = [
    "f12",
    "f14",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f15",
    "f16",
    "f17",
    "f18",
    "f8",
    "f10",
    "f20",
    "f21",
]

FIELD_MAP = {
    "code": ["f12", "代码", "证券代码", "symbol", "code"],
    "name": ["f14", "名称", "股票名称", "name"],
    "price": ["f2", "最新价", "现价", "price"],
    "pct_chg": ["f3", "涨跌幅", "涨幅", "changepercent", "pct_chg"],
    "change": ["f4", "涨跌额", "pricechange", "change"],
    "volume": ["f5", "成交量", "volume"],
    "amount": ["f6", "成交额", "amount"],
    "amplitude": ["f7", "振幅", "amplitude"],
    "high": ["f15", "最高", "high"],
    "low": ["f16", "最低", "low"],
    "open": ["f17", "今开", "开盘", "open"],
    "prev_close": ["f18", "昨收", "prev_close"],
    "turnover": ["f8", "换手率", "换手", "turnover", "turnover_rate"],
    "volume_ratio": ["f10", "量比", "volume_ratio"],
    "market_cap": ["f20", "总市值", "market_cap"],
    "float_market_cap": ["f21", "流通市值", "float_market_cap"],
}


class AkshareSource(BaseDataSource):
    """Read A-share spot and history quotes from public sources."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.source_attempts: list[dict[str, Any]] = []
        self.last_spot_meta: dict[str, Any] = self._empty_meta()
        self.sina_last_diagnostics: dict[str, Any] = {}

    def fetch_spot(self) -> pd.DataFrame:
        """Fetch full-market spot quotes, preferring Eastmoney push2."""
        self.source_attempts = []
        self.last_spot_meta = self._empty_meta()
        ak = self._load_akshare()

        attempts = [
            ("东方财富 push2 原始接口", "A", True, True, self.direct_eastmoney_push2_fetch),
            ("东方财富全市场", "B", True, True, ak.stock_zh_a_spot_em),
            ("东方财富分市场合并", "C", True, True, lambda: self._fetch_eastmoney_parts(ak)),
            ("新浪备用源", "D", False, False, self._fetch_sina_spot_robust),
        ]

        last_errors: list[str] = []
        for source_name, level, full_featured, allow_strategy, fetcher in attempts:
            print(f"正在尝试行情源: {source_name}")
            try:
                raw = fetcher()
                if raw is None or raw.empty:
                    raise RuntimeError("返回空数据")
                data, meta = self._normalize_spot(raw, source_name, level, full_featured, allow_strategy)
                if source_name == "新浪备用源":
                    meta["sina_diagnostics"] = self.sina_last_diagnostics.copy()
                if data.empty:
                    raise RuntimeError("字段标准化后为空")
                self._record_attempt(source_name, True, len(data), None)
                meta["source_attempts"] = self.source_attempts.copy()
                self.last_spot_meta = meta
                if source_name == "东方财富 push2 原始接口" and meta["allow_strategy_candidate"]:
                    self._write_spot_cache(data, meta)
                print(f"行情源成功: {source_name}，获取 {len(data)} 只股票")
                print(f"是否识别到换手率: {bool(meta.get('turnover_field'))}")
                print(f"是否允许生成策略候选股: {meta.get('allow_strategy_candidate')}")
                return data
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                last_errors.append(f"{source_name}: {error}")
                self._record_attempt(source_name, False, 0, error)
                self.logger.warning("Spot source %s failed: %s", source_name, error)
                print(f"行情源失败: {source_name}，原因: {error}")
                if source_name == "东方财富 push2 原始接口":
                    cached = self._read_spot_cache()
                    if cached is not None:
                        data, meta = cached
                        self._record_attempt("东方财富 push2 缓存数据", True, len(data), None)
                        meta["source_attempts"] = self.source_attempts.copy()
                        self.last_spot_meta = meta
                        print("已使用最近一次东方财富完整缓存行情，可能不是最新实时数据。")
                        return data

        self.last_spot_meta = self._empty_meta(errors=last_errors, source_attempts=self.source_attempts.copy())
        return pd.DataFrame(columns=SPOT_COLUMNS)

    def direct_eastmoney_push2_fetch(self) -> pd.DataFrame:
        """Fetch Eastmoney push2 clist data directly with paging and retries."""
        hosts = [
            "https://82.push2.eastmoney.com/api/qt/clist/get",
            "https://push2.eastmoney.com/api/qt/clist/get",
        ]
        page_size = int(self.config.get("page_size") or self.config.get("push2_page_size") or 50)
        max_pages = int(self.config.get("push2_max_pages") or 20)
        retry_times = int(self.config.get("retry_times") or self.config.get("retry_count") or 5)
        retry_wait = float(self.config.get("retry_wait_seconds") or 1)
        timeout = float(self.config.get("timeout") or self.config.get("timeout_seconds") or 12)
        proxy_modes = self._proxy_modes()
        fields = ",".join(PUSH2_FIELDS)
        base_params = {
            "pn": 1,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f12",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
            "fields": fields,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://quote.eastmoney.com/",
            "Connection": "close",
        }
        errors: list[str] = []

        def fetch_fs(fs_expr: str, label: str) -> pd.DataFrame:
            for proxy_mode in proxy_modes:
                for host in hosts:
                    rows: list[dict[str, Any]] = []
                    total: int | None = None
                    completed = False
                    with requests.Session() as session:
                        session.trust_env = proxy_mode == "system"
                        session.headers.update(headers)
                        for page in range(1, max_pages + 1):
                            params = dict(base_params, fs=fs_expr, pn=page)
                            payload = None
                            for attempt in range(1, retry_times + 1):
                                try:
                                    response = session.get(host, params=params, timeout=timeout)
                                    response.raise_for_status()
                                    payload = response.json()
                                    break
                                except Exception as exc:
                                    error = f"{label} {proxy_mode} {host} page={page} try={attempt}: {exc}"
                                    errors.append(error)
                                    self.logger.warning("Eastmoney push2 request failed: %s", error)
                                    time.sleep(min(retry_wait * attempt, 5))
                            if payload is None:
                                break
                            data = (payload or {}).get("data") or {}
                            diff = data.get("diff") or []
                            if isinstance(diff, dict):
                                diff = list(diff.values())
                            if total is None:
                                total = int(data.get("total") or 0) or None
                            if not diff:
                                completed = True
                                break
                            rows.extend(diff)
                            if len(diff) < page_size or (total and len(rows) >= total):
                                completed = True
                                break
                    if rows and completed:
                        self.last_spot_meta["proxy_mode_used"] = proxy_mode
                        frame = pd.DataFrame(rows).drop_duplicates("f12")
                        self.logger.info("Eastmoney push2 %s fetched %s/%s rows", label, len(frame), total or "?")
                        return frame
                    if rows:
                        error = f"{label} incomplete rows={len(rows)} total={total or '?'}"
                        errors.append(error)
                        self.logger.warning("Eastmoney push2 partial result ignored: %s", error)
            return pd.DataFrame()

        full_fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
        full_market = fetch_fs(full_fs, "all")
        if not full_market.empty:
            return full_market

        parts = []
        for label, fs_expr in [
            ("sh", "m:1+t:2,m:1+t:23"),
            ("sz", "m:0+t:6,m:0+t:80"),
            ("bj", "m:0+t:81+s:2048"),
        ]:
            part = fetch_fs(fs_expr, label)
            if not part.empty:
                parts.append(part)

        if parts:
            merged = pd.concat(parts, ignore_index=True).drop_duplicates("f12")
            has_sh = merged["f12"].astype(str).str.startswith(("6", "9")).any()
            has_sz = merged["f12"].astype(str).str.startswith(("0", "3")).any()
            has_bj = merged["f12"].astype(str).str.startswith(("4", "8")).any()
            if has_sh and has_sz:
                self.logger.info(
                    "Eastmoney push2 partition merge fetched rows=%s sh=%s sz=%s bj=%s",
                    len(merged),
                    has_sh,
                    has_sz,
                    has_bj,
                )
                return merged
            errors.append(f"partition fetch incomplete rows={len(merged)} sh={has_sh} sz={has_sz} bj={has_bj}")
        raise RuntimeError("; ".join(errors[-5:]) or "东方财富 push2 原始接口请求失败")

    def fetch_history(self, code: str, days: int = 60) -> pd.DataFrame:
        """Fetch daily history for one stock code."""
        ak = self._load_akshare()
        symbol = self._normalize_code(code)
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - pd.Timedelta(days=max(days * 3, 120))).strftime("%Y%m%d")
        try:
            raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
        except Exception as exc:
            self.logger.warning("Failed to fetch history for %s: %s", symbol, exc)
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        data = pd.DataFrame()
        data["date"] = raw[_find_column(raw, ["日期", "date"])] if _find_column(raw, ["日期", "date"]) else pd.NA
        data["open"] = raw[_find_column(raw, ["开盘", "open"])] if _find_column(raw, ["开盘", "open"]) else pd.NA
        data["close"] = raw[_find_column(raw, ["收盘", "close"])] if _find_column(raw, ["收盘", "close"]) else pd.NA
        data["high"] = raw[_find_column(raw, ["最高", "high"])] if _find_column(raw, ["最高", "high"]) else pd.NA
        data["low"] = raw[_find_column(raw, ["最低", "low"])] if _find_column(raw, ["最低", "low"]) else pd.NA
        data["volume"] = raw[_find_column(raw, ["成交量", "volume"])] if _find_column(raw, ["成交量", "volume"]) else pd.NA
        data["amount"] = raw[_find_column(raw, ["成交额", "amount"])] if _find_column(raw, ["成交额", "amount"]) else pd.NA
        data["turnover"] = raw[_find_column(raw, ["换手率", "turnover"])] if _find_column(raw, ["换手率", "turnover"]) else pd.NA
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        for column in HISTORY_COLUMNS:
            if column != "date":
                data[column] = pd.to_numeric(data[column], errors="coerce")
        return data.dropna(subset=["date"]).tail(days).reset_index(drop=True)

    def fetch_industry_sectors(self) -> pd.DataFrame:
        """Fetch Eastmoney industry board data through AKShare."""
        try:
            return self._load_akshare().stock_board_industry_name_em()
        except Exception as exc:
            self.logger.warning("Failed to fetch industry sectors: %s", exc)
            return pd.DataFrame()

    def fetch_concept_sectors(self) -> pd.DataFrame:
        """Fetch Eastmoney concept board data through AKShare."""
        try:
            return self._load_akshare().stock_board_concept_name_em()
        except Exception as exc:
            self.logger.warning("Failed to fetch concept sectors: %s", exc)
            return pd.DataFrame()

    def fetch_main_indices(self) -> pd.DataFrame:
        """Fetch main index spot data when available."""
        try:
            raw = self._load_akshare().stock_zh_index_spot_em()
        except Exception as exc:
            self.logger.warning("Failed to fetch index data: %s", exc)
            return pd.DataFrame(columns=["index_code", "index_name", "pct_chg"])
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["index_code", "index_name", "pct_chg"])
        code_col = _find_column(raw, ["代码", "index_code", "code"])
        name_col = _find_column(raw, ["名称", "index_name", "name"])
        pct_col = _find_column(raw, ["涨跌幅", "pct_chg"])
        if not code_col:
            return pd.DataFrame(columns=["index_code", "index_name", "pct_chg"])
        return pd.DataFrame(
            {
                "index_code": raw[code_col].astype(str).str.extract(r"(\d{6})", expand=False),
                "index_name": raw[name_col].astype(str) if name_col else pd.NA,
                "pct_chg": pd.to_numeric(raw[pct_col], errors="coerce") if pct_col else pd.NA,
            }
        )

    def _fetch_sina_spot_robust(self) -> pd.DataFrame:
        """Fetch Sina quote pages while isolating malformed records and batches.

        AKShare's bundled Sina helper assumes the count response always contains
        a number and indexes the first regex match directly. This implementation
        validates the count response and every record before building a frame.
        """
        from akshare.stock.cons import zh_sina_a_stock_count_url, zh_sina_a_stock_payload, zh_sina_a_stock_url
        from akshare.utils import demjson

        timeout = float(self.config.get("timeout") or self.config.get("timeout_seconds") or 12)
        diagnostics: dict[str, Any] = {
            "raw_response_available": False,
            "count_response_length": 0,
            "page_count_fallback_used": False,
            "page_count_error": "",
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": [],
            "raw_record_count": 0,
            "parsed_rows": 0,
            "skipped_bad_rows": 0,
            "empty_rows": 0,
            "field_count_distribution": {},
            "normal_record_samples": [],
            "abnormal_record_samples": [],
            "error_message": "",
        }

        try:
            count_response = requests.get(zh_sina_a_stock_count_url, timeout=timeout)
            count_response.raise_for_status()
            count_text = count_response.text or ""
            diagnostics["count_response_length"] = len(count_text)
            count_matches = re.findall(r"\d+", count_text)
            if not count_matches:
                raise RuntimeError("Sina page-count response contained no numeric page total")
            page_count = max(1, (int(count_matches[0]) + 79) // 80)
        except Exception as exc:
            # The count endpoint occasionally returns an empty body or 502
            # while the quote pages remain available. Use the normal A-share
            # page count as a bounded fallback and keep per-page diagnostics.
            page_count = int(self.config.get("sina_fallback_page_count") or 70)
            diagnostics["page_count_fallback_used"] = True
            diagnostics["page_count_error"] = f"{type(exc).__name__}: {exc}"
            self.logger.warning(
                "Sina page-count request failed; falling back to %s pages: %s",
                page_count,
                diagnostics["page_count_error"],
            )

        diagnostics["total_batches"] = page_count
        field_counts: Counter[int] = Counter()
        records: list[dict[str, Any]] = []
        payload = zh_sina_a_stock_payload.copy()

        for page in range(1, page_count + 1):
            try:
                payload["page"] = page
                response = requests.get(zh_sina_a_stock_url, params=payload, timeout=timeout)
                response.raise_for_status()
                text = response.text or ""
                diagnostics["raw_response_available"] = diagnostics["raw_response_available"] or bool(text.strip())
                decoded = demjson.decode(text)
                if not isinstance(decoded, list):
                    raise ValueError(f"decoded page is {type(decoded).__name__}, expected list")
                if not decoded:
                    diagnostics["failed_batches"].append({"batch_id": page, "error": "empty decoded record list"})
                    continue

                diagnostics["successful_batches"] += 1
                diagnostics["raw_record_count"] += len(decoded)
                for raw_record in decoded:
                    if isinstance(raw_record, dict):
                        field_count = len(raw_record)
                        field_counts[field_count] += 1
                        row_data = {
                            "代码": raw_record.get("code") or raw_record.get("symbol"),
                            "名称": raw_record.get("name"),
                            "最新价": raw_record.get("trade"),
                            "涨跌额": raw_record.get("pricechange"),
                            "涨跌幅": raw_record.get("changepercent"),
                            "买入": raw_record.get("buy"),
                            "卖出": raw_record.get("sell"),
                            "昨收": raw_record.get("settlement"),
                            "今开": raw_record.get("open"),
                            "最高": raw_record.get("high"),
                            "最低": raw_record.get("low"),
                            "成交量": raw_record.get("volume"),
                            "成交额": raw_record.get("amount"),
                            "时间戳": raw_record.get("ticktime"),
                        }
                    elif isinstance(raw_record, (list, tuple)):
                        field_count = len(raw_record)
                        field_counts[field_count] += 1
                        if field_count < len(SINA_RAW_COLUMNS):
                            diagnostics["skipped_bad_rows"] += 1
                            if field_count == 0:
                                diagnostics["empty_rows"] += 1
                            if len(diagnostics["abnormal_record_samples"]) < 5:
                                diagnostics["abnormal_record_samples"].append(
                                    {"reason": f"field_count={field_count}, expected={len(SINA_RAW_COLUMNS)}", "record": list(raw_record)}
                                )
                            continue
                        row_data = dict(zip(SINA_RAW_COLUMNS, list(raw_record[: len(SINA_RAW_COLUMNS)])))
                    else:
                        diagnostics["skipped_bad_rows"] += 1
                        if len(diagnostics["abnormal_record_samples"]) < 5:
                            diagnostics["abnormal_record_samples"].append({"reason": "record is not a list", "record": str(raw_record)[:500]})
                        continue

                    if not str(row_data.get("代码") or "").strip() or not str(row_data.get("名称") or "").strip():
                        diagnostics["skipped_bad_rows"] += 1
                        if len(diagnostics["abnormal_record_samples"]) < 5:
                            diagnostics["abnormal_record_samples"].append({"reason": "missing code or name", "record": row_data})
                        continue
                    records.append(row_data)
                    if len(diagnostics["normal_record_samples"]) < 5:
                        diagnostics["normal_record_samples"].append(row_data)
            except Exception as exc:
                diagnostics["failed_batches"].append({"batch_id": page, "error": f"{type(exc).__name__}: {exc}"})
                self.logger.warning("Sina quote batch %s failed: %s", page, exc)
                continue

        diagnostics["field_count_distribution"] = {str(key): value for key, value in sorted(field_counts.items())}
        diagnostics["parsed_rows"] = len(records)
        if not records:
            diagnostics["error_message"] = "All Sina quote batches failed or contained no valid records"
            self.sina_last_diagnostics = diagnostics
            raise RuntimeError(diagnostics["error_message"])

        self.sina_last_diagnostics = diagnostics
        return pd.DataFrame(records)

    def _normalize_spot(
        self,
        raw: pd.DataFrame,
        source_name: str,
        level: str,
        full_featured: bool,
        allow_strategy: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        turnover_field = _find_column(raw, FIELD_MAP["turnover"])
        amount_field = _find_column(raw, FIELD_MAP["amount"])
        data = pd.DataFrame(index=raw.index)
        for target, aliases in FIELD_MAP.items():
            col = _find_column(raw, aliases)
            data[target] = raw[col] if col else pd.NA
        data["code"] = data["code"].map(self._normalize_code)
        data["name"] = data["name"].fillna("").astype(str)
        for column in [c for c in FIELD_MAP if c not in {"code", "name"}]:
            # Preserve quote placeholders as missing values instead of zero.
            data[column] = _clean_numeric_series(data[column])
        data = data[data["code"].str.len() == 6].drop_duplicates("code")
        data["amount_display"] = data["amount"].map(format_amount)
        data["turnover_display"] = data["turnover"].map(format_turnover)
        data["data_source"] = source_name
        data["data_source_level"] = level
        data["is_full_featured"] = bool(full_featured and turnover_field)
        data["allow_strategy_candidate"] = bool(allow_strategy and turnover_field)
        missing = []
        warnings = []
        if not turnover_field:
            missing.append("turnover")
            warnings.append("当前数据源缺少换手率字段")
        if not amount_field:
            missing.append("amount")
            warnings.append("当前数据源缺少成交额字段")
        if source_name == "新浪备用源":
            warnings.append("新浪备用源缺少换手率，本轮仅生成活跃观察池。")
        meta = {
            "data_source": source_name,
            "data_source_level": level,
            "is_full_featured": bool(full_featured and turnover_field),
            "allow_strategy_candidate": bool(allow_strategy and turnover_field),
            "raw_columns": [str(c) for c in raw.columns],
            "turnover_field": turnover_field,
            "amount_field": amount_field,
            "missing_fields": missing,
            "errors": [],
            "warnings": warnings,
            "used_cache": source_name == "东方财富 push2 缓存数据",
            "has_turnover": bool(turnover_field),
            "proxy_mode_used": self.last_spot_meta.get("proxy_mode_used"),
        }
        return data[SPOT_COLUMNS].reset_index(drop=True), meta

    def _fetch_eastmoney_parts(self, ak: Any) -> pd.DataFrame:
        parts = []
        for fetcher in [ak.stock_sh_a_spot_em, ak.stock_sz_a_spot_em, ak.stock_bj_a_spot_em]:
            try:
                frame = fetcher()
                if frame is not None and not frame.empty:
                    parts.append(frame)
            except Exception as exc:
                self.logger.warning("Eastmoney market part failed: %s", exc)
        if not parts:
            return pd.DataFrame()
        merged = pd.concat(parts, ignore_index=True)
        code_col = _find_column(merged, FIELD_MAP["code"])
        if not code_col:
            return pd.DataFrame()
        codes = merged[code_col].astype(str).str.extract(r"(\d{6})", expand=False).fillna("")
        has_sh = codes.str.startswith(("6", "9")).any()
        has_sz = codes.str.startswith(("0", "3")).any()
        if len(merged) < 4000 or not (has_sh and has_sz):
            self.logger.warning(
                "Eastmoney market parts incomplete: rows=%s has_sh=%s has_sz=%s",
                len(merged),
                has_sh,
                has_sz,
            )
            return pd.DataFrame()
        return merged

    def _write_spot_cache(self, data: pd.DataFrame, meta: dict[str, Any]) -> None:
        try:
            codes = data["code"].astype(str) if "code" in data.columns else pd.Series(dtype=str)
            has_sh = codes.str.startswith(("6", "9")).any()
            has_sz = codes.str.startswith(("0", "3")).any()
            if len(data) < 4000 or not (has_sh and has_sz):
                self.logger.warning(
                    "Skip writing incomplete spot cache: rows=%s has_sh=%s has_sz=%s",
                    len(data),
                    has_sh,
                    has_sz,
                )
                return
            SPOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data.to_csv(SPOT_CACHE_CSV, index=False, encoding="utf-8-sig")
            SPOT_CACHE_JSON.write_text(
                json.dumps(
                    {
                        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rows": len(data),
                        "has_turnover": True,
                        "data_source": meta.get("data_source"),
                        "source_level": meta.get("data_source_level"),
                        "columns": list(data.columns),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("Failed to write spot cache: %s", exc)

    def _read_spot_cache(self) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        if not SPOT_CACHE_CSV.exists() or not SPOT_CACHE_JSON.exists():
            return None
        try:
            info = json.loads(SPOT_CACHE_JSON.read_text(encoding="utf-8"))
            fetch_time = pd.to_datetime(info.get("fetch_time"), errors="coerce")
            max_age = int(self.config.get("max_cache_age_minutes") or 15)
            if pd.isna(fetch_time) or (datetime.now() - fetch_time.to_pydatetime()).total_seconds() > max_age * 60:
                return None
            if not info.get("has_turnover"):
                return None
            data = pd.read_csv(SPOT_CACHE_CSV, dtype={"code": str})
            if data.empty or "turnover" not in data or data["turnover"].isna().all():
                return None
            codes = data["code"].astype(str) if "code" in data.columns else pd.Series(dtype=str)
            has_sh = codes.str.startswith(("6", "9")).any()
            has_sz = codes.str.startswith(("0", "3")).any()
            if len(data) < 4000 or not (has_sh and has_sz):
                self.logger.warning(
                    "Ignore incomplete spot cache: rows=%s has_sh=%s has_sz=%s",
                    len(data),
                    has_sh,
                    has_sz,
                )
                return None
            data["data_source"] = "东方财富 push2 缓存数据"
            data["data_source_level"] = "A_CACHE"
            data["is_full_featured"] = True
            data["allow_strategy_candidate"] = True
            meta = {
                "data_source": "东方财富 push2 缓存数据",
                "data_source_level": "A_CACHE",
                "is_full_featured": True,
                "allow_strategy_candidate": True,
                "raw_columns": list(data.columns),
                "turnover_field": "turnover",
                "amount_field": "amount",
                "missing_fields": [],
                "errors": [],
                "warnings": ["当前使用最近一次东方财富完整缓存行情，可能不是最新实时数据。"],
                "used_cache": True,
                "cache_time": info.get("fetch_time"),
                "has_turnover": True,
            }
            return data[SPOT_COLUMNS].reset_index(drop=True), meta
        except Exception as exc:
            self.logger.warning("Failed to read spot cache: %s", exc)
            return None

    def _record_attempt(self, source: str, success: bool, rows: int, error: str | None) -> None:
        self.source_attempts.append({"source": source, "success": success, "rows": int(rows), "error": error})

    def _empty_meta(self, errors: list[str] | None = None, source_attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "data_source": "--",
            "data_source_level": "--",
            "is_full_featured": False,
            "allow_strategy_candidate": False,
            "raw_columns": [],
            "turnover_field": None,
            "amount_field": None,
            "missing_fields": ["spot"],
            "errors": errors or [],
            "warnings": [],
            "source_attempts": source_attempts or [],
            "used_cache": False,
            "has_turnover": False,
        }

    def _proxy_modes(self) -> list[str]:
        mode = str(self.config.get("proxy_mode") or "").lower()
        if mode == "system" or self.config.get("use_env_proxy") is True:
            return ["system"]
        if mode == "direct" or self.config.get("use_env_proxy") is False:
            return ["direct"]
        return ["direct", "system"]

    @staticmethod
    def _load_akshare() -> Any:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run pip install -r requirements.txt first.") from exc
        return ak

    @staticmethod
    def _normalize_code(value: Any) -> str:
        digits = re.sub(r"\D", "", str(value))
        return digits[-6:].zfill(6) if digits else ""


def format_amount(value: Any) -> str:
    amount = pd.to_numeric(value, errors="coerce")
    if pd.isna(amount) or amount <= 0:
        return "--"
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:.2f} 亿"
    if amount >= 10_000:
        return f"{amount / 10_000:.2f} 万"
    return f"{amount:.0f}"


def _clean_numeric_series(values: pd.Series) -> pd.Series:
    """Convert quote values safely while preserving invalid values as missing."""
    text = values.astype("string").str.strip().str.replace(",", "", regex=False)
    missing = {"", "--", "-", "None", "none", "null", "NULL", "nan", "NaN"}
    return pd.to_numeric(text.mask(text.isin(missing)), errors="coerce")


def format_turnover(value: Any) -> str:
    turnover = pd.to_numeric(value, errors="coerce")
    if pd.isna(turnover):
        return "--"
    return f"{turnover:.2f}%"


def _find_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    columns = [str(c) for c in frame.columns]
    for alias in aliases:
        if alias in columns:
            return alias
    normalized = {_normalize_name(alias) for alias in aliases}
    for column in columns:
        if _normalize_name(column) in normalized:
            return column
    return None


def _normalize_name(value: str) -> str:
    return re.sub(r"[\s_%（）()]", "", str(value)).lower()
