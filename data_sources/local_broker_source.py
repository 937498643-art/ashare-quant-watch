"""Read-only local broker quote table source.

This module normalizes行情表格 copied/exported from the local 招商证券
software. It only reads CSV/table data and does not access accounts, passwords,
broker trading pages, order entry, cancellation, or fund transfer functions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .base_source import BaseDataSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "local_broker" / "cmbc_quote_sample.csv"

OUTPUT_COLUMNS = [
    "code",
    "name",
    "price",
    "pct_chg",
    "amount",
    "amount_display",
    "turnover",
    "turnover_display",
    "volume_ratio",
    "data_source",
    "data_source_level",
    "is_full_featured",
    "allow_strategy_candidate",
]

FIELD_ALIASES = {
    "code": ["code", "股票代码", "代码", "证券代码"],
    "name": ["name", "股票名称", "名称", "证券简称"],
    "price": ["price", "最新价", "现价", "成交价"],
    "pct_chg": ["pct_chg", "涨跌幅", "涨幅", "涨跌%"],
    "amount": ["amount", "成交额", "成交金额", "金额"],
    "turnover": ["turnover", "换手率", "换手", "换手%"],
    "volume_ratio": ["volume_ratio", "量比"],
}


class LocalBrokerSource(BaseDataSource):
    """Normalize local 招商证券 quote tables into project fields."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_SAMPLE_PATH
        self.last_spot_meta: dict[str, Any] = {}

    def fetch_spot(self) -> pd.DataFrame:
        """Read and normalize the local quote sample CSV."""
        if not self.path.exists():
            self.last_spot_meta = {
                "data_source": "招商证券本地行情",
                "data_source_level": "LOCAL",
                "is_full_featured": False,
                "allow_strategy_candidate": False,
                "missing_fields": ["file"],
                "errors": [f"本地行情文件不存在: {self.path}"],
                "warnings": [],
            }
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        raw = pd.read_csv(self.path, dtype=str, encoding="utf-8-sig")
        data = normalize_local_quote_table(raw)
        missing_fields = []
        warnings = []
        if data.empty:
            missing_fields.append("spot")
            warnings.append("招商证券本地行情文件为空或字段无法识别。")
        if "turnover" not in data.columns or data["turnover"].isna().all():
            missing_fields.append("turnover")
            warnings.append("招商证券本地行情缺少换手率，只能生成活跃观察池。")
        if "volume_ratio" not in data.columns or data["volume_ratio"].isna().all():
            missing_fields.append("volume_ratio")
            warnings.append("招商证券本地行情缺少量比。")

        has_turnover = "turnover" in data.columns and data["turnover"].notna().any()
        self.last_spot_meta = {
            "data_source": "招商证券本地行情",
            "data_source_level": "LOCAL",
            "is_full_featured": bool(has_turnover),
            "allow_strategy_candidate": bool(has_turnover),
            "raw_columns": [str(column) for column in raw.columns],
            "turnover_field": _find_column(raw, FIELD_ALIASES["turnover"]),
            "amount_field": _find_column(raw, FIELD_ALIASES["amount"]),
            "missing_fields": missing_fields,
            "errors": [],
            "warnings": warnings,
            "source_attempts": [
                {
                    "source": "招商证券本地行情",
                    "success": not data.empty,
                    "rows": int(len(data)),
                    "error": None if not data.empty else "本地行情文件为空或字段无法识别",
                }
            ],
        }
        return data

    def fetch_history(self, code: str, days: int = 60) -> pd.DataFrame:
        """Use the existing public read-only history source for indicators."""
        from .akshare_source import AkshareSource

        return AkshareSource({}).fetch_history(code, days=days)

    def fetch_industry_sectors(self) -> pd.DataFrame:
        """Local copied quote tables do not provide sector overview."""
        return pd.DataFrame()

    def fetch_concept_sectors(self) -> pd.DataFrame:
        """Local copied quote tables do not provide concept overview."""
        return pd.DataFrame()

    def fetch_main_indices(self) -> pd.DataFrame:
        """Local copied quote tables do not provide index quotes."""
        return pd.DataFrame(columns=["index_code", "index_name", "pct_chg"])


def normalize_local_quote_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a copied/exported local quote table to unified quote fields."""
    output = pd.DataFrame(index=raw.index)
    for target, aliases in FIELD_ALIASES.items():
        column = _find_column(raw, aliases)
        output[target] = raw[column] if column else pd.NA

    output["code"] = output["code"].map(_normalize_code)
    output["name"] = output["name"].fillna("").astype(str)
    for column in ["price", "pct_chg", "amount", "turnover", "volume_ratio"]:
        output[column] = output[column].map(_clean_number)

    has_turnover = "turnover" in output.columns and output["turnover"].notna().any()
    output["amount_display"] = output["amount"].map(format_amount)
    output["turnover_display"] = output["turnover"].map(format_turnover)
    output["data_source"] = "招商证券本地行情"
    output["data_source_level"] = "LOCAL"
    output["is_full_featured"] = bool(has_turnover)
    output["allow_strategy_candidate"] = bool(has_turnover)
    output = output[output["code"].astype(str).str.len() == 6]
    return output[OUTPUT_COLUMNS].reset_index(drop=True)


def format_amount(value: Any) -> str:
    """Format amount in yuan into Chinese display units."""
    amount = pd.to_numeric(value, errors="coerce")
    if pd.isna(amount) or amount <= 0:
        return "--"
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:.2f} 亿"
    if amount >= 10_000:
        return f"{amount / 10_000:.2f} 万"
    return f"{amount:.0f}"


def format_turnover(value: Any) -> str:
    """Format turnover percentage while preserving missing values."""
    turnover = pd.to_numeric(value, errors="coerce")
    if pd.isna(turnover):
        return "--"
    return f"{turnover:.2f}%"


def _find_column(raw: pd.DataFrame, aliases: list[str]) -> str | None:
    columns = [str(column) for column in raw.columns]
    normalized_aliases = {_normalize_name(alias) for alias in aliases}
    for column in columns:
        if column in aliases or _normalize_name(column) in normalized_aliases:
            return column
    return None


def _normalize_name(value: str) -> str:
    return re.sub(r"[\s_%（）()：:]", "", str(value)).lower()


def _normalize_code(value: Any) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits[-6:].zfill(6) if digits else ""


def _clean_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "-", "nan", "None"}:
        return None
    multiplier = 1.0
    if text.endswith("亿"):
        multiplier = 100_000_000
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10_000
        text = text[:-1]
    number = pd.to_numeric(text, errors="coerce")
    if pd.isna(number):
        return None
    return float(number) * multiplier
