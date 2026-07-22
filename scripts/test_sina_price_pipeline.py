"""Diagnose the read-only Sina quote normalization and filtering pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import akshare as ak  # noqa: E402
from core.filters import filter_stock_pool  # noqa: E402
from data_sources.akshare_source import AkshareSource  # noqa: E402


PRICE_ALIASES = ("最新价", "现价", "price")
MISSING_MARKERS = {"", "--", "-", "None", "none", "null", "NULL", "nan", "NaN"}


def clean_number(values: pd.Series) -> pd.Series:
    """Return numeric values without converting quote placeholders to zero."""
    text = values.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(text.mask(text.isin(MISSING_MARKERS)), errors="coerce")


def main() -> None:
    raw = ak.stock_zh_a_spot()
    price_column = next((column for column in PRICE_ALIASES if column in raw.columns), None)
    raw_price = raw[price_column] if price_column else pd.Series(dtype="object")
    raw_numeric_price = clean_number(raw_price)

    print("Sina raw quote diagnosis")
    print(f"raw_count: {len(raw)}")
    print(f"raw_columns: {list(raw.columns)}")
    print("raw_head:")
    print(raw.head(10).to_string(index=False))
    print(f"price_source_column: {price_column!r}")
    print(f"price_source_dtype: {raw_price.dtype}")
    print(f"price_source_first20: {raw_price.head(20).tolist()}")
    print(f"price_source_null_count: {int(raw_price.isna().sum())}")
    print(f"price_source_zero_count: {int((raw_numeric_price == 0).sum())}")
    print(f"price_source_numeric_count: {int(raw_numeric_price.notna().sum())}")

    normalized, metadata = AkshareSource()._normalize_spot(raw, "Sina fallback", "D", False, False)
    active = filter_stock_pool(
        normalized,
        {"filters": {"min_price": 3.0, "min_amount": 100_000_000}},
    )
    prices = pd.to_numeric(normalized["price"], errors="coerce")
    columns = ["code", "name", "price", "pct_chg", "amount"]

    print("\nNormalized quote diagnosis")
    print(f"normalized_count: {len(normalized)}")
    print(f"normalized_columns: {list(normalized.columns)}")
    print(f"price_dtype: {normalized['price'].dtype}")
    print(f"valid_price_count: {int((prices > 0).sum())}")
    print(f"price_0_5_to_5000_count: {int(prices.between(0.5, 5000).sum())}")
    print(f"price_null_count: {int(prices.isna().sum())}")
    print(f"price_zero_count: {int((prices == 0).sum())}")
    print("normalized_head:")
    print(normalized[columns].head(20).to_string(index=False))

    print("\nActive pool diagnosis")
    print(f"active_count: {len(active)}")
    print(f"price_min: {prices.min()}")
    print(f"price_max: {prices.max()}")
    print(f"price_median: {prices.median()}")
    print("active_head:")
    print(active[columns].head(10).to_string(index=False))
    print(f"turnover_field: {metadata.get('turnover_field')!r}")
    print("Sina has no realtime turnover; no formal strategy candidates are generated.")


if __name__ == "__main__":
    main()
