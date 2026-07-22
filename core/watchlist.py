"""User-maintained watchlist and holdings helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_watchlist(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load user-maintained watchlist.yaml."""
    if not path.exists():
        return {"watchlist": [], "holdings": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "watchlist": data.get("watchlist") or [],
        "holdings": data.get("holdings") or [],
    }


def build_watchlist_quotes(spot_quotes: pd.DataFrame, items: list[dict[str, Any]]) -> pd.DataFrame:
    """Match configured watchlist items with current spot quotes."""
    return _match_items(spot_quotes, items, holding=False)


def build_holding_quotes(spot_quotes: pd.DataFrame, items: list[dict[str, Any]]) -> pd.DataFrame:
    """Match configured holdings with quotes and calculate P/L fields."""
    data = _match_items(spot_quotes, items, holding=True)
    if data.empty:
        return data
    data["cost_price"] = pd.to_numeric(data.get("cost_price"), errors="coerce").fillna(0)
    data["shares"] = pd.to_numeric(data.get("shares"), errors="coerce").fillna(0)
    data["price"] = pd.to_numeric(data.get("price"), errors="coerce")
    data["market_value"] = data["price"] * data["shares"]
    data["floating_pnl"] = (data["price"] - data["cost_price"]) * data["shares"]
    data["floating_pnl_ratio"] = (data["price"] / data["cost_price"].replace(0, pd.NA) - 1) * 100
    buy_date = data["buy_date"] if "buy_date" in data.columns else pd.Series("", index=data.index)
    data["t1_sellable_hint"] = buy_date.fillna("").map(lambda value: "仅提示：请自行确认 T+1 状态" if value else "--")
    return data


def _match_items(spot_quotes: pd.DataFrame, items: list[dict[str, Any]], holding: bool) -> pd.DataFrame:
    rows = []
    quotes = spot_quotes.copy()
    if "code" in quotes.columns:
        quotes["code"] = quotes["code"].astype(str).str.zfill(6)
    for item in items:
        code = str(item.get("code") or "").zfill(6)
        matched = quotes[quotes["code"] == code].head(1) if "code" in quotes.columns else pd.DataFrame()
        row = dict(item)
        row["code"] = code
        if matched.empty:
            row["quote_status"] = "暂无行情"
        else:
            row.update(matched.iloc[0].to_dict())
            row["quote_status"] = "已匹配"
        rows.append(row)
    return pd.DataFrame(rows)
