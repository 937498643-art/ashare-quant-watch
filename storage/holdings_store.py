"""Local file store for manually entered holdings.

This module only reads and writes a project-local CSV file. It does not connect
to broker software, trading accounts, funds, orders, or any trading interface.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDINGS_PATH = PROJECT_ROOT / "data" / "user" / "holdings.csv"
HOLDING_COLUMNS = ["code", "name", "cost_price", "shares", "buy_date", "tag", "note"]


def ensure_holdings_file(path: Path = HOLDINGS_PATH) -> Path:
    """Ensure the local holdings CSV exists with the expected header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=HOLDING_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_user_holdings(path: Path = HOLDINGS_PATH) -> pd.DataFrame:
    """Load manually entered local holdings."""

    ensure_holdings_file(path)
    try:
        data = pd.read_csv(path, dtype={"code": "string", "buy_date": "string"})
    except Exception:
        data = pd.DataFrame(columns=HOLDING_COLUMNS)
    for column in HOLDING_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data["code"] = data["code"].astype("string").str.zfill(6)
    data["cost_price"] = pd.to_numeric(data["cost_price"], errors="coerce").fillna(0)
    data["shares"] = pd.to_numeric(data["shares"], errors="coerce").fillna(0)
    return data[HOLDING_COLUMNS]


def upsert_user_holding(
    code: str,
    name: str,
    cost_price: float,
    shares: float,
    buy_date: str,
    tag: str = "持仓",
    note: str = "",
    path: Path = HOLDINGS_PATH,
) -> tuple[bool, str]:
    """Add or update one manually entered holding."""

    normalized_code = str(code or "").strip().zfill(6)
    if not normalized_code or normalized_code == "000000":
        return False, "股票代码无效"
    data = load_user_holdings(path)
    mask = data["code"].astype(str).str.zfill(6) == normalized_code
    row = {
        "code": normalized_code,
        "name": str(name or "").strip(),
        "cost_price": float(cost_price or 0),
        "shares": float(shares or 0),
        "buy_date": str(buy_date or "").strip(),
        "tag": str(tag or "持仓").strip(),
        "note": str(note or "").strip(),
    }
    if mask.any():
        for column, value in row.items():
            data.loc[mask, column] = value
        message = "已更新持仓"
    else:
        data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
        message = "已新增持仓"
    ensure_holdings_file(path)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return True, message


def remove_user_holding(code: str, path: Path = HOLDINGS_PATH) -> tuple[bool, str]:
    """Remove one manually entered holding."""

    normalized_code = str(code or "").strip().zfill(6)
    data = load_user_holdings(path)
    before = len(data)
    data = data[data["code"].astype(str).str.zfill(6) != normalized_code]
    if len(data) == before:
        return False, "持仓不存在"
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return True, "已删除持仓"
