"""Local file store for user watchlist.

This module only reads and writes a project-local CSV file. It does not connect
to broker software, trading accounts, or any trading interface.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCHLIST_PATH = PROJECT_ROOT / "data" / "user" / "watchlist.csv"
WATCHLIST_COLUMNS = ["code", "name", "add_time", "tag", "note"]


def ensure_watchlist_file(path: Path = WATCHLIST_PATH) -> Path:
    """Ensure the local watchlist CSV exists with the expected header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=WATCHLIST_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_user_watchlist(path: Path = WATCHLIST_PATH) -> pd.DataFrame:
    """Load the local user watchlist."""
    ensure_watchlist_file(path)
    try:
        data = pd.read_csv(path, dtype={"code": "string"})
    except Exception:
        data = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    for column in WATCHLIST_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data["code"] = data["code"].astype("string").str.zfill(6)
    return data[WATCHLIST_COLUMNS]


def add_user_watchlist_stock(
    code: str,
    name: str,
    tag: str = "自选",
    note: str = "",
    path: Path = WATCHLIST_PATH,
) -> tuple[bool, str]:
    """Add a stock to the local watchlist if it is not already present."""
    normalized_code = str(code or "").strip().zfill(6)
    if not normalized_code or normalized_code == "000000":
        return False, "股票代码无效"

    data = load_user_watchlist(path)
    if normalized_code in set(data["code"].astype(str).str.zfill(6)):
        return False, "该股票已在自选股中"

    row = {
        "code": normalized_code,
        "name": str(name or "").strip(),
        "add_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "note": note,
    }
    data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
    ensure_watchlist_file(path)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return True, "已加入自选股"


def update_user_watchlist_stock(
    code: str,
    tag: str,
    note: str,
    path: Path = WATCHLIST_PATH,
) -> tuple[bool, str]:
    """Update the tag and note for a local watchlist stock."""
    normalized_code = str(code or "").strip().zfill(6)
    data = load_user_watchlist(path)
    mask = data["code"].astype(str).str.zfill(6) == normalized_code
    if not mask.any():
        return False, "自选股不存在"
    data.loc[mask, "tag"] = str(tag or "").strip()
    data.loc[mask, "note"] = str(note or "").strip()
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return True, "已更新自选股"


def remove_user_watchlist_stock(code: str, path: Path = WATCHLIST_PATH) -> tuple[bool, str]:
    """Remove a stock from the local watchlist."""
    normalized_code = str(code or "").strip().zfill(6)
    data = load_user_watchlist(path)
    before = len(data)
    data = data[data["code"].astype(str).str.zfill(6) != normalized_code]
    if len(data) == before:
        return False, "自选股不存在"
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return True, "已移除自选股"
