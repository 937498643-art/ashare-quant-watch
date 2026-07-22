"""Stock code helpers for read-only market classification."""

from __future__ import annotations

from typing import Any

import pandas as pd


BOARD_TYPE_DISPLAY = {
    "sh_main": "上证主板",
    "sz_main": "深证主板",
    "chi_next": "创业板",
    "star_market": "科创板",
    "bj": "北交所",
    "other": "其他",
}

BOARD_DISPLAY_ORDER = ["上证主板", "深证主板", "创业板", "科创板", "北交所", "其他"]


def normalize_stock_code(code: Any) -> str:
    """Normalize common A-share code inputs to six digits."""
    digits = "".join(ch for ch in str(code or "") if ch.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def classify_board(code: Any) -> tuple[str, str]:
    """Return internal board type and display name for one stock code."""
    normalized = normalize_stock_code(code)
    if not normalized:
        return "other", BOARD_TYPE_DISPLAY["other"]

    if normalized.startswith(("600", "601", "603", "605")):
        board_type = "sh_main"
    elif normalized.startswith(("688", "689")):
        board_type = "star_market"
    elif normalized.startswith(("000", "001", "002", "003")):
        board_type = "sz_main"
    elif normalized.startswith(("300", "301")):
        board_type = "chi_next"
    elif normalized.startswith(("8", "4", "9")):
        board_type = "bj"
    else:
        board_type = "other"

    return board_type, BOARD_TYPE_DISPLAY[board_type]


def to_market_code(code: Any) -> str:
    """Convert a six-digit A-share code to market-suffixed code."""
    normalized = normalize_stock_code(code)
    if len(normalized) != 6:
        return ""

    board_type, _ = classify_board(normalized)
    if board_type in {"sh_main", "star_market"}:
        return f"{normalized}.SH"
    if board_type in {"sz_main", "chi_next"}:
        return f"{normalized}.SZ"
    if board_type == "bj":
        return f"{normalized}.BJ"
    return normalized


def add_board_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add market_code, board_type, and board_type_display columns."""
    output = data.copy()
    if "code" not in output.columns:
        output["market_code"] = pd.NA
        output["board_type"] = "other"
        output["board_type_display"] = BOARD_TYPE_DISPLAY["other"]
        return output

    classifications = output["code"].map(classify_board)
    output["market_code"] = output["code"].map(to_market_code)
    output["board_type"] = classifications.map(lambda value: value[0])
    output["board_type_display"] = classifications.map(lambda value: value[1])
    return output


def board_counts(data: pd.DataFrame) -> dict[str, int]:
    """Count rows by board display name in a stable display order."""
    if data.empty:
        return {name: 0 for name in BOARD_DISPLAY_ORDER}

    classified = add_board_columns(data)
    counts = classified["board_type_display"].value_counts(dropna=False).to_dict()
    return {name: int(counts.get(name, 0)) for name in BOARD_DISPLAY_ORDER}
