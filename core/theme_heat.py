"""Theme heat keyword matching."""

from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_KEYWORDS = ["AI", "算力", "光模块", "CPO", "半导体", "芯片", "机器人", "商业航天", "低空经济", "军工", "证券", "电力", "固态电池", "新能源", "数据中心", "液冷", "国产替代"]


def enrich_theme_heat(candidates: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add theme tags by matching keywords in available text fields."""
    if candidates.empty:
        return candidates.copy()
    keywords = (config or {}).get("theme_keywords") or DEFAULT_KEYWORDS
    rows = []
    for _, row in candidates.iterrows():
        output = row.to_dict()
        output.update(match_theme(row, keywords))
        rows.append(output)
    return pd.DataFrame(rows)


def match_theme(row: pd.Series | dict[str, Any], keywords: list[str]) -> dict[str, Any]:
    text = " ".join(str(row.get(key, "")) for key in ["name", "sector_name", "sector_summary", "latest_announcement_titles"] if hasattr(row, "get"))
    tags = [kw for kw in keywords if kw and kw.lower() in text.lower()]
    score = min(len(tags) * 10, 30)
    if score >= 20:
        level = "高"
    elif score >= 10:
        level = "中"
    elif tags:
        level = "低"
    else:
        level = "暂无"
    return {
        "theme_tags": " / ".join(tags) if tags else "暂无题材标签",
        "theme_heat_score": score,
        "theme_heat_level": level,
        "theme_heat_summary": f"命中关键词：{' / '.join(tags)}" if tags else "暂无题材标签",
    }
