"""Announcement risk keyword matching."""

from __future__ import annotations

from typing import Any

import pandas as pd


RISK_KEYWORDS = ["减持", "立案", "监管函", "问询函", "业绩预亏", "亏损", "诉讼", "冻结", "质押", "退市风险", "ST", "债务逾期", "被调查", "违规", "处罚"]
STRONG_KEYWORDS = {"立案", "退市风险", "债务逾期", "被调查"}


def enrich_announcement_risk(candidates: pd.DataFrame, announcements: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Add announcement risk fields from already fetched titles."""
    if candidates.empty:
        return candidates.copy()
    announcements = announcements or {}
    rows = []
    for _, row in candidates.iterrows():
        code = str(row.get("code") or "").zfill(6)
        titles = announcements.get(code, [])
        output = row.to_dict()
        output.update(analyze_titles(titles))
        rows.append(output)
    return pd.DataFrame(rows)


def analyze_titles(titles: list[str]) -> dict[str, Any]:
    if not titles:
        return {
            "announcement_risk_level": "未知",
            "announcement_risk_keywords": "",
            "announcement_risk_summary": "公告数据暂不可用",
            "latest_announcement_titles": "",
        }
    text = " ".join(titles)
    hits = [kw for kw in RISK_KEYWORDS if kw in text]
    if any(kw in STRONG_KEYWORDS for kw in hits):
        level = "高"
    elif hits:
        level = "中"
    else:
        level = "低"
    return {
        "announcement_risk_level": level,
        "announcement_risk_keywords": " / ".join(hits),
        "announcement_risk_summary": "命中公告风险关键词" if hits else "最近公告未命中重点风险关键词",
        "latest_announcement_titles": "；".join(titles[:5]),
    }
