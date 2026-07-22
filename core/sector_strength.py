"""Sector strength overview helpers for read-only market context."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


SECTOR_COLUMNS = [
    "sector_name",
    "sector_pct_chg",
    "sector_amount",
    "rise_count",
    "fall_count",
    "limit_up_count",
    "sector_rank",
    "sector_strength_level",
]


def normalize_sector_frame(raw: pd.DataFrame, sector_type: str) -> pd.DataFrame:
    """Normalize AKShare Eastmoney industry/concept board rows."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=[*SECTOR_COLUMNS, "sector_type"])

    aliases = {
        "sector_name": ["板块名称", "名称", "name"],
        "sector_pct_chg": ["涨跌幅", "涨幅", "pct_chg"],
        "sector_amount": ["成交额", "amount"],
        "rise_count": ["上涨家数", "涨家数", "上涨数"],
        "fall_count": ["下跌家数", "跌家数", "下跌数"],
        "limit_up_count": ["涨停家数", "涨停数"],
    }
    data = pd.DataFrame(index=raw.index)
    for target, choices in aliases.items():
        source = _first_column(raw, choices)
        data[target] = raw[source] if source else pd.NA

    data["sector_type"] = sector_type
    for column in ["sector_pct_chg", "sector_amount", "rise_count", "fall_count", "limit_up_count"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["sector_name"] = data["sector_name"].fillna("").astype(str)
    data = data[data["sector_name"] != ""].copy()
    data = data.sort_values(["sector_pct_chg", "sector_amount"], ascending=[False, False]).reset_index(drop=True)
    data["sector_rank"] = range(1, len(data) + 1)
    data["sector_strength_level"] = data.apply(_strength_level, axis=1)
    return data[[*SECTOR_COLUMNS, "sector_type"]]


def build_sector_overview(industry: pd.DataFrame, concept: pd.DataFrame) -> dict[str, Any]:
    """Build serializable sector overview payload for latest status JSON."""
    industry_top = _top_records(industry)
    concept_top = _top_records(concept)
    warnings = []
    if not industry_top:
        warnings.append("行业板块数据暂不可用")
    if not concept_top:
        warnings.append("概念板块数据暂不可用")
    return {
        "industry_top10": industry_top,
        "concept_top10": concept_top,
        "warnings": warnings,
    }


def build_realtime_sector_ranks(
    spot_quotes: pd.DataFrame,
    industry_cache_dir: str | Path,
) -> pd.DataFrame:
    """Rank current stock industries from realtime quotes and a local mapping.

    ``sector_rank`` uses the same method as the historical backtest: mean
    stock return for every industry, ranked descending across industries.  The
    industry membership is read from the existing local cache so no placeholder
    rank is fabricated when a live board-to-stock mapping is unavailable.
    """
    output_columns = [
        "code",
        "sector_name",
        "sector_pct_chg",
        "sector_rank",
        "sector_strength_level",
        "industry_rank",
        "industry_count",
        "industry_up_ratio",
        "industry_limit_up_count",
    ]
    if spot_quotes is None or spot_quotes.empty:
        return pd.DataFrame(columns=output_columns)

    industry = load_latest_industry_mapping(industry_cache_dir)
    if industry.empty or "code" not in spot_quotes.columns:
        return pd.DataFrame(columns=output_columns)

    quotes = pd.DataFrame(
        {
            "code": spot_quotes["code"].map(_normalize_stock_code),
            "pct_chg": pd.to_numeric(spot_quotes.get("pct_chg"), errors="coerce"),
        }
    )
    usable = quotes.merge(industry, on="code", how="left").dropna(subset=["industry", "pct_chg"])
    usable = usable[usable["industry"].astype(str).str.strip().ne("")]
    if usable.empty:
        return pd.DataFrame(columns=output_columns)

    usable["is_up"] = usable["pct_chg"].gt(0)
    usable["is_limit_up"] = usable.apply(
        lambda row: row["pct_chg"] >= _limit_up_threshold(str(row["code"])),
        axis=1,
    )
    sector = usable.groupby("industry", as_index=False).agg(
        sector_pct_chg=("pct_chg", "mean"),
        industry_stock_count=("code", "nunique"),
        industry_up_count=("is_up", "sum"),
        industry_limit_up_count=("is_limit_up", "sum"),
    )
    sector["sector_rank"] = sector["sector_pct_chg"].rank(method="min", ascending=False)
    sector["industry_rank"] = sector["sector_rank"]
    sector["industry_count"] = int(len(sector))
    sector["industry_up_ratio"] = sector["industry_up_count"] / sector["industry_stock_count"].replace(0, pd.NA)
    output = usable[["code", "industry"]].drop_duplicates("code").merge(sector, on="industry", how="left")
    output = output.rename(columns={"industry": "sector_name"})
    output["sector_strength_level"] = output.apply(_strength_level, axis=1)
    return output[output_columns]


def load_latest_industry_mapping(industry_cache_dir: str | Path) -> pd.DataFrame:
    """Load the newest local industry membership cache, with a stable fallback."""
    root = Path(industry_cache_dir)
    paths = sorted(root.glob("industry_*.csv"), reverse=True)
    fallback = root.parent / "stock_basic_industry.csv"
    if fallback.exists():
        paths.append(fallback)

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        code_column = _first_column(raw, ["ts_code", "code", "股票代码"])
        industry_column = _first_column(raw, ["industry", "行业", "所属行业"])
        if not code_column or not industry_column:
            continue
        data = pd.DataFrame(
            {
                "code": raw[code_column].map(_normalize_stock_code),
                "industry": raw[industry_column].astype("string").str.strip(),
            }
        ).dropna(subset=["code", "industry"])
        data = data[data["industry"].ne("")]
        if not data.empty:
            frames.append(data)
        # The newest dated cache normally covers the full universe.  Do not
        # repeatedly read older files unless it is incomplete for a code.
        if frames and path.name.startswith("industry_"):
            break
    if not frames:
        return pd.DataFrame(columns=["code", "industry"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("code", keep="first")


def enrich_candidates_with_sector(
    candidates: pd.DataFrame,
    sector_overview: dict[str, Any],
    stock_sector_ranks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach stock-level realtime industry ranks when a local mapping exists."""
    output = candidates.copy()
    ranked = stock_sector_ranks.copy() if stock_sector_ranks is not None else pd.DataFrame()
    if not ranked.empty and "code" in output.columns:
        ranked = ranked.copy()
        ranked["code"] = ranked["code"].map(_normalize_stock_code)
        output["_sector_code"] = output["code"].map(_normalize_stock_code)
        output = output.merge(
            ranked.rename(columns={"code": "_sector_code"}),
            on="_sector_code",
            how="left",
            suffixes=("", "_cached"),
        )
        for column in (
            "sector_name",
            "sector_pct_chg",
            "sector_rank",
            "sector_strength_level",
            "industry_rank",
            "industry_count",
            "industry_up_ratio",
            "industry_limit_up_count",
        ):
            cached_column = f"{column}_cached"
            if cached_column in output.columns:
                output[column] = output[cached_column].combine_first(output.get(column, pd.Series(index=output.index, dtype="object")))
                output = output.drop(columns=[cached_column])
        output = output.drop(columns=["_sector_code"])

    output["sector_name"] = output.get("sector_name", pd.Series(index=output.index, dtype="object")).fillna("暂未匹配")
    output["sector_pct_chg"] = pd.to_numeric(output.get("sector_pct_chg", pd.Series(index=output.index, dtype="float64")), errors="coerce")
    output["sector_rank"] = pd.to_numeric(output.get("sector_rank", pd.Series(index=output.index, dtype="float64")), errors="coerce")
    output["industry_rank"] = pd.to_numeric(output.get("industry_rank", pd.Series(index=output.index, dtype="float64")), errors="coerce")
    output["industry_count"] = pd.to_numeric(output.get("industry_count", pd.Series(index=output.index, dtype="float64")), errors="coerce")
    output["industry_up_ratio"] = pd.to_numeric(output.get("industry_up_ratio", pd.Series(index=output.index, dtype="float64")), errors="coerce")
    output["industry_limit_up_count"] = pd.to_numeric(output.get("industry_limit_up_count", pd.Series(index=output.index, dtype="float64")), errors="coerce")
    output["sector_strength_level"] = output.get("sector_strength_level", pd.Series(index=output.index, dtype="object")).fillna("暂无")
    has_overview = bool(sector_overview.get("industry_top10") or sector_overview.get("concept_top10"))
    output["sector_summary"] = (
        "实时行业涨幅排名来自本地行业映射 + 全市场实时涨跌幅"
        if output["sector_rank"].notna().any()
        else "个股所属板块匹配暂未完成；已展示板块强弱总览"
        if has_overview
        else "板块数据暂不可用"
    )
    return output


def _top_records(data: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    if data is None or data.empty:
        return []
    records = data.head(n).copy()
    return records.where(pd.notna(records), None).to_dict(orient="records")


def _strength_level(row: pd.Series) -> str:
    rank = pd.to_numeric(row.get("sector_rank"), errors="coerce")
    pct = pd.to_numeric(row.get("sector_pct_chg"), errors="coerce")
    if pd.notna(rank) and rank <= 10 and pd.notna(pct) and pct > 0:
        return "强"
    if pd.notna(pct) and pct >= 0:
        return "中"
    return "弱"


def _first_column(data: pd.DataFrame, choices: list[str]) -> str | None:
    columns = [str(column) for column in data.columns]
    for choice in choices:
        if choice in columns:
            return choice
    normalized = {_normalize(choice) for choice in choices}
    for column in columns:
        if _normalize(column) in normalized:
            return column
    return None


def _normalize(value: str) -> str:
    return str(value).replace(" ", "").replace("_", "").lower()


def _normalize_stock_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits[-6:].zfill(6) if digits else ""


def _limit_up_threshold(code: str) -> float:
    """Return the daily limit-up threshold used by the cache/backtest path."""
    if str(code).startswith(("300", "301", "688", "689")):
        return 19.5
    if str(code).startswith(("4", "8", "9")):
        return 29.5
    return 9.5
