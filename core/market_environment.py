"""Market environment analysis based on read-only quote fields."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_market_environment(spot_quotes: pd.DataFrame, index_quotes: pd.DataFrame | None = None) -> dict[str, Any]:
    """Summarize broad market breadth and classify the short-term environment."""
    if spot_quotes is None or spot_quotes.empty:
        return _empty_environment("行情数据为空，无法生成市场环境")

    pct = pd.to_numeric(spot_quotes.get("pct_chg"), errors="coerce")
    amount = pd.to_numeric(spot_quotes.get("amount"), errors="coerce")
    valid = pct.dropna()
    total = int(len(valid))
    up = int((valid > 0).sum())
    down = int((valid < 0).sum())
    flat = int((valid == 0).sum())
    limit_up = int((valid >= 9.5).sum())
    limit_down = int((valid <= -9.5).sum())
    env = _classify_environment(up, limit_down)
    warnings = []
    if limit_down >= 50:
        warnings.append("跌停家数偏多，市场风险上升")
    return {
        "market_environment": env,
        "up_count": up,
        "down_count": down,
        "flat_count": flat,
        "limit_up_count": limit_up,
        "limit_down_count": limit_down,
        "market_amount": float(amount.sum(skipna=True)),
        "up_ratio": float(up / total * 100) if total else None,
        "down_ratio": float(down / total * 100) if total else None,
        "index_quotes": _index_records(index_quotes),
        "warnings": warnings,
    }


def build_realtime_market_score_fields(
    market_environment: dict[str, Any],
    index_quotes: pd.DataFrame | None,
    market_cache_dir: str | Path,
) -> dict[str, Any]:
    """Build V5.3 market-score inputs from current quotes plus local history.

    The current index percentage and market breadth come from the active scan.
    The local market cache supplies the prior closes and prior 20 market
    amounts needed for MA5, MA10 and the point-in-time amount baseline.
    """
    history = _load_market_history(Path(market_cache_dir))
    latest = history.iloc[-1] if not history.empty else pd.Series(dtype="object")
    current_pct = _shanghai_index_pct_change(index_quotes)
    cached_close = _number(latest.get("sh_close"))
    cached_pct = _number(latest.get("sh_pct_chg"))
    pct = current_pct if pd.notna(current_pct) else cached_pct
    if pd.notna(cached_close) and pd.notna(current_pct):
        sh_close = cached_close * (1 + current_pct / 100)
        close_series = pd.concat(
            [history["sh_close"].dropna().tail(19), pd.Series([sh_close])],
            ignore_index=True,
        )
        sh_ma5 = float(close_series.tail(5).mean()) if len(close_series) >= 5 else float("nan")
        sh_ma10 = float(close_series.tail(10).mean()) if len(close_series) >= 10 else float("nan")
        sh_ma20 = float(close_series.tail(20).mean()) if len(close_series) >= 20 else float("nan")
        index_source = "实时上证涨跌幅 + 本地市场缓存"
    else:
        sh_close = cached_close
        sh_ma5 = _number(latest.get("sh_ma5"))
        sh_ma10 = _number(latest.get("sh_ma10"))
        closes = history["sh_close"].dropna().tail(20) if not history.empty else pd.Series(dtype="float64")
        sh_ma20 = float(closes.mean()) if len(closes) >= 20 else float("nan")
        index_source = "本地市场缓存" if not history.empty else "缺失"

    market_amount = _number(market_environment.get("market_amount"))
    if pd.isna(market_amount):
        market_amount = _number(latest.get("market_amount"))
    prior_amounts = history["market_amount"].dropna().tail(20) if not history.empty else pd.Series(dtype="float64")
    market_amount_ma20 = float(prior_amounts.mean()) if len(prior_amounts) >= 20 else float("nan")

    up_count = _number(market_environment.get("up_count"))
    down_count = _number(market_environment.get("down_count"))
    directional_count = up_count + down_count if pd.notna(up_count) and pd.notna(down_count) else float("nan")
    up_ratio = up_count / directional_count if pd.notna(directional_count) and directional_count > 0 else float("nan")

    return {
        "sh_close": sh_close,
        "sh_pct_chg": pct,
        "sh_ma5": sh_ma5,
        "sh_ma10": sh_ma10,
        "sh_ma20": sh_ma20,
        "up_count": up_count,
        "down_count": down_count,
        "up_ratio": up_ratio,
        "limit_up_count": _number(market_environment.get("limit_up_count")),
        "market_amount": market_amount,
        "market_amount_ma20": market_amount_ma20,
        "market_environment": str(market_environment.get("market_environment") or "未知"),
        "market_score_input_source": index_source,
    }


def attach_market_score_fields(candidates: pd.DataFrame, fields: dict[str, Any]) -> pd.DataFrame:
    """Copy shared market-score inputs onto every row before V5.3 scoring."""
    output = candidates.copy()
    if output.empty:
        return output
    for key, value in fields.items():
        output[key] = value
    return output


def apply_market_level_cap(candidates: pd.DataFrame, environment: dict[str, Any]) -> pd.DataFrame:
    """Cap displayed levels under weak market environments."""
    if candidates.empty:
        return candidates
    output = candidates.copy()
    env = str(environment.get("market_environment") or "")
    if env == "偏弱":
        output["level_display"] = output["level_display"].map(lambda value: _cap_level(value, "加入观察"))
        output["risk_summary"] = output["risk_summary"].fillna("").astype(str) + "；市场偏弱，候选股等级上限为加入观察"
    elif env == "极弱":
        output["level_display"] = output["level_display"].map(lambda value: _cap_level(value, "普通观察"))
        output["risk_summary"] = output["risk_summary"].fillna("").astype(str) + "；市场极弱，短线需谨慎"
    elif env == "震荡":
        output["risk_summary"] = output["risk_summary"].fillna("").astype(str) + "；市场震荡"
    return output


def _classify_environment(up_count: int, limit_down_count: int) -> str:
    if up_count > 3500:
        env = "强势"
    elif up_count >= 2500:
        env = "偏强"
    elif up_count >= 1500:
        env = "震荡"
    elif up_count >= 800:
        env = "偏弱"
    else:
        env = "极弱"
    if limit_down_count >= 80 and env in {"强势", "偏强"}:
        return "震荡"
    if limit_down_count >= 80 and env == "震荡":
        return "偏弱"
    if limit_down_count >= 120:
        return "极弱"
    return env


def _cap_level(value: Any, cap: str) -> str:
    order = {"重点关注": 4, "加入观察": 3, "普通观察": 2, "暂不关注": 1}
    text = str(value or "")
    return cap if order.get(text, 0) > order.get(cap, 0) else text


def _index_records(index_quotes: pd.DataFrame | None) -> list[dict[str, Any]]:
    if index_quotes is None or index_quotes.empty:
        return []
    return index_quotes.where(pd.notna(index_quotes), None).to_dict(orient="records")


def _empty_environment(reason: str) -> dict[str, Any]:
    return {
        "market_environment": "未知",
        "up_count": 0,
        "down_count": 0,
        "flat_count": 0,
        "limit_up_count": 0,
        "limit_down_count": 0,
        "market_amount": 0,
        "up_ratio": None,
        "down_ratio": None,
        "index_quotes": [],
        "warnings": [reason],
    }


def _load_market_history(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("market_*.csv")):
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        data = pd.DataFrame(
            {
                "date": pd.to_datetime(raw.get("trade_date"), errors="coerce"),
                "sh_close": pd.to_numeric(raw.get("sh_close"), errors="coerce"),
                "sh_pct_chg": pd.to_numeric(raw.get("sh_pct_chg"), errors="coerce"),
                "sh_ma5": pd.to_numeric(raw.get("sh_ma5"), errors="coerce"),
                "sh_ma10": pd.to_numeric(raw.get("sh_ma10"), errors="coerce"),
                "market_amount": pd.to_numeric(raw.get("market_amount"), errors="coerce"),
            }
        ).dropna(subset=["date"])
        if not data.empty:
            frames.append(data)
    if not frames:
        return pd.DataFrame(columns=["date", "sh_close", "sh_pct_chg", "sh_ma5", "sh_ma10", "market_amount"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _shanghai_index_pct_change(index_quotes: pd.DataFrame | None) -> float:
    if index_quotes is None or index_quotes.empty or "pct_chg" not in index_quotes.columns:
        return float("nan")
    data = index_quotes.copy()
    code = data.get("index_code", pd.Series(index=data.index, dtype="object")).astype(str).map(_normalize_index_code)
    name = data.get("index_name", pd.Series(index=data.index, dtype="object")).astype(str)
    pct = pd.to_numeric(data.get("pct_chg"), errors="coerce")
    matches = data[code.eq("000001") | name.str.contains("上证指数", na=False)]
    if matches.empty:
        return float("nan")
    value = pd.to_numeric(matches["pct_chg"], errors="coerce").dropna()
    return float(value.iloc[0]) if not value.empty else float("nan")


def _normalize_index_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits[-6:].zfill(6) if digits else ""


def _number(value: Any) -> float:
    try:
        return float(pd.to_numeric(value, errors="coerce"))
    except (TypeError, ValueError):
        return float("nan")
