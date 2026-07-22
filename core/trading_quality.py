"""Shared V5.5/V5.8 trading-quality calculations layered on base scores.

The functions here do not alter core.scoring weights.  They return independent
adjustments that a caller may enable or disable during research or backtests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


QualityMode = Literal["legacy_v5_5", "realtime", "backtest"]

# V5.8 keeps V5.5's existing quality components and adds only the requested
# turnover-quality component. These bounds describe the possible *adjustment*
# range, not a change to core.scoring.py's base-score weights.
V58_QUALITY_ADJUSTMENT_MIN = -25
V58_QUALITY_ADJUSTMENT_MAX = 26


@dataclass(frozen=True)
class TradingQualityConfig:
    enabled: bool = True
    market_enabled: bool = True
    trend_stage_enabled: bool = True
    buy_point_enabled: bool = True
    sector_linkage_enabled: bool = True


def _number(data: dict[str, Any], key: str) -> float:
    try:
        value = float(data.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value


def _known(value: float) -> bool:
    return pd.notna(value)


def _truthy(value: Any) -> bool:
    """Interpret common dataframe boolean values without treating NA as true."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}
    return bool(value)


def market_score(market: dict[str, Any], enabled: bool = True) -> dict[str, Any]:
    """Return the independent [-8, +8] market-environment adjustment."""
    if not enabled:
        return {"score": 0, "detail": "市场环境模块关闭"}
    close = _number(market, "sh_close")
    ma20 = _number(market, "sh_ma20")
    up_ratio = _number(market, "up_ratio")
    limit_up = _number(market, "limit_up_count")
    score = 0
    detail: list[str] = []
    if _known(close) and _known(ma20):
        if close > ma20:
            score += 3
            detail.append("上证收盘>MA20 +3")
        elif close < ma20:
            score -= 3
            detail.append("上证收盘<MA20 -3")
    if _known(up_ratio):
        if up_ratio > 0.60:
            score += 3
            detail.append("全市场上涨比例>60% +3")
        elif up_ratio < 0.40:
            score -= 3
            detail.append("全市场上涨比例<40% -3")
    if _known(limit_up):
        if limit_up > 50:
            score += 2
            detail.append("涨停数>50 +2")
        elif limit_up < 20:
            score -= 2
            detail.append("涨停数<20 -2")
    return {"score": max(-8, min(8, score)), "detail": "；".join(detail) or "市场数据不足，不调整"}


def trend_stage_score(row: dict[str, Any], enabled: bool = True) -> dict[str, Any]:
    """Identify start, main-rise, and high-acceleration stages."""
    if not enabled:
        return {
            "score": 0,
            "risk_penalty": 0,
            "stage": "模块关闭",
            "detail": "趋势阶段模块关闭",
        }
    pct20 = _number(row, "pct_chg_20d")
    pct10 = _number(row, "pct_chg_10d")
    price = _number(row, "price")
    ma20 = _number(row, "ma20")
    prior_price = _number(row, "prior_price")
    prior_ma20 = _number(row, "prior_ma20")
    up_days = _number(row, "up_days_5d")
    bonus = 0
    risk = 0
    stage = "震荡/未定义"
    detail: list[str] = []
    broke_ma20 = (
        _known(price)
        and _known(ma20)
        and price > ma20
        and _known(prior_price)
        and _known(prior_ma20)
        and prior_price <= prior_ma20
    )
    if _known(pct20) and pct20 < 40 and broke_ma20:
        bonus = 3
        stage = "启动阶段"
        detail.append("20日涨幅<40%且突破MA20 +3")
    elif _known(pct20) and 40 <= pct20 <= 80:
        stage = "主升阶段"
        detail.append("20日涨幅40%–80%，不调整")
    elif _known(pct20) and pct20 > 80:
        stage = "高位加速"
        risk -= 5
        detail.append("20日涨幅>80% -5")
    if _known(up_days) and up_days >= 4:
        risk -= 3
        detail.append("近5日上涨>=4天 -3")
    if _known(pct10) and pct10 > 50:
        risk -= 5
        detail.append("近10日涨幅>50% -5")
    risk = max(-10, risk)
    return {
        "score": bonus + risk,
        "risk_penalty": risk,
        "stage": stage,
        "detail": "；".join(detail) or "趋势阶段无调整",
    }


def buy_point_score(row: dict[str, Any], enabled: bool = True) -> dict[str, Any]:
    """Return an independent buy-point quality adjustment in [-5, +5]."""
    if not enabled:
        return {"score": 0, "detail": "买点优化模块关闭"}
    volume = _number(row, "volume")
    avg_volume20 = _number(row, "avg_volume_20d")
    price = _number(row, "price")
    ma20 = _number(row, "ma20")
    low_volume_pullback = _truthy(row.get("recent_low_volume_pullback", False))
    score = 0
    detail: list[str] = []
    if _known(volume) and _known(avg_volume20) and avg_volume20 > 0 and volume > avg_volume20 * 1.5:
        score += 3
        detail.append("成交量>20日均量1.5倍 +3")
    if low_volume_pullback:
        score += 2
        detail.append("连续上涨后缩量调整 +2")
    if _known(price) and _known(ma20) and ma20 > 0 and (price / ma20 - 1) * 100 > 25:
        score -= 5
        detail.append("偏离MA20超过25% -5")
    return {"score": max(-5, min(5, score)), "detail": "；".join(detail) or "买点无调整"}


def sector_linkage_score(industry: dict[str, Any], enabled: bool = True) -> dict[str, Any]:
    """Return the optional sector-linkage adjustment, maximum +7."""
    if not enabled:
        return {"score": 0, "detail": "板块联动模块关闭"}
    rank = _number(industry, "industry_rank")
    count = _number(industry, "industry_count")
    up_ratio = _number(industry, "industry_up_ratio")
    limit_up_count = _number(industry, "industry_limit_up_count")
    score = 0
    detail: list[str] = []
    if _known(rank) and _known(count) and count > 0 and rank <= count * 0.20:
        score += 3
        detail.append("行业涨幅前20% +3")
    if _known(up_ratio) and up_ratio > 0.60:
        score += 2
        detail.append("行业上涨比例>60% +2")
    if _known(limit_up_count) and limit_up_count > 3:
        score += 2
        detail.append("行业涨停数>3 +2")
    return {"score": min(7, score), "detail": "；".join(detail) or "行业联动无调整"}


def turnover_quality_score(row: dict[str, Any], enabled: bool = True) -> dict[str, Any]:
    """Evaluate the V5.8 turnover-quality band without changing base scoring.

    The 5%--30% range remains the hard eligibility range.  Within it, 8%--25%
    is the preferred participation band.  Scores outside the hard range are
    marked ineligible; they are not allowed into the Top50 even if another
    component produces a high final score.
    """
    if not enabled:
        return {
            "score": 0,
            "eligible": True,
            "detail": "换手质量模块关闭",
        }

    turnover = _number(row, "turnover")
    if not _known(turnover):
        return {
            "score": 0,
            "eligible": False,
            "detail": "换手率缺失，不能进入Top50",
        }
    if turnover < 5 or turnover > 30:
        return {
            "score": 0,
            "eligible": False,
            "detail": f"换手率 {turnover:.2f}% 不在5%–30%允许范围，不能进入Top50",
        }
    if 8 <= turnover <= 25:
        return {
            "score": 3,
            "eligible": True,
            "detail": f"换手率 {turnover:.2f}% 位于8%–25%最佳区间 +3",
        }
    return {
        "score": -2,
        "eligible": True,
        "detail": f"换手率 {turnover:.2f}% 位于允许但非最佳区间 -2",
    }


def normalize_trading_quality_score(adjustment: float) -> float:
    """Map V5.8's signed quality adjustment onto a stable 0--100 scale."""
    value = _number({"adjustment": adjustment}, "adjustment")
    if not _known(value):
        return float("nan")
    bounded = max(V58_QUALITY_ADJUSTMENT_MIN, min(V58_QUALITY_ADJUSTMENT_MAX, value))
    span = V58_QUALITY_ADJUSTMENT_MAX - V58_QUALITY_ADJUSTMENT_MIN
    return round((bounded - V58_QUALITY_ADJUSTMENT_MIN) / span * 100, 2)


def calculate_final_trade_score(base_score: float, trading_quality_score: float) -> float:
    """Calculate the requested V5.8 70/30 final-trade score formula."""
    base = _number({"base_score": base_score}, "base_score")
    quality = _number({"trading_quality_score": trading_quality_score}, "trading_quality_score")
    if not _known(base) or not _known(quality):
        return float("nan")
    return round(base * 0.7 + quality * 0.3, 2)


def prepare_v58_quality_row(row: dict[str, Any]) -> dict[str, Any]:
    """Use realtime point-in-time helper fields when present.

    The V5.3 base row remains untouched.  These aliases only give the shared
    V5.8 quality layer the current-session equivalent of V5.5's historical
    inputs (for example, current price versus the close ten sessions ago).
    """
    output = dict(row)
    aliases = {
        "quality_pct_chg_10d": "pct_chg_10d",
        "quality_pct_chg_20d": "pct_chg_20d",
        "quality_up_days_5d": "up_days_5d",
    }
    for source, target in aliases.items():
        value = _number(output, source)
        if _known(value):
            output[target] = value
    return output


def score_data_quality(row: dict[str, Any], mode: QualityMode = "realtime") -> dict[str, Any]:
    """Score field completeness and expose Top50-critical missing fields.

    A complete row starts at 100.  Missing critical fields deduct 15 points
    each and block Top50 admission.  Missing supporting score inputs deduct
    five points each but do not silently manufacture a value.  In backtest
    mode, historical volume-and-average-volume is the equivalent of a live
    volume-ratio field because a point-in-time intraday ratio is unavailable.
    """
    _validate_mode(mode)
    critical_checks = {
        "换手率": _known(_number(row, "turnover")),
        "成交额": _known(_number(row, "amount")),
        "资金强度": _known(_number(row, "capital_attack_ratio")) or any(
            _known(_number(row, field))
            for field in ("float_market_cap", "float_market_cap_for_score", "circ_mv", "float_share")
        ),
        "涨跌幅": _known(_number(row, "pct_chg")),
        "行业信息": _industry_available(row),
    }
    if mode == "realtime":
        critical_checks["量比"] = _realtime_ratio_available(row)
        critical_checks["换手率"] = critical_checks["换手率"] and _realtime_turnover_available(row)
    else:
        critical_checks["量比"] = all(
            _known(_number(row, field))
            for field in ("volume", "avg_volume_5d", "avg_volume_10d")
        )

    support_checks = {
        "趋势均线": all(_known(_number(row, field)) for field in ("ma5", "ma10", "ma20")),
        "涨停活跃": _known(_number(row, "limit_up_count_20d")),
        "买点历史": all(
            _known(_number(row, field))
            for field in ("prior_20d_high", "avg_volume_20d")
        ),
        "市场环境": all(
            _known(_number(row, field))
            for field in ("sh_close", "sh_ma20", "up_ratio", "limit_up_count")
        ),
    }

    missing_critical = [label for label, available in critical_checks.items() if not available]
    missing_support = [label for label, available in support_checks.items() if not available]
    score = max(0, 100 - 15 * len(missing_critical) - 5 * len(missing_support))
    status = (
        "完整"
        if score == 100
        else "关键字段缺失" if missing_critical else "部分字段缺失"
    )
    return {
        "data_quality_score": score,
        "data_quality_status": status,
        "data_quality_missing_critical": "；".join(missing_critical),
        "data_quality_missing_support": "；".join(missing_support),
        "top50_data_eligible": not missing_critical,
    }


def evaluate_trading_quality(
    row: dict[str, Any],
    market: dict[str, Any],
    industry: dict[str, Any],
    config: TradingQualityConfig | None = None,
    mode: QualityMode = "legacy_v5_5",
) -> dict[str, Any]:
    """Evaluate shared quality modules without touching the V5.3 base score.

    ``legacy_v5_5`` preserves historical V5.5 results. ``realtime`` and
    ``backtest`` are the V5.8 parity modes and include the turnover-quality
    adjustment requested for the Top50 trading-observation pool.
    """
    _validate_mode(mode)
    config = config or TradingQualityConfig()
    if not config.enabled:
        return {
            "adjustment": 0,
            "market_score": 0,
            "trend_stage_score": 0,
            "trend_stage_risk": 0,
            "trend_stage": "全部模块关闭",
            "buy_point_score": 0,
            "sector_linkage_score": 0,
            "turnover_quality_score": 0,
            "turnover_quality_eligible": True,
            "trading_quality_score": normalize_trading_quality_score(0),
            "mode": mode,
            "detail": "交易质量优化模块关闭",
        }
    quality_row = prepare_v58_quality_row(row) if mode in {"realtime", "backtest"} else row
    market_result = market_score(market, config.market_enabled)
    trend_result = trend_stage_score(quality_row, config.trend_stage_enabled)
    buy_result = buy_point_score(quality_row, config.buy_point_enabled)
    sector_result = sector_linkage_score(industry, config.sector_linkage_enabled)
    turnover_result = (
        turnover_quality_score(quality_row)
        if mode in {"realtime", "backtest"}
        else {"score": 0, "eligible": True, "detail": "V5.5旧口径不计换手质量调整"}
    )
    adjustment = (
        market_result["score"]
        + trend_result["score"]
        + buy_result["score"]
        + sector_result["score"]
        + turnover_result["score"]
    )
    return {
        "adjustment": adjustment,
        "market_score": market_result["score"],
        "trend_stage_score": trend_result["score"],
        "trend_stage_risk": trend_result["risk_penalty"],
        "trend_stage": trend_result["stage"],
        "buy_point_score": buy_result["score"],
        "sector_linkage_score": sector_result["score"],
        "turnover_quality_score": turnover_result["score"],
        "turnover_quality_eligible": turnover_result["eligible"],
        "trading_quality_score": normalize_trading_quality_score(adjustment),
        "mode": mode,
        "detail": "；".join(
            [
                market_result["detail"],
                trend_result["detail"],
                buy_result["detail"],
                sector_result["detail"],
                turnover_result["detail"],
            ]
        ),
    }


def _validate_mode(mode: QualityMode) -> None:
    if mode not in {"legacy_v5_5", "realtime", "backtest"}:
        raise ValueError(f"Unsupported trading-quality mode: {mode}")


def _industry_available(row: dict[str, Any]) -> bool:
    rank = _number(row, "sector_rank")
    raw_name = row.get("sector_name")
    name = "" if raw_name is None or pd.isna(raw_name) else str(raw_name).strip()
    return _known(rank) and rank >= 1 and name not in {"", "暂无", "暂未匹配", "nan", "None"}


def _realtime_turnover_available(row: dict[str, Any]) -> bool:
    flag = row.get("is_realtime_turnover")
    return _truthy(flag) if flag is not None and not pd.isna(flag) else True


def _realtime_ratio_available(row: dict[str, Any]) -> bool:
    if not _known(_number(row, "volume_ratio")):
        return False
    flag = row.get("is_realtime_volume_ratio")
    return _truthy(flag) if flag is not None and not pd.isna(flag) else True
