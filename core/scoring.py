"""Explainable V5.9 base scoring for read-only stock watch results.

V5.9 scores a stock's base capability with five independent factors only:
capital attack, trend structure, turnover participation, volume expansion and
recent limit-up activity.  Market environment, sector linkage, buy-point
quality and entry risk remain outside this module for the V5.8 trading-quality
layer.  Batch scoring then calibrates the raw score against the same-day full
market cross section without forcing weak-market stocks into high score bands.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


RISK_PENALTY_CAP = 15

BASE_SCORE_COMPONENT_MAX = {
    "capital_attack_score": 30,
    "trend_score": 25,
    "turnover_score": 25,
    "volume_score": 15,
    "limit_up_activity_score": 5,
}
RAW_BASE_SCORE_MAX = sum(BASE_SCORE_COMPONENT_MAX.values())


def score_candidates(candidates: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Score a same-day stock universe and apply V5.9 percentile calibration.

    ``score_stock`` remains usable for a single row and returns its raw base
    capability score.  A cross-sectional percentile only has meaning when a
    batch is supplied, so this function performs the second pass that assigns
    ``base_percentile`` and the final ``base_score``/``score`` values.
    """
    if candidates.empty:
        scored = candidates.copy()
        for column, dtype in {
            "score": "float64",
            "raw_base_score": "float64",
            "base_score": "float64",
            "base_percentile": "float64",
            "level": "object",
            "score_detail": "object",
            "score_breakdown": "object",
        }.items():
            scored[column] = pd.Series(dtype=dtype)
        return scored

    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        output_row = row.to_dict()
        output_row.update(score_stock(row, config))
        rows.append(output_row)
    return _apply_base_percentile_scores(pd.DataFrame(rows)).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)


def score_stock(row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Calculate one stock's V5.9 raw base score without a universe ranking.

    The public function signature is unchanged for compatibility.  A single
    row has no valid full-market percentile, therefore ``score`` and
    ``base_score`` equal ``raw_base_score`` here.  Call ``score_candidates``
    for the calibrated same-day score used by the scanner and backtest.
    """
    del config  # Preserve the established public signature.

    detail: list[str] = []
    capital_attack_score, capital_attack_ratio, float_market_cap = _score_capital_attack(row, detail)
    turnover_score, turnover_source = _score_turnover_activity(row, detail)
    trend_score = _score_trend(row, detail)
    volume_score, volume_source = _score_volume_change(row, detail)
    limit_up_activity_score, limit_up_count_20d = _score_limit_up_activity(row, detail)
    _, _, risk_flags, risk_detail = _score_risk_penalties(row)
    confidence, confidence_level, confidence_notes = _data_confidence(
        row,
        volume_source,
        float_market_cap,
        limit_up_count_20d,
    )

    raw_base_score = (
        capital_attack_score
        + turnover_score
        + trend_score
        + volume_score
        + limit_up_activity_score
    )
    raw_base_score = max(0, min(RAW_BASE_SCORE_MAX, int(raw_base_score)))

    # Risk remains visible for downstream quality/entry evaluation, but V5.9
    # does not deduct it from the stock's base capability score.
    detail.extend([f"风险提示（不计入基础分）：{item}" for item in risk_detail])
    breakdown = (
        f"V5.9 基础评分拆解：原始基础分 {raw_base_score}/{RAW_BASE_SCORE_MAX}；"
        f"资金攻击 {capital_attack_score}/30；"
        f"换手 {turnover_score}/25；趋势 {trend_score}/25；量能 {volume_score}/15；"
        f"涨停活跃 {limit_up_activity_score}/5；"
        "最终基础分需在 score_candidates() 中按全市场百分位校准；"
        f"数据可信度 {confidence}/100（{confidence_level}）"
    )
    detail.append(breakdown)

    return {
        "score": raw_base_score,
        "raw_base_score": raw_base_score,
        "base_score": raw_base_score,
        "base_percentile": float("nan"),
        "level": score_level(raw_base_score),
        "capital_attack_score": capital_attack_score,
        "capital_attack_ratio": capital_attack_ratio,
        "float_market_cap_for_score": float_market_cap,
        "turnover_score": turnover_score,
        "turnover_activity_score": turnover_score,
        "turnover_score_source": turnover_source,
        "trend_score": trend_score,
        "volume_ratio_score": volume_score,
        "volume_score": volume_score,
        "volume_ratio_score_source": volume_source,
        "limit_up_activity_score": limit_up_activity_score,
        "limit_up_count_20d": limit_up_count_20d,
        # These V5.3 compatibility fields are intentionally no longer part of
        # the V5.9 base score.  Their logic belongs to V5.8 quality/entry use.
        "sector_heat_score": 0,
        "sector_rank_for_score": _num(row, "sector_rank", "板块涨幅排名"),
        "buy_point_quality_score": 0,
        "high_20d_drawdown_pct": _num(row, "high_20d_drawdown_pct"),
        "buy_point_pattern_score": 0,
        "trend_calibration_score": 0,
        "turnover_calibration_score": 0,
        "continuous_strength_score": 0,
        "up_days_5d_for_score": _num(row, "up_days_5d", "近5日上涨天数"),
        "market_environment_score": 0,
        "market_environment_max": 0,
        "market_environment_detail": "V5.9 基础评分不计市场环境；由 V5.8 交易质量层处理",
        "raw_positive_score": raw_base_score,
        "score_normalization_base": RAW_BASE_SCORE_MAX,
        # Retained for downstream schema compatibility. V5.3 does not assign
        # a positive score for recent strength.
        "continuous_activity_score": 0,
        "consecutive_count_for_score": _num(row, "consecutive_count", "consecutive_selection_days", "连续入选天数"),
        "risk_score": 0,
        "risk_penalty": 0,
        "position_risk_penalty": 0,
        "data_confidence": confidence,
        "data_confidence_level": confidence_level,
        "data_confidence_notes": "；".join(confidence_notes),
        "risk_flags": "；".join(dict.fromkeys(risk_flags)),
        "market_score_cap_applied": False,
        "score_breakdown": breakdown,
        "score_detail": "；".join(detail),
        # Compatibility fields: V5.9 does not score these modules in base.
        "amount_score": 0,
        "liquidity_score": 0,
        "fund_score": capital_attack_score,
        "pct_chg_score": 0,
        "market_cap_score": 0,
        "strategy_score": 0,
    }


def _apply_base_percentile_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Convert raw base scores into guarded same-day percentile scores.

    Percentile bands create a stable cross-sectional distribution, while raw
    floors and component gates ensure a weak market cannot mechanically create
    90-point stocks merely because every other stock is weaker.
    """
    output = scored.copy()
    raw = pd.to_numeric(output["raw_base_score"], errors="coerce").fillna(0).clip(0, RAW_BASE_SCORE_MAX)
    percentile = _stable_market_percentile(output, raw)
    output["base_percentile"] = percentile.round(3)

    base_scores: list[int] = []
    for index, raw_score in raw.items():
        row = output.loc[index]
        base_scores.append(_calibrated_base_score(float(raw_score), float(percentile.loc[index]), row))

    output["base_score"] = base_scores
    output["score"] = output["base_score"]
    output["level"] = output["base_score"].map(score_level)
    output["score_breakdown"] = output.apply(_append_percentile_breakdown, axis=1)
    output["score_detail"] = output.apply(_append_percentile_detail, axis=1)
    return output


def _stable_market_percentile(data: pd.DataFrame, raw: pd.Series) -> pd.Series:
    """Return a deterministic percentile when discrete component scores tie.

    Component tiers naturally create many equal raw scores.  Using an average
    rank would put every tied row in the same high band and defeat the intended
    Top25/Top125/Top450 distribution.  This ranking keeps raw score primary,
    then uses already available continuous base inputs solely as deterministic
    tie breakers; it does not add an extra score component or weight.
    """
    ranking = pd.DataFrame(index=data.index)
    ranking["raw_base_score"] = raw
    ranking["capital_attack_ratio"] = pd.to_numeric(
        data.get("capital_attack_ratio", pd.Series(index=data.index, dtype="float64")),
        errors="coerce",
    ).fillna(float("-inf"))
    ranking["turnover"] = pd.to_numeric(
        data.get("turnover", pd.Series(index=data.index, dtype="float64")),
        errors="coerce",
    ).fillna(float("-inf"))
    ranking["volume_ratio"] = pd.to_numeric(
        data.get("volume_ratio", pd.Series(index=data.index, dtype="float64")),
        errors="coerce",
    ).fillna(float("-inf"))
    ranking["amount"] = pd.to_numeric(
        data.get("amount", pd.Series(index=data.index, dtype="float64")),
        errors="coerce",
    ).fillna(float("-inf"))
    ranking["code"] = data.get("code", pd.Series(index=data.index, dtype="object")).fillna("").astype(str)
    ranked_index = ranking.sort_values(
        ["raw_base_score", "capital_attack_ratio", "turnover", "volume_ratio", "amount", "code"],
        ascending=[True, True, True, True, True, True],
        kind="stable",
    ).index
    rank = pd.Series(range(1, len(ranked_index) + 1), index=ranked_index, dtype="float64")
    return rank.reindex(data.index).div(len(data)).mul(100)


def _calibrated_base_score(raw_score: float, percentile: float, row: pd.Series) -> int:
    """Map a raw score to V5.9 bands with absolute-quality safeguards."""
    if percentile > 99.5:
        score = 90 + round((percentile - 99.5) / 0.5 * 10)
    elif percentile > 97:
        score = 80 + round((percentile - 97) / 2.5 * 9)
    elif percentile > 88:
        score = 70 + round((percentile - 88) / 9 * 9)
    else:
        # Below the observation threshold, retain an absolute raw-score
        # expression rather than manufacturing a high relative score.
        score = min(69, round(raw_score * 0.9))

    capital = _num(row, "capital_attack_score")
    trend = _num(row, "trend_score")
    turnover = _num(row, "turnover_score")
    volume = _num(row, "volume_score")

    # Absolute raw floors prevent a universally weak day from filling the
    # upper percentile bands with weak stocks.
    if raw_score < 40:
        score = min(score, 69)
    elif raw_score < 55:
        score = min(score, 79)
    elif raw_score < 70:
        score = min(score, 89)

    # The agreed 90-point gate is deliberately stricter than rank alone.
    if not (
        capital >= 21
        and trend >= 15
        and turnover >= 10
        and volume >= 5
    ):
        score = min(score, 89)

    return max(0, min(100, int(score)))


def _append_percentile_breakdown(row: pd.Series) -> str:
    """Add calibration facts to the raw V5.9 scoring breakdown."""
    raw = pd.to_numeric(row.get("raw_base_score"), errors="coerce")
    percentile = pd.to_numeric(row.get("base_percentile"), errors="coerce")
    score = pd.to_numeric(row.get("base_score"), errors="coerce")
    raw_text = "--" if pd.isna(raw) else f"{raw:.0f}/{RAW_BASE_SCORE_MAX}"
    percentile_text = "--" if pd.isna(percentile) else f"{percentile:.3f}%"
    score_text = "--" if pd.isna(score) else f"{score:.0f}/100"
    return (
        f"V5.9 基础评分拆解：原始基础分 {raw_text}；"
        f"全市场百分位 {percentile_text}；最终基础分 {score_text}；"
        "90分须资金攻击≥21、趋势≥15、换手≥10、量能≥5"
    )


def _append_percentile_detail(row: pd.Series) -> str:
    """Preserve raw details while explaining the batch percentile result."""
    original = str(row.get("score_detail") or "")
    breakdown = str(row.get("score_breakdown") or "")
    return "；".join(part for part in (original, breakdown) if part)


def score_level(score: int | float) -> str:
    """Map a numeric score to a Chinese watch level."""
    if score >= 90:
        return "龙头强势池"
    if score >= 85:
        return "主力关注池"
    if score >= 80:
        return "观察池"
    return "过滤"


def _score_capital_attack(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, float, float]:
    """Score amount / float-market-cap attack strength, maximum 30 points."""
    amount = _num(row, "amount", "成交额")
    float_market_cap = _float_market_cap(row)
    if pd.isna(amount) or pd.isna(float_market_cap) or float_market_cap <= 0:
        detail.append("成交额/流通市值数据不足，资金攻击不评分")
        return 0, float("nan"), float_market_cap
    ratio = amount / float_market_cap * 100
    if ratio < 1:
        score = 0
    elif ratio < 3:
        score = 9
    elif ratio < 8:
        score = 21
    elif ratio <= 15:
        score = 30
    else:
        score = 24
    detail.append(f"成交额/流通市值 {ratio:.2f}% +{score}")
    return score, ratio, float_market_cap


def _score_turnover_activity(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, str]:
    """Score participation heat from realtime turnover only, maximum 25."""
    turnover = _num(row, "realtime_turnover_value", "turnover", "换手率")
    if pd.isna(turnover):
        detail.append("实时换手率缺失，参与热度不评分")
        return 0, "missing"
    if _is_growth_board(row):
        limits = (2.0, 5.0, 12.0, 25.0, 35.0)
        label = "创业板/科创板"
    else:
        limits = (1.0, 3.0, 8.0, 20.0, 30.0)
        label = "普通主板"
    low, normal, active, high, peak = limits
    if turnover < low:
        score = 0
    elif turnover < normal:
        score = 5
    elif turnover < active:
        score = 12
    elif turnover <= high:
        score = 20
    elif turnover <= peak:
        score = 25
    else:
        score = 15
    detail.append(f"{label}换手率 {turnover:.2f}% +{score}")
    return score, "realtime"


def _score_trend(row: pd.Series | dict[str, Any], detail: list[str]) -> int:
    """Trend strength, maximum 25 points."""
    price = _num(row, "price", "最新价")
    ma5 = _num(row, "ma5", "MA5")
    ma10 = _num(row, "ma10", "MA10")
    ma20 = _num(row, "ma20", "MA20")
    price_above_ma5 = _gt(price, ma5)
    ma5_above_ma10 = _gt(ma5, ma10)
    ma10_above_ma20 = _gt(ma10, ma20)
    score = 0
    if price_above_ma5:
        score += 7
        detail.append("最新价大于 MA5 +7")
    if ma5_above_ma10:
        score += 7
        detail.append("MA5 大于 MA10 +7")
    if ma10_above_ma20:
        score += 5
        detail.append("MA10 大于 MA20 +5")
    if price_above_ma5 and ma5_above_ma10 and ma10_above_ma20:
        score += 6
        detail.append("趋势多头排列 +6")
    return score


def _score_trend_calibration(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, int]:
    """Give a small bonus only to a fully aligned four-level trend."""
    price = _num(row, "price", "最新价")
    ma5 = _num(row, "ma5", "MA5")
    ma10 = _num(row, "ma10", "MA10")
    ma20 = _num(row, "ma20", "MA20")
    if any(pd.isna(value) for value in (price, ma5, ma10, ma20)):
        detail.append("趋势校准数据缺失，不评分")
        return 0, 0
    if price > ma5 > ma10 > ma20:
        detail.append("收盘价>MA5>MA10>MA20，趋势校准 +5")
        return 5, 5
    detail.append("未满足完整多头排列，趋势校准 +0")
    return 0, 5


def _score_turnover_calibration(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, int]:
    """Reward healthy, not merely high, turnover inside the calibration band."""
    turnover = _num(row, "realtime_turnover_value", "turnover", "换手率")
    if pd.isna(turnover):
        detail.append("换手率缺失，换手校准不评分")
        return 0, 0
    if _is_growth_board(row):
        low, high, label = 5.0, 25.0, "创业板/科创板"
    else:
        low, high, label = 3.0, 20.0, "普通主板"
    if low <= turnover <= high:
        detail.append(f"{label}换手率 {turnover:.2f}% 健康，换手校准 +3")
        return 3, 3
    detail.append(f"{label}换手率 {turnover:.2f}% 未在健康区间，换手校准 +0")
    return 0, 3


def _score_volume_change(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, str]:
    """Score incremental volume, maximum 15; avoid rewarding climax volume."""
    ratio = _num(row, "volume_ratio", "量比")
    if pd.notna(ratio):
        if ratio < 1.2:
            score = 0
        elif ratio < 1.5:
            score = 5
        elif ratio < 2:
            score = 9
        elif ratio <= 3:
            score = 15
        else:
            score = 10
        detail.append(f"实时量比 {ratio:.2f} +{score}")
        return score, "realtime"

    volume = _num(row, "volume", "成交量")
    avg_volume_5d = _num(row, "avg_volume_5d", "5日均量")
    avg_volume_10d = _num(row, "avg_volume_10d", "10日均量")
    score = 0
    available = False
    if _gt(volume, avg_volume_5d):
        score += 7
        available = True
        detail.append("今日成交量大于 5 日均量 +7")
    elif pd.notna(volume) and pd.notna(avg_volume_5d):
        available = True
    if _gt(volume, avg_volume_10d):
        score += 8
        available = True
        detail.append("今日成交量大于 10 日均量 +8")
    elif pd.notna(volume) and pd.notna(avg_volume_10d):
        available = True
    if not available:
        detail.append("实时量比及历史量能数据缺失，量能不评分")
        return 0, "missing"
    return min(score, 15), "history"


def _score_limit_up_activity(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, float]:
    """Score individual limit-up frequency in the latest 20 days, maximum 5."""
    count = _num(
        row,
        "limit_up_count_20d",
        "limit_up_times_20d",
        "recent_20d_limit_up_count",
        "近20日涨停次数",
    )
    if pd.isna(count):
        detail.append("近20日涨停次数缺失，涨停活跃不评分")
        return 0, float("nan")
    if count <= 0:
        score = 0
    elif count == 1:
        score = 2
    elif count == 2:
        score = 4
    else:
        score = 5
    detail.append(f"近20日涨停 {int(count)} 次 +{score}")
    return score, count


def _score_sector_heat(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, float]:
    """Score stock-level mapped sector performance rank, maximum 10 points."""
    rank = _num(row, "sector_rank", "板块涨幅排名")
    if pd.isna(rank):
        detail.append("所属板块涨幅排名缺失，板块热度不评分")
        return 0, float("nan")
    if rank < 1:
        detail.append("所属板块涨幅排名无效，板块热度不评分")
        return 0, rank
    if rank <= 5:
        score = 10
    elif rank <= 10:
        score = 6
    else:
        score = 0
    detail.append(f"所属板块涨幅排名第 {int(rank)} +{score}")
    return score, rank


def _score_buy_point_quality(row: pd.Series | dict[str, Any], detail: list[str]) -> tuple[int, float, int]:
    """Score 20-day position and low-volume-pullback entry quality, max 20.

    The position and pattern portions are deliberately capped together.  This
    lets a well-supported breakout rank above a generic high-position stock
    without allowing the two descriptions to inflate the 20-point module.
    """
    price = _num(row, "price", "最新价")
    prior_high_20d = _num(row, "prior_20d_high", "20日前最高价", "前20日最高价")
    high_20d = _num(row, "high_20d", "20日最高价")
    reference_high = prior_high_20d if pd.notna(prior_high_20d) and prior_high_20d > 0 else high_20d
    volume = _num(row, "volume", "成交量")
    avg_volume_5d = _num(row, "avg_volume_5d", "5日均量")
    pct_chg = _num(row, "pct_chg", "涨跌幅")
    pct_chg_5d = _num(row, "pct_chg_5d", "5日涨幅")
    ma20 = _num(row, "ma20", "MA20")
    recent_pullback = _truthy(_value(row, "recent_low_volume_pullback", "近期缩量回调"))

    if pd.isna(price) or pd.isna(reference_high) or reference_high <= 0:
        detail.append("20日最高价数据不足，买点位置不评分")
        return 0, float("nan"), 0

    drawdown = max(0.0, (reference_high - price) / reference_high * 100)
    breakout = price > reference_high
    volume_expansion = _gt(volume, avg_volume_5d * 1.3)
    position_score = 0
    if breakout and volume_expansion:
        position_score = 10
        detail.append("突破前20日高点且放量 +10")
    elif breakout:
        position_score = 8
        detail.append("重新突破前20日高点 +8")
    elif drawdown <= 5:
        detail.append(f"距20日高点 {drawdown:.2f}%，位置偏高 +0")
    elif drawdown <= 10:
        position_score = 3
        detail.append(f"距20日高点回调 {drawdown:.2f}% +3")
    elif drawdown <= 20:
        position_score = 5
        detail.append(f"距20日高点回调 {drawdown:.2f}% +5")
    else:
        detail.append(f"距20日高点回调 {drawdown:.2f}% 过深 +0")

    pattern_score = 0
    if breakout and volume_expansion and recent_pullback:
        pattern_score = 10
        detail.append("缩量调整后重新放量突破 +10")
    elif (
        pd.notna(pct_chg)
        and pct_chg <= 0
        and pd.notna(pct_chg_5d)
        and pct_chg_5d > 0
        and pd.notna(volume)
        and pd.notna(avg_volume_5d)
        and volume < avg_volume_5d
        and _gt(price, ma20)
    ):
        pattern_score = 5
        detail.append("上涨后缩量调整 +5")

    return min(20, position_score + pattern_score), drawdown, pattern_score


def _score_market_environment(
    row: pd.Series | dict[str, Any], detail: list[str]
) -> tuple[int, int, str]:
    """Score available market context, with a maximum contribution of 10.

    The 100 stock-factor points remain intact. The available maximum is used
    as the extra normalization capacity, so missing market fields cannot lower
    an otherwise identical stock's score.
    """
    score = 0
    available_max = 0
    reasons: list[str] = []
    sh_close = _num(row, "sh_close", "上证收盘价")
    sh_ma5 = _num(row, "sh_ma5", "上证5日均线")
    sh_ma10 = _num(row, "sh_ma10", "上证10日均线")
    sh_pct_chg = _num(row, "sh_pct_chg", "上证涨跌幅")
    market_amount = _num(row, "market_amount", "市场成交额")
    market_amount_ma20 = _num(row, "market_amount_ma20", "市场20日均额")
    up_count = _num(row, "up_count", "上涨家数")
    down_count = _num(row, "down_count", "下跌家数")

    if pd.notna(sh_close) and pd.notna(sh_ma5):
        available_max += 3
        if sh_close > sh_ma5:
            score += 3
            reasons.append("上证收盘高于MA5 +3")
        else:
            reasons.append("上证收盘未高于MA5 +0")
    else:
        reasons.append("上证收盘/MA5缺失，不评分")
    if pd.notna(sh_ma5) and pd.notna(sh_ma10):
        available_max += 2
        if sh_ma5 > sh_ma10:
            score += 2
            reasons.append("上证MA5高于MA10 +2")
        else:
            reasons.append("上证MA5未高于MA10 +0")
    else:
        reasons.append("上证MA5/MA10缺失，不评分")

    if pd.notna(sh_pct_chg):
        available_max += 2
        if sh_pct_chg > 1:
            score += 2
            reasons.append(f"上证当日涨幅 {sh_pct_chg:.2f}% +2")
        elif sh_pct_chg >= 0:
            score += 1
            reasons.append(f"上证当日涨幅 {sh_pct_chg:.2f}% +1")
        elif sh_pct_chg < -1:
            score -= 2
            reasons.append(f"上证当日跌幅 {sh_pct_chg:.2f}% -2")
        else:
            reasons.append(f"上证当日涨跌 {sh_pct_chg:.2f}% +0")
    else:
        reasons.append("上证当日涨跌幅缺失，不评分")

    if pd.notna(market_amount) and pd.notna(market_amount_ma20) and market_amount_ma20 > 0:
        available_max += 3
        amount_ratio = market_amount / market_amount_ma20
        if amount_ratio >= 1.2:
            score += 3
            reasons.append(f"市场成交额为20日均额 {amount_ratio:.2f} 倍，放量 +3")
        elif amount_ratio >= 0.8:
            score += 1
            reasons.append(f"市场成交额为20日均额 {amount_ratio:.2f} 倍，正常 +1")
        else:
            reasons.append(f"市场成交额为20日均额 {amount_ratio:.2f} 倍，缩量 +0")
    else:
        reasons.append("市场成交额/20日均额缺失，不评分")

    # Breadth reuses the prior environment convention. It is optional and
    # only adjusts the available 10-point market module; absence has no cost.
    if pd.notna(up_count) and pd.notna(down_count) and up_count + down_count > 0:
        breadth_ratio = up_count / (up_count + down_count)
        if breadth_ratio > 0.6:
            score += 3
            reasons.append(f"上涨家数占比 {breadth_ratio:.1%}，市场宽度强 +3")
        elif breadth_ratio >= 0.4:
            score += 1
            reasons.append(f"上涨家数占比 {breadth_ratio:.1%}，市场宽度中性 +1")
        else:
            score -= 2
            reasons.append(f"上涨家数占比 {breadth_ratio:.1%}，市场宽度弱 -2")
        available_max = min(10, available_max + 3)
    else:
        reasons.append("上涨/下跌家数缺失，市场宽度不评分")

    score = max(-4, min(10, score))
    detail.append(f"市场环境：{score}/{available_max or 10}（{'；'.join(reasons)}）")
    return score, available_max, "；".join(reasons)


def _score_risk_penalties(row: pd.Series | dict[str, Any]) -> tuple[int, int, list[str], list[str]]:
    """Apply the four retained V5.3 risk deductions, capped at -15."""
    raw_penalty = 0
    position_penalty = 0
    flags: list[str] = []
    detail: list[str] = []
    name = str(_value(row, "name", "名称") or "")
    pct_chg_20d = _num(row, "pct_chg_20d", "20日涨幅")
    amount = _num(row, "amount", "成交额")
    avg_amount_5d = _num(row, "avg_amount_5d", "5日平均成交额", "5日均成交额")
    limit_up_days = _num(row, "consecutive_limit_up_days", "limit_up_days", "连续涨停天数")

    if "ST" in name.upper():
        raw_penalty += 15
        flags.append("ST股票")
        detail.append("ST 股票 -15")
    if pd.notna(limit_up_days) and limit_up_days > 3:
        raw_penalty += 5
        flags.append("连续涨停超过3天")
        detail.append("连续涨停超过 3 天 -5")
    if _gt(amount, avg_amount_5d * 5):
        raw_penalty += 3
        flags.append("成交额超过5日均值5倍")
        detail.append("成交额突然放大超过历史均值 5 倍 -3")

    if pd.notna(pct_chg_20d) and pct_chg_20d > 80:
        position_penalty = 15
        flags.append("20日涨幅超过80%")
        detail.append("近20交易日涨幅超过 80% -15")
    elif pd.notna(pct_chg_20d) and pct_chg_20d > 50:
        position_penalty = 10
        flags.append("20日涨幅超过50%")
        detail.append("近20交易日涨幅超过 50% -10")
    elif pd.notna(pct_chg_20d) and pct_chg_20d > 30:
        position_penalty = 5
        flags.append("20日涨幅超过30%")
        detail.append("近20交易日涨幅超过 30% -5")
    raw_penalty += position_penalty


    applied = min(RISK_PENALTY_CAP, raw_penalty)
    if raw_penalty > RISK_PENALTY_CAP:
        detail.append("风险扣分封顶 -15")
    return -applied, -position_penalty, flags, detail


def _data_confidence(
    row: pd.Series | dict[str, Any],
    volume_source: str,
    float_market_cap: float,
    limit_up_count_20d: float,
) -> tuple[int, str, list[str]]:
    """Report completeness of V5.9 base-score inputs without deductions."""
    deductions = 0
    notes: list[str] = []
    for key, label in (("price", "价格"), ("ma5", "MA5"), ("ma10", "MA10"), ("ma20", "MA20")):
        if pd.isna(_num(row, key)):
            deductions += 5
            notes.append(f"缺少{label}")
    if pd.isna(_num(row, "amount", "成交额")):
        deductions += 10
        notes.append("缺少成交额")
    if pd.isna(_num(row, "turnover", "换手率")):
        deductions += 10
        notes.append("缺少实时换手率")
    if pd.isna(float_market_cap):
        deductions += 15
        notes.append("缺少流通市值")
    if volume_source == "missing":
        deductions += 10
        notes.append("缺少量能数据")
    if pd.isna(limit_up_count_20d):
        deductions += 5
        notes.append("缺少20日涨停次数")
    confidence = max(0, 100 - deductions)
    level = "高" if confidence >= 85 else "中" if confidence >= 60 else "低"
    return confidence, level, notes or ["评分所需关键数据完整"]


def _float_market_cap(row: pd.Series | dict[str, Any]) -> float:
    """Return float market cap in yuan from realtime or Tushare reference fields."""
    realtime_cap = _num(row, "float_market_cap", "流通市值", "circulating_market_cap")
    if pd.notna(realtime_cap) and realtime_cap > 0:
        return realtime_cap
    # Tushare daily_basic circ_mv is quoted in 万元.
    tushare_cap = _num(row, "circ_mv")
    if pd.notna(tushare_cap) and tushare_cap > 0:
        return tushare_cap * 10_000
    # Tushare float_share is in 万股; combine it with the current price.
    float_share = _num(row, "float_share")
    price = _num(row, "price", "最新价")
    if pd.notna(float_share) and pd.notna(price) and float_share > 0 and price > 0:
        return float_share * 10_000 * price
    return float("nan")


def _is_growth_board(row: pd.Series | dict[str, Any]) -> bool:
    board = str(_value(row, "board_type", "board_type_display", "板块类型") or "")
    return board in {"chi_next", "star_market", "创业板", "科创板"}


def _value(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    """Get the first non-null value from a row-like object."""
    for key in keys:
        value = row.get(key) if hasattr(row, "get") else None
        if value is not None and not pd.isna(value):
            return value
    return None


def _num(row: pd.Series | dict[str, Any], *keys: str) -> float:
    value = _value(row, *keys)
    if value is None:
        return float("nan")
    return float(pd.to_numeric(value, errors="coerce"))


def _gt(left: float, right: float) -> bool:
    return pd.notna(left) and pd.notna(right) and left > right


def _truthy(value: Any) -> bool:
    """Interpret common dataframe boolean encodings without treating NA as true."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)
