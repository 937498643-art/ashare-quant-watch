"""Streamlit dashboard for the read-only A-share watch system."""

from __future__ import annotations

import json
import re
import sys
from html import escape
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from backtest.paper_account import PaperAccountConfig, run_v59_paper_account
from core.detail_analysis import build_detail_analysis, prepare_history_detail
from core.t1_risk import analyze_t1_risk
from data_sources.akshare_source import AkshareSource
from data_sources.tushare_kline_cache_reader import read_tushare_kline_from_cache, update_stock_kline_cache
from storage.holdings_store import load_user_holdings, remove_user_holding, upsert_user_holding
from storage.strategy_tracking_store import (
    add_strategy_tracking_stock,
    build_strategy_tracking_statistics,
    end_strategy_tracking_stock,
    load_strategy_tracking_pool,
)
from storage.watchlist_store import add_user_watchlist_stock, load_user_watchlist, remove_user_watchlist_stock, update_user_watchlist_stock


DATA_DIR = PROJECT_ROOT / "data"
LATEST_CANDIDATES_PATH = DATA_DIR / "latest_candidates.csv"
LATEST_MARKET_STATUS_PATH = DATA_DIR / "latest_market_status.json"
LATEST_WATCHLIST_PATH = DATA_DIR / "latest_watchlist.csv"
LATEST_HOLDINGS_PATH = DATA_DIR / "latest_holdings.csv"
REFERENCE_CANDIDATES_PATH = DATA_DIR / "output" / "reference_candidates.csv"
# main.py currently publishes the V5.9 full-market Top50 snapshot at this
# established output location.  ``load_v59_top50`` validates the V5.9-specific
# fields so a legacy V5.8-only snapshot is never presented as V5.9 data.
V59_TOP50_PATH = DATA_DIR / "output" / "v5_8_top50.csv"
V59_REQUIRED_OUTPUT_COLUMNS = {
    "code",
    "name",
    "price",
    "pct_chg",
    "amount",
    "turnover",
    "raw_base_score",
    "base_percentile",
    "base_score",
    "trading_quality_score",
    "final_trade_score",
}
CALCULATED_VOLUME_RATIO_PATH = DATA_DIR / "diagnostics" / "calculated_volume_ratio_sample.csv"
V59_CANDIDATE_HISTORY_DIR = DATA_DIR / "history" / "candidate_pools"
HISTORICAL_DAILY_DIR = DATA_DIR / "history" / "daily"
STRATEGY_TRACKING_HISTORY_PATH = DATA_DIR / "user" / "strategy_tracking_history.csv"
V59_PERFORMANCE_HOLDING_DAYS = (3, 5, 10)

SOURCE_TYPE_STRATEGY = "strategy_candidate"
SOURCE_TYPE_ACTIVE = "active_watchlist"
SOURCE_TYPE_REFERENCE = "reference_candidate"

SOURCE_TYPE_LABELS = {
    SOURCE_TYPE_STRATEGY: "策略候选股",
    SOURCE_TYPE_ACTIVE: "市场异动观察池",
    SOURCE_TYPE_REFERENCE: "参考候选股",
}
STRATEGY_LABELS = {
    "trend_bullish": "趋势多头",
    "volume_breakout": "放量突破",
    "pullback_low_volume": "缩量回踩",
    "active_watchlist": "市场异动观察池",
}
LEVEL_LABELS = {
    "key_watch": "重点关注",
    "watch": "加入观察",
    "normal_watch": "普通观察",
    "ignore": "暂不关注",
    "重点关注": "重点关注",
    "加入观察": "加入观察",
    "普通观察": "普通观察",
    "暂不关注": "暂不关注",
}
BOOL_LABELS = {"True": "是", "False": "否", True: "是", False: "否"}
BOARD_DISPLAY_ORDER = ["上证主板", "深证主板", "创业板", "科创板", "北交所", "其他"]

DISPLAY_COLUMNS = [
    "source_type_display",
    "board_type_display",
    "code",
    "market_code",
    "name",
    "price_display",
    "pct_chg_display",
    "amount_display",
    "turnover_display",
    "turnover_level_display",
    "volume_ratio_display",
    "money_strength_level",
    "sector_name",
    "sector_strength_level",
    "theme_tags",
    "strategy_names_display",
    "score_display",
    "base_score_display",
    "trading_quality_score_display",
    "final_trade_score_display",
    "level_display",
    "action_state_display",
    "position_risk_level",
    "t1_risk_level",
    "tracking_summary",
    "reason",
    "risk_summary",
    "data_source",
]

ACTIVE_DISPLAY_COLUMNS = [
    "name",
    "code",
    "price_display",
    "pct_chg_display",
    "amount_display",
    "turnover_display",
    "volume_ratio_display",
    "score_display",
    "base_score_display",
    "trading_quality_score_display",
    "final_trade_score_display",
    "level_display",
    "tracking_summary",
    "data_status_display",
    "position_risk_level",
    "t1_risk_level",
    "action_state_display",
]

V59_TOP50_DISPLAY_COLUMNS = [
    "code",
    "name",
    "price_display",
    "pct_chg_display",
    "amount_display",
    "turnover_display",
    "raw_base_score",
    "base_percentile",
    "base_score",
    "trading_quality_score",
    "final_trade_score",
]

TODAY_TRADE_DISPLAY_COLUMNS = [
    "name",
    "code",
    "price_display",
    "final_trade_score_display",
    "base_score_display",
    "turnover_display",
    "volume_ratio_display",
    "pct_chg_display",
    "amount_display",
    "money_strength_level",
    "data_quality_level_display",
    "trade_risk_level",
    "t1_risk_level",
    "buy_point_status_display",
    "buy_status_display",
]

ACTIVE_EXTRA_COLUMNS = [
    "code",
    "name",
    "turnover_display",
    "turnover_source_display",
    "volume_ratio_source_display",
    "volume_ratio_ref_display",
    "total_mv_display",
    "circ_mv_display",
    "reference_trade_date_display",
    "risk_summary",
    "data_source",
]

REFERENCE_DISPLAY_COLUMNS = [
    "name",
    "code",
    "price_display",
    "pct_chg_display",
    "amount_display",
    "turnover_display",
    "volume_ratio_display",
    "data_status_display",
]

COLUMN_LABELS = {
    "source_type_display": "数据类型",
    "board_type_display": "市场归类",
    "code": "股票代码",
    "market_code": "市场代码",
    "name": "股票名称",
    "price": "最新价",
    "price_display": "最新价",
    "pct_chg_display": "涨跌幅",
    "amount_display": "成交额",
    "turnover_display": "换手率",
    "turnover_source_display": "换手率来源",
    "turnover_level_display": "换手状态",
    "volume_ratio_display": "量比",
    "volume_ratio_source_display": "量比来源",
    "data_status_display": "数据状态",
    "volume_ratio_ref_display": "参考量比",
    "total_mv_display": "总市值",
    "circ_mv_display": "流通市值",
    "reference_trade_date_display": "参考数据日期",
    "money_strength_level": "资金强度",
    "sector_name": "板块",
    "sector_strength_level": "板块强度",
    "theme_tags": "题材标签",
    "strategy_names_display": "触发策略",
    "score": "评分",
    "score_display": "评分",
    "raw_base_score": "原始基础分",
    "base_percentile": "全市场百分位",
    "base_score": "基础评分",
    "trading_quality_score": "交易质量评分",
    "final_trade_score": "最终交易评分",
    "base_score_display": "基础评分",
    "trading_quality_score_display": "交易质量评分",
    "final_trade_score_display": "最终交易评分",
    "data_quality_score_display": "数据完整性",
    "data_quality_status": "数据完整状态",
    "data_quality_level_display": "数据质量",
    "level_display": "等级",
    "action_state_display": "操作状态",
    "trade_risk_level": "风险等级",
    "buy_point_status_display": "买点状态",
    "buy_status_display": "买入状态",
    "position_risk_level": "位置风险",
    "t1_risk_level": "T+1 风险",
    "tracking_summary": "连续入选",
    "reason": "触发原因",
    "risk_summary": "风险提示",
    "data_source": "数据源",
    "volume_ratio_ref_display": "参考量比",
    "total_mv_display": "总市值",
    "circ_mv_display": "流通市值",
    "reference_reason": "触发原因",
    "reference_warning": "风险提示",
    "warning_message": "数据说明",
}

HOLDING_COLUMN_LABELS = {
    "code": "股票代码",
    "name": "股票名称",
    "cost_price": "成本价",
    "price": "当前价",
    "shares": "持仓数量",
    "market_value": "持仓市值",
    "floating_pnl": "浮动盈亏",
    "floating_pnl_ratio": "盈亏比例",
    "buy_date": "买入日期",
    "tag": "标签",
    "note": "备注",
    "data_status": "数据状态",
}


def apply_dashboard_style() -> None:
    """Apply a compact, terminal-inspired visual system across the dashboard."""
    st.markdown(
        """
        <style>
        :root {
            --terminal-navy: #102a43;
            --terminal-blue: #1d4ed8;
            --terminal-muted: #667085;
            --terminal-surface: #ffffff;
            --terminal-border: #dce3ec;
        }
        .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #f8fafc 34%, #f6f8fb 100%);
            color: #102a43;
        }
        .main .block-container {
            max-width: 1720px;
            padding: 0.8rem 1.15rem 1.5rem;
        }
        section[data-testid="stSidebar"] {
            background: #f8fafc;
        }
        h1 {
            color: var(--terminal-navy);
            font-size: 1.72rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0 0 0.08rem;
            letter-spacing: -0.025em;
        }
        h2, h3 {
            color: var(--terminal-navy);
            letter-spacing: -0.012em;
            margin-top: 0.95rem;
            margin-bottom: 0.42rem;
        }
        h2 { font-size: 1.18rem; }
        h3 { font-size: 1.02rem; }
        div[data-testid="stCaptionContainer"] {
            color: #6b7280;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.18rem;
            border-bottom: 1px solid var(--terminal-border);
            margin: 0.35rem 0 0.7rem;
        }
        div[data-testid="stTabs"] button {
            font-size: 0.88rem;
            font-weight: 650;
            color: #5d6878;
            height: 2.16rem;
            padding: 0 0.72rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--terminal-blue);
        }
        div[data-testid="stAlert"] {
            border-radius: 10px;
            padding: 0.48rem 0.72rem;
            margin: 0.35rem 0 0.6rem;
            border-width: 1px;
        }
        div[data-testid="stAlert"] p {
            font-size: 0.88rem;
            line-height: 1.4;
        }
        details[data-testid="stExpander"], div[data-testid="stExpander"] {
            border: 1px solid var(--terminal-border);
            border-radius: 10px;
            background: var(--terminal-surface);
        }
        details[data-testid="stExpander"] summary {
            color: #344054;
            font-weight: 600;
        }
        div[data-testid="stMetric"] {
            background: var(--terminal-surface);
            border: 1px solid var(--terminal-border);
            border-radius: 10px;
            padding: 9px 11px;
            min-height: 70px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--terminal-muted);
            font-size: 0.76rem;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: var(--terminal-navy);
            font-size: 1.14rem;
            font-weight: 740;
        }
        .stDataFrame {
            border: 1px solid var(--terminal-border);
            border-radius: 10px;
            background: var(--terminal-surface);
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.035);
        }
        .watch-card {
            background: var(--terminal-surface);
            border: 1px solid var(--terminal-border);
            border-radius: 10px;
            padding: 10px 12px;
            min-height: 72px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.055);
        }
        .watch-card-red {
            border-color: #f3cece;
            background: #fff9f9;
        }
        .watch-card-green {
            border-color: #cce7d3;
            background: #fbfefc;
        }
        .watch-card-risk {
            border-color: #e7bbbb;
            background: #fff7f7;
        }
        .watch-card-gray {
            border-color: #e5e7eb;
            background: #fbfbfc;
        }
        .watch-card-label {
            color: var(--terminal-muted);
            font-size: 0.75rem;
            font-weight: 620;
            margin-bottom: 5px;
        }
        .watch-card-value {
            font-size: 1.14rem;
            font-weight: 760;
            line-height: 1.25;
            word-break: break-word;
        }
        .tone-red { color: #d62728; }
        .tone-green { color: #118642; }
        .tone-gray { color: #6b7280; }
        .tone-neutral { color: #111827; }
        .tone-risk { color: #991b1b; }
        .small-hint {
            color: #4b5563;
            font-size: 0.86rem;
            line-height: 1.35;
            margin: 4px 0 8px 0;
        }
        .small-hint-ok { color: #1f8f4d; }
        .small-hint-muted { color: #6b7280; }
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        @media (max-width: 900px) {
            .main .block-container {
                padding: 0.7rem 0.7rem 1.2rem;
            }
            h1 { font-size: 1.45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render dashboard."""
    st.set_page_config(
        page_title="A股智能量化盯盘系统 V5.9",
        page_icon=":material/query_stats:",
        layout="wide",
    )
    apply_dashboard_style()
    initialize_dashboard_state()
    st.title("A股智能量化盯盘系统 V5.9")
    st.caption("实时选股观察 · 风险复核 · 策略跟踪｜仅供辅助观察，不构成交易建议。")

    status = load_market_status()
    candidates = load_candidates()
    v59_top50 = load_v59_top50()
    reference_candidates = load_reference_candidates()
    watchlist = load_auxiliary_csv(LATEST_WATCHLIST_PATH)
    holdings = load_auxiliary_csv(LATEST_HOLDINGS_PATH)
    strategy_candidates = candidates[candidates["source_type"] == SOURCE_TYPE_STRATEGY]
    today_trade_pool, top50_source_count = build_today_trade_pool(
        v59_top50,
        market_environment=str(status.get("market_environment") or ""),
    )
    detail_candidates = build_detail_candidates(
        v59_top50,
        today_trade_pool,
        candidates,
        reference_candidates,
    )
    render_top_status(status, candidate_count=len(today_trade_pool))
    tabs = st.tabs(
        ["今日交易候选池", "V5.9 Top50交易观察池", "策略候选池", "策略跟踪池", "自选股 / 持仓股", "板块与大盘", "个股详情", "策略统计"],
        key="main_dashboard_tabs",
        on_change="rerun",
    )
    if tabs[0].open:
        with tabs[0]:
            render_today_trade_pool_tab(today_trade_pool, top50_source_count)
    if tabs[1].open:
        with tabs[1]:
            render_v59_top50_tab(v59_top50, status)
    if tabs[2].open:
        with tabs[2]:
            st.info("策略候选池：展示所有模型评分结果，用于观察。基础评分、交易质量评分和最终交易评分会分开显示；未进入全市场基础评分Top100的股票不计算交易质量。")
            render_table_tab(strategy_candidates, "当前没有策略候选股。")
    if tabs[3].open:
        with tabs[3]:
            render_strategy_tracking_pool_tab()
    if tabs[4].open:
        with tabs[4]:
            render_watchlist_holdings(watchlist, holdings, candidates)
    if tabs[5].open:
        with tabs[5]:
            render_sector_market_tab(status, candidates)
    if tabs[6].open:
        with tabs[6]:
            render_stock_detail_panel(detail_candidates, status)
    if tabs[7].open:
        with tabs[7]:
            render_strategy_statistics_tab()
    render_developer_diagnostics(
        status,
        candidates[candidates["source_type"] == SOURCE_TYPE_ACTIVE],
        reference_candidates,
    )


def load_candidates(
    path: Path = LATEST_CANDIDATES_PATH,
    *,
    attach_metric_samples: bool = True,
) -> pd.DataFrame:
    """Load latest candidates with stable display fields."""
    data = load_auxiliary_csv(path, dtype={"code": "string"})
    if data.empty:
        return _empty_candidates()
    missing_columns = [column for column in _candidate_columns() if column not in data.columns]
    if missing_columns:
        placeholders = pd.DataFrame(
            {column: pd.Series(pd.NA, index=data.index, dtype="object") for column in missing_columns},
            index=data.index,
        )
        data = pd.concat([data, placeholders], axis=1)
    data["code"] = data["code"].astype("string").str.zfill(6)
    if attach_metric_samples:
        data = attach_realtime_metric_fields(data)
    # Consolidate the wide CSV frame once before adding presentation columns.
    # This avoids repeated single-column inserts fragmenting DataFrames on every rerun.
    data = data.copy()
    data["source_type"] = data["source_type"].fillna(SOURCE_TYPE_ACTIVE).astype(str)
    data["source_type_display"] = data["source_type"].map(SOURCE_TYPE_LABELS).fillna(data["source_type_display"].fillna(data["source_type"]))
    data["strategy_names_display"] = data["strategy_names"].map(format_strategy_names)
    data["level_display"] = data["level"].map(format_level)
    data["pct_chg_display"] = data["pct_chg"].map(format_change_percent)
    data["price_display"] = data["price"].map(format_price)
    data["turnover_display"] = data.apply(format_preferred_turnover, axis=1)
    data["amount_display"] = data.apply(lambda row: row["amount_display"] if _usable(row.get("amount_display")) else format_amount(row.get("amount")), axis=1)
    data["volume_ratio_display"] = data.apply(format_preferred_volume_ratio, axis=1)
    data["volume_ratio_ref_display"] = data.get("volume_ratio_ref", pd.Series(index=data.index, dtype="float64")).map(format_reference_number)
    data["total_mv_display"] = data.get("total_mv", pd.Series(index=data.index, dtype="float64")).map(format_tushare_market_value)
    data["circ_mv_display"] = data.get("circ_mv", pd.Series(index=data.index, dtype="float64")).map(format_tushare_market_value)
    data["reference_trade_date_display"] = data.get("reference_trade_date", pd.Series(index=data.index, dtype="object")).map(format_reference_date)
    data["turnover_source_display"] = data.apply(format_turnover_source_display, axis=1)
    data["volume_ratio_source_display"] = data.apply(format_volume_ratio_source_display, axis=1)
    data["data_status_display"] = data.apply(format_data_status, axis=1)
    data["score"] = pd.to_numeric(data["score"], errors="coerce").fillna(0)
    data["score_display"] = data["score"].map(format_score_display)
    for column in ["base_score", "trading_quality_score", "final_trade_score", "data_quality_score"]:
        data[column] = pd.to_numeric(
            data.get(column, pd.Series(pd.NA, index=data.index)),
            errors="coerce",
        )
        data[f"{column}_display"] = data[column].map(format_score_display)
    data["data_quality_level_display"] = data.get(
        "data_quality_level", pd.Series("C", index=data.index)
    ).map(format_data_quality_level)
    for column in ["sector_name", "sector_strength_level", "theme_tags", "money_strength_level", "action_state_display", "position_risk_level", "t1_risk_level", "tracking_summary", "data_quality_status"]:
        data[column] = data[column].fillna("--").astype(str).replace({"": "--", "nan": "--"})
    return data.copy().sort_values("score", ascending=False).reset_index(drop=True)


def load_v59_top50() -> pd.DataFrame:
    """Load only a V5.9-provenance Top50 snapshot from main.py output."""
    raw = load_auxiliary_csv(V59_TOP50_PATH, dtype={"code": "string"})
    if raw.empty or not V59_REQUIRED_OUTPUT_COLUMNS.issubset(raw.columns):
        return _empty_candidates()

    # Column names alone are insufficient: an old or partial file must not be
    # displayed as V5.9 data when its calibrated-score fields contain no rows.
    required_scores = ["raw_base_score", "base_percentile", "base_score"]
    if any(pd.to_numeric(raw[column], errors="coerce").notna().sum() == 0 for column in required_scores):
        return _empty_candidates()
    data = load_candidates(V59_TOP50_PATH, attach_metric_samples=False)
    data["source_type"] = "v59_top50"
    data["source_type_display"] = "V5.9 Top50交易观察池"
    return data


def load_reference_candidates(path: Path = REFERENCE_CANDIDATES_PATH) -> pd.DataFrame:
    """Load reference candidates generated from Sina realtime + Tushare cache."""
    data = load_auxiliary_csv(path, dtype={"code": "string"})
    if data.empty:
        return data
    data["code"] = data["code"].astype("string").str.zfill(6)
    data = attach_realtime_metric_fields(data)
    data["source_type"] = data.get("source_type", SOURCE_TYPE_REFERENCE)
    data["source_type_display"] = "参考候选股"
    data["pct_chg_display"] = data.get("pct_chg", pd.Series(dtype="float64")).map(format_change_percent)
    data["price_display"] = data.get("price", pd.Series(dtype="float64")).map(format_price)
    data["amount_display"] = data.apply(
        lambda row: row["amount_display"] if _usable(row.get("amount_display")) else format_amount(row.get("amount")),
        axis=1,
    )
    data["volume_ratio_ref_display"] = data.get("volume_ratio_ref", pd.Series(dtype="float64")).map(format_reference_number)
    data["total_mv_display"] = data.get("total_mv", pd.Series(dtype="float64")).map(format_tushare_market_value)
    data["circ_mv_display"] = data.get("circ_mv", pd.Series(dtype="float64")).map(format_tushare_market_value)
    data["turnover_display"] = data.apply(format_preferred_turnover, axis=1)
    data["volume_ratio_display"] = data.apply(format_preferred_volume_ratio, axis=1)
    data["turnover_source_display"] = data.apply(format_turnover_source_display, axis=1)
    data["volume_ratio_source_display"] = data.apply(format_volume_ratio_source_display, axis=1)
    data["data_status_display"] = data.apply(format_data_status, axis=1)
    data["data_source"] = data.get("reference_source", "Tushare 本地缓存")
    data["strategy_names_display"] = "--"
    data["level_display"] = "参考"
    data["score"] = pd.NA
    data["score_display"] = "--"
    for column in ["reference_reason", "reference_warning", "warning_message", "market_code", "name", "price"]:
        if column not in data.columns:
            data[column] = pd.NA
    return data.reset_index(drop=True)


def attach_realtime_metric_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Preserve persisted realtime turnover while enriching only volume-ratio display."""
    output = data.copy()
    output["code"] = output["code"].astype("string").str.zfill(6)
    realtime_turnover = pd.to_numeric(
        output.get("realtime_turnover_value", pd.Series(pd.NA, index=output.index)),
        errors="coerce",
    )
    valid_turnover = realtime_turnover.between(0, 100)
    output["realtime_turnover_value"] = realtime_turnover.where(valid_turnover)
    existing_source = output.get("realtime_turnover_source", pd.Series(pd.NA, index=output.index))
    output["realtime_turnover_source"] = existing_source.where(valid_turnover, pd.NA)

    realtime_ratio = pd.to_numeric(output.get("volume_ratio", pd.Series(pd.NA, index=output.index)), errors="coerce")
    ratio_realtime = _truthy_series(output.get("is_realtime_volume_ratio", pd.Series(False, index=output.index)))
    output["realtime_volume_ratio_value"] = realtime_ratio.where(ratio_realtime & realtime_ratio.between(0, 100))
    output["realtime_volume_ratio_source"] = output.get("volume_ratio_source", pd.Series(pd.NA, index=output.index))

    ratio_sample = load_auxiliary_csv(CALCULATED_VOLUME_RATIO_PATH, dtype={"code": "string"})
    if ratio_sample.empty or "code" not in ratio_sample.columns:
        return output
    ratio_sample["code"] = ratio_sample["code"].astype("string").str.zfill(6)
    sample_value = pd.to_numeric(ratio_sample.get("calculated_volume_ratio"), errors="coerce")
    sample_valid = _truthy_series(ratio_sample.get("is_realtime_volume_ratio", pd.Series(False, index=ratio_sample.index)))
    lookup = pd.DataFrame({
        "code": ratio_sample["code"],
        "sample_realtime_volume_ratio": sample_value.where(sample_valid & sample_value.between(0, 100)),
        "sample_volume_ratio_source": ratio_sample.get("volume_ratio_source", pd.Series(pd.NA, index=ratio_sample.index)),
    }).drop_duplicates("code", keep="last")
    output = output.merge(lookup, on="code", how="left")
    output["realtime_volume_ratio_value"] = output["realtime_volume_ratio_value"].where(
        output["realtime_volume_ratio_value"].notna(), output["sample_realtime_volume_ratio"]
    )
    output["realtime_volume_ratio_source"] = output["realtime_volume_ratio_source"].where(
        output["realtime_volume_ratio_source"].notna(), output["sample_volume_ratio_source"]
    )
    output["is_realtime_volume_ratio"] = output["realtime_volume_ratio_value"].notna()
    return output.drop(columns=["sample_realtime_volume_ratio", "sample_volume_ratio_source"], errors="ignore")


def _truthy_series(values: pd.Series) -> pd.Series:
    """Normalize boolean CSV values without treating missing values as realtime."""
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_auxiliary_csv(path: Path, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load CSV safely."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception:
        return pd.DataFrame()


def load_market_status(path: Path = LATEST_MARKET_STATUS_PATH) -> dict[str, Any]:
    """Load latest market status JSON."""
    if not path.exists() or path.stat().st_size == 0:
        return {"data_status": "行情源失败", "errors": ["尚未生成 latest_market_status.json"]}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        return {"data_status": "行情源失败", "errors": ["状态文件不是有效对象"]}
    except Exception as exc:
        return {"data_status": "行情源失败", "errors": [f"读取状态文件失败: {type(exc).__name__}: {exc}"]}


def initialize_dashboard_state() -> None:
    """Initialize cross-page UI state once, with safe defaults for reruns."""
    defaults = {
        "selected_stock_code": "",
        "detail_focus_requested": False,
        "strategy_tracking_feedback": None,
        "strategy_tracking_exit_tracking_id": "",
        "watchlist_feedback": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_top_status(status: dict[str, Any], *, candidate_count: int) -> None:
    """Render the professional top-line operational snapshot."""
    raw_market = status.get("market_environment_detail")
    market = raw_market if isinstance(raw_market, dict) else {}
    data_quality_level = str(status.get("data_quality_level") or "C").upper()
    data_quality_text = _data_quality_level_text(status)
    market_env = str(status.get("market_environment", "--"))
    columns = st.columns(4)
    render_status_card(columns[0], "数据更新时间", _format_dashboard_timestamp(status.get("scan_time")), "neutral")
    render_status_card(columns[1], "当前数据源", str(status.get("data_source", "--")), "neutral")
    render_status_card(
        columns[2],
        "市场状态",
        market_env,
        "red" if market_env in {"强势", "偏强"} else "green" if market_env in {"偏弱", "极弱"} else "neutral",
    )
    render_status_card(columns[3], "今日交易候选", format_quantity(candidate_count), "red" if candidate_count else "gray")
    st.caption(
        f"数据质量：{data_quality_text} ｜ 上涨 {format_quantity(market.get('up_count'))} ｜ "
        f"下跌 {format_quantity(market.get('down_count'))} ｜ "
        f"涨停 {format_quantity(market.get('limit_up_count'))} ｜ "
        f"跌停 {format_quantity(market.get('limit_down_count'))}"
    )
    if data_quality_level == "B":
        st.warning("当前换手率/量比来自计算，不是交易所实时字段。")
    elif data_quality_level == "C":
        st.warning("当前数据质量为 C级（关键字段缺失），缺失行不能进入正式候选池。")


def render_status_card(container: Any, label: str, value: Any, tone: str = "neutral") -> None:
    """Render a compact colored status card."""
    tone_class = {
        "red": "tone-red",
        "green": "tone-green",
        "gray": "tone-gray",
        "risk": "tone-risk",
        "neutral": "tone-neutral",
    }.get(tone, "tone-neutral")
    card_class = {
        "red": "watch-card-red",
        "green": "watch-card-green",
        "gray": "watch-card-gray",
        "risk": "watch-card-risk",
        "neutral": "",
    }.get(tone, "")
    safe_label = escape(str(label))
    safe_value = escape(display_text(value))
    container.markdown(
        f"""
        <div class="watch-card {card_class}">
            <div class="watch-card-label">{safe_label}</div>
            <div class="watch-card-value {tone_class}">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_small_hint(message: str, tone: str = "muted") -> None:
    """Render compact inline feedback instead of a full alert block."""
    tone_class = "small-hint-ok" if tone == "ok" else "small-hint-muted"
    st.markdown(f'<div class="small-hint {tone_class}">{escape(str(message))}</div>', unsafe_allow_html=True)


def _data_source_mode_text(status: dict[str, Any]) -> str:
    quality_level = str(status.get("data_quality_level") or "C").upper()
    if quality_level == "A":
        return "接口实时字段"
    if quality_level == "B":
        return "计算字段模式"
    if quality_level == "C":
        return "关键字段缺失"
    level = str(status.get("data_source_level") or "").strip().upper()
    source = str(status.get("data_source") or "")
    if level in {"D", "B_BASIC"} or "新浪" in source:
        return "基础行情模式"
    if bool(status.get("allow_strategy_candidate")):
        return "完整策略模式"
    return str(status.get("data_status") or "--")


def _data_quality_level_text(status: dict[str, Any]) -> str:
    """Show final scoring-input quality, not raw source field availability."""
    level = str(status.get("data_quality_level") or "C").upper()
    labels = {
        "A": "A级（接口实时字段）",
        "B": "B级（计算字段）",
        "C": "C级（关键字段缺失）",
    }
    return str(status.get("data_quality_label") or labels.get(level, labels["C"]))


def _format_dashboard_timestamp(value: Any) -> str:
    """Display a valid scan timestamp compactly without assuming its format."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return pd.Timestamp(parsed).strftime("%Y-%m-%d %H:%M")
    return display_text(value)


def _missing_realtime_turnover_or_ratio(status: dict[str, Any]) -> bool:
    return str(status.get("data_quality_level") or "C").upper() == "C"


def render_table_tab(data: pd.DataFrame, empty_text: str) -> None:
    """Render a candidate/active table with filters."""
    if data.empty:
        st.info(empty_text)
        return
    board_options = sorted([x for x in data["board_type_display"].dropna().unique() if x])
    selected_boards = st.multiselect("市场归类筛选", board_options, key=f"board_filter_{empty_text}")
    shown = data[data["board_type_display"].isin(selected_boards)] if selected_boards else data
    render_compact_stock_dataframe(display_frame(shown))
    with st.expander("长文本风险提示 / 评分解释"):
        columns = [c for c in ["code", "name", "score_detail", "risk_summary", "turnover_summary", "sector_summary", "action_summary", "t1_risk_summary"] if c in shown.columns]
        st.dataframe(shown[columns], width="stretch", hide_index=True)


def build_today_trade_pool(
    v59_top50: pd.DataFrame,
    market_environment: str = "",
) -> tuple[pd.DataFrame, int]:
    """Apply display-only safety checks to the V5.9 Top50 source snapshot.

    The Top50 has already passed V5.9's final-trade-score admission in
    ``main.py``.  This function does not rescore or replace that admission;
    it merely removes rows that are unsuitable for today's focused monitoring.
    """
    source_count = len(v59_top50)
    if v59_top50.empty:
        return v59_top50.copy(), source_count

    data = v59_top50.copy()
    data = attach_t1_risk_if_missing(data, market_environment)
    data["trade_risk_level"] = data.apply(derive_live_v56_risk_level, axis=1)
    data["trade_data_complete"] = data.apply(is_v59_trade_data_complete, axis=1)
    data["trade_not_st"] = data.apply(is_not_st_stock, axis=1)
    data["trade_not_suspended"] = data.apply(is_not_suspended_stock, axis=1)
    data["trade_t1_allowed"] = data.apply(is_t1_risk_allowed, axis=1)
    data["buy_point_ready"] = data.apply(is_buy_point_ready, axis=1)
    data["buy_point_status_display"] = data["buy_point_ready"].map(
        {True: "满足", False: "不满足"}
    )
    data["trade_not_filtered"] = data.apply(is_not_filtered_trade_status, axis=1)
    data["buy_status_display"] = data.apply(derive_buy_status, axis=1)

    output = data[
        data["trade_data_complete"]
        & data["trade_not_st"]
        & data["trade_not_suspended"]
        & data["trade_risk_level"].ne("高风险")
        & data["trade_t1_allowed"]
        & data["buy_point_ready"]
        & data["trade_not_filtered"]
    ].copy()
    return output.sort_values(["final_trade_score", "base_score"], ascending=[False, False]).reset_index(drop=True), source_count


def build_detail_candidates(
    v59_top50: pd.DataFrame,
    today_trade_pool: pd.DataFrame,
    candidates: pd.DataFrame,
    reference_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Build detail rows with the same-day V5.9 Top50 record taking priority.

    This is a Dashboard-only provenance order: it never recalculates a score
    and uses no quote request.  When a V5.9 Top50 stock also survives today's
    display-only safety filter, its existing risk/buy-status display fields are
    attached to that same Top50 record.
    """
    top50 = v59_top50.copy()
    if not top50.empty:
        top50["detail_score_source"] = "V5.9 Top50交易观察池（当天生成）"
        top50["code"] = top50["code"].astype("string").str.zfill(6)
        if not today_trade_pool.empty and "code" in today_trade_pool.columns:
            state_columns = [
                column
                for column in ["trade_risk_level", "t1_risk_level", "t1_risk_summary", "buy_point_status_display", "buy_status_display"]
                if column in today_trade_pool.columns
            ]
            if state_columns:
                states = today_trade_pool[["code", *state_columns]].copy()
                states["code"] = states["code"].astype("string").str.zfill(6)
                states = states.drop_duplicates("code", keep="first").set_index("code")
                for column in state_columns:
                    top50[column] = top50["code"].map(states[column]).where(
                        top50["code"].map(states[column]).notna(),
                        top50.get(column, pd.Series(pd.NA, index=top50.index)),
                    )
        if "buy_status_display" not in top50.columns:
            top50["buy_status_display"] = pd.NA
        top50["buy_status_display"] = top50["buy_status_display"].where(
            top50["buy_status_display"].notna(),
            "未进入今日交易候选",
        )

    layers: list[pd.DataFrame] = [top50]
    for frame, source_label in (
        (candidates, "策略候选池 / 市场异动观察池（当天生成）"),
        (reference_candidates, "数据参考池（当天生成）"),
    ):
        if frame.empty:
            continue
        output = frame.copy()
        output["detail_score_source"] = source_label
        layers.append(output)

    available_layers = [frame for frame in layers if not frame.empty]
    if not available_layers:
        return _empty_candidates()
    output = pd.concat(available_layers, ignore_index=True, sort=False)
    output["code"] = output["code"].astype("string").str.zfill(6)
    return output.drop_duplicates("code", keep="first").reset_index(drop=True)


def attach_t1_risk_if_missing(data: pd.DataFrame, market_environment: str) -> pd.DataFrame:
    """Use the existing T+1 rule for Top50 rows lacking its display fields."""
    output = data.copy()
    if "t1_risk_level" not in output.columns:
        output["t1_risk_level"] = pd.NA
    if "t1_risk_summary" not in output.columns:
        output["t1_risk_summary"] = pd.NA

    missing = output["t1_risk_level"].map(_is_blank_value)
    for index, row in output.loc[missing].iterrows():
        result = analyze_t1_risk(row, market_environment)
        output.at[index, "t1_risk_level"] = result["t1_risk_level"]
        output.at[index, "t1_risk_summary"] = result["t1_risk_summary"]
    return output


def derive_live_v56_risk_level(row: pd.Series) -> str:
    """Project V5.6 entry risks using only fields available in today's snapshot."""
    pct_chg = pd.to_numeric(row.get("pct_chg"), errors="coerce")
    position = display_text(row.get("position_risk_level"))
    chase = display_text(row.get("chase_risk_level"))
    risk_text = display_text(row.get("risk_summary"))
    if (
        pd.notna(pct_chg)
        and pct_chg > 9
    ) or position == "高" or chase == "高" or "巨量风险" in risk_text or "冲高回落" in risk_text:
        return "高风险"
    if (
        pd.notna(pct_chg)
        and pct_chg >= 5
    ) or position == "中" or chase == "中" or "不追高" in risk_text:
        return "中风险"
    return "低风险"


def is_v59_trade_data_complete(row: pd.Series) -> bool:
    """Require a V5.9 Top50 row with complete A/B-level final inputs."""
    base_score = pd.to_numeric(row.get("base_score"), errors="coerce")
    final_score = pd.to_numeric(row.get("final_trade_score"), errors="coerce")
    price = pd.to_numeric(row.get("price"), errors="coerce")
    quality_level = _normalize_data_quality_level(row.get("data_quality_level"))
    top50_data_eligible = _truthy_value(row.get("top50_data_eligible"))
    return bool(
        pd.notna(base_score)
        and pd.notna(final_score)
        and pd.notna(price)
        and price > 0
        and quality_level in {"A", "B"}
        and top50_data_eligible
    )


def is_not_st_stock(row: pd.Series) -> bool:
    return "ST" not in display_text(row.get("name")).upper()


def is_not_suspended_stock(row: pd.Series) -> bool:
    price = pd.to_numeric(row.get("price"), errors="coerce")
    risk_text = display_text(row.get("risk_summary"))
    return bool(pd.notna(price) and price > 0 and "停牌" not in risk_text and "SUSPEND" not in risk_text.upper())


def is_t1_risk_allowed(row: pd.Series) -> bool:
    return display_text(row.get("t1_risk_level")) in {"低", "中"}


def is_buy_point_ready(row: pd.Series) -> bool:
    """Accept neutral-or-better existing buy-point quality; reject a negative signal."""
    score = pd.to_numeric(row.get("quality_buy_point_score"), errors="coerce")
    return bool(pd.notna(score) and score >= 0)


def is_not_filtered_trade_status(row: pd.Series) -> bool:
    """Exclude rows explicitly marked as filtered by the existing strategy."""
    blocked_values = {"过滤", "暂不关注", "ignore", "filtered", "filter"}
    def normalized(value: Any) -> str:
        return "" if value is None or pd.isna(value) else str(value).strip().lower()

    level = normalized(row.get("level"))
    level_display = normalized(row.get("level_display"))
    action_state = normalized(row.get("action_state"))
    return (
        level not in blocked_values
        and level_display not in blocked_values
        and action_state not in {"weak_ignore", "data_insufficient", "filtered", "filter"}
    )


def derive_buy_status(row: pd.Series) -> str:
    """Label the surviving V5.9 Top50 subset without changing its score."""
    final_score = pd.to_numeric(row.get("final_trade_score"), errors="coerce")
    risk = display_text(row.get("trade_risk_level"))
    if risk == "低风险" and pd.notna(final_score) and final_score >= 90:
        return "强烈关注"
    if risk == "中风险":
        return "谨慎"
    return "观察等待"


def render_today_trade_pool_tab(data: pd.DataFrame, top50_source_count: int) -> None:
    """Render the safety-filtered subset of V5.9 Top50."""
    st.info(
        "今日交易候选池：仅从 V5.9 Top50交易观察池生成，不再读取策略候选池。"
        "在 Top50 基础上保留数据完整、非ST、非停牌、非高风险、T+1风险允许且买点状态满足的股票。"
    )
    st.caption(
        f"本轮 V5.9 Top50 上游共 {top50_source_count} 只；今日交易候选为其二次安全筛选结果。"
    )
    if data.empty:
        st.info("今日暂无符合交易条件股票")
        return
    render_strategy_tracking_action_table(
        data,
        display_today_trade_frame(data),
        source_page="今日交易候选池",
        context_key="today_trade_pool",
    )
    with st.expander("准入说明 / 风险提示", expanded=False):
        columns = [
            column
            for column in [
                "code",
                "name",
                "base_score",
                "final_trade_score",
                "data_quality_level",
                "trade_risk_level",
                "t1_risk_level",
                "t1_risk_summary",
                "quality_buy_point_score",
                "buy_point_status_display",
                "buy_status_display",
                "position_risk_level",
                "chase_risk_level",
                "risk_summary",
                "score_detail",
            ]
            if column in data.columns
        ]
        st.dataframe(data[columns], width="stretch", hide_index=True)


def render_v59_top50_tab(data: pd.DataFrame, status: dict[str, Any]) -> None:
    """Render the V5.9 full-market, quality-gated Top50 observation pool."""
    st.info(
        "V5.9 Top50交易观察池：先对全市场计算原始基础分，再按全市场百分位生成基础评分；"
        "仅对基础评分Top100计算交易质量。"
        "最终交易评分 = 基础评分 × 0.7 + 交易质量评分 × 0.3。"
        "入池必须同时满足基础评分≥70、最终交易评分≥85、数据完整且换手率处于5%–30%。"
    )
    st.caption(
        "原始基础分用于展示绝对能力；全市场百分位用于同日横截面对比；基础评分为校准后的最终基础层结果。"
    )
    if data.empty:
        st.info("当前没有满足V5.9交易条件股票")
        return

    shown = data.sort_values(["final_trade_score", "base_score"], ascending=[False, False])
    st.caption(
        f"本轮全市场基础评分 {int(status.get('v58_full_market_count') or 0)} 只；"
        f"交易质量评分Top100 {int(status.get('v58_top100_count') or 0)} 只；"
        f"正式Top50 {len(shown)} 只。"
    )
    columns = [column for column in V59_TOP50_DISPLAY_COLUMNS if column in shown.columns]
    render_strategy_tracking_action_table(
        shown,
        clean_display_frame(shown[columns].rename(columns=COLUMN_LABELS)),
        source_page="V5.9 Top50交易观察池",
        context_key="v59_top50",
    )
    with st.expander("V5.9 准入与评分拆解", expanded=False):
        detail_columns = [
            column
            for column in [
                "code",
                "name",
                "raw_base_score",
                "base_percentile",
                "base_score",
                "trading_quality_score",
                "final_trade_score",
                "quality_adjustment",
                "quality_market_score",
                "quality_trend_stage_score",
                "quality_buy_point_score",
                "quality_sector_linkage_score",
                "turnover_quality_score",
                "data_quality_score",
                "data_quality_status",
                "v58_quality_detail",
                "v58_admission_reason",
            ]
            if column in shown.columns
        ]
        st.dataframe(shown[detail_columns], width="stretch", hide_index=True)


def render_strategy_tracking_action_table(
    source_data: pd.DataFrame,
    display_data: pd.DataFrame,
    *,
    source_page: str,
    context_key: str,
) -> None:
    """Render a read-only stock table with one tracking action per source row."""
    source_rows = source_data.reset_index(drop=True).copy()
    table = display_data.reset_index(drop=True).copy()
    if len(source_rows) != len(table):
        st.error("策略跟踪表格行数不一致，当前未展示加入操作。")
        render_compact_stock_dataframe(table)
        return

    action_column = "操作"
    table[action_column] = ":material/bookmark_add: 加入跟踪"
    click_key = f"strategy_tracking_table_click_{context_key}"
    render_strategy_tracking_feedback()
    with st.container(horizontal_alignment="center"):
        st.dataframe(
            style_watch_table(table),
            column_config={
                **compact_table_column_config(table),
                action_column: st.column_config.ButtonColumn(
                    action_column,
                    width=96,
                    help="保存该股票当日的完整策略快照到本地策略跟踪池。",
                    type="secondary",
                    on_click=_handle_strategy_tracking_table_click,
                    args=(source_rows, source_page, click_key),
                    key=click_key,
                ),
            },
            width="content",
            hide_index=True,
        )


def _handle_strategy_tracking_table_click(
    source_rows: pd.DataFrame,
    source_page: str,
    click_key: str,
) -> None:
    """Persist the exact raw row selected from a tracking-action table."""
    click = st.session_state.get(click_key)
    try:
        row_index = int(click.get("row"))
    except (AttributeError, TypeError, ValueError):
        return
    if row_index < 0 or row_index >= len(source_rows):
        return

    row = source_rows.iloc[row_index]
    name = display_text(row.get("name"))
    added, message = add_strategy_tracking_stock(row, source_page=source_page)
    st.session_state["strategy_tracking_feedback"] = {
        "message": f"已加入策略跟踪：{name}" if added else message,
        "tone": "ok" if added else "muted",
    }


def render_active_watchlist_tab(data: pd.DataFrame) -> None:
    """Render a compact active watchlist table for daily monitoring."""
    st.info(
        "市场异动观察池：来源于 data/latest_candidates.csv，用于观察市场活跃股票。"
        "该页面不参与 V5.9 评分，也不参与今日交易候选生成。"
        "换手率仅展示当天生成的实时换手率字段；字段缺失时显示“--”。"
    )
    if data.empty:
        st.info("当前没有市场异动观察池数据。")
        return
    board_options = sorted([x for x in data["board_type_display"].dropna().unique() if x])
    selected_boards = st.multiselect("市场归类筛选", board_options, key="board_filter_active_watchlist")
    shown = data[data["board_type_display"].isin(selected_boards)] if selected_boards else data
    active_options = build_stock_select_options(shown)
    selected_row = None
    selected_stock = st.selectbox("选择个股查看详情", [""] + list(active_options.keys()), key="active_watchlist_stock_select")
    if selected_stock:
        selected_code = str(active_options[selected_stock]).zfill(6)
        st.session_state["selected_stock_code"] = selected_code
        selected_row = find_stock_row(shown, selected_code)
        render_stock_selection_actions(selected_row, "active_watchlist")
    else:
        st.caption("选择个股后，请切换到“个股详情”查看 K 线和技术信息。")
    display = display_active_frame(shown)
    render_compact_stock_dataframe(display)
    with st.expander("更多行情字段 / 触发原因 / 风险提示", expanded=False):
        render_selected_reason_panel(selected_row)


def render_reference_candidates_tab(data: pd.DataFrame) -> None:
    """Render reference candidates without presenting them as official strategy picks."""
    st.info("数据参考池：用于补充核对，不等于正式策略候选或今日交易候选。换手率仅展示当天生成的实时字段；日级换手率仅保留在数据文件中，不在页面展示。")
    if data.empty:
        st.info("当前没有参考候选股。")
        return

    shown = data.copy()
    options = build_stock_select_options(shown)
    selected_row = None
    selected_stock = st.selectbox("选择个股查看详情", [""] + list(options.keys()), key="reference_stock_select")
    if selected_stock:
        selected_code = str(options[selected_stock]).zfill(6)
        st.session_state["selected_stock_code"] = selected_code
        selected_row = find_stock_row(shown, selected_code)
        render_stock_selection_actions(selected_row, "reference_candidate")
    else:
        st.caption("选择个股后，请切换到“个股详情”查看 K 线和技术信息。")

    display = display_reference_frame(shown)
    render_compact_stock_dataframe(display)
    with st.expander("更多行情字段 / 触发原因 / 风险提示", expanded=False):
        render_selected_reason_panel(selected_row)


def render_watchlist_holdings(watchlist: pd.DataFrame, holdings: pd.DataFrame, candidates: pd.DataFrame) -> None:
    """Render user-maintained watchlist and holdings."""
    st.subheader("自选股")
    user_watchlist = load_user_watchlist()
    if user_watchlist.empty:
        st.info("暂无本地自选股。可在“系统诊断 / 开发工具”中的市场异动观察池选择个股后加入自选股。")
    else:
        st.dataframe(style_watch_table(format_user_watchlist_table(user_watchlist, candidates, watchlist)), width="stretch", hide_index=True)
        render_watchlist_editor(user_watchlist)
    st.subheader("持仓股")
    user_holdings = load_user_holdings()
    if user_holdings.empty:
        st.info("暂未配置持仓股。可在下方手动新增，本系统不会读取券商真实持仓。")
    else:
        st.dataframe(style_watch_table(format_user_holdings_table(user_holdings, candidates, watchlist)), width="stretch", hide_index=True)
    render_holdings_editor(user_holdings, candidates, watchlist)


def render_strategy_tracking_pool_tab() -> None:
    """Render the independent local strategy-observation records."""
    st.info(
        "策略跟踪池：保存加入时的V5.9交易评分与行情快照，并在每次 main.py 运行后更新。"
        "仅用于观察和复盘，不改变评分、选股或交易规则。"
    )
    render_strategy_tracking_feedback()
    tracking_pool = load_strategy_tracking_pool()
    if tracking_pool.empty:
        st.info("当前策略跟踪池为空。可在“今日交易候选池”“V5.9 Top50交易观察池”或“个股详情”加入股票。")
        return
    render_strategy_tracking_exit_action_table(
        tracking_pool,
        format_strategy_tracking_table(tracking_pool),
    )
    render_strategy_tracking_exit_dialog(tracking_pool)


def render_strategy_statistics_tab() -> None:
    """Render local tracking-ledger and V5.9 candidate-snapshot statistics."""
    st.info(
        "策略统计：仅汇总策略跟踪池中已保存的价格、评分和结束记录，用于验证历史表现。"
        "已结束记录使用结束收益率；进行中记录使用当前收益率。"
        "V5.9 历史候选池统计只读取本地候选快照和历史日线，不会重新评分。"
    )
    tracking_pool = load_strategy_tracking_pool()
    if tracking_pool.empty:
        st.info("暂无策略跟踪数据，加入并完成至少一轮扫描后即可生成统计。")
    else:
        statistics = build_strategy_tracking_statistics(tracking_pool)
        overview = statistics["overview"]
        cards = [
            ("当前跟踪股票数量", str(overview["current_tracking_count"])),
            ("已完成跟踪数量", str(overview["completed_tracking_count"])),
            ("盈利数量", str(overview["profit_count"])),
            ("亏损数量", str(overview["loss_count"])),
            ("胜率", format_plain_percent(overview["win_rate"])),
            ("平均收益率", format_change_percent(overview["average_return_pct"])),
            ("最大盈利", format_change_percent(overview["max_profit_pct"])),
            ("最大亏损", format_change_percent(overview["max_loss_pct"])),
        ]
        st.subheader("策略跟踪池表现")
        for offset in range(0, len(cards), 3):
            columns = st.columns(3)
            for column, (label, value) in zip(columns, cards[offset : offset + 3]):
                column.metric(label, value)

        st.subheader("收益分布")
        st.dataframe(pd.DataFrame(statistics["return_distribution"]), width="stretch", hide_index=True)

        st.subheader("按加入评分统计")
        st.caption("按加入时 V5.9 最终交易评分分组。")
        st.dataframe(format_strategy_group_statistics(statistics["score_groups"]), width="stretch", hide_index=True)

        st.subheader("按换手率统计")
        st.caption("按加入时换手率分组。")
        st.dataframe(format_strategy_group_statistics(statistics["turnover_groups"]), width="stretch", hide_index=True)

        st.subheader("按量比统计")
        st.caption("按加入时量比分组。")
        st.dataframe(format_strategy_group_statistics(statistics["volume_ratio_groups"]), width="stretch", hide_index=True)

        st.subheader("入选原因统计")
        st.caption("标签基于加入策略跟踪时永久保存的入选快照，不会受后续行情更新影响。")
        st.dataframe(format_strategy_reason_statistics(statistics["reason_groups"]), width="stretch", hide_index=True)

    render_v59_historical_candidate_statistics()


def format_strategy_group_statistics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Format pre-aggregated tracking groups without altering their values."""
    data = pd.DataFrame(rows)
    if data.empty:
        return pd.DataFrame(columns=["分组", "股票数量", "胜率", "平均收益率"])
    data["胜率"] = data["胜率"].map(format_plain_percent)
    data["平均收益率"] = data["平均收益率"].map(format_change_percent)
    return clean_display_frame(data)


def format_strategy_reason_statistics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Format immutable entry-reason aggregates for the strategy statistics tab."""
    data = pd.DataFrame(rows)
    if data.empty:
        return pd.DataFrame(columns=["标签", "股票数量", "胜率", "平均收益率"])
    data["胜率"] = data["胜率"].map(format_plain_percent)
    data["平均收益率"] = data["平均收益率"].map(format_change_percent)
    return clean_display_frame(data)


def render_v59_historical_candidate_statistics() -> None:
    """Render offline V5.9 group statistics from immutable candidate snapshots."""
    snapshot_version = _local_csv_version(V59_CANDIDATE_HISTORY_DIR, "v5_9_candidate_*.csv")
    daily_version = _local_csv_version(HISTORICAL_DAILY_DIR, "daily_*.csv")
    tracking_history_version = _local_csv_version(STRATEGY_TRACKING_HISTORY_PATH.parent, STRATEGY_TRACKING_HISTORY_PATH.name)
    performance, coverage = load_v59_candidate_pool_performance(snapshot_version, daily_version)
    tracking_history = load_strategy_tracking_history_summary(tracking_history_version)

    st.subheader("V5.9 历史候选池验证")
    st.caption(
        "收益口径：使用候选池快照中的当日价格作为买入价，并以本地历史日线第 3、5、10 个后续交易日收盘价计算最终收益。"
        "每个持有窗口同时统计最高价收益与相对买入价的最大回撤；评分、换手率和量比均直接读取快照，不重新计算。"
    )
    source_cards = [
        ("候选池快照文件", str(coverage["snapshot_files"])),
        ("总样本数", str(len(performance))),
        ("3日收益样本", str(coverage["completed_rows"]["3"])),
        ("5日收益样本", str(coverage["completed_rows"]["5"])),
        ("10日收益样本", str(coverage["completed_rows"]["10"])),
        ("策略跟踪历史快照", str(tracking_history["snapshot_rows"])),
    ]
    for offset in range(0, len(source_cards), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, source_cards[offset : offset + 4]):
            column.metric(label, value)
    if tracking_history["latest_snapshot_date"]:
        st.caption(
            f"策略跟踪历史已保存 {tracking_history['tracking_count']} 个跟踪对象，"
            f"最新快照日期：{tracking_history['latest_snapshot_date']}。"
        )

    if coverage["snapshot_files"] == 0:
        st.info("暂无 V5.9 每日候选池快照。后续成功运行 main.py --once 后将自动累积。")
        return
    if all(count == 0 for count in coverage["completed_rows"].values()):
        st.info(
            "已读取候选池快照，但 3/5/10 日收益样本均不足。"
            f"缺少信号日历史：{coverage['missing_signal_date']}；"
            f"3/5/10 日后续数据不足或收盘价缺失："
            f"{coverage['missing_future_price']['3']}/{coverage['missing_future_price']['5']}/{coverage['missing_future_price']['10']}。"
        )

    render_v59_overall_return_summary(performance)
    render_v59_strategy_effectiveness_analysis(performance)
    render_v59_paper_trading_account(snapshot_version, daily_version)

    st.subheader("按最终交易评分分组")
    st.caption("final_trade_score：90–100、80–90、70–80、<70。“10个交易日”行的平均最终收益率即最终10日收益率。")
    st.dataframe(
        format_v59_historical_group_statistics(v59_historical_group_statistics(performance, "final_trade_score")),
        width="stretch",
        hide_index=True,
    )

    st.subheader("按换手率分组")
    st.caption("turnover_rate：<5%、5%–10%、10%–20%、20%–40%、>40%。")
    st.dataframe(
        format_v59_historical_group_statistics(v59_historical_group_statistics(performance, "turnover_rate")),
        width="stretch",
        hide_index=True,
    )

    st.subheader("按量比分组")
    st.caption("volume_ratio：<1、1–2、2–5、>5。")
    st.dataframe(
        format_v59_historical_group_statistics(v59_historical_group_statistics(performance, "volume_ratio")),
        width="stretch",
        hide_index=True,
    )

    with st.expander("逐只候选池收益记录", expanded=False):
        st.caption("记录直接来自每日候选池快照；收益为入选日收盘价买入后第 3、5、10 个交易日的收盘收益。")
        st.dataframe(
            format_v59_candidate_performance_table(performance),
            width="stretch",
            hide_index=True,
        )


def _local_csv_version(directory: Path, pattern: str) -> tuple[tuple[str, int, int], ...]:
    """Build a lightweight cache key from local immutable-report files."""
    if not directory.exists():
        return ()
    return tuple(
        (file_path.name, file_path.stat().st_mtime_ns, file_path.stat().st_size)
        for file_path in sorted(directory.glob(pattern))
        if file_path.is_file()
    )


@st.cache_data(show_spinner=False, max_entries=4)
def load_v59_paper_trading_account(
    snapshot_version: tuple[tuple[str, int, int], ...],
    daily_version: tuple[tuple[str, int, int], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load a cached, local-only V5.9 top-five paper-account simulation."""
    del snapshot_version, daily_version  # Cache inputs intentionally track local source-file changes.
    return run_v59_paper_account(
        PaperAccountConfig(
            candidate_dir=V59_CANDIDATE_HISTORY_DIR,
            daily_dir=HISTORICAL_DAILY_DIR,
        )
    )


def render_v59_paper_trading_account(
    snapshot_version: tuple[tuple[str, int, int], ...],
    daily_version: tuple[tuple[str, int, int], ...],
) -> None:
    """Render the offline account without creating any real trading action."""
    trades, equity_curve, summary, coverage = load_v59_paper_trading_account(snapshot_version, daily_version)
    st.subheader("V5.9 模拟交易账户")
    st.caption(
        "纯历史模拟：每日收盘按最终交易评分买入前 5 只，持有 5 个后续交易日收盘卖出。"
        "初始净值为 100，采用五日滚动仓位；每只股票占初始净值的 4%。"
        "不连接券商、不发送订单、不使用真实资金；未计手续费、滑点和税费。"
    )

    cards = [
        ("完成模拟交易", str(summary["trade_count"])),
        ("5日平均收益", format_change_percent(summary["average_return_pct"])),
        ("5日胜率", format_plain_percent(summary["win_rate_pct"])),
        ("最大回撤", format_change_percent(summary["max_drawdown_pct"])),
        ("期末净值", _format_paper_net_value(summary["final_net_value"])),
    ]
    for column, (label, value) in zip(st.columns(5), cards):
        column.metric(label, value)

    if trades.empty or equity_curve.empty:
        st.info(
            "当前候选池与本地日线尚不足以完成 5 日模拟交易。"
            f"候选池交易日：{coverage['candidate_pool_dates']}；"
            f"已选信号：{coverage['selected_rows']}；"
            f"后续日线不足：{coverage['insufficient_future_history']}；"
            f"价格缺失：{coverage['missing_price_history']}。"
        )
        return

    st.markdown("**资金曲线（归一化净值）**")
    st.line_chart(
        equity_curve.set_index("date")[["net_value"]],
        height=240,
    )
    with st.expander("模拟交易明细", expanded=False):
        st.dataframe(
            format_v59_paper_trade_table(trades),
            width="stretch",
            hide_index=True,
        )


def format_v59_paper_trade_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Format paper-account trades without modifying their simulated values."""
    output = trades.copy().rename(
        columns={
            "code": "股票代码",
            "buy_date": "买入日期",
            "exit_date": "卖出日期",
            "buy_price": "买入价格",
            "exit_price": "卖出价格",
            "buy_score": "买入评分",
            "return_5d": "5日收益率",
            "allocation_pct": "初始资金占比",
            "candidate_source": "候选池快照",
        }
    )
    for column in ("买入日期", "卖出日期"):
        output[column] = pd.to_datetime(output[column], errors="coerce").dt.strftime("%Y-%m-%d")
    output["股票代码"] = output["股票代码"].map(
        lambda value: display_text(value).zfill(6) if display_text(value) != "--" else "--"
    )
    output["买入价格"] = output["买入价格"].map(format_price)
    output["卖出价格"] = output["卖出价格"].map(format_price)
    output["买入评分"] = output["买入评分"].map(_format_v59_numeric_score)
    output["5日收益率"] = output["5日收益率"].map(format_change_percent)
    output["初始资金占比"] = output["初始资金占比"].map(format_plain_percent)
    return clean_display_frame(output)


def _format_paper_net_value(value: Any) -> str:
    """Present the normalized, non-monetary paper-account net value."""
    number = pd.to_numeric(value, errors="coerce")
    return "--" if pd.isna(number) else f"{float(number):.2f}"


@st.cache_data(show_spinner=False, max_entries=4)
def load_v59_candidate_pool_performance(
    snapshot_version: tuple[tuple[str, int, int], ...],
    daily_version: tuple[tuple[str, int, int], ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate offline 3/5/10-day returns without querying or re-scoring."""
    del snapshot_version, daily_version  # Cache inputs intentionally track local-file changes.
    snapshot_files = sorted(V59_CANDIDATE_HISTORY_DIR.glob("v5_9_candidate_*.csv"))
    coverage = {
        "snapshot_files": len(snapshot_files),
        "candidate_rows": 0,
        "completed_rows": {str(days): 0 for days in V59_PERFORMANCE_HOLDING_DAYS},
        "missing_signal_date": 0,
        "missing_future_price": {str(days): 0 for days in V59_PERFORMANCE_HOLDING_DAYS},
    }
    frames: list[pd.DataFrame] = []
    for file_path in snapshot_files:
        try:
            data = pd.read_csv(file_path, dtype={"code": "string"})
        except Exception:
            continue
        coverage["candidate_rows"] += len(data)
        required = {"code", "trade_date", "price", "final_trade_score", "turnover_rate", "volume_ratio"}
        if data.empty or not required.issubset(data.columns):
            continue
        frame = data.reindex(
            columns=[
                "code",
                "name",
                "trade_date",
                "price",
                "final_trade_score",
                "turnover_rate",
                "volume_ratio",
            ]
        ).copy()
        frame["code"] = _normalize_history_codes(frame["code"])
        frame["signal_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        for column in ("price", "final_trade_score", "turnover_rate", "volume_ratio"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[(frame["code"] != "") & frame["signal_date"].notna() & (frame["price"] > 0)]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return _empty_v59_performance_frame(), coverage
    candidates = pd.concat(frames, ignore_index=True)
    candidates["sample_id"] = pd.RangeIndex(len(candidates), dtype="int64")
    daily_files = _daily_history_files_by_date()
    trading_dates = sorted(daily_files)
    date_positions = {trade_date: index for index, trade_date in enumerate(trading_dates)}
    for days in V59_PERFORMANCE_HOLDING_DAYS:
        candidates[f"exit_date_{days}d"] = pd.NaT
    for index, signal_date in candidates["signal_date"].items():
        position = date_positions.get(signal_date)
        if position is None:
            coverage["missing_signal_date"] += 1
            continue
        for days in V59_PERFORMANCE_HOLDING_DAYS:
            if position + days >= len(trading_dates):
                coverage["missing_future_price"][str(days)] += 1
                continue
            candidates.at[index, f"exit_date_{days}d"] = trading_dates[position + days]

    price_requests: list[pd.DataFrame] = []
    for days in V59_PERFORMANCE_HOLDING_DAYS:
        exit_date_column = f"exit_date_{days}d"
        request = candidates.loc[candidates[exit_date_column].notna(), ["code", exit_date_column]].rename(
            columns={exit_date_column: "exit_date"}
        )
        if not request.empty:
            price_requests.append(request)
    exit_prices = _load_v59_exit_prices(
        pd.concat(price_requests, ignore_index=True).drop_duplicates(["code", "exit_date"])
        if price_requests
        else pd.DataFrame(columns=["code", "exit_date"]),
        daily_files,
    )
    for days in V59_PERFORMANCE_HOLDING_DAYS:
        exit_date_column = f"exit_date_{days}d"
        exit_price_column = f"exit_close_{days}d"
        candidates = candidates.merge(
            exit_prices.rename(columns={"exit_date": exit_date_column, "exit_close": exit_price_column}),
            on=["code", exit_date_column],
            how="left",
        )
        return_column = f"return_{days}d"
        candidates[return_column] = (candidates[exit_price_column] / candidates["price"] - 1) * 100
        valid_exit_date = candidates[exit_date_column].notna()
        missing_prices = valid_exit_date & candidates[return_column].isna()
        coverage["missing_future_price"][str(days)] += int(missing_prices.sum())
        coverage["completed_rows"][str(days)] = int(candidates[return_column].notna().sum())
    holding_extremes = _load_v59_holding_extremes(candidates, daily_files, date_positions, trading_dates)
    for days in V59_PERFORMANCE_HOLDING_DAYS:
        period = holding_extremes[holding_extremes["holding_days"] == days].rename(
            columns={
                "period_high": f"period_high_{days}d",
                "period_low": f"period_low_{days}d",
            }
        )
        candidates = candidates.merge(
            period[["sample_id", f"period_high_{days}d", f"period_low_{days}d"]],
            on="sample_id",
            how="left",
        )
        high_return_column = f"highest_return_{days}d"
        drawdown_column = f"max_drawdown_{days}d"
        candidates[high_return_column] = (candidates[f"period_high_{days}d"] / candidates["price"] - 1) * 100
        raw_drawdown = (candidates[f"period_low_{days}d"] / candidates["price"] - 1) * 100
        candidates[drawdown_column] = raw_drawdown.clip(upper=0)
    return candidates.reset_index(drop=True), coverage


def _daily_history_files_by_date() -> dict[pd.Timestamp, Path]:
    """Map only locally cached trading days; no network fallback is allowed."""
    files: dict[pd.Timestamp, Path] = {}
    for file_path in HISTORICAL_DAILY_DIR.glob("daily_*.csv"):
        matched = re.search(r"daily_(\d{8})$", file_path.stem)
        if not matched:
            continue
        trade_date = pd.to_datetime(matched.group(1), format="%Y%m%d", errors="coerce")
        if pd.notna(trade_date):
            files[pd.Timestamp(trade_date).normalize()] = file_path
    return files


def _load_v59_exit_prices(candidates: pd.DataFrame, daily_files: dict[pd.Timestamp, Path]) -> pd.DataFrame:
    """Read only the needed local future-day close files for the candidate rows."""
    frames: list[pd.DataFrame] = []
    for exit_date, day_candidates in candidates.groupby("exit_date", sort=True):
        file_path = daily_files.get(pd.Timestamp(exit_date).normalize())
        if file_path is None:
            continue
        try:
            daily = pd.read_csv(file_path, usecols=["ts_code", "close"])
        except Exception:
            continue
        daily["code"] = _normalize_history_codes(daily["ts_code"])
        daily["exit_close"] = pd.to_numeric(daily["close"], errors="coerce")
        codes = set(day_candidates["code"].astype(str))
        daily = daily[daily["code"].isin(codes) & (daily["exit_close"] > 0)]
        if not daily.empty:
            daily["exit_date"] = pd.Timestamp(exit_date).normalize()
            frames.append(daily[["code", "exit_date", "exit_close"]])
    if not frames:
        return pd.DataFrame(
            {
                "code": pd.Series(dtype="string"),
                "exit_date": pd.Series(dtype="datetime64[ns]"),
                "exit_close": pd.Series(dtype="float64"),
            }
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(["code", "exit_date"], keep="last")


def _load_v59_holding_extremes(
    candidates: pd.DataFrame,
    daily_files: dict[pd.Timestamp, Path],
    date_positions: dict[pd.Timestamp, int],
    trading_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Read only local high/low bars needed for each 3/5/10-day holding window."""
    requests: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date).normalize()
        position = date_positions.get(signal_date)
        if position is None:
            continue
        for days in V59_PERFORMANCE_HOLDING_DAYS:
            if pd.isna(getattr(row, f"exit_date_{days}d")):
                continue
            for bar_date in trading_dates[position + 1 : position + days + 1]:
                requests.append(
                    {
                        "sample_id": row.sample_id,
                        "code": row.code,
                        "holding_days": days,
                        "bar_date": bar_date,
                    }
                )
    if not requests:
        return _empty_v59_holding_extremes()

    requested = pd.DataFrame(requests)
    bar_frames: list[pd.DataFrame] = []
    for bar_date, day_requests in requested.groupby("bar_date", sort=True):
        file_path = daily_files.get(pd.Timestamp(bar_date).normalize())
        if file_path is None:
            continue
        try:
            bars = pd.read_csv(file_path, usecols=["ts_code", "high", "low"])
        except Exception:
            continue
        bars["code"] = _normalize_history_codes(bars["ts_code"])
        bars["high"] = pd.to_numeric(bars["high"], errors="coerce")
        bars["low"] = pd.to_numeric(bars["low"], errors="coerce")
        codes = set(day_requests["code"].astype(str))
        bars = bars[bars["code"].isin(codes) & (bars["high"] > 0) & (bars["low"] > 0)]
        if not bars.empty:
            bars["bar_date"] = pd.Timestamp(bar_date).normalize()
            bar_frames.append(bars[["code", "bar_date", "high", "low"]])
    if not bar_frames:
        return _empty_v59_holding_extremes()

    windows = requested.merge(
        pd.concat(bar_frames, ignore_index=True).drop_duplicates(["code", "bar_date"], keep="last"),
        on=["code", "bar_date"],
        how="left",
    )
    return (
        windows.groupby(["sample_id", "holding_days"], as_index=False)
        .agg(period_high=("high", "max"), period_low=("low", "min"))
    )


def _empty_v59_holding_extremes() -> pd.DataFrame:
    """Return a stable empty schema for unavailable local high/low windows."""
    return pd.DataFrame(
        {
            "sample_id": pd.Series(dtype="int64"),
            "holding_days": pd.Series(dtype="int64"),
            "period_high": pd.Series(dtype="float64"),
            "period_low": pd.Series(dtype="float64"),
        }
    )


def _normalize_history_codes(values: pd.Series) -> pd.Series:
    """Normalize six-digit codes shared by candidate and historical-bar caches."""
    codes = values.astype("string").fillna("").str.extract(r"(\d{6})", expand=False).fillna("")
    return codes.astype("string")


def _empty_v59_performance_frame() -> pd.DataFrame:
    """Return a stable, empty offline-performance schema."""
    return pd.DataFrame(
        {
            "code": pd.Series(dtype="string"),
            "name": pd.Series(dtype="string"),
            "signal_date": pd.Series(dtype="datetime64[ns]"),
            "price": pd.Series(dtype="float64"),
            "final_trade_score": pd.Series(dtype="float64"),
            "turnover_rate": pd.Series(dtype="float64"),
            "volume_ratio": pd.Series(dtype="float64"),
            **{
                f"exit_date_{days}d": pd.Series(dtype="datetime64[ns]")
                for days in V59_PERFORMANCE_HOLDING_DAYS
            },
            **{
                f"exit_close_{days}d": pd.Series(dtype="float64")
                for days in V59_PERFORMANCE_HOLDING_DAYS
            },
            **{
                f"return_{days}d": pd.Series(dtype="float64")
                for days in V59_PERFORMANCE_HOLDING_DAYS
            },
            **{
                f"period_high_{days}d": pd.Series(dtype="float64")
                for days in V59_PERFORMANCE_HOLDING_DAYS
            },
            **{
                f"period_low_{days}d": pd.Series(dtype="float64")
                for days in V59_PERFORMANCE_HOLDING_DAYS
            },
            **{
                f"highest_return_{days}d": pd.Series(dtype="float64")
                for days in V59_PERFORMANCE_HOLDING_DAYS
            },
            **{
                f"max_drawdown_{days}d": pd.Series(dtype="float64")
                for days in V59_PERFORMANCE_HOLDING_DAYS
            },
        }
    )


@st.cache_data(show_spinner=False, max_entries=4)
def load_strategy_tracking_history_summary(
    history_version: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Summarize existing local tracking snapshots for data-source traceability."""
    del history_version  # The version argument invalidates cached local CSV reads.
    empty = {"snapshot_rows": 0, "tracking_count": 0, "latest_snapshot_date": ""}
    if not STRATEGY_TRACKING_HISTORY_PATH.exists():
        return empty
    try:
        data = pd.read_csv(STRATEGY_TRACKING_HISTORY_PATH, dtype={"tracking_id": "string"})
    except Exception:
        return empty
    if data.empty:
        return empty
    dates = pd.to_datetime(data.get("snapshot_date"), errors="coerce")
    return {
        "snapshot_rows": int(len(data)),
        "tracking_count": int(data.get("tracking_id", pd.Series(dtype="string")).dropna().nunique()),
        "latest_snapshot_date": dates.max().date().isoformat() if dates.notna().any() else "",
    }


def v59_historical_group_statistics(performance: pd.DataFrame, field: str) -> pd.DataFrame:
    """Aggregate existing 3/5/10-day returns by one immutable snapshot field."""
    rules = _v59_historical_group_rules(field)
    values = pd.to_numeric(performance.get(field, pd.Series(dtype="float64")), errors="coerce")
    rows: list[dict[str, Any]] = []
    for days in V59_PERFORMANCE_HOLDING_DAYS:
        returns = pd.to_numeric(performance.get(f"return_{days}d", pd.Series(dtype="float64")), errors="coerce")
        highest_returns = pd.to_numeric(
            performance.get(f"highest_return_{days}d", pd.Series(dtype="float64")),
            errors="coerce",
        )
        max_drawdowns = pd.to_numeric(
            performance.get(f"max_drawdown_{days}d", pd.Series(dtype="float64")),
            errors="coerce",
        )
        for label, matches in rules:
            group_mask = _v59_boolean_mask(matches(values))
            completed_mask = group_mask & returns.notna()
            group_returns = returns[completed_mask]
            group_highest_returns = highest_returns[completed_mask & highest_returns.notna()]
            group_max_drawdowns = max_drawdowns[completed_mask & max_drawdowns.notna()]
            sample_count = int(len(group_returns))
            rows.append(
                {
                    "持有周期": f"{days}个交易日",
                    "分组": label,
                    "样本数量": sample_count,
                    "盈利数量": int((group_returns > 0).sum()),
                    "胜率": float((group_returns > 0).mean() * 100) if sample_count else pd.NA,
                    "平均最终收益率": float(group_returns.mean()) if sample_count else pd.NA,
                    "平均最高收益": float(group_highest_returns.mean()) if not group_highest_returns.empty else pd.NA,
                    "平均最大回撤": float(group_max_drawdowns.mean()) if not group_max_drawdowns.empty else pd.NA,
                    "最大收益": float(group_returns.max()) if sample_count else pd.NA,
                    "最大亏损": float(group_returns.min()) if sample_count else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def _v59_historical_group_rules(field: str) -> list[tuple[str, Any]]:
    """Return the fixed snapshot group definitions shared by V5.9 reports."""
    rules = {
        "final_trade_score": [
            ("90-100", lambda value: value >= 90),
            ("80-90", lambda value: (value >= 80) & (value < 90)),
            ("70-80", lambda value: (value >= 70) & (value < 80)),
            ("<70", lambda value: value < 70),
        ],
        "turnover_rate": [
            ("<5%", lambda value: value < 5),
            ("5%-10%", lambda value: (value >= 5) & (value < 10)),
            ("10%-20%", lambda value: (value >= 10) & (value < 20)),
            ("20%-40%", lambda value: (value >= 20) & (value <= 40)),
            (">40%", lambda value: value > 40),
        ],
        "volume_ratio": [
            ("<1", lambda value: value < 1),
            ("1-2", lambda value: (value >= 1) & (value < 2)),
            ("2-5", lambda value: (value >= 2) & (value <= 5)),
            (">5", lambda value: value > 5),
        ],
    }
    if field not in rules:
        raise ValueError(f"Unsupported V5.9 historical group field: {field}")
    return rules[field]


def _v59_boolean_mask(values: pd.Series) -> pd.Series:
    """Normalize nullable pandas comparison results before row filtering."""
    return values.fillna(False).astype(bool)


def render_v59_overall_return_summary(performance: pd.DataFrame) -> None:
    """Render the requested all-sample return summary without changing signals."""
    st.subheader("整体收益表现")
    st.caption("按全部有效候选样本汇总；各持有周期仅统计已具备对应后续收盘价的记录。")
    st.dataframe(
        format_v59_overall_return_statistics(v59_overall_return_statistics(performance)),
        width="stretch",
        hide_index=True,
    )


def v59_overall_return_statistics(performance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed local candidate-pool returns for each holding period."""
    rows: list[dict[str, Any]] = []
    for days in V59_PERFORMANCE_HOLDING_DAYS:
        returns = pd.to_numeric(performance.get(f"return_{days}d", pd.Series(dtype="float64")), errors="coerce")
        completed_returns = returns[returns.notna()]
        sample_count = int(len(completed_returns))
        rows.append(
            {
                "持有周期": f"{days}个交易日",
                "样本数量": sample_count,
                "盈利数量": int((completed_returns > 0).sum()),
                "胜率": float((completed_returns > 0).mean() * 100) if sample_count else pd.NA,
                "平均收益": float(completed_returns.mean()) if sample_count else pd.NA,
                "最大收益": float(completed_returns.max()) if sample_count else pd.NA,
                "最大亏损": float(completed_returns.min()) if sample_count else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def format_v59_overall_return_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Format all-sample V5.9 return statistics for the strategy page."""
    output = data.copy()
    output["统计状态"] = output["样本数量"].map(
        lambda count: "样本不足" if pd.isna(count) or int(count) < 3 else "已统计"
    )
    output["胜率"] = output["胜率"].map(format_plain_percent)
    for column in ("平均收益", "最大收益", "最大亏损"):
        output[column] = output[column].map(format_change_percent)
    return clean_display_frame(output)


def format_v59_candidate_performance_table(performance: pd.DataFrame) -> pd.DataFrame:
    """Present immutable per-stock snapshot fields alongside local holding returns."""
    columns = [
        "code",
        "name",
        "signal_date",
        "price",
        "final_trade_score",
        "turnover_rate",
        "volume_ratio",
        "return_3d",
        "return_5d",
        "return_10d",
    ]
    output = performance.reindex(columns=columns).copy().rename(
        columns={
            "code": "股票代码",
            "name": "股票名称",
            "signal_date": "入选日期",
            "price": "入选价格",
            "final_trade_score": "最终交易评分",
            "turnover_rate": "换手率",
            "volume_ratio": "量比",
            "return_3d": "3日收益率",
            "return_5d": "5日收益率",
            "return_10d": "10日收益率",
        }
    )
    output["入选日期"] = pd.to_datetime(output["入选日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    output["股票代码"] = output["股票代码"].map(lambda value: display_text(value).zfill(6) if display_text(value) != "--" else "--")
    output["股票名称"] = output["股票名称"].map(display_text)
    output["入选价格"] = output["入选价格"].map(format_price)
    output["最终交易评分"] = output["最终交易评分"].map(_format_v59_numeric_score)
    output["换手率"] = output["换手率"].map(format_change_percent)
    output["量比"] = output["量比"].map(_format_v59_ratio)
    for column in ("3日收益率", "5日收益率", "10日收益率"):
        output[column] = output[column].map(format_change_percent)
    return clean_display_frame(output)


def _format_v59_numeric_score(value: Any) -> str:
    """Format a score for historical display without applying score labels."""
    number = pd.to_numeric(value, errors="coerce")
    return "--" if pd.isna(number) else f"{float(number):.2f}"


def _format_v59_ratio(value: Any) -> str:
    """Format a raw volume ratio without treating it as a percentage."""
    number = pd.to_numeric(value, errors="coerce")
    return "--" if pd.isna(number) else f"{float(number):.2f}"


def render_v59_strategy_effectiveness_analysis(performance: pd.DataFrame) -> None:
    """Show compact V5.9 effectiveness tables from immutable local snapshots."""
    score_statistics = v59_strategy_effectiveness_statistics(performance, "final_trade_score")
    turnover_statistics = v59_strategy_effectiveness_statistics(performance, "turnover_rate")

    st.subheader("策略有效性分析")
    st.caption(
        "总样本为具备代码、日期和买入价的本地候选池快照记录。各持有期胜率和平均收益仅使用该持有期已具备后续收盘价的样本。"
    )
    st.metric("总样本数量", str(len(performance)))

    st.markdown("**按最终交易评分区间**")
    st.dataframe(
        format_v59_score_effectiveness_statistics(score_statistics),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**按换手率区间（5日表现）**")
    st.dataframe(
        format_v59_turnover_effectiveness_statistics(turnover_statistics),
        width="stretch",
        hide_index=True,
    )


def v59_strategy_effectiveness_statistics(performance: pd.DataFrame, field: str) -> pd.DataFrame:
    """Aggregate per-horizon effectiveness without recalculating scores or signals."""
    values = pd.to_numeric(performance.get(field, pd.Series(dtype="float64")), errors="coerce")
    rows: list[dict[str, Any]] = []
    for label, matches in _v59_historical_group_rules(field):
        group_mask = _v59_boolean_mask(matches(values))
        row: dict[str, Any] = {
            "分组": label,
            "样本数量": int(group_mask.sum()),
        }
        for days in V59_PERFORMANCE_HOLDING_DAYS:
            returns = pd.to_numeric(performance.get(f"return_{days}d", pd.Series(dtype="float64")), errors="coerce")
            completed_returns = returns[group_mask & returns.notna()]
            sample_count = int(len(completed_returns))
            row[f"{days}日胜率"] = (
                float((completed_returns > 0).mean() * 100) if sample_count else pd.NA
            )
            row[f"{days}日平均收益"] = float(completed_returns.mean()) if sample_count else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def format_v59_score_effectiveness_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Format score-band effectiveness values while keeping each horizon explicit."""
    output = data.copy()
    for days in V59_PERFORMANCE_HOLDING_DAYS:
        output[f"{days}日胜率"] = output[f"{days}日胜率"].map(format_plain_percent)
        output[f"{days}日平均收益"] = output[f"{days}日平均收益"].map(format_change_percent)
    return clean_display_frame(output)


def format_v59_turnover_effectiveness_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the turnover effectiveness table focused on the requested 5-day result."""
    output = data[["分组", "样本数量", "5日胜率", "5日平均收益"]].copy()
    output["5日胜率"] = output["5日胜率"].map(format_plain_percent)
    output["5日平均收益"] = output["5日平均收益"].map(format_change_percent)
    return clean_display_frame(output)


def format_v59_historical_group_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Format offline V5.9 group statistics for presentation only."""
    output = data.copy()
    output["统计状态"] = output["样本数量"].map(lambda count: "样本不足" if pd.isna(count) or int(count) < 3 else "已统计")
    for column in ("胜率", "平均最终收益率", "平均最高收益", "平均最大回撤", "最大收益", "最大亏损"):
        output[column] = output[column].map(format_change_percent if column != "胜率" else format_plain_percent)
    return clean_display_frame(output)


def render_strategy_tracking_exit_action_table(
    tracking_pool: pd.DataFrame,
    display_data: pd.DataFrame,
) -> None:
    """Show the tracking ledger with a row-specific, local close action."""
    source_rows = tracking_pool.reset_index(drop=True).copy()
    table = display_data.reset_index(drop=True).copy()
    if len(source_rows) != len(table):
        st.error("策略跟踪表格行数不一致，当前未展示结束操作。")
        render_compact_stock_dataframe(table)
        return

    is_closed = source_rows["status"].fillna("").astype(str).eq("已结束")
    action_column = "操作"
    table[action_column] = pd.Series("结束跟踪", index=table.index).where(~is_closed, "已结束")
    click_key = "strategy_tracking_exit_table_click"
    with st.container(horizontal_alignment="center"):
        st.dataframe(
            style_watch_table(table),
            column_config={
                **compact_table_column_config(table),
                action_column: st.column_config.ButtonColumn(
                    action_column,
                    width=96,
                    help="结束该股票的本地策略跟踪；不会发送交易指令。",
                    type="secondary",
                    on_click=_handle_strategy_tracking_exit_table_click,
                    args=(source_rows, click_key),
                    key=click_key,
                ),
            },
            width="content",
            hide_index=True,
        )


def _handle_strategy_tracking_exit_table_click(source_rows: pd.DataFrame, click_key: str) -> None:
    """Open the close-confirmation dialog only for the clicked tracking row."""
    click = st.session_state.get(click_key)
    try:
        row_index = int(click.get("row"))
    except (AttributeError, TypeError, ValueError):
        return
    if row_index < 0 or row_index >= len(source_rows):
        return

    row = source_rows.iloc[row_index]
    if str(row.get("status") or "").strip() == "已结束":
        st.session_state["strategy_tracking_feedback"] = {
            "message": "该股票的策略跟踪已结束。",
            "tone": "muted",
        }
        return

    tracking_id = str(row.get("tracking_id") or "").strip()
    if not tracking_id:
        st.session_state["strategy_tracking_feedback"] = {
            "message": "策略跟踪记录缺少唯一标识，无法结束。",
            "tone": "muted",
        }
        return
    st.session_state["strategy_tracking_exit_tracking_id"] = tracking_id


def _clear_strategy_tracking_exit_dialog() -> None:
    """Clear the row-specific dialog selection after dismissal or completion."""
    st.session_state.pop("strategy_tracking_exit_tracking_id", None)


@st.dialog("确认结束策略跟踪", width="small", on_dismiss=_clear_strategy_tracking_exit_dialog)
def _render_strategy_tracking_exit_dialog(selected_row: pd.Series) -> None:
    """Collect one local close record without touching trade or score logic."""
    tracking_id = str(selected_row.get("tracking_id") or "").strip()
    code = str(selected_row.get("code") or "").zfill(6)
    name = display_text(selected_row.get("name"))
    current_price = pd.to_numeric(selected_row.get("current_price"), errors="coerce")
    default_price = float(current_price) if pd.notna(current_price) and current_price > 0 else 0.0

    st.caption("该操作只记录本地观察结束信息，不会发送交易指令。")
    with st.form(f"strategy_tracking_exit_form_{tracking_id}"):
        st.text_input("股票代码", value=code, disabled=True)
        st.text_input("股票名称", value=name, disabled=True)
        st.number_input(
            "当前价格",
            min_value=0.0,
            value=default_price,
            step=0.01,
            format="%.3f",
            disabled=True,
        )
        exit_price = st.number_input(
            "结束价格",
            min_value=0.0,
            value=default_price,
            step=0.01,
            format="%.3f",
        )
        exit_date = st.date_input("结束日期", value=date.today())
        exit_reason = st.selectbox("结束原因", ["止盈", "止损", "时间退出", "人工观察结束", "其他"])
        confirm_column, cancel_column = st.columns(2)
        confirmed = confirm_column.form_submit_button("确认结束跟踪", type="primary", width="stretch")
        cancelled = cancel_column.form_submit_button("取消", width="stretch")

    if cancelled:
        _clear_strategy_tracking_exit_dialog()
        st.rerun()
    if not confirmed:
        return

    ended, message = end_strategy_tracking_stock(
        tracking_id,
        exit_price=exit_price,
        exit_date=exit_date,
        exit_reason=exit_reason,
    )
    if ended:
        _clear_strategy_tracking_exit_dialog()
        st.session_state["strategy_tracking_feedback"] = {"message": message, "tone": "ok"}
        st.rerun()
    _show_compact_feedback(message, tone="muted")


def render_strategy_tracking_exit_dialog(tracking_pool: pd.DataFrame) -> None:
    """Resolve the selected row and invoke at most one Streamlit dialog."""
    tracking_id = str(st.session_state.get("strategy_tracking_exit_tracking_id") or "").strip()
    if not tracking_id:
        return
    selected = tracking_pool[tracking_pool["tracking_id"].astype(str) == tracking_id]
    if selected.empty:
        _clear_strategy_tracking_exit_dialog()
        st.session_state["strategy_tracking_feedback"] = {
            "message": "策略跟踪记录已不存在，无法结束。",
            "tone": "muted",
        }
        st.rerun()
    row = selected.iloc[0]
    if str(row.get("status") or "").strip() == "已结束":
        _clear_strategy_tracking_exit_dialog()
        st.session_state["strategy_tracking_feedback"] = {
            "message": "该股票的策略跟踪已结束。",
            "tone": "muted",
        }
        st.rerun()
    _render_strategy_tracking_exit_dialog(row)


def render_watchlist_editor(user_watchlist: pd.DataFrame) -> None:
    """Render local watchlist edit controls."""
    with st.expander("管理本地自选股", expanded=False):
        options = {
            f"{str(row.get('code') or '').zfill(6)} - {row.get('name', '--')}": row
            for _, row in user_watchlist.iterrows()
        }
        selected = st.selectbox("选择自选股", [""] + list(options.keys()), key="edit_watchlist_stock")
        if not selected:
            st.caption("选择一只自选股后，可以修改标签、备注，或从本地自选股移除。")
            return

        row = options[selected]
        code = str(row.get("code") or "").zfill(6)
        tag = st.text_input(
            "标签",
            value=str(row.get("tag") or ""),
            placeholder="例如：光通信观察、AI硬件、半导体反弹、明日重点、高位风险",
            key=f"watchlist_tag_{code}",
        )
        note = st.text_area("备注", value=str(row.get("note") or ""), height=80, key=f"watchlist_note_{code}")
        col1, col2 = st.columns([1, 1])
        if col1.button("保存标签和备注", key=f"save_watchlist_{code}"):
            ok, message = update_user_watchlist_stock(code, tag=tag, note=note)
            render_small_hint(message, tone="ok" if ok else "muted")
            if ok:
                st.rerun()
        if col2.button("移除自选股", key=f"remove_watchlist_{code}"):
            ok, message = remove_user_watchlist_stock(code)
            render_small_hint(message, tone="ok" if ok else "muted")
            if ok:
                st.rerun()


def render_holdings_editor(user_holdings: pd.DataFrame, candidates: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    """Render local manual holdings edit controls."""
    with st.expander("手动管理本地持仓股", expanded=False):
        st.caption("持仓数据只保存在本地 CSV，不读取券商账户、资金或真实持仓。T+1 状态仅作风险提示。")
        options = {
            f"{str(row.get('code') or '').zfill(6)} - {row.get('name', '--')}": row
            for _, row in user_holdings.iterrows()
        }
        selected = st.selectbox("选择已有持仓编辑", ["新增持仓"] + list(options.keys()), key="edit_holding_stock")
        row = options.get(selected)

        default_code = str(row.get("code") or "").zfill(6) if row is not None else ""
        default_name = str(row.get("name") or "") if row is not None else ""
        default_cost = float(row.get("cost_price") or 0) if row is not None else 0.0
        default_shares = float(row.get("shares") or 0) if row is not None else 0.0
        default_buy_date = str(row.get("buy_date") or "") if row is not None else ""
        default_tag = str(row.get("tag") or "持仓") if row is not None else "持仓"
        default_note = str(row.get("note") or "") if row is not None else ""

        col1, col2 = st.columns(2)
        code = col1.text_input("股票代码", value=default_code, key="holding_code")
        name = col2.text_input("股票名称", value=default_name, key="holding_name")
        col3, col4 = st.columns(2)
        cost_price = col3.number_input("成本价", min_value=0.0, value=default_cost, step=0.01, format="%.3f", key="holding_cost")
        shares = col4.number_input("持股数量", min_value=0.0, value=default_shares, step=100.0, key="holding_shares")
        col5, col6 = st.columns(2)
        buy_date = col5.text_input("买入日期", value=default_buy_date, placeholder="YYYY-MM-DD", key="holding_buy_date")
        tag = col6.text_input("标签", value=default_tag, key="holding_tag")
        note = st.text_area("备注", value=default_note, height=80, key="holding_note")

        btn1, btn2 = st.columns(2)
        if btn1.button("保存持仓", key="save_holding"):
            ok, message = upsert_user_holding(code, name, cost_price, shares, buy_date, tag=tag, note=note)
            render_small_hint(message, tone="ok" if ok else "muted")
            if ok:
                st.rerun()
        if row is not None and btn2.button("删除持仓", key=f"remove_holding_{default_code}"):
            ok, message = remove_user_holding(default_code)
            render_small_hint(message, tone="ok" if ok else "muted")
            if ok:
                st.rerun()


def render_market_rankings(candidates: pd.DataFrame) -> None:
    """Render distribution and ranking views from current dashboard data."""
    st.subheader("观察池涨跌与成交排行")
    if candidates.empty:
        st.info("暂无观察池数据，无法生成涨跌分布和排行。")
        return

    data = candidates.copy()
    data["pct_chg_num"] = pd.to_numeric(data.get("pct_chg"), errors="coerce")
    data["amount_num"] = pd.to_numeric(data.get("amount"), errors="coerce")
    data = data.dropna(subset=["pct_chg_num"])
    if data.empty:
        st.info("当前观察池缺少涨跌幅数据，无法生成分布和排行。")
        return

    st.caption("以下列表基于当前已生成的策略候选股/市场异动观察池，不代表全市场完整排行。")
    render_pct_distribution(data)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 成交额排行")
        render_rank_table(data.sort_values("amount_num", ascending=False).head(10))
    with col2:
        st.markdown("##### 涨幅前十")
        render_rank_table(data.sort_values("pct_chg_num", ascending=False).head(10))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### 跌幅前十")
        render_rank_table(data.sort_values("pct_chg_num", ascending=True).head(10))
    with col4:
        st.markdown("##### 涨停 / 跌停")
        limit_up = data[data["pct_chg_num"] >= 9.5].sort_values("pct_chg_num", ascending=False)
        limit_down = data[data["pct_chg_num"] <= -9.5].sort_values("pct_chg_num", ascending=True)
        st.write("涨停股列表")
        render_rank_table(limit_up)
        st.write("跌停股列表")
        render_rank_table(limit_down)


def render_pct_distribution(data: pd.DataFrame) -> None:
    """Render percentage-change distribution for current watch data."""
    bins = [-100, -9.5, -5, -2, 0, 2, 5, 9.5, 100]
    labels = ["跌停附近", "-9.5%~-5%", "-5%~-2%", "-2%~0", "0~2%", "2%~5%", "5%~9.5%", "涨停附近"]
    dist = data.copy()
    dist["区间"] = pd.cut(dist["pct_chg_num"], bins=bins, labels=labels, include_lowest=True)
    summary = dist.groupby("区间", observed=False).size().reset_index(name="数量")
    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("区间:N", title="涨跌幅区间", sort=labels),
            y=alt.Y("数量:Q", title="股票数量"),
            color=alt.Color(
                "区间:N",
                scale=alt.Scale(
                    domain=labels,
                    range=["#1f8f4d", "#35a566", "#7bcf97", "#b7d7c2", "#f4b6b6", "#ef7d7d", "#df4b4b", "#c91f1f"],
                ),
                legend=None,
            ),
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def render_rank_table(data: pd.DataFrame) -> None:
    """Render a compact Chinese ranking table."""
    if data.empty:
        st.info("暂无符合条件的股票。")
        return
    columns = [column for column in ["code", "name", "pct_chg_display", "amount_display", "tracking_summary", "risk_summary"] if column in data.columns]
    output = data[columns].rename(columns=COLUMN_LABELS)
    st.dataframe(style_watch_table(clean_display_frame(output)), width="stretch", hide_index=True)


def render_sector_market_tab(status: dict[str, Any], candidates: pd.DataFrame) -> None:
    """Render sector strength and market environment."""
    st.subheader("大盘环境")
    raw_market = status.get("market_environment_detail")
    market = raw_market if isinstance(raw_market, dict) else {}
    cols = st.columns(6)
    market_env = str(status.get("market_environment", "--"))
    render_status_card(cols[0], "市场环境", market_env, "red" if market_env in {"强势", "偏强"} else "green" if market_env in {"偏弱", "极弱"} else "neutral")
    render_status_card(cols[1], "上涨家数", int(market.get("up_count") or 0), "red")
    render_status_card(cols[2], "下跌家数", int(market.get("down_count") or 0), "green")
    render_status_card(cols[3], "全市场成交额", format_amount(market.get("market_amount")), "neutral")
    render_status_card(cols[4], "上涨占比", format_percent(market.get("up_ratio")), "red")
    render_status_card(cols[5], "下跌占比", format_percent(market.get("down_ratio")), "green")
    index_rows = market.get("index_quotes") or []
    if index_rows:
        st.dataframe(pd.DataFrame(index_rows), width="stretch", hide_index=True)
    st.subheader("板块强度")
    raw_sector = status.get("sector_strength")
    sector = raw_sector if isinstance(raw_sector, dict) else {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 强势行业板块前 10")
        _render_records(sector.get("industry_top10") or [])
    with col2:
        st.markdown("##### 强势概念板块前 10")
        _render_records(sector.get("concept_top10") or [])
    render_market_rankings(candidates)


def render_developer_diagnostics(
    status: dict[str, Any],
    active_watchlist: pd.DataFrame,
    reference_candidates: pd.DataFrame,
) -> None:
    """Render source diagnostics and opt-in access to secondary data pools."""
    diagnostics = st.expander(
        "系统诊断 / 开发工具",
        expanded=False,
        icon=":material/developer_mode:",
        key="developer_diagnostics",
        on_change="rerun",
    )
    if not diagnostics.open:
        return
    with diagnostics:
        cols = st.columns(3)
        cols[0].metric("扫描时间", str(status.get("scan_time", "--")))
        cols[1].metric("当前数据源", str(status.get("data_source", "--")))
        cols[2].metric("当前数据状态", str(status.get("data_status", "--")))

        cols = st.columns(2)
        cols[0].metric("实时换手率覆盖率", _format_coverage(status.get("realtime_turnover_coverage")))
        cols[1].metric("实时量比覆盖率", _format_coverage(_realtime_volume_ratio_coverage(status)))

        st.caption("最近异常摘要")
        reasons = _friendly_diagnostic_reasons(status)
        if reasons:
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("暂无明显异常。")

        st.divider()
        st.markdown("**隐藏数据池入口**")
        st.caption("以下页面保留原有展示与交互，仅从主导航移入开发工具，不参与交易候选生成。")
        selected_pool = st.selectbox(
            "选择数据池",
            ["不显示", "市场异动观察池", "数据参考池"],
            key="developer_hidden_data_pool",
        )
        if selected_pool == "市场异动观察池":
            render_active_watchlist_tab(active_watchlist)
        elif selected_pool == "数据参考池":
            render_reference_candidates_tab(reference_candidates)


def _yes_no(value: Any) -> str:
    return "是" if bool(value) else "否"


def _format_coverage(value: Any) -> str:
    """Format a stored coverage ratio for display without changing its value."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    if 0 <= number <= 1:
        number *= 100
    return f"{number:.2f}%"


def _realtime_volume_ratio_coverage(status: dict[str, Any]) -> Any:
    """Read an existing local validation sample only when status lacks coverage."""
    stored = status.get("realtime_volume_ratio_coverage")
    if pd.notna(pd.to_numeric(stored, errors="coerce")):
        return stored

    sample = load_auxiliary_csv(CALCULATED_VOLUME_RATIO_PATH, dtype={"code": "string"})
    if sample.empty or "calculated_volume_ratio" not in sample.columns:
        return pd.NA
    values = pd.to_numeric(sample["calculated_volume_ratio"], errors="coerce")
    return values.between(0, 100).mean()


def _has_realtime_turnover(status: dict[str, Any]) -> bool:
    source = str(status.get("data_source") or "")
    missing_fields = {str(item) for item in status.get("missing_fields") or []}
    if "新浪" in source or "turnover" in missing_fields:
        return False
    return bool(status.get("turnover_field") or status.get("allow_strategy_candidate"))


def _has_realtime_volume_ratio(status: dict[str, Any]) -> bool:
    source = str(status.get("data_source") or "")
    missing_fields = {str(item) for item in status.get("missing_fields") or []}
    if "新浪" in source or "volume_ratio" in missing_fields:
        return False
    return bool(status.get("allow_strategy_candidate"))


def _diagnostic_reasons(status: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for item in status.get("warnings") or []:
        text = str(item).strip()
        if text and text not in reasons:
            reasons.append(text)
    for item in status.get("errors") or []:
        text = str(item).strip()
        if text and text not in reasons:
            reasons.append(text)
    for attempt in status.get("source_attempts") or []:
        if not isinstance(attempt, dict) or attempt.get("success"):
            continue
        source = str(attempt.get("source") or "未知数据源")
        error = str(attempt.get("error") or "未返回可用数据")
        summary = f"{source} 不可用：{error[:180]}"
        if summary not in reasons:
            reasons.append(summary)
    return reasons


def _friendly_diagnostic_reasons(status: dict[str, Any]) -> list[str]:
    """Keep the collapsed status useful without exposing raw tracebacks."""
    reasons: list[str] = []
    for warning in status.get("warnings") or []:
        text = str(warning).strip()
        if text:
            reasons.append(text[:120])
    for attempt in status.get("source_attempts") or []:
        if not isinstance(attempt, dict) or attempt.get("success"):
            continue
        source = str(attempt.get("source") or "公开行情源")
        message = f"{source}本轮连接未成功，已继续尝试其他可用行情源。"
        if message not in reasons:
            reasons.append(message)
    return reasons[:3]


def render_stock_detail_panel(candidates: pd.DataFrame, status: dict[str, Any] | None = None) -> None:
    """Render selected stock K-line detail panel."""
    if candidates.empty:
        st.info("请选择一只股票查看 K 线和详细技术信息。")
        return
    selected_code = str(st.session_state.get("selected_stock_code") or "").zfill(6)
    row = find_stock_row(candidates, selected_code) if selected_code and selected_code != "000000" else None
    options = {f"{row_data.get('name', '--')}（{str(row_data.get('code') or '').zfill(6)} / {row_data.get('source_type_display', '--')}）": row_data for _, row_data in candidates.iterrows()}
    if row is None:
        selected = st.selectbox("选择股票查看详情", [""] + list(options.keys()))
        if selected:
            row = options[selected]
            st.session_state["selected_stock_code"] = str(row.get("code") or "").zfill(6)
    else:
        render_small_hint(f"当前查看：{row.get('name', '--')}（{selected_code}）", tone="muted")
        selected = st.selectbox("切换查看其他股票", ["当前选择"] + list(options.keys()))
        if selected and selected != "当前选择":
            row = options[selected]
            st.session_state["selected_stock_code"] = str(row.get("code") or "").zfill(6)

    if row is None:
        st.info("请选择一只股票查看 K 线和详细技术信息。")
        return

    render_detail_header(row)
    render_detail_state_cards(row)
    render_detail_reason_and_risk(row)
    with st.expander("数据来源", expanded=False):
        render_detail_data_sources(row, status or {})

    code = str(row.get("code") or "").zfill(6)
    if render_tushare_cache_detail(row, code):
        return

    history = load_history_for_detail(code, 60)
    if history.empty:
        st.caption("历史K线数据暂不足，当前先展示实时行情与量化指标。")
        return
    detail = prepare_history_detail(history)
    if detail.empty:
        st.caption("历史K线数据暂不足，当前先展示实时行情与量化指标。")
        return
    render_kline_chart(row, detail)
    render_volume_chart(detail)
    analysis = build_detail_analysis(row, detail)
    st.subheader("辅助标签")
    label_cols = st.columns(5)
    for idx, (name, value) in enumerate(analysis["labels"].items()):
        label_cols[idx % 5].metric(name, value)
    st.subheader("技术摘要")
    for item in analysis["summary"]:
        st.write(f"- {item}")
    st.subheader("风险提示")
    for item in analysis["risks"]:
        st.write(f"- {item}")


def render_tushare_cache_detail(row: pd.Series, code: str) -> bool:
    """Render detail charts from local Tushare daily cache when available."""
    window_options = {"最近30日": 30, "最近60日": 60, "最近120日": 120}
    selected_window = st.selectbox(
        "K线范围",
        list(window_options),
        index=1,
        key=f"kline_window_{code}",
    )
    if st.button("更新该股历史K线", key=f"update_kline_{code}"):
        _, update_error = update_stock_kline_cache(code, days=120)
        if update_error:
            render_small_hint(update_error, tone="muted")
        else:
            _show_compact_feedback("该股历史K线缓存已更新。", tone="ok")

    kline = read_tushare_kline_from_cache(code, days=window_options[selected_window])
    if kline.empty:
        return False

    st.caption("K线来自本地Tushare历史缓存，非盘中实时数据。")
    chart_data = kline.copy()
    chart_data["date"] = pd.to_datetime(chart_data["trade_date"], errors="coerce")
    for column in ["open", "high", "low", "close", "vol"]:
        if column in chart_data.columns:
            chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    required_ohlc = ["open", "high", "low", "close"]
    chart_data = chart_data.dropna(subset=["date", *required_ohlc])
    chart_data = chart_data[(chart_data[required_ohlc] > 0).all(axis=1)]
    chart_data = chart_data[chart_data["high"] >= chart_data["low"]]
    chart_data = chart_data.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    if len(chart_data) >= 5:
        render_tushare_kline_or_trend(row, chart_data)
        render_tushare_technical_summary(chart_data)
    else:
        st.caption("历史K线数据暂不足，当前先展示实时行情与量化指标。")
    return True


def render_tushare_reference_info(row: pd.Series) -> None:
    """Render data provenance without presenting references as realtime values."""
    metrics = [
        ("换手率来源", format_turnover_source_display(row)),
        ("量比来源", format_volume_ratio_source_display(row)),
        ("参考数据日期", row.get("reference_trade_date_display", "--")),
    ]
    cols = st.columns(3)
    for idx, (label, value) in enumerate(metrics):
        render_status_card(cols[idx], label, str(value) if _usable(value) else "--", "neutral")


def render_tushare_kline_or_trend(row: pd.Series, data: pd.DataFrame) -> None:
    """Render a K-line chart with a date-aligned volume chart when available."""
    st.subheader(f"{row.get('name', '--')}（{row.get('code', '--')}）日K线")
    data = data.copy()
    data["date_label"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data = data.dropna(subset=["date_label"]).reset_index(drop=True)
    date_labels = data["date_label"].tolist()
    has_ohlc = all(column in data.columns and data[column].notna().any() for column in ["open", "high", "low", "close"])
    volume_data = data[[column for column in ["date", "date_label", "open", "close", "vol"] if column in data.columns]].copy()
    if "vol" in volume_data.columns:
        volume_data["vol"] = pd.to_numeric(volume_data["vol"], errors="coerce")
        volume_data = volume_data[volume_data["vol"] >= 0].dropna(subset=["date", "vol"])
        volume_data["volume_wan_lot"] = volume_data["vol"] / 10000
    else:
        volume_data = pd.DataFrame()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.04,
        )
        if has_ohlc:
            figure.add_trace(
                go.Candlestick(
                    x=data["date_label"],
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    increasing_line_color="#d62728",
                    increasing_fillcolor="#d62728",
                    decreasing_line_color="#1f8f4d",
                    decreasing_fillcolor="#1f8f4d",
                    name="K 线",
                ),
                row=1,
                col=1,
            )
        else:
            figure.add_trace(go.Scatter(x=data["date_label"], y=data["close"], mode="lines", name="收盘价", showlegend=False), row=1, col=1)
        for ma_col, label in [("ma5", "MA5"), ("ma10", "MA10"), ("ma20", "MA20"), ("ma60", "MA60")]:
            if ma_col in data.columns and data[ma_col].notna().any():
                figure.add_trace(
                    go.Scatter(
                        x=data["date_label"],
                        y=data[ma_col],
                        mode="lines",
                        name=label,
                        hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
        if not volume_data.empty:
            volume_colors = [
                "#d62728" if close >= open else "#1f8f4d"
                for open, close in zip(volume_data["open"], volume_data["close"])
            ]
            figure.add_trace(
                go.Bar(
                    x=volume_data["date_label"],
                    y=volume_data["volume_wan_lot"],
                    marker_color=volume_colors,
                    name="成交量",
                    showlegend=False,
                    hovertemplate="日期: %{x|%Y-%m-%d}<br>成交量: %{y:.2f} 万手<extra></extra>",
                ),
                row=2,
                col=1,
            )
        figure.update_layout(
            height=570 if not volume_data.empty else 430,
            title="" if has_ohlc else "收盘价走势",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=35, b=10),
        )
        figure.update_yaxes(title_text="价格", row=1, col=1)
        if not volume_data.empty:
            figure.update_yaxes(title_text="成交量（万手）", row=2, col=1)
        figure.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=date_labels,
            tickangle=-45,
            row=1,
            col=1,
        )
        figure.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=date_labels,
            tickangle=-45,
            row=2,
            col=1,
        )
        st.plotly_chart(figure, width="stretch")
    except Exception:
        if has_ohlc:
            fallback_data = data.copy()
            fallback_data["direction"] = fallback_data.apply(
                lambda item: "上涨" if item["close"] >= item["open"] else "下跌", axis=1
            )
            category_sort = alt.SortField(field="date", order="ascending")
            x_axis = alt.X(
                "date_label:N",
                title=None,
                sort=category_sort,
                axis=alt.Axis(labelAngle=-45, labelOverlap=False),
            )
            base = alt.Chart(fallback_data).encode(x=x_axis)
            wick = base.mark_rule().encode(
                y=alt.Y("low:Q", title="价格", scale=alt.Scale(zero=False)),
                y2="high:Q",
                color=alt.Color("direction:N", scale=alt.Scale(domain=["上涨", "下跌"], range=["#d62728", "#1f8f4d"]), legend=None),
                tooltip=[
                    alt.Tooltip("date_label:N", title="日期"),
                    alt.Tooltip("open:Q", title="开盘", format=".2f"),
                    alt.Tooltip("high:Q", title="最高", format=".2f"),
                    alt.Tooltip("low:Q", title="最低", format=".2f"),
                    alt.Tooltip("close:Q", title="收盘", format=".2f"),
                    alt.Tooltip("ma5:Q", title="MA5", format=".2f"),
                    alt.Tooltip("ma10:Q", title="MA10", format=".2f"),
                    alt.Tooltip("ma20:Q", title="MA20", format=".2f"),
                    alt.Tooltip("ma60:Q", title="MA60", format=".2f"),
                ],
            )
            candle = base.mark_bar(size=8).encode(
                y=alt.Y("open:Q", scale=alt.Scale(zero=False)),
                y2="close:Q",
                color=alt.Color("direction:N", scale=alt.Scale(domain=["上涨", "下跌"], range=["#d62728", "#1f8f4d"]), legend=None),
            )
            ma_data = fallback_data.melt(
                id_vars=["date", "date_label"], value_vars=["ma5", "ma10", "ma20", "ma60"], var_name="均线", value_name="value"
            ).dropna(subset=["value"])
            ma_lines = alt.Chart(ma_data).mark_line(strokeWidth=1.6).encode(
                x=alt.X("date_label:N", sort=category_sort, title=None, axis=None),
                y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
                color="均线:N",
            )
            kline_chart = (wick + candle + ma_lines).properties(height=430)
            if not volume_data.empty:
                volume_data = volume_data.copy()
                volume_data["direction"] = volume_data.apply(
                    lambda item: "上涨" if item["close"] >= item["open"] else "下跌", axis=1
                )
                volume_chart = alt.Chart(volume_data).mark_bar().encode(
                    x=alt.X(
                        "date_label:N",
                        title="交易日期",
                        sort=category_sort,
                        axis=alt.Axis(labelAngle=-45, labelOverlap=False),
                    ),
                    y=alt.Y("volume_wan_lot:Q", title="成交量（万手）"),
                    color=alt.Color("direction:N", scale=alt.Scale(domain=["上涨", "下跌"], range=["#d62728", "#1f8f4d"]), legend=None),
                    tooltip=[
                        alt.Tooltip("date_label:N", title="日期"),
                        alt.Tooltip("volume_wan_lot:Q", title="成交量（万手）", format=".2f"),
                    ],
                ).properties(height=180)
                st.altair_chart(
                    alt.vconcat(kline_chart, volume_chart, spacing=4).resolve_scale(x="shared"),
                    width="stretch",
                )
            else:
                st.altair_chart(kline_chart, width="stretch")
                st.caption("当前历史缓存缺少有效成交量数据。")
        else:
            trend_columns = [column for column in ["close", "ma5", "ma10", "ma20", "ma60"] if column in data.columns]
            st.caption("收盘价走势")
            st.line_chart(data.set_index("date")[trend_columns])


def render_tushare_technical_summary(data: pd.DataFrame) -> None:
    """Render technical summary from cached K-line rows."""
    latest = data.dropna(subset=["close"]).iloc[-1]
    close = latest.get("close")
    ma5 = latest.get("ma5")
    ma10 = latest.get("ma10")
    ma20 = latest.get("ma20")
    ma60 = latest.get("ma60")
    pct_5 = calc_period_return(data, 5)
    pct_20 = calc_period_return(data, 20)
    st.subheader("技术摘要")
    summary = [
        f"最新收盘价：{format_number(close)}",
        f"MA5：{format_number(ma5)}",
        f"MA10：{format_number(ma10)}",
        f"MA20：{format_number(ma20)}",
        f"MA60：{format_number(ma60)}",
        f"是否站上 MA5：{'是' if pd.notna(ma5) and close > ma5 else '否' if pd.notna(ma5) else '--'}",
        f"是否站上 MA20：{'是' if pd.notna(ma20) and close > ma20 else '否' if pd.notna(ma20) else '--'}",
        f"近 5 日涨跌幅：{format_change_percent(pct_5)}",
        f"近 20 日涨跌幅：{format_change_percent(pct_20)}",
    ]
    for item in summary:
        st.write(f"- {item}")


def calc_period_return(data: pd.DataFrame, period: int) -> float | None:
    """Calculate period return from cached close prices."""
    valid = data.dropna(subset=["close"])
    if len(valid) <= period:
        return None
    latest_close = valid["close"].iloc[-1]
    base_close = valid["close"].iloc[-period - 1]
    if pd.isna(base_close) or base_close == 0:
        return None
    return (latest_close / base_close - 1) * 100


def render_detail_header(row: pd.Series) -> None:
    """Render a compact quote-first header for daily stock review."""
    name = display_text(row.get("name"))
    code = str(row.get("code") or "").zfill(6)
    market_code = display_text(row.get("market_code"))
    board = display_text(row.get("board_type_display"))
    st.subheader(f"{name}  {code}")
    st.caption(f"{market_code} · {board}")

    metrics = [
        ("最新价", format_price(row.get("price")), "neutral"),
        ("涨跌幅", display_text(row.get("pct_chg_display")), _pct_tone(row.get("pct_chg"))),
        ("成交额", display_text(row.get("amount_display")), "neutral"),
        ("换手率", format_preferred_turnover(row), "neutral"),
        ("量比", format_preferred_volume_ratio(row), "neutral"),
        ("评分", format_score_display(row.get("score")), _score_tone(row.get("score"))),
    ]
    cols = st.columns(6)
    for index, (label, value, tone) in enumerate(metrics):
        render_status_card(cols[index], label, value, tone)

    render_detail_watchlist_action(row)
    render_strategy_tracking_action(row, "个股详情", "stock_detail")


def render_detail_state_cards(row: pd.Series) -> None:
    """Surface status without changing existing status or risk calculations."""
    consecutive = pd.to_numeric(row.get("consecutive_selection_days"), errors="coerce")
    if pd.isna(consecutive):
        consecutive = pd.to_numeric(row.get("consecutive_count"), errors="coerce")
    states: list[tuple[str, str, str]] = [
        ("评分等级", display_text(row.get("level_display")), "neutral"),
        ("数据状态", format_data_status(row), _data_status_tone(row)),
        ("位置风险", display_text(row.get("position_risk_level")), _risk_tone(row.get("position_risk_level"))),
        ("T+1风险", display_text(row.get("t1_risk_level")), _risk_tone(row.get("t1_risk_level"))),
    ]
    if pd.notna(consecutive):
        states.append(("连续入选", f"{consecutive:.0f}个交易日", "neutral"))

    total = pd.to_numeric(row.get("selection_days_total"), errors="coerce")
    if pd.notna(total):
        states.append(("累计入选", f"{total:.0f}个交易日", "neutral"))

    cols = st.columns(len(states))
    for index, (label, value, tone) in enumerate(states):
        render_status_card(cols[index], label, value, tone)

    v59_scores = [
        ("原始基础分", row.get("raw_base_score")),
        ("全市场百分位", row.get("base_percentile")),
        ("基础评分", row.get("base_score")),
        ("最终交易评分", row.get("final_trade_score")),
    ]
    if any(pd.notna(pd.to_numeric(value, errors="coerce")) for _, value in v59_scores):
        st.subheader("V5.9 当日评分")
        score_cols = st.columns(4)
        for index, (label, value) in enumerate(v59_scores):
            render_status_card(score_cols[index], label, format_score_display(value), _score_tone(value))

    trade_states = [
        ("风险等级", display_text(row.get("trade_risk_level")), _risk_tone(row.get("trade_risk_level"))),
        ("买入状态", display_text(row.get("buy_status_display")), "neutral"),
    ]
    available_trade_states = [item for item in trade_states if item[1] != "--"]
    if available_trade_states:
        trade_cols = st.columns(len(available_trade_states))
        for index, (label, value, tone) in enumerate(available_trade_states):
            render_status_card(trade_cols[index], label, value, tone)


def render_detail_reason_and_risk(row: pd.Series) -> None:
    """Keep selection context and risk context distinct for fast review."""
    reason = display_text(row.get("reason"))
    if reason == "--":
        reason = display_text(row.get("reference_reason"))
    risk = display_text(row.get("risk_summary"))
    if risk == "--":
        risk = display_text(row.get("reference_warning"))
    if risk == "--":
        risk = display_text(row.get("warning_message"))

    reason_col, risk_col = st.columns(2)
    with reason_col:
        st.subheader("为什么进入观察池")
        st.write(reason)
    with risk_col:
        st.subheader("风险提示")
        st.write(risk)


def render_detail_data_sources(row: pd.Series, status: dict[str, Any]) -> None:
    """Explain displayed values without exposing internal source enums."""
    score_source = display_text(row.get("detail_score_source"))
    if score_source != "--":
        st.caption(f"评分、风险和买入状态来源：{score_source}")
    latest_source = display_text(row.get("data_source"))
    update_time = display_text(row.get("quote_time"))
    if update_time == "--":
        update_time = display_text(row.get("scan_time"))
    if update_time == "--":
        update_time = display_text(status.get("scan_time"))
    metrics = [
        ("最新价来源", latest_source),
        ("换手率来源", format_turnover_source_display(row)),
        ("量比来源", format_volume_ratio_source_display(row)),
        ("实时数据更新时间", update_time),
        ("参考数据日期", display_text(row.get("reference_trade_date_display"))),
    ]
    cols = st.columns(len(metrics))
    for index, (label, value) in enumerate(metrics):
        render_status_card(cols[index], label, value, "neutral")


def render_detail_watchlist_action(row: pd.Series) -> None:
    """Provide a local watchlist action without any external account integration."""
    code = str(row.get("code") or "").zfill(6)
    name = display_text(row.get("name"))
    watchlist = load_user_watchlist()
    codes = set(watchlist.get("code", pd.Series(dtype="string")).astype(str).str.zfill(6))
    if code in codes:
        st.button("已在自选", key=f"detail_watchlist_exists_{code}", disabled=True)
        return

    if st.button("加入自选股", key=f"detail_add_watchlist_{code}"):
        added, message = add_user_watchlist_stock(code, name)
        if added:
            st.session_state["watchlist_feedback"] = f"已加入自选：{name}"
            st.rerun()
        _show_compact_feedback("该股票已在自选中。" if not added else message, tone="muted")


def render_strategy_tracking_feedback() -> None:
    """Render one persisted acknowledgement after a tracking-pool update."""
    feedback = st.session_state.pop("strategy_tracking_feedback", "")
    if isinstance(feedback, dict):
        message = str(feedback.get("message") or "")
        tone = str(feedback.get("tone") or "muted")
    else:
        message = str(feedback or "")
        tone = "ok"
    if message:
        _show_compact_feedback(message, tone=tone)


def render_strategy_tracking_action(
    row: pd.Series | None,
    source_page: str,
    context_key: str,
) -> None:
    """Save the exact displayed quote/score snapshot into the tracking pool."""
    if row is None:
        return
    code = str(row.get("code") or "").zfill(6)
    name = display_text(row.get("name"))
    tracking_pool = load_strategy_tracking_pool()
    tracked_codes = set(tracking_pool.get("code", pd.Series(dtype="string")).astype(str).str.zfill(6))
    if code in tracked_codes:
        st.button("已在策略跟踪池", key=f"strategy_tracking_exists_{context_key}_{code}", disabled=True)
        return

    if st.button("加入策略跟踪", key=f"add_strategy_tracking_{context_key}_{code}"):
        added, message = add_strategy_tracking_stock(row, source_page=source_page)
        if added:
            st.session_state["strategy_tracking_feedback"] = f"已加入策略跟踪：{name}"
            st.rerun()
        _show_compact_feedback(message, tone="muted")


def _pct_tone(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "gray"
    return "red" if number > 0 else "green" if number < 0 else "gray"


def _score_tone(value: Any) -> str:
    score = pd.to_numeric(value, errors="coerce")
    if pd.isna(score):
        return "gray"
    if score >= 50:
        return "red"
    if score < 30:
        return "gray"
    return "neutral"


def _risk_tone(value: Any) -> str:
    text = str(value or "")
    if "高" in text:
        return "risk"
    if "中" in text:
        return "neutral"
    if "低" in text:
        return "green"
    return "gray"


def _data_status_tone(row: pd.Series) -> str:
    status = format_data_status(row)
    if status == "完整实时":
        return "green"
    if status in {"部分实时", "日级参考"}:
        return "neutral"
    return "gray"


def render_kline_chart(row: pd.Series, history: pd.DataFrame) -> None:
    """Render K-line chart with MA lines."""
    st.subheader(f"K 线图：{row.get('name', '--')}（{row.get('code', '--')}）")
    data = history.copy()
    data["direction"] = data.apply(lambda item: "上涨" if item["close"] >= item["open"] else "下跌", axis=1)
    base = alt.Chart(data).encode(x=alt.X("date:T", title="日期"))
    wick = base.mark_rule().encode(y=alt.Y("low:Q", title="价格", scale=alt.Scale(zero=False)), y2="high:Q", color=alt.Color("direction:N", scale=alt.Scale(domain=["上涨", "下跌"], range=["#d62728", "#2ca02c"])))
    candle = base.mark_bar(size=7).encode(y=alt.Y("open:Q", scale=alt.Scale(zero=False)), y2="close:Q", color=alt.Color("direction:N", scale=alt.Scale(domain=["上涨", "下跌"], range=["#d62728", "#2ca02c"])))
    ma = data.melt(id_vars=["date"], value_vars=["ma5", "ma10", "ma20"], var_name="均线", value_name="value").dropna(subset=["value"])
    lines = alt.Chart(ma).mark_line(strokeWidth=1.6).encode(x="date:T", y=alt.Y("value:Q", scale=alt.Scale(zero=False)), color="均线:N")
    st.altair_chart((wick + candle + lines).properties(height=420), width="stretch")


def render_volume_chart(history: pd.DataFrame) -> None:
    """Render volume chart."""
    st.subheader("成交量")
    data = history.copy()
    bars = alt.Chart(data).mark_bar().encode(x=alt.X("date:T", title="日期"), y=alt.Y("volume:Q", title="成交量"))
    avg = data.melt(id_vars=["date"], value_vars=["avg_volume_5d", "avg_volume_20d"], var_name="均量", value_name="value").dropna(subset=["value"])
    lines = alt.Chart(avg).mark_line().encode(x="date:T", y="value:Q", color="均量:N")
    st.altair_chart((bars + lines).properties(height=260), width="stretch")


@st.cache_data(show_spinner=False, ttl=3600)
def load_history_for_detail(code: str, days: int = 60) -> pd.DataFrame:
    """Load historical K-line data for selected stock."""
    try:
        return AkshareSource({"retry_count": 1, "timeout": 8}).fetch_history(code, days=days)
    except Exception:
        return pd.DataFrame()


def display_frame(data: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DISPLAY_COLUMNS if column in data.columns]
    return clean_display_frame(data[columns].rename(columns=COLUMN_LABELS))


def display_active_frame(data: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ACTIVE_DISPLAY_COLUMNS if column in data.columns]
    return clean_display_frame(data[columns].rename(columns=COLUMN_LABELS))


def display_today_trade_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Format only the fields required for the intraday focus list."""
    columns = [
        column for column in TODAY_TRADE_DISPLAY_COLUMNS if column in data.columns
    ]
    return clean_display_frame(data[columns].rename(columns=COLUMN_LABELS))


def compact_table_column_config(data: pd.DataFrame) -> dict[str, Any]:
    """Set stable, content-appropriate widths for the stock-list columns."""
    widths = {
        "股票名称": 120,
        "股票代码": 82,
        "最新价": 78,
        "涨跌幅": 78,
        "成交额": 100,
        "换手率": 82,
        "量比": 70,
        "评分": 70,
        "原始基础分": 88,
        "全市场百分位": 100,
        "基础评分": 82,
        "交易质量评分": 96,
        "最终交易评分": 96,
        "数据完整性": 82,
        "数据完整状态": 100,
        "等级": 70,
        "连续入选": 92,
        "数据状态": 88,
        "位置风险": 80,
        "T+1 风险": 80,
        "操作状态": 88,
        "风险等级": 82,
        "买点状态": 82,
        "买入状态": 96,
    }
    numeric_labels = {"最新价", "涨跌幅", "成交额", "换手率", "量比", "评分", "原始基础分", "全市场百分位", "基础评分", "交易质量评分", "最终交易评分", "数据完整性"}
    centered_labels = {"股票代码", "等级", "连续入选", "数据状态", "数据完整状态", "位置风险", "T+1 风险", "操作状态", "风险等级", "买点状态", "买入状态"}
    return {
        label: st.column_config.TextColumn(
            label,
            width=width,
            alignment="right" if label in numeric_labels else "center" if label in centered_labels else "left",
            pinned=label in {"股票名称", "股票代码"},
        )
        for label, width in widths.items()
        if label in data.columns
    }


def render_compact_stock_dataframe(data: pd.DataFrame) -> None:
    """Render a centered stock list without stretching short columns on wide screens."""
    with st.container(horizontal_alignment="center"):
        st.dataframe(
            style_watch_table(data),
            column_config=compact_table_column_config(data),
            width="content",
            hide_index=True,
        )


def render_selected_reason_panel(row: pd.Series | None) -> None:
    """Show long-form context only after the user has selected one stock."""
    if row is None:
        st.caption("请选择一只股票查看触发原因、风险提示和数据来源。")
        return

    name = display_text(row.get("name"))
    code = display_text(row.get("code"))
    reason = display_text(row.get("reason"))
    if reason == "--":
        reason = display_text(row.get("reference_reason"))

    risk_summary = display_text(row.get("risk_summary"))
    if risk_summary == "--":
        risk_summary = display_text(row.get("reference_warning"))
    if risk_summary == "--":
        risk_summary = display_text(row.get("warning_message"))

    st.markdown(f"#### {name}（{code}）")
    reason_col, risk_col = st.columns(2)
    with reason_col:
        st.caption("触发原因")
        st.write(reason)
    with risk_col:
        st.caption("风险提示")
        st.write(risk_summary)

    source_col_1, source_col_2, source_col_3 = st.columns(3)
    with source_col_1:
        st.caption("实时换手率来源")
        st.write(format_turnover_source_display(row))
    with source_col_2:
        st.caption("实时量比来源")
        st.write(format_volume_ratio_source_display(row))
        st.caption("历史参考量比")
        st.write(display_text(row.get("volume_ratio_ref_display")))
    with source_col_3:
        st.caption("参考数据日期")
        st.write(display_text(row.get("reference_trade_date_display")))


def display_reference_frame(data: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in REFERENCE_DISPLAY_COLUMNS if column in data.columns]
    return clean_display_frame(data[columns].rename(columns=COLUMN_LABELS))


def style_watch_table(data: pd.DataFrame) -> Any:
    """Apply A-share watch colors to display tables."""
    styled = data.style
    if "股票名称" in data.columns:
        styled = apply_styler_map(styled, lambda _: "font-weight: 700;", subset=["股票名称"])
    if "涨跌幅" in data.columns:
        styled = apply_styler_map(styled, _pct_chg_style, subset=["涨跌幅"])
    for column in ["浮盈浮亏", "浮盈浮亏比例"]:
        if column in data.columns:
            styled = apply_styler_map(styled, _pct_chg_style, subset=[column])
    for column in ["评分", "基础评分", "最终交易评分"]:
        if column in data.columns:
            styled = apply_styler_map(styled, _score_badge_style, subset=[column])
    if "等级" in data.columns:
        styled = apply_styler_map(styled, _level_badge_style, subset=["等级"])
    if "操作状态" in data.columns:
        styled = apply_styler_map(styled, _action_state_style, subset=["操作状态"])
    for column in ["风险等级", "位置风险", "T+1 风险"]:
        if column in data.columns:
            styled = apply_styler_map(styled, _risk_badge_style, subset=[column])
    if "数据状态" in data.columns:
        styled = apply_styler_map(styled, _data_status_style, subset=["数据状态"])
    for column in ["换手率", "量比"]:
        if column in data.columns:
            styled = apply_styler_map(styled, _activity_value_style, subset=[column])
    return styled


def apply_styler_map(styled: Any, func: Any, subset: list[str]) -> Any:
    """Apply element-wise style across pandas versions."""
    if hasattr(styled, "map"):
        return styled.map(func, subset=subset)
    return styled.applymap(func, subset=subset)


def _pct_chg_style(value: Any) -> str:
    number = parse_percent_number(value)
    if number is None:
        return "color: #6b7280;"
    if number > 0:
        return "color: #d62728; font-weight: 700;"
    if number < 0:
        return "color: #1f8f4d; font-weight: 700;"
    return "color: #6b7280;"


def _score_badge_style(value: Any) -> str:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    number = pd.to_numeric(match.group(0), errors="coerce") if match else pd.NA
    if pd.isna(number):
        return "color: #6b7280;"
    if number >= 60:
        return "background-color: #fde8e8; color: #991b1b; font-weight: 700; border-radius: 4px;"
    if number >= 50:
        return "background-color: #fff1e8; color: #c2410c; font-weight: 700; border-radius: 4px;"
    if number >= 40:
        return "background-color: #fff7df; color: #a16207; font-weight: 700; border-radius: 4px;"
    if number >= 30:
        return "background-color: #eaf2ff; color: #1d4ed8; font-weight: 700; border-radius: 4px;"
    return "background-color: #f3f4f6; color: #6b7280;"


def _level_badge_style(value: Any) -> str:
    text = str(value or "")
    if "重点" in text:
        return "background-color: #fff1e8; color: #c2410c; font-weight: 700; border-radius: 4px;"
    if "加入观察" in text:
        return "background-color: #fff7df; color: #a16207; font-weight: 700; border-radius: 4px;"
    if "普通" in text:
        return "background-color: #eaf2ff; color: #1d4ed8; border-radius: 4px;"
    if "暂不" in text:
        return "background-color: #f3f4f6; color: #6b7280; border-radius: 4px;"
    return "color: #374151;"


def _risk_badge_style(value: Any) -> str:
    text = str(value or "")
    if "高" in text:
        return "background-color: #fde8e8; color: #b91c1c; font-weight: 700; border-radius: 4px;"
    if "中" in text:
        return "background-color: #fff7df; color: #a16207; font-weight: 700; border-radius: 4px;"
    if "低" in text:
        return "background-color: #f0fdf4; color: #166534; border-radius: 4px;"
    return "color: #6b7280;"


def _data_status_style(value: Any) -> str:
    text = str(value or "")
    if text == "完整实时":
        return "background-color: #f0fdf4; color: #166534; font-weight: 700; border-radius: 4px;"
    if text == "部分实时":
        return "background-color: #eaf2ff; color: #1d4ed8; border-radius: 4px;"
    if text == "日级参考":
        return "background-color: #eef2f7; color: #475569; border-radius: 4px;"
    return "background-color: #fff7df; color: #a16207; border-radius: 4px;"


def _action_state_style(value: Any) -> str:
    text = str(value or "")
    if "数据不足" in text or text in {"--", ""}:
        return "color: #9ca3af;"
    if "风险" in text or "不追高" in text:
        return "color: #92400e; font-weight: 700;"
    return "color: #374151;"


def _activity_value_style(value: Any) -> str:
    """Keep reference activity values visibly quieter than realtime values."""
    text = str(value or "")
    if "参考" in text or text == "--":
        return "color: #9ca3af;"
    return "color: #111827;"


def build_stock_select_options(data: pd.DataFrame) -> dict[str, str]:
    """Build compact stock options for cross-tab detail selection."""
    options: dict[str, str] = {}
    for _, row in data.iterrows():
        code = display_text(row.get("code"))
        if code == "--" or code == "000000":
            continue
        code = code.zfill(6)
        pct_chg = display_text(row.get("pct_chg_display"))
        if pct_chg == "--":
            pct_chg = format_percent(row.get("pct_chg"))
        score = pd.to_numeric(row.get("score"), errors="coerce")
        score_text = "--" if pd.isna(score) else f"{score:.0f}分"
        label = f"{display_text(row.get('name'))}（{code}）｜{pct_chg}｜{score_text}"
        options[label] = code
    return options


def render_stock_selection_actions(row: pd.Series | None, context_key: str) -> None:
    """Keep stock selection, detail viewing, and local watchlist actions together."""
    if row is None:
        return

    code = str(row.get("code") or "").zfill(6)
    name = display_text(row.get("name"))
    st.caption(f"当前选择：{name}（{code}）")

    feedback = st.session_state.pop("watchlist_feedback", "")
    if feedback:
        _show_compact_feedback(str(feedback), tone="ok")

    detail_col, watchlist_col = st.columns(2)
    if detail_col.button("查看个股详情", key=f"open_detail_{context_key}_{code}"):
        st.session_state["selected_stock_code"] = code
        st.session_state["detail_focus_requested"] = True
        _show_compact_feedback(f"已选择：{name}。请打开“个股详情”查看。", tone="muted")

    user_watchlist = load_user_watchlist()
    is_in_watchlist = code in set(user_watchlist.get("code", pd.Series(dtype="string")).astype(str).str.zfill(6))
    if is_in_watchlist:
        watchlist_col.button("已在自选", key=f"watchlist_exists_{context_key}_{code}", disabled=True)
        return

    if watchlist_col.button("加入自选股", key=f"add_watchlist_{context_key}_{code}"):
        added, message = add_user_watchlist_stock(code, name)
        if added:
            st.session_state["watchlist_feedback"] = f"已加入自选：{name}"
            st.rerun()
        _show_compact_feedback("该股票已在自选中。" if not added else message, tone="muted")


def _show_compact_feedback(message: str, tone: str = "muted") -> None:
    """Prefer a short toast, with the existing compact hint as fallback."""
    if hasattr(st, "toast"):
        st.toast(message)
        return
    render_small_hint(message, tone=tone)


def find_stock_row(data: pd.DataFrame, code: str) -> pd.Series | None:
    """Find a stock row by six-digit code."""
    if data.empty or not code:
        return None
    matched = data[data["code"].astype(str).str.zfill(6) == str(code).zfill(6)]
    if matched.empty:
        return None
    return matched.iloc[0]


def _render_records(records: list[dict[str, Any]]) -> None:
    if not records:
        st.info("当前公开源暂未获取到板块强度数据，后续可接入东方财富板块源或其它公开源。")
        return
    st.dataframe(clean_display_frame(pd.DataFrame(records)), width="stretch", hide_index=True)


def _format_aux_table(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    if "code" in output.columns:
        output["code"] = output["code"].astype(str).str.zfill(6)
    if "amount" in output.columns:
        output["amount_display"] = output["amount"].map(format_amount)
    if "turnover" in output.columns:
        output["turnover_display"] = output["turnover"].map(format_percent)
    return output


def clean_display_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Remove noisy empty values from display-only tables."""
    output = data.copy()
    output = output.replace({None: "", "None": "", "nan": "", "NaN": "", pd.NA: ""})
    for column in output.columns:
        output[column] = output[column].map(lambda value: "" if pd.isna(value) else value)
    return output


def format_user_watchlist_table(user_watchlist: pd.DataFrame, candidates: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Show local watchlist with latest known quote fields."""
    base = user_watchlist.copy()
    base["code"] = base["code"].astype(str).str.zfill(6)
    quote_source = build_quote_lookup_frame(candidates, snapshot)
    if not quote_source.empty:
        base = base.merge(quote_source, on="code", how="left", suffixes=("", "_quote"))
    if "name_quote" in base.columns:
        base["name"] = base["name"].where(base["name"].astype(str).str.strip() != "", base["name_quote"])
    for column in ["price", "pct_chg_display", "amount_display"]:
        if column not in base.columns:
            base[column] = "--"
    base["data_status"] = base["price"].apply(lambda value: "暂无行情" if not _usable(value) else "有行情")
    output = pd.DataFrame(
        {
            "股票代码": base["code"],
            "股票名称": base["name"].fillna("--"),
            "标签": base["tag"].fillna("--"),
            "备注": base["note"].fillna("--"),
            "加入时间": base["add_time"].fillna("--"),
            "最新价": base["price"].fillna("--"),
            "涨跌幅": base["pct_chg_display"].fillna("--"),
            "成交额": base["amount_display"].fillna("--"),
            "数据状态": base["data_status"],
        }
    )
    return clean_display_frame(output)


def format_strategy_tracking_table(tracking_pool: pd.DataFrame) -> pd.DataFrame:
    """Format persisted strategy-tracking fields without recalculating a score."""
    data = tracking_pool.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    holding_days = pd.to_numeric(data.get("holding_days", pd.Series(index=data.index, dtype="float64")), errors="coerce")
    entry_score = pd.to_numeric(data.get("entry_final_trade_score", pd.Series(index=data.index, dtype="float64")), errors="coerce")
    current_score = pd.to_numeric(data.get("current_final_trade_score", pd.Series(index=data.index, dtype="float64")), errors="coerce")
    score_change = current_score - entry_score
    output = pd.DataFrame(
        {
            "股票名称": data.get("name", pd.Series("--", index=data.index)).fillna("--"),
            "股票代码": data["code"],
            "来源页面": data.get("source_page", pd.Series("--", index=data.index)).fillna("--"),
            "策略类型": data.get("strategy_type", pd.Series("--", index=data.index)).fillna("--"),
            "加入日期": data.get("added_date", pd.Series("--", index=data.index)).fillna("--"),
            "加入价格": data.get("entry_price", pd.Series(index=data.index, dtype="float64")).map(format_price),
            "当前价格": data.get("current_price", pd.Series(index=data.index, dtype="float64")).map(format_price),
            "当前收益率": data.get("current_return_pct", pd.Series(index=data.index, dtype="float64")).map(format_change_percent),
            "持有天数": holding_days.map(lambda value: "--" if pd.isna(value) else f"{int(value)}天"),
            "最大收益率": data.get("highest_return_pct", pd.Series(index=data.index, dtype="float64")).map(format_change_percent),
            "最大回撤": data.get("max_drawdown_pct", pd.Series(index=data.index, dtype="float64")).map(format_change_percent),
            "加入时评分": entry_score.map(format_number),
            "当前评分": current_score.map(format_number),
            "当前评分变化": score_change.map(format_score_change),
            "结束日期": data.get("exit_date", pd.Series("--", index=data.index)).fillna("--"),
            "结束价格": data.get("exit_price", pd.Series(index=data.index, dtype="float64")).map(format_price),
            "结束收益率": data.get("profit_loss", pd.Series(index=data.index, dtype="float64")).map(format_change_percent),
            "结束原因": data.get("exit_reason", pd.Series("--", index=data.index)).fillna("--"),
            "当前状态": data.get("status", pd.Series("--", index=data.index)).fillna("--"),
            "数据状态": data.get("data_status", pd.Series("--", index=data.index)).fillna("--"),
            "更新时间": data.get("last_update_at", pd.Series("--", index=data.index)).fillna("--"),
        }
    )
    return clean_display_frame(output)


def has_real_holdings(holdings: pd.DataFrame) -> bool:
    """Return whether holdings contains anything other than placeholder rows."""
    if holdings.empty or "code" not in holdings.columns:
        return False
    codes = holdings["code"].astype(str).str.zfill(6)
    return bool((codes != "000000").any())


def format_user_holdings_table(user_holdings: pd.DataFrame, candidates: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Show manually entered holdings with latest known quote fields."""
    base = user_holdings.copy()
    base["code"] = base["code"].astype(str).str.zfill(6)
    quote_source = build_quote_lookup_frame(candidates, snapshot)
    if not quote_source.empty:
        base = base.merge(quote_source, on="code", how="left", suffixes=("", "_quote"))
    if "name_quote" in base.columns:
        base["name"] = base["name"].where(base["name"].astype(str).str.strip() != "", base["name_quote"])

    base["cost_price"] = pd.to_numeric(base.get("cost_price"), errors="coerce").fillna(0)
    base["shares"] = pd.to_numeric(base.get("shares"), errors="coerce").fillna(0)
    base["price_num"] = pd.to_numeric(base.get("price"), errors="coerce")
    base["market_value"] = base["price_num"] * base["shares"]
    base["floating_pnl"] = (base["price_num"] - base["cost_price"]) * base["shares"]
    cost_amount = base["cost_price"] * base["shares"]
    base["floating_pnl_ratio"] = base.apply(
        lambda row: (row["floating_pnl"] / (row["cost_price"] * row["shares"]) * 100)
        if row["cost_price"] > 0 and row["shares"] > 0 and pd.notna(row["floating_pnl"])
        else pd.NA,
        axis=1,
    )
    base["t1_status"] = base["buy_date"].map(format_t1_status)
    output = pd.DataFrame(
        {
            "股票代码": base["code"],
            "股票名称": base["name"].fillna("--"),
            "成本价": base["cost_price"].map(format_number),
            "持股数量": base["shares"].map(format_quantity),
            "买入日期": base["buy_date"].fillna("--"),
            "当前价": base["price"].fillna("--") if "price" in base.columns else "--",
            "当前市值": base["market_value"].map(format_money),
            "浮盈浮亏": base["floating_pnl"].map(format_signed_money),
            "浮盈浮亏比例": base["floating_pnl_ratio"].map(format_change_percent),
            "T+1 状态": base["t1_status"],
            "备注": base["note"].fillna("--"),
        }
    )
    return clean_display_frame(output)


def format_holdings_table(holdings: pd.DataFrame) -> pd.DataFrame:
    """Format configured holdings with Chinese columns."""
    output = _format_aux_table(holdings)
    if "code" in output.columns:
        output = output[output["code"].astype(str).str.zfill(6) != "000000"]
    columns = [column for column in HOLDING_COLUMN_LABELS if column in output.columns]
    return clean_display_frame(output[columns].rename(columns=HOLDING_COLUMN_LABELS))


def build_quote_lookup_frame(candidates: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Build latest quote lookup from already-loaded dashboard data."""
    frames = []
    for data in [candidates, snapshot]:
        if data is None or data.empty or "code" not in data.columns:
            continue
        frame = data.copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if "pct_chg_display" not in frame.columns and "pct_chg" in frame.columns:
            frame["pct_chg_display"] = frame["pct_chg"].map(format_change_percent)
        if "amount_display" not in frame.columns and "amount" in frame.columns:
            frame["amount_display"] = frame["amount"].map(format_amount)
        for column in ["name", "price", "pct_chg_display", "amount_display"]:
            if column not in frame.columns:
                frame[column] = pd.NA
        frames.append(frame[["code", "name", "price", "pct_chg_display", "amount_display"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"], keep="first")


def format_strategy_names(value: Any) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "--"
    names = []
    for item in re.split(r"[,/，、\s]+", str(value)):
        key = item.strip()
        if key:
            names.append(STRATEGY_LABELS.get(key, key))
    return " / ".join(names) if names else "--"


def display_text(value: Any) -> str:
    """Render scalar text safely for optional dashboard fields."""
    if value is None:
        return "--"
    try:
        if bool(pd.isna(value)):
            return "--"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "--" if text in {"", "nan", "None", "<NA>"} else text


def format_level(value: Any) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "--"
    return LEVEL_LABELS.get(str(value).strip(), str(value).strip())


def format_score_display(value: Any) -> str:
    """Add a presentation-only attention label to an unchanged score."""
    score = pd.to_numeric(value, errors="coerce")
    if pd.isna(score):
        return "--"
    if score >= 60:
        label = "高关注"
    elif score >= 50:
        label = "重点观察"
    elif score >= 40:
        label = "关注"
    elif score >= 30:
        label = "观察"
    else:
        label = "弱关注"
    number = f"{score:.0f}" if float(score).is_integer() else f"{score:.1f}"
    return f"{number}  {label}"


def format_amount(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or number <= 0:
        return "--"
    if number >= 100_000_000:
        return f"{number / 100_000_000:.2f} 亿"
    if number >= 10_000:
        return f"{number / 10_000:.2f} 万"
    return f"{number:.0f}"


def format_price(value: Any) -> str:
    """Format a quote compactly without changing its underlying value."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def format_money(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return format_amount(number)


def format_signed_money(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    prefix = "+" if number > 0 else ""
    return f"{prefix}{format_amount(abs(number)) if number != 0 else '0'}" if number >= 0 else f"-{format_amount(abs(number))}"


def format_percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.2f}%"


def format_change_percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    if number > 0:
        return f"↑ +{number:.2f}%"
    if number < 0:
        return f"↓ {number:.2f}%"
    return "0.00%"


def parse_percent_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text == "--":
        return None
    text = text.replace("↑", "").replace("↓", "").replace("%", "").replace("+", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.2f}"


def format_plain_percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "--" if pd.isna(number) else f"{number:.1f}%"


def format_trading_days(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "--" if pd.isna(number) else f"{number:.1f}个交易日"


def format_score_change(value: Any) -> str:
    """Format a score delta without reinterpreting its trading meaning."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"+{number:.2f}" if number > 0 else f"{number:.2f}"


def format_quantity(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.0f}"


def format_t1_status(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "买入日期未填写，仅作提示"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return "买入日期格式无效，仅作提示"
    today = pd.Timestamp.today().normalize()
    if parsed.normalize() == today:
        return "今日买入，T+1 不可卖"
    if parsed.normalize() < today:
        return "可卖"
    return "买入日期晚于今天，请检查"


def format_reference_percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.2f}%（参考）"


def format_preferred_turnover(row: pd.Series) -> str:
    """Display only the persisted same-day realtime turnover field."""
    realtime = pd.to_numeric(row.get("realtime_turnover_value"), errors="coerce")
    if pd.notna(realtime) and 0 <= realtime <= 100:
        return f"{realtime:.2f}%"
    return "--"


def format_preferred_volume_ratio(row: pd.Series) -> str:
    """Show a validated realtime ratio before the Tushare daily reference."""
    realtime = pd.to_numeric(row.get("realtime_volume_ratio_value"), errors="coerce")
    if pd.notna(realtime):
        return f"{realtime:.2f}"
    return format_reference_number(row.get("volume_ratio_ref"))


def format_turnover_source_display(row: pd.Series) -> str:
    """Translate source names without exposing internal provider enums."""
    if pd.notna(pd.to_numeric(row.get("realtime_turnover_value"), errors="coerce")):
        source = str(row.get("realtime_turnover_source") or "")
        if "eastmoney" in source.lower():
            return "东方财富实时"
        if "sina" in source.lower():
            return "自主实时计算"
        return "实时行情"
    return "--"


def format_volume_ratio_source_display(row: pd.Series) -> str:
    """Translate realtime/reference volume-ratio provenance for the UI."""
    if pd.notna(pd.to_numeric(row.get("realtime_volume_ratio_value"), errors="coerce")):
        return "实时计算"
    if pd.notna(pd.to_numeric(row.get("volume_ratio_ref"), errors="coerce")):
        return "Tushare日级参考"
    return "--"


def format_data_status(row: pd.Series) -> str:
    """Summarize whether the two activity indicators are realtime or fallback."""
    quality_level = _normalize_data_quality_level(row.get("data_quality_level"), default="")
    if quality_level in {"A", "B", "C"}:
        return format_data_quality_level(quality_level)
    realtime_turnover = pd.notna(pd.to_numeric(row.get("realtime_turnover_value"), errors="coerce"))
    realtime_ratio = pd.notna(pd.to_numeric(row.get("realtime_volume_ratio_value"), errors="coerce"))
    reference_ratio = pd.notna(pd.to_numeric(row.get("volume_ratio_ref"), errors="coerce"))
    if realtime_turnover and realtime_ratio:
        return "完整实时"
    if realtime_turnover or realtime_ratio:
        return "部分实时"
    if reference_ratio:
        return "日级参考"
    return "数据不足"


def format_data_quality_level(value: Any) -> str:
    """Format the final-row completeness classification for display."""
    level = _normalize_data_quality_level(value)
    return {
        "A": "A级（接口实时字段）",
        "B": "B级（计算字段）",
        "C": "C级（关键字段缺失）",
    }.get(level, "C级（关键字段缺失）")


def _normalize_data_quality_level(value: Any, default: str = "C") -> str:
    """Normalize nullable CSV values without treating a missing level as A/B."""
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        return default
    text = str(value).strip().upper()
    return text if text in {"A", "B", "C"} else default


def _is_blank_value(value: Any) -> bool:
    """Return whether a nullable scalar lacks a usable display value."""
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        return True
    return str(value).strip() in {"", "--", "nan", "None", "<NA>"}


def _truthy_value(value: Any) -> bool:
    """Interpret persisted boolean values without treating missing values as true."""
    if _is_blank_value(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}
    return bool(value)


def format_reference_number(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "--"
    return f"{number:.2f}（参考）"


def format_reference_date(value: Any) -> str:
    """Format the Tushare cache date without implying intraday freshness."""
    text = str(value or "").strip()
    if text in {"", "nan", "None", "<NA>"}:
        return "--"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}（参考）"
    return f"{text}（参考）"


def format_tushare_market_value(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or number <= 0:
        return "--"
    return f"{number / 10000:.2f} 亿"


def _usable(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() not in {"", "nan"}


def _candidate_columns() -> list[str]:
    return list(dict.fromkeys([*DISPLAY_COLUMNS, "source_type", "strategy_names", "level", "turnover", "realtime_turnover_value", "realtime_turnover_source", "amount", "volume_ratio", "score_detail", "turnover_summary", "sector_summary", "action_summary", "t1_risk_summary"]))


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=_candidate_columns())


if __name__ == "__main__":
    main()
