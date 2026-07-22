"""Command entry for quant_stock_watch."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.action_state import enrich_action_state
from core.announcement_risk import enrich_announcement_risk
from core.filters import filter_strategy_eligible
from core.indicators import enrich_indicators_from_local_daily_cache
from core.market_environment import (
    analyze_market_environment,
    apply_market_level_cap,
    attach_market_score_fields,
    build_realtime_market_score_fields,
)
from core.money_flow_strength import enrich_money_strength
from core.position_risk import enrich_position_risk
from core.reference_candidates import REFERENCE_WARNING, build_reference_candidates
from core.risk_check import assert_no_trading_capability, check_risks
from core.scoring import score_candidates
from core.sector_strength import (
    build_realtime_sector_ranks,
    build_sector_overview,
    enrich_candidates_with_sector,
    normalize_sector_frame,
)
from core.stock_utils import add_board_columns, board_counts
from core.strategies import apply_strategy_tags
from core.t1_risk import enrich_t1_risk
from core.theme_heat import enrich_theme_heat
from core.tracking import enrich_tracking
from core.trading_quality import (
    calculate_final_trade_score,
    evaluate_trading_quality,
    score_data_quality,
)
from core.universe import build_universe
from core.watchlist import build_holding_quotes, build_watchlist_quotes, load_watchlist
from data_sources.akshare_source import AkshareSource, format_amount, format_turnover
from data_sources.calculated_turnover_source import apply_realtime_turnover_priority
from data_sources.calculated_volume_ratio_source import apply_realtime_volume_ratio_priority
from data_sources.merged_reference_source import merge_sina_with_tushare_cache
from storage.database import CandidateDatabase
from storage.strategy_tracking_store import update_strategy_tracking_pool


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
WATCHLIST_PATH = PROJECT_ROOT / "config" / "watchlist.yaml"
LATEST_CANDIDATES_PATH = DATA_DIR / "latest_candidates.csv"
LATEST_MARKET_STATUS_PATH = DATA_DIR / "latest_market_status.json"
LATEST_WATCHLIST_PATH = DATA_DIR / "latest_watchlist.csv"
LATEST_HOLDINGS_PATH = DATA_DIR / "latest_holdings.csv"
OUTPUT_DIR = DATA_DIR / "output"
REFERENCE_CANDIDATES_PATH = OUTPUT_DIR / "reference_candidates.csv"
V58_TOP50_PATH = OUTPUT_DIR / "v5_8_top50.csv"
V58_TOP50_REQUIRED_COLUMNS = [
    "code",
    "name",
    "price",
    "pct_chg",
    "turnover",
    "realtime_turnover_value",
    "realtime_turnover_source",
    "volume_ratio",
    "base_score",
    "trading_quality_score",
    "final_trade_score",
    "data_quality_score",
    "data_quality_status",
    "v58_top50_eligible",
]
TUSHARE_CACHE_STATUS_PATH = DATA_DIR / "cache" / "tushare" / "tushare_cache_status.json"
LOCAL_DAILY_HISTORY_DIR = DATA_DIR / "history" / "daily"
LOCAL_INDUSTRY_HISTORY_DIR = DATA_DIR / "history" / "industry"
LOCAL_MARKET_HISTORY_DIR = DATA_DIR / "history" / "market"

TOP_N = 20
ACTIVE_WATCHLIST_SIZE = 30
HISTORY_CALC_LIMIT = 300
SOURCE_TYPE_STRATEGY = "strategy_candidate"
SOURCE_TYPE_ACTIVE = "active_watchlist"

SOURCE_TYPE_DISPLAY = {
    SOURCE_TYPE_STRATEGY: "策略候选股",
    SOURCE_TYPE_ACTIVE: "活跃观察池",
}

STRATEGY_DISPLAY = {
    "trend_bullish": "趋势多头",
    "volume_breakout": "放量突破",
    "pullback_low_volume": "缩量回踩",
    "active_watchlist": "活跃观察池",
}

LEVEL_DISPLAY = {
    "key_watch": "重点关注",
    "watch": "加入观察",
    "normal_watch": "普通观察",
    "ignore": "暂不关注",
    "重点关注": "重点关注",
    "加入观察": "加入观察",
    "普通观察": "普通观察",
    "暂不关注": "暂不关注",
}

RESULT_COLUMNS = [
    "source_type",
    "source_type_display",
    "board_type",
    "board_type_display",
    "code",
    "market_code",
    "name",
    "price",
    "pct_chg",
    "amount",
    "amount_display",
    "volume",
    "turnover",
    "turnover_display",
    "turnover_source",
    "is_realtime_turnover",
    "realtime_turnover_value",
    "realtime_turnover_source",
    "float_share",
    "turnover_rate_ref",
    "volume_ratio_ref",
    "total_mv",
    "circ_mv",
    "reference_trade_date",
    "turnover_level",
    "turnover_level_display",
    "turnover_score",
    "turnover_summary",
    "volume_ratio",
    "volume_ratio_source",
    "is_realtime_volume_ratio",
    "amount_ratio_5d",
    "amount_ratio_20d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "ma5",
    "ma10",
    "ma20",
    "avg_volume_5d",
    "avg_volume_10d",
    "avg_amount_5d",
    "high_20d",
    "prior_20d_high",
    "distance_to_20d_high_pct",
    "volume_breakout",
    "recent_low_volume_pullback",
    "limit_up_count_20d",
    "consecutive_limit_up_days",
    "money_strength_level",
    "money_strength_summary",
    "sector_name",
    "sector_pct_chg",
    "sector_rank",
    "sector_strength_level",
    "sector_summary",
    "sh_close",
    "sh_pct_chg",
    "sh_ma5",
    "sh_ma10",
    "up_count",
    "down_count",
    "market_amount",
    "market_amount_ma20",
    "sh_ma20",
    "up_ratio",
    "limit_up_count",
    "market_environment",
    "market_score_input_source",
    "industry_rank",
    "industry_count",
    "industry_up_ratio",
    "industry_limit_up_count",
    "theme_tags",
    "theme_heat_score",
    "theme_heat_level",
    "theme_heat_summary",
    "strategy_names",
    "strategy_names_display",
    "raw_base_score",
    "base_percentile",
    "score",
    "base_score",
    "quality_adjustment",
    "trading_quality_score",
    "final_trade_score",
    "quality_market_score",
    "quality_trend_stage_score",
    "quality_trend_stage_risk",
    "quality_buy_point_score",
    "quality_sector_linkage_score",
    "turnover_quality_score",
    "turnover_quality_eligible",
    "data_quality_score",
    "data_quality_status",
    "data_quality_missing_critical",
    "data_quality_missing_support",
    "top50_data_eligible",
    "v58_quality_evaluated",
    "v58_top50_eligible",
    "v58_quality_detail",
    "v58_admission_reason",
    "level",
    "level_display",
    "action_state",
    "action_state_display",
    "action_summary",
    "position_risk_level",
    "position_risk_summary",
    "chase_risk_level",
    "chase_risk_summary",
    "t1_risk_level",
    "t1_risk_summary",
    "first_seen_today",
    "consecutive_count",
    "selection_trade_date",
    "selection_days_total",
    "consecutive_selection_days",
    "score_trend",
    "level_trend",
    "upgraded_today",
    "downgraded_today",
    "tracking_summary",
    "announcement_risk_level",
    "announcement_risk_keywords",
    "announcement_risk_summary",
    "latest_announcement_titles",
    "reason",
    "risk_summary",
    "data_source",
    "data_source_level",
    "is_full_featured",
    "allow_strategy_candidate",
    "data_quality_level",
    "data_quality_label",
    "data_quality_note",
    "score_detail",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "scan_interval_seconds": 180,
        "history_days": 60,
        "history_calc_limit": HISTORY_CALC_LIMIT,
        "active_watchlist_size": ACTIVE_WATCHLIST_SIZE,
    },
    "data_source": {
        "current": "akshare",
        "primary": "akshare",
        "retry_count": 2,
        "request_interval_seconds": 0.5,
        "local_broker_enabled": False,
        "local_broker_path": "data/local_broker/cmbc_quote_sample.csv",
    },
    "universe": {
        "exclude_st": True,
        "exclude_delisting_risk": True,
        "exclude_suspended": True,
        "min_price": 3.0,
        "min_amount": 100_000_000,
        "max_pct_chg": 9.5,
        "min_pct_chg": -9.5,
    },
    "filters": {
        "min_price": 3.0,
        "min_amount": 100_000_000,
        "max_pct_chg": 9.5,
        "min_pct_chg": -9.5,
    },
    "risk_check": {
        "min_price": 3.0,
        "min_amount": 100_000_000,
        "high_pct_chg": 8.0,
        "low_pct_chg": -8.0,
        "high_turnover": 20.0,
    },
    "risk_boundary": {
        "allow_trading": False,
    },
    "feature_flags": {
        "enable_kline_detail": True,
        "enable_sector_strength": True,
        "enable_market_environment": True,
        "enable_turnover_scoring": True,
        "enable_position_risk": True,
        "enable_tracking": True,
        "enable_money_strength": True,
        "enable_announcement_risk": False,
        "enable_theme_heat": True,
        "enable_action_state": True,
        "enable_t1_risk": True,
        "enable_watchlist": True,
    },
    "theme_keywords": [
        "AI", "算力", "光模块", "CPO", "半导体", "芯片", "机器人", "商业航天",
        "低空经济", "军工", "证券", "电力", "固态电池", "新能源", "数据中心", "液冷", "国产替代",
    ],
    "board_filter": {
        "include": ["sh_main", "sz_main", "chi_next", "star_market", "bj"],
        "exclude": [],
    },
}


def setup_logging() -> None:
    """Write logs to logs/app.log and the console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read YAML config and merge it with stable defaults."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing PyYAML. Run `pip install -r requirements.txt`.") from exc

    if not path.exists():
        logging.warning("Config file not found: %s. Defaults will be used.", path)
        return DEFAULT_CONFIG.copy()

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a YAML mapping.")

    return _merge_dicts(DEFAULT_CONFIG, loaded)


def run_once(config: dict[str, Any]) -> pd.DataFrame:
    """Run one full scan cycle and return ranked watch results."""
    logger = logging.getLogger("quant_stock_watch")
    started_at = datetime.now()

    assert_no_trading_capability(config)
    _print_header(started_at)

    data_source = AkshareSource(config.get("data_source", {}))
    spot_quotes = pd.DataFrame()
    active_pool = pd.DataFrame()
    enriched = pd.DataFrame()
    strategy_candidates = pd.DataFrame()
    active_watchlist = pd.DataFrame()
    reference_candidates = pd.DataFrame()
    ranked = pd.DataFrame(columns=RESULT_COLUMNS)
    v58_top50 = pd.DataFrame()
    v58_scored = pd.DataFrame()
    status_errors: list[str] = []
    status_warnings: list[str] = []
    sector_overview: dict[str, Any] = {"industry_top10": [], "concept_top10": [], "warnings": []}
    stock_sector_ranks = pd.DataFrame()
    market_environment: dict[str, Any] = {}
    market_score_fields: dict[str, Any] = {}
    watchlist_rows = pd.DataFrame()
    holding_rows = pd.DataFrame()
    reference_candidate_count = 0
    reference_candidate_enabled = False
    reference_candidate_source = "新浪实时行情 + Tushare 本地缓存"
    reference_candidate_warning = REFERENCE_WARNING

    try:
        logger.info("Fetching full-market realtime quotes.")
        spot_quotes = data_source.fetch_spot()
        source_meta = data_source.last_spot_meta or {}
        if should_try_local_broker_source(source_meta, config):
            local_source = LocalBrokerSource(resolve_local_broker_path(config))
            local_quotes = local_source.fetch_spot()
            local_meta = local_source.last_spot_meta or {}
            if not local_quotes.empty:
                spot_quotes = local_quotes
                data_source = local_source
                source_meta = local_meta
                status_warnings.append("东方财富实时行情不可用，已切换到招商证券本地只读行情。")
                print("已切换到招商证券本地只读行情。")
            else:
                status_warnings.extend(local_meta.get("warnings", []))
                status_errors.extend(local_meta.get("errors", []))
                print("招商证券本地行情不可用，继续使用原行情源结果。")

        spot_quotes, turnover_stats = apply_realtime_turnover_priority(
            spot_quotes,
            str(source_meta.get("data_source") or ""),
        )
        spot_quotes = attach_realtime_turnover_display_field(spot_quotes)
        source_meta.update(turnover_stats)
        spot_quotes, volume_ratio_stats = apply_realtime_volume_ratio_priority(
            spot_quotes,
            str(source_meta.get("data_source") or ""),
        )
        source_meta.update(volume_ratio_stats)
        spot_quotes, final_data_quality_meta = assess_final_score_data_quality(spot_quotes)
        source_meta.update(final_data_quality_meta)
        if source_meta["data_quality_level"] in {"A", "B"}:
            # Keep raw-source diagnostics available separately, while making
            # the current status reflect the final fields actually supplied
            # to scoring rather than a superseded API-schema omission.
            source_meta["raw_missing_fields"] = list(source_meta.get("missing_fields", []))
            source_meta["missing_fields"] = []
            source_meta["warnings"] = [
                warning
                for warning in source_meta.get("warnings", [])
                if "缺少换手率" not in str(warning) and "仅生成活跃观察池" not in str(warning)
            ]
        source_meta["allow_strategy_candidate"] = is_official_strategy_source_ready(spot_quotes, source_meta)
        source_meta["is_full_featured"] = bool(source_meta["allow_strategy_candidate"])
        data_source.last_spot_meta = source_meta
        print(f"全市场实时行情数量: {len(spot_quotes)}")
        print(f"当前数据源: {source_meta.get('data_source', '--')}")
        print(f"数据源等级: {source_meta.get('data_source_level', '--')}")

        if spot_quotes.empty:
            status_errors.append("所有行情源均失败或返回空数据")
            save_candidates(ranked)
            save_reference_candidates(reference_candidates)
            save_v58_top50(v58_top50)
            save_market_status(
                build_market_status(
                    started_at=started_at,
                    data_source=data_source,
                    spot_count=0,
                    active_count=0,
                    enriched_count=0,
                    strategy_candidate_count=0,
                    active_watchlist_count=0,
                    board_count_map={},
                    sector_overview=sector_overview,
                    market_environment=market_environment,
                    watchlist_count=0,
                    holdings_count=0,
                    reference_candidate_count=reference_candidate_count,
                    reference_candidate_enabled=reference_candidate_enabled,
                    reference_candidate_source=reference_candidate_source,
                    reference_candidate_warning=reference_candidate_warning,
                    errors=status_errors,
                    warnings=status_warnings,
                )
            )
            print_candidates(ranked.head(TOP_N))
            return ranked

        if feature_enabled(config, "enable_sector_strength"):
            sector_overview = fetch_sector_overview(data_source)
            status_warnings.extend(sector_overview.get("warnings", []))
            stock_sector_ranks = build_realtime_sector_ranks(
                spot_quotes,
                LOCAL_INDUSTRY_HISTORY_DIR,
            )
            if stock_sector_ranks.empty:
                status_warnings.append("本地行业映射不可用，实时板块排名将标记为缺失。")

        if feature_enabled(config, "enable_market_environment"):
            index_quotes = data_source.fetch_main_indices()
            market_environment = analyze_market_environment(spot_quotes, index_quotes)
            market_score_fields = build_realtime_market_score_fields(
                market_environment,
                index_quotes,
                LOCAL_MARKET_HISTORY_DIR,
            )
            status_warnings.extend(market_environment.get("warnings", []))

        if feature_enabled(config, "enable_watchlist"):
            watch_config = load_watchlist(WATCHLIST_PATH)
            watchlist_rows = build_watchlist_quotes(spot_quotes, watch_config.get("watchlist", []))
            holding_rows = build_holding_quotes(spot_quotes, watch_config.get("holdings", []))

        # V5.9 percentile calibration must use the complete same-day market
        # cross section.  Keep this universe separate from ``active_pool`` so
        # the existing board/universe filters continue to govern only the
        # established strategy and active-observation pools.
        full_market_pool = add_board_columns(spot_quotes)
        active_pool = apply_board_filter(add_board_columns(build_universe(spot_quotes, config)), config)
        print(f"过滤后活跃股票数量: {len(active_pool)}")

        reference_candidates = build_and_save_reference_candidates(spot_quotes, config)
        reference_candidate_count = len(reference_candidates)
        reference_candidate_enabled = True
        print(f"参考候选股数量: {reference_candidate_count}")

        allow_strategy = is_official_strategy_source_ready(active_pool, source_meta)
        if allow_strategy:
            strategy_pool = filter_strategy_eligible(active_pool, config)
            history_pool = _limit_history_pool(strategy_pool, config)
            print(f"历史指标计划计算股票数量: {len(history_pool)}")

            history_days = _history_days(config)
            enriched = enrich_indicators_from_local_daily_cache(
                realtime_quotes=history_pool,
                daily_dir=LOCAL_DAILY_HISTORY_DIR,
                days=history_days,
            )
            print(f"完成历史指标计算数量: {len(enriched)}")

            strategy_tagged = apply_strategy_tags(enriched, config)
            if "triggered" in strategy_tagged.columns:
                strategy_candidates = strategy_tagged[strategy_tagged["triggered"] == True].copy()
            else:
                strategy_candidates = strategy_tagged.head(0).copy()
            strategy_candidates["source_type"] = SOURCE_TYPE_STRATEGY
            print(f"策略触发候选数量: {len(strategy_candidates)}")
        else:
            warning = "当前数据源不是东方财富 push2 实时完整数据，或缺少实时换手率/量比/价格/成交额，本轮不生成正式策略候选股。"
            status_warnings.append(warning)
            print(warning)
            logger.warning(warning)

        active_watchlist = build_active_watchlist(active_pool, config)
        active_watchlist = enrich_active_watchlist_with_reference_fields(
            active_watchlist,
            spot_quotes,
            source_meta,
        )
        # Even when the spot source is downgraded and the formal strategy pool
        # is intentionally disabled, active-watchlist rows still go through
        # score_candidates().  Give them the same local, read-only history
        # inputs as formal candidates so score_stock() never sees a different
        # field shape solely because of the source tier.
        if not active_watchlist.empty:
            active_watchlist = enrich_indicators_from_local_daily_cache(
                realtime_quotes=active_watchlist,
                daily_dir=LOCAL_DAILY_HISTORY_DIR,
                days=_history_days(config),
            )
        print(f"活跃股票观察池数量: {len(active_watchlist)}")

        combined = pd.concat([strategy_candidates, active_watchlist], ignore_index=True)
        combined = enrich_optional_fields(
            combined,
            config,
            sector_overview,
            market_environment,
            stock_sector_ranks=stock_sector_ranks,
        )
        combined = attach_market_score_fields(combined, market_score_fields)
        risk_checked = check_risks(combined, config)
        print_score_input_checks(risk_checked)
        ranked = risk_checked.copy()

        # V5.9 calibrates base scores only after the complete current market
        # has received raw scores.  Existing strategy and active-observation
        # pools are left unchanged; their rows simply receive the matching
        # full-market raw score, percentile and calibrated base score.  The
        # existing V5.8 quality layer remains limited to that Top100.
        try:
            v58_scored, v58_top50 = build_v58_realtime_scores(
                full_market_pool=full_market_pool,
                config=config,
                sector_overview=sector_overview,
                stock_sector_ranks=stock_sector_ranks,
                market_environment=market_environment,
                market_score_fields=market_score_fields,
                source_meta=source_meta,
            )
            ranked = attach_v58_fields_to_candidates(ranked, v58_scored)
            print(
                "V5.9 全市场评分=%s，百分位基础分Top100=%s，交易质量Top50=%s"
                % (
                    len(v58_scored),
                    int(v58_scored.get("v58_quality_evaluated", pd.Series(dtype="bool")).sum()),
                    len(v58_top50),
                )
            )
        except Exception as exc:
            status_warnings.append(f"V5.9 全市场评分未完成：{type(exc).__name__}: {exc}")
            logger.exception("V5.9 full-market score pipeline failed.")
            # Keep the legacy pools usable if the independent full-market
            # processing fails.  This fallback is intentionally not used in
            # the normal V5.9 path because its percentile is only local.
            ranked = score_candidates(risk_checked, config)

        # Strategy tracking is a separate local observation record.  It uses
        # this scan's existing full-market score frame (or the quote snapshot
        # when V5.9 generation failed) and never changes scoring or selection.
        tracking_snapshot = v58_scored if not v58_scored.empty else spot_quotes
        tracking_stats = update_strategy_tracking_pool(tracking_snapshot, started_at)
        if tracking_stats["tracked"]:
            logger.info(
                "Strategy tracking updated. tracked=%s updated=%s unmatched=%s price_missing=%s",
                tracking_stats["tracked"],
                tracking_stats["updated"],
                tracking_stats["unmatched"],
                tracking_stats["price_missing"],
            )

        ranked = enrich_display_fields(ranked, source_meta)

        ranked = post_score_enrichment(
            ranked,
            config,
            market_environment,
            selection_trade_date=started_at.date().isoformat(),
        )
        ranked = ensure_result_columns(ranked)

        status = build_market_status(
            started_at=started_at,
            data_source=data_source,
            spot_count=len(spot_quotes),
            active_count=len(active_pool),
            enriched_count=len(enriched),
            strategy_candidate_count=len(strategy_candidates),
            active_watchlist_count=len(active_watchlist),
            board_count_map=board_counts(active_pool),
            sector_overview=sector_overview,
            market_environment=market_environment,
            watchlist_count=len(watchlist_rows),
            holdings_count=len(holding_rows),
            reference_candidate_count=reference_candidate_count,
            reference_candidate_enabled=reference_candidate_enabled,
            reference_candidate_source=reference_candidate_source,
            reference_candidate_warning=reference_candidate_warning,
            v58_full_market_count=len(v58_scored),
            v58_top100_count=int(v58_scored.get("v58_quality_evaluated", pd.Series(dtype="bool")).sum()),
            v58_top50_count=len(v58_top50),
            errors=status_errors,
            warnings=status_warnings,
        )
        save_candidates(ranked)
        save_reference_candidates(reference_candidates)
        save_v58_top50(v58_top50)
        save_auxiliary_csv(watchlist_rows, LATEST_WATCHLIST_PATH, "自选股")
        save_auxiliary_csv(holding_rows, LATEST_HOLDINGS_PATH, "持仓股")
        save_market_status(status)
        CandidateDatabase().save_candidates(ranked, started_at)
        print_candidates(ranked.head(TOP_N))

        logger.info(
            "Scan finished. spot=%s active=%s enriched=%s strategy=%s active_watchlist=%s saved=%s",
            len(spot_quotes),
            len(active_pool),
            len(enriched),
            len(strategy_candidates),
            len(active_watchlist),
            LATEST_CANDIDATES_PATH,
        )
        return ranked
    except Exception as exc:
        logger.exception("Scan cycle failed.")
        status_errors.append(f"{type(exc).__name__}: {exc}")
        save_candidates(ranked)
        save_v58_top50(v58_top50)
        save_market_status(
            build_market_status(
                started_at=started_at,
                data_source=data_source,
                spot_count=len(spot_quotes),
                active_count=len(active_pool),
                enriched_count=len(enriched),
                strategy_candidate_count=len(strategy_candidates),
                active_watchlist_count=len(active_watchlist),
                board_count_map=board_counts(active_pool),
                sector_overview=sector_overview,
                market_environment=market_environment,
                watchlist_count=len(watchlist_rows),
                holdings_count=len(holding_rows),
                reference_candidate_count=reference_candidate_count,
                reference_candidate_enabled=reference_candidate_enabled,
                reference_candidate_source=reference_candidate_source,
                reference_candidate_warning=reference_candidate_warning,
                errors=status_errors,
                warnings=status_warnings,
            )
        )
        return ranked


def build_v58_realtime_scores(
    *,
    full_market_pool: pd.DataFrame,
    config: dict[str, Any],
    sector_overview: dict[str, Any],
    stock_sector_ranks: pd.DataFrame,
    market_environment: dict[str, Any],
    market_score_fields: dict[str, Any],
    source_meta: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build V5.9 full-market base scores plus the existing V5.8 Top50 layer.

    The complete same-day quote universe first receives a V5.9 raw base score
    and cross-sectional percentile-calibrated ``base_score``.  Only the
    resulting Top100 then goes through the unchanged shared V5.8
    trading-quality calculation and 70/30 final-score formula.  Existing
    strategy triggers and observation-pool membership are not changed.
    """
    if full_market_pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    full_market = enrich_indicators_from_local_daily_cache(
        realtime_quotes=full_market_pool,
        daily_dir=LOCAL_DAILY_HISTORY_DIR,
        days=_history_days(config),
    )
    full_market = enrich_optional_fields(
        full_market,
        config,
        sector_overview,
        market_environment,
        stock_sector_ranks=stock_sector_ranks,
    )
    full_market = attach_market_score_fields(full_market, market_score_fields)
    full_market = check_risks(full_market, config)
    scored = score_candidates(full_market, config)
    scored = enrich_display_fields(scored, source_meta)
    scored["base_score"] = pd.to_numeric(scored["score"], errors="coerce")
    scored["v58_quality_evaluated"] = False
    scored["v58_top50_eligible"] = False

    top100 = scored.nlargest(100, "base_score").copy()
    quality_rows: list[dict[str, Any]] = []
    for _, row in top100.iterrows():
        row_dict = row.to_dict()
        quality = evaluate_trading_quality(
            row=row_dict,
            market={
                key: row_dict.get(key)
                for key in ("sh_close", "sh_ma20", "up_ratio", "limit_up_count")
            },
            industry={
                key: row_dict.get(key)
                for key in (
                    "industry_rank",
                    "industry_count",
                    "industry_up_ratio",
                    "industry_limit_up_count",
                )
            },
            mode="realtime",
        )
        data_quality = score_data_quality(row_dict, mode="realtime")
        final_trade_score = calculate_final_trade_score(
            row_dict.get("base_score"),
            quality["trading_quality_score"],
        )
        source_ready = _v58_truthy(row_dict.get("allow_strategy_candidate"))
        top50_eligible = bool(
            row_dict.get("base_score", 0) >= 70
            and final_trade_score >= 85
            and quality["turnover_quality_eligible"]
            and data_quality["top50_data_eligible"]
            and source_ready
        )
        if not source_ready:
            admission_reason = "当前行情源未达到正式实时策略字段标准"
        elif not data_quality["top50_data_eligible"]:
            admission_reason = f"关键字段缺失：{data_quality['data_quality_missing_critical']}"
        elif not quality["turnover_quality_eligible"]:
            admission_reason = "换手率不在5%–30%允许范围"
        elif row_dict.get("base_score", 0) < 70:
            admission_reason = "基础评分低于70"
        elif final_trade_score < 85:
            admission_reason = "最终交易评分低于85"
        else:
            admission_reason = "满足V5.8 Top50准入条件"

        quality_rows.append(
            {
                "code": _normalize_stock_code(row_dict.get("code")),
                "base_score": row_dict.get("base_score"),
                "quality_adjustment": quality["adjustment"],
                "trading_quality_score": quality["trading_quality_score"],
                "final_trade_score": final_trade_score,
                "quality_market_score": quality["market_score"],
                "quality_trend_stage_score": quality["trend_stage_score"],
                "quality_trend_stage_risk": quality["trend_stage_risk"],
                "quality_buy_point_score": quality["buy_point_score"],
                "quality_sector_linkage_score": quality["sector_linkage_score"],
                "turnover_quality_score": quality["turnover_quality_score"],
                "turnover_quality_eligible": quality["turnover_quality_eligible"],
                "v58_quality_detail": quality["detail"],
                "v58_quality_evaluated": True,
                "v58_top50_eligible": top50_eligible,
                "v58_admission_reason": admission_reason,
                **data_quality,
            }
        )

    if quality_rows:
        quality_frame = pd.DataFrame(quality_rows).drop_duplicates("code", keep="first")
        scored["_v58_code"] = scored["code"].map(_normalize_stock_code)
        scored = scored.merge(
            quality_frame.rename(columns={"code": "_v58_code"}),
            on="_v58_code",
            how="left",
            suffixes=("", "_v58"),
        )
        for field in ("base_score", "v58_quality_evaluated", "v58_top50_eligible"):
            v58_field = f"{field}_v58"
            if v58_field in scored.columns:
                scored[field] = scored[v58_field].combine_first(scored[field])
                scored = scored.drop(columns=[v58_field])
        scored = scored.drop(columns=["_v58_code"])

    top50 = scored[scored.get("v58_top50_eligible", False).fillna(False)].copy()
    top50 = top50.sort_values(["final_trade_score", "base_score"], ascending=[False, False]).head(50)
    return scored.reset_index(drop=True), top50.reset_index(drop=True)


def attach_v58_fields_to_candidates(candidates: pd.DataFrame, full_market_scores: pd.DataFrame) -> pd.DataFrame:
    """Expose V5.9/V5.8 values on existing rows without changing pool membership."""
    output = candidates.copy()
    if output.empty:
        return output
    if full_market_scores.empty or "code" not in full_market_scores.columns:
        return output

    v59_base_fields = [
        "raw_base_score",
        "base_percentile",
        "base_score",
        "score",
        "level",
        "capital_attack_score",
        "capital_attack_ratio",
        "float_market_cap_for_score",
        "turnover_score",
        "turnover_activity_score",
        "turnover_score_source",
        "trend_score",
        "volume_score",
        "volume_ratio_score",
        "volume_ratio_score_source",
        "limit_up_activity_score",
        "limit_up_count_20d",
        "raw_positive_score",
        "score_normalization_base",
        "score_breakdown",
        "score_detail",
        "data_confidence",
        "data_confidence_level",
        "data_confidence_notes",
        "risk_flags",
    ]
    v58_quality_fields = [
        "quality_adjustment",
        "trading_quality_score",
        "final_trade_score",
        "quality_market_score",
        "quality_trend_stage_score",
        "quality_trend_stage_risk",
        "quality_buy_point_score",
        "quality_sector_linkage_score",
        "turnover_quality_score",
        "turnover_quality_eligible",
        "data_quality_score",
        "data_quality_status",
        "data_quality_missing_critical",
        "data_quality_missing_support",
        "top50_data_eligible",
        "v58_quality_evaluated",
        "v58_top50_eligible",
        "v58_quality_detail",
        "v58_admission_reason",
    ]
    fields = [*v59_base_fields, *v58_quality_fields]
    available = [field for field in fields if field in full_market_scores.columns]
    if not available:
        return output
    lookup = full_market_scores.copy()
    lookup["_v58_code"] = lookup["code"].map(_normalize_stock_code)
    lookup = lookup.drop_duplicates("_v58_code", keep="first").set_index("_v58_code")
    output_codes = output["code"].map(_normalize_stock_code)
    for field in available:
        matched = output_codes.map(lookup[field])
        if field in output.columns:
            output[field] = matched.combine_first(output[field])
        else:
            output[field] = matched
    return output


def _normalize_stock_code(value: Any) -> str:
    """Normalize a code for snapshot joins without inferring a market suffix."""
    if value is None or pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits[-6:].zfill(6) if digits else ""


def _v58_truthy(value: Any) -> bool:
    """Interpret nullable dataframe values conservatively for admission gates."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}
    return bool(value)


def save_candidates(candidates: pd.DataFrame) -> None:
    """Save the latest ranked candidates to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_result_columns(candidates).to_csv(LATEST_CANDIDATES_PATH, index=False, encoding="utf-8-sig")
    print(f"候选股结果已保存: {LATEST_CANDIDATES_PATH}")


def save_reference_candidates(candidates: pd.DataFrame) -> None:
    """Save reference candidates to a separate CSV snapshot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(REFERENCE_CANDIDATES_PATH, index=False, encoding="utf-8-sig")
    print(f"参考候选股结果已保存: {REFERENCE_CANDIDATES_PATH}")


def save_v58_top50(candidates: pd.DataFrame) -> None:
    """Save a schema-stable V5.8 Top50 snapshot, including the empty case."""
    output = candidates.copy()
    for column in V58_TOP50_REQUIRED_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    save_auxiliary_csv(output, V58_TOP50_PATH, "V5.8 Top50")


def build_and_save_reference_candidates(spot_quotes: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build reference candidates from realtime quotes plus local Tushare cache."""
    try:
        trade_date = resolve_tushare_cache_trade_date()
        if not trade_date:
            logging.getLogger("quant_stock_watch").warning("No Tushare cache trade date found.")
            return build_reference_candidates(pd.DataFrame(), config)

        realtime_quotes = fetch_sina_quotes_for_reference()
        if realtime_quotes.empty:
            realtime_quotes = spot_quotes
        merged = merge_sina_with_tushare_cache(realtime_quotes, trade_date)
        return build_reference_candidates(merged, config, limit=50)
    except Exception as exc:
        logging.getLogger("quant_stock_watch").warning("Reference candidate build failed: %s", exc)
        return build_reference_candidates(pd.DataFrame(), config)


def fetch_sina_quotes_for_reference() -> pd.DataFrame:
    """Fetch Sina realtime quotes for the reference-candidate test layer."""
    try:
        import akshare as ak

        data = ak.stock_zh_a_spot()
        if data is None or data.empty:
            return pd.DataFrame()
        return data
    except Exception as exc:
        logging.getLogger("quant_stock_watch").warning("Sina reference quote fetch failed: %s", exc)
        return pd.DataFrame()


def resolve_tushare_cache_trade_date() -> str:
    """Resolve the Tushare cache trade date from status or cached daily files."""
    try:
        if TUSHARE_CACHE_STATUS_PATH.exists():
            status = json.loads(TUSHARE_CACHE_STATUS_PATH.read_text(encoding="utf-8"))
            trade_date = str(status.get("trade_date") or "").strip()
            if trade_date:
                return trade_date
    except Exception:
        logging.getLogger("quant_stock_watch").warning("Failed to read Tushare cache status.")

    cache_dir = TUSHARE_CACHE_STATUS_PATH.parent
    daily_files = sorted(cache_dir.glob("daily_*.csv"), reverse=True)
    if not daily_files:
        return ""
    return daily_files[0].stem.replace("daily_", "")


def save_market_status(status: dict[str, Any]) -> None:
    """Save latest market status for the dashboard."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LATEST_MARKET_STATUS_PATH.open("w", encoding="utf-8") as file:
        json.dump(status, file, ensure_ascii=False, indent=2)
    print(f"行情状态已保存: {LATEST_MARKET_STATUS_PATH}")


def save_auxiliary_csv(data: pd.DataFrame, path: Path, label: str) -> None:
    """Save optional dashboard CSV snapshots."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{label}快照已保存: {path}")
    except Exception:
        logging.getLogger("quant_stock_watch").exception("Failed to save %s snapshot: %s", label, path)


def is_official_strategy_source_ready(quotes: pd.DataFrame, source_meta: dict[str, Any]) -> bool:
    """Check final scoring inputs, rather than raw API field declarations.

    The legacy function name is retained for its existing callers.  A source
    may legitimately supply intraday turnover and volume ratio through the
    already-validated calculation path, so source branding and raw-field
    metadata must not override the final per-row data-quality result.
    """
    if quotes is None or quotes.empty:
        return False
    del source_meta
    levels = quotes.get("data_quality_level", pd.Series("C", index=quotes.index)).astype(str).str.upper()
    return bool(levels.isin({"A", "B"}).any())


def attach_realtime_turnover_display_field(quotes: pd.DataFrame) -> pd.DataFrame:
    """Persist the validated turnover selected by the existing source priority.

    ``apply_realtime_turnover_priority`` has already selected either the
    Eastmoney f8 value or the Sina-volume/Tushare-float-share calculation in
    ``turnover``.  This helper only exposes that validated value for display;
    it neither changes the scoring field nor promotes ``turnover_rate_ref``.
    """
    output = quotes.copy()
    if output.empty:
        output["realtime_turnover_value"] = pd.Series(dtype="float64")
        output["realtime_turnover_source"] = pd.Series(dtype="object")
        return output

    turnover = pd.to_numeric(output.get("turnover", pd.Series(pd.NA, index=output.index)), errors="coerce")
    realtime_flag = output.get("is_realtime_turnover", pd.Series(False, index=output.index))
    is_realtime = realtime_flag.fillna(False).astype(bool)
    valid = is_realtime & turnover.between(0, 100)
    output["realtime_turnover_value"] = turnover.where(valid)
    source = output.get("turnover_source", pd.Series(pd.NA, index=output.index))
    output["realtime_turnover_source"] = source.where(valid, pd.NA)
    return output


def assess_final_score_data_quality(quotes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify final scoring rows after calculated fields have been attached.

    A is reserved for direct interface turnover and volume-ratio fields.  B
    accepts the existing validated calculation path when the final row has
    turnover, volume ratio, amount and a usable capital-strength basis.  C is
    missing one or more of those inputs and remains ineligible.  This affects
    only data admission and provenance; it does not alter any score formula.
    """
    output = quotes.copy()
    if output.empty:
        for column, dtype in {
            "data_quality_level": "object",
            "data_quality_label": "object",
            "data_quality_note": "object",
            "allow_strategy_candidate": "bool",
        }.items():
            output[column] = pd.Series(dtype=dtype)
        return output, {
            "data_quality_level": "C",
            "data_quality_label": "C级（关键字段缺失）",
            "data_quality_note": "关键评分字段缺失，不能进入正式候选池。",
            "data_quality_counts": {"A": 0, "B": 0, "C": 0},
            "data_quality_ready_count": 0,
        }

    turnover = _first_numeric_column(output, ("turnover", "turnover_rate", "换手率"))
    volume_ratio = _first_numeric_column(output, ("volume_ratio", "量比"))
    amount = _first_numeric_column(output, ("amount", "成交额"))
    # Downstream strategy and quality modules use the normalized English
    # fields.  Preserve any populated canonical value and only fill it from a
    # same-meaning final input when the canonical field is absent.
    for column, values in {
        "turnover": turnover,
        "volume_ratio": volume_ratio,
        "amount": amount,
    }.items():
        existing = pd.to_numeric(output.get(column, pd.Series(pd.NA, index=output.index)), errors="coerce")
        output[column] = existing.where(existing.notna(), values)
    capital_basis = _first_numeric_column(
        output,
        ("capital_attack_ratio", "float_market_cap_for_score", "float_market_cap", "circ_mv", "float_share"),
    )
    money_strength = output.get("money_strength_level", pd.Series(pd.NA, index=output.index))
    money_strength_available = money_strength.fillna("").astype(str).str.strip().isin(
        {"正常", "温和放量", "放量", "爆量", "缩量"}
    )
    final_fields_complete = (
        turnover.between(0, 100)
        & volume_ratio.between(0, 100)
        & amount.gt(0)
        & (capital_basis.gt(0) | money_strength_available)
    )

    turnover_source = output.get("turnover_source", pd.Series("", index=output.index)).fillna("").astype(str)
    volume_ratio_source = output.get("volume_ratio_source", pd.Series("", index=output.index)).fillna("").astype(str)
    direct_interface = (
        final_fields_complete
        & turnover_source.eq("eastmoney_realtime")
        & volume_ratio_source.eq("eastmoney_realtime")
    )
    calculated_fields = final_fields_complete & ~direct_interface

    output["data_quality_level"] = "C"
    output.loc[calculated_fields, "data_quality_level"] = "B"
    output.loc[direct_interface, "data_quality_level"] = "A"
    output["data_quality_label"] = output["data_quality_level"].map(
        {
            "A": "A级（接口实时字段）",
            "B": "B级（计算字段）",
            "C": "C级（关键字段缺失）",
        }
    )
    output["data_quality_note"] = output["data_quality_level"].map(
        {
            "A": "换手率/量比来自实时接口字段。",
            "B": "当前换手率/量比来自计算，不是交易所实时字段。",
            "C": "关键评分字段缺失，不能进入正式候选池。",
        }
    )
    output["allow_strategy_candidate"] = output["data_quality_level"].isin({"A", "B"})

    counts = output["data_quality_level"].value_counts().to_dict()
    level_counts = {level: int(counts.get(level, 0)) for level in ("A", "B", "C")}
    if level_counts["B"]:
        overall_level = "B"
    elif level_counts["A"]:
        overall_level = "A"
    else:
        overall_level = "C"
    labels = {"A": "A级（接口实时字段）", "B": "B级（计算字段）", "C": "C级（关键字段缺失）"}
    notes = {
        "A": "换手率/量比来自实时接口字段。",
        "B": "当前换手率/量比来自计算，不是交易所实时字段。",
        "C": "关键评分字段缺失，不能进入正式候选池。",
    }
    return output, {
        "data_quality_level": overall_level,
        "data_quality_label": labels[overall_level],
        "data_quality_note": notes[overall_level],
        "data_quality_counts": level_counts,
        "data_quality_ready_count": level_counts["A"] + level_counts["B"],
    }


def _first_numeric_column(data: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Return the first usable numeric value across same-meaning aliases."""
    values = pd.Series(float("nan"), index=data.index, dtype="float64")
    for column in columns:
        if column in data.columns:
            candidate = pd.to_numeric(data[column], errors="coerce")
            values = values.where(values.notna(), candidate)
    return values


def should_try_local_broker_source(source_meta: dict[str, Any], config: dict[str, Any]) -> bool:
    """Local broker software ingestion is disabled; use public data sources only."""
    return False


def resolve_local_broker_path(config: dict[str, Any]) -> Path:
    """Resolve the configured local broker quote CSV path."""
    raw_path = str(config.get("data_source", {}).get("local_broker_path") or "data/local_broker/cmbc_quote_sample.csv")
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_market_status(
    *,
    started_at: datetime,
    data_source: AkshareSource,
    spot_count: int,
    active_count: int,
    enriched_count: int,
    strategy_candidate_count: int,
    active_watchlist_count: int,
    board_count_map: dict[str, int],
    sector_overview: dict[str, Any],
    market_environment: dict[str, Any],
    watchlist_count: int,
    holdings_count: int,
    errors: list[str],
    warnings: list[str],
    reference_candidate_count: int = 0,
    reference_candidate_enabled: bool = False,
    reference_candidate_source: str = "",
    reference_candidate_warning: str = "",
    v58_full_market_count: int = 0,
    v58_top100_count: int = 0,
    v58_top50_count: int = 0,
) -> dict[str, Any]:
    """Build the latest market status payload."""
    meta = data_source.last_spot_meta or {}
    missing_fields = list(dict.fromkeys(meta.get("missing_fields", [])))
    all_errors = list(dict.fromkeys([*meta.get("errors", []), *errors]))
    all_warnings = list(dict.fromkeys([*meta.get("warnings", []), *warnings]))
    is_full_featured = bool(meta.get("is_full_featured"))
    allow_strategy_candidate = bool(meta.get("allow_strategy_candidate"))

    data_quality_level = str(meta.get("data_quality_level") or "C").upper()
    if spot_count == 0:
        data_status = "行情源失败"
    elif data_quality_level == "C" or not allow_strategy_candidate:
        data_status = "部分字段缺失"
    else:
        data_status = "正常"

    return {
        "scan_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": meta.get("data_source", "--"),
        "data_source_level": meta.get("data_source_level", "--"),
        "data_status": data_status,
        "is_full_featured": is_full_featured,
        "allow_strategy_candidate": allow_strategy_candidate,
        "data_quality_level": data_quality_level,
        "data_quality_label": meta.get("data_quality_label", "C级（关键字段缺失）"),
        "data_quality_note": meta.get("data_quality_note", "关键评分字段缺失，不能进入正式候选池。"),
        "data_quality_counts": meta.get("data_quality_counts", {"A": 0, "B": 0, "C": spot_count}),
        "data_quality_ready_count": int(meta.get("data_quality_ready_count") or 0),
        "spot_count": spot_count,
        "active_count": active_count,
        "enriched_count": enriched_count,
        "strategy_candidate_count": strategy_candidate_count,
        "active_watchlist_count": active_watchlist_count,
        "reference_candidate_count": reference_candidate_count,
        "reference_candidate_enabled": reference_candidate_enabled,
        "reference_candidate_source": reference_candidate_source,
        "reference_candidate_warning": reference_candidate_warning,
        "v58_full_market_count": v58_full_market_count,
        "v58_top100_count": v58_top100_count,
        "v58_top50_count": v58_top50_count,
        "watchlist_count": watchlist_count,
        "holdings_count": holdings_count,
        "board_counts": board_count_map,
        "sector_strength": sector_overview,
        "market_environment_detail": market_environment,
        "market_environment": market_environment.get("market_environment", "未知") if market_environment else "未知",
        "missing_fields": missing_fields,
        "errors": all_errors,
        "warnings": all_warnings,
        "source_attempts": meta.get("source_attempts", []),
        "turnover_field": meta.get("turnover_field"),
        "amount_field": meta.get("amount_field"),
        "realtime_turnover_available": bool(meta.get("realtime_turnover_available")),
        "realtime_turnover_source": meta.get("realtime_turnover_source", "unavailable"),
        "realtime_turnover_count": int(meta.get("realtime_turnover_count") or 0),
        "calculated_turnover_count": int(meta.get("calculated_turnover_count") or 0),
        "eastmoney_turnover_count": int(meta.get("eastmoney_turnover_count") or 0),
        "reference_turnover_count": int(meta.get("reference_turnover_count") or 0),
        "realtime_turnover_coverage": float(meta.get("realtime_turnover_coverage") or 0.0),
        "realtime_volume_ratio_available": bool(meta.get("realtime_volume_ratio_available")),
        "realtime_volume_ratio_source": meta.get("realtime_volume_ratio_source", "unavailable"),
        "realtime_volume_ratio_count": int(meta.get("realtime_volume_ratio_count") or 0),
        "calculated_volume_ratio_count": int(meta.get("calculated_volume_ratio_count") or 0),
        "eastmoney_volume_ratio_count": int(meta.get("eastmoney_volume_ratio_count") or 0),
        "reference_volume_ratio_count": int(meta.get("reference_volume_ratio_count") or 0),
        "realtime_volume_ratio_coverage": float(meta.get("realtime_volume_ratio_coverage") or 0.0),
    }


def build_active_watchlist(active_pool: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build a fallback watchlist from the most liquid active stocks."""
    if active_pool.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    watchlist_size = _active_watchlist_size(config)
    watchlist = active_pool.copy()
    if "amount" in watchlist.columns:
        watchlist["amount"] = pd.to_numeric(watchlist["amount"], errors="coerce").fillna(0)
        watchlist = watchlist.sort_values("amount", ascending=False)

    watchlist = watchlist.head(watchlist_size).reset_index(drop=True)
    watchlist["source_type"] = SOURCE_TYPE_ACTIVE
    watchlist["strategy_name"] = "active_watchlist"
    watchlist["strategy_names"] = "active_watchlist"
    watchlist["triggered"] = False
    watchlist["reason"] = "行情正常但未作为策略候选，按成交额排名展示活跃股票观察池。"
    watchlist["risk"] = "活跃观察池只表示成交活跃，不是操作建议。"
    return watchlist


def enrich_active_watchlist_with_reference_fields(
    active_watchlist: pd.DataFrame,
    spot_quotes: pd.DataFrame,
    source_meta: dict[str, Any],
) -> pd.DataFrame:
    """Attach Tushare daily-cache reference fields to Sina active watchlist rows.

    These values remain explicitly separate from realtime ``turnover`` and
    ``volume_ratio``. They are display-only reference data and have no role in
    scoring or official strategy-candidate eligibility.
    """
    if active_watchlist.empty or spot_quotes.empty:
        return active_watchlist

    if {"turnover_rate_ref", "volume_ratio_ref", "reference_trade_date"}.issubset(active_watchlist.columns):
        return active_watchlist

    source_name = str(source_meta.get("data_source") or "")
    if "新浪" not in source_name:
        return active_watchlist

    trade_date = resolve_tushare_cache_trade_date()
    if not trade_date:
        return active_watchlist

    try:
        merged = merge_sina_with_tushare_cache(spot_quotes, trade_date)
        reference_columns = [
            "code",
            "turnover_rate_ref",
            "volume_ratio_ref",
            "total_mv",
            "circ_mv",
            "tushare_trade_date",
        ]
        if merged.empty or not set(reference_columns).issubset(merged.columns):
            return active_watchlist

        reference = merged[reference_columns].drop_duplicates("code").rename(
            columns={"tushare_trade_date": "reference_trade_date"}
        )
        output = active_watchlist.copy()
        output["code"] = output["code"].astype(str).str.zfill(6)
        reference["code"] = reference["code"].astype(str).str.zfill(6)
        return output.merge(reference, on="code", how="left")
    except Exception as exc:
        logging.getLogger("quant_stock_watch").warning(
            "Active watchlist reference-field merge failed: %s", exc
        )
        return active_watchlist


def apply_board_filter(stocks: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply optional board include/exclude filters; defaults keep all boards."""
    if stocks.empty or "board_type" not in stocks.columns:
        return stocks

    board_config = config.get("board_filter", {})
    include = board_config.get("include") or []
    exclude = board_config.get("exclude") or []

    filtered = stocks.copy()
    if include:
        filtered = filtered[filtered["board_type"].isin(include)]
    if exclude:
        filtered = filtered[~filtered["board_type"].isin(exclude)]
    return filtered.reset_index(drop=True)


def enrich_display_fields(results: pd.DataFrame, source_meta: dict[str, Any]) -> pd.DataFrame:
    """Add display-only fields while preserving raw calculation columns."""
    output = add_board_columns(results.copy())
    output["source_type"] = output.get("source_type", pd.Series(dtype="object")).fillna(SOURCE_TYPE_STRATEGY)
    output["source_type_display"] = output["source_type"].map(SOURCE_TYPE_DISPLAY).fillna(output["source_type"])
    output["amount_display"] = output.get("amount", pd.Series(dtype="float64")).map(format_amount)
    output["turnover_display"] = output.get("turnover", pd.Series(dtype="float64")).map(format_turnover)
    output["strategy_names"] = output.get("strategy_names", output.get("strategy_name", pd.Series(dtype="object")))
    output["strategy_names"] = output["strategy_names"].fillna(output.get("strategy_name", ""))
    output["strategy_names_display"] = output["strategy_names"].map(format_strategy_names)
    output["level_display"] = output.get("level", pd.Series(dtype="object")).map(format_level)

    output["data_source"] = output.get("data_source", pd.Series(dtype="object")).fillna(
        source_meta.get("data_source", "--")
    )
    output["data_source_level"] = output.get("data_source_level", pd.Series(dtype="object")).fillna(
        source_meta.get("data_source_level", "--")
    )
    output["is_full_featured"] = output.get("is_full_featured", pd.Series(dtype="object")).fillna(
        bool(source_meta.get("is_full_featured"))
    )
    output["allow_strategy_candidate"] = output.get(
        "allow_strategy_candidate", pd.Series(dtype="object")
    ).fillna(bool(source_meta.get("allow_strategy_candidate")))
    return output


def ensure_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Ensure the CSV keeps stable dashboard columns even when data is empty."""
    output = results.copy()
    for column in RESULT_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    if "score" in output.columns:
        output["score"] = pd.to_numeric(output["score"], errors="coerce").fillna(0)
        source_order = output["source_type"].map({SOURCE_TYPE_STRATEGY: 0, SOURCE_TYPE_ACTIVE: 1}).fillna(2)
        output["_source_order"] = source_order
        output = output.sort_values(["_source_order", "score"], ascending=[True, False]).drop(columns=["_source_order"])

    return output[RESULT_COLUMNS].reset_index(drop=True)


def format_strategy_names(value: Any) -> str:
    """Translate comma-separated strategy names to Chinese labels."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "--"
    names = []
    for item in re.split(r"[,/，、\s]+", str(value)):
        key = item.strip()
        if key:
            names.append(STRATEGY_DISPLAY.get(key, key))
    return " / ".join(names) if names else "--"


def format_level(value: Any) -> str:
    """Translate level enums to Chinese labels."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "--"
    text = str(value).strip()
    return LEVEL_DISPLAY.get(text, text)


def print_candidates(candidates: pd.DataFrame) -> None:
    """Print a clear top candidate table."""
    print("\n排名前 20 观察结果:")
    if candidates.empty:
        print("本轮暂无可展示数据。")
        return

    display_columns = [
        "source_type_display",
        "board_type_display",
        "code",
        "market_code",
        "name",
        "price",
        "pct_chg",
        "amount_display",
        "turnover_display",
        "volume_ratio",
        "strategy_names_display",
        "score",
        "level_display",
        "reason",
        "risk_summary",
        "data_source",
    ]
    existing_columns = [column for column in display_columns if column in candidates.columns]
    print(candidates[existing_columns].to_string(index=False))


def feature_enabled(config: dict[str, Any], name: str) -> bool:
    """Read a feature flag; defaults to True for known enhancement modules."""
    return bool(config.get("feature_flags", {}).get(name, True))


def fetch_sector_overview(data_source: AkshareSource) -> dict[str, Any]:
    """Fetch industry and concept sector overview, with graceful fallback."""
    try:
        industry = normalize_sector_frame(data_source.fetch_industry_sectors(), "industry")
    except Exception as exc:
        logging.getLogger("quant_stock_watch").warning("Industry sector overview failed: %s", exc)
        industry = pd.DataFrame()
    try:
        concept = normalize_sector_frame(data_source.fetch_concept_sectors(), "concept")
    except Exception as exc:
        logging.getLogger("quant_stock_watch").warning("Concept sector overview failed: %s", exc)
        concept = pd.DataFrame()
    return build_sector_overview(industry, concept)


def enrich_optional_fields(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    sector_overview: dict[str, Any],
    market_environment: dict[str, Any],
    stock_sector_ranks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply optional analysis modules before risk/scoring."""
    output = candidates.copy()
    if output.empty:
        return output
    if feature_enabled(config, "enable_sector_strength"):
        output = enrich_candidates_with_sector(output, sector_overview, stock_sector_ranks)
    if feature_enabled(config, "enable_turnover_scoring"):
        from core.turnover_scoring import enrich_turnover_fields

        output = enrich_turnover_fields(output)
    if feature_enabled(config, "enable_money_strength"):
        output = enrich_money_strength(output)
    if feature_enabled(config, "enable_position_risk"):
        output = enrich_position_risk(output)
    if feature_enabled(config, "enable_announcement_risk"):
        output = enrich_announcement_risk(output, {})
    else:
        output["announcement_risk_level"] = "未启用"
        output["announcement_risk_keywords"] = ""
        output["announcement_risk_summary"] = "公告风险功能未启用"
        output["latest_announcement_titles"] = ""
    if feature_enabled(config, "enable_theme_heat"):
        output = enrich_theme_heat(output, config)
    return output


def print_score_input_checks(candidates: pd.DataFrame) -> None:
    """Print the exact V5.3 inputs for every row before scoring.

    This is a read-only runtime diagnostic.  It exposes input availability and
    values without changing score calculation, thresholds, or candidate order.
    """
    if candidates.empty:
        print("评分输入检查：当前没有待评分股票。")
        return

    print("评分输入检查：以下为每只股票传入 V5.3 的模块字段。")
    for _, row in candidates.iterrows():
        payload = {
            "code": _score_input_value(row, "code"),
            "name": _score_input_value(row, "name"),
            "资金攻击": {
                "amount": _score_input_value(row, "amount"),
                "circ_mv": _score_input_value(row, "circ_mv"),
                "float_share": _score_input_value(row, "float_share"),
            },
            "换手": {"turnover": _score_input_value(row, "turnover")},
            "趋势": {
                "price": _score_input_value(row, "price"),
                "ma5": _score_input_value(row, "ma5"),
                "ma10": _score_input_value(row, "ma10"),
                "ma20": _score_input_value(row, "ma20"),
            },
            "量能": {
                "volume_ratio": _score_input_value(row, "volume_ratio"),
                "volume": _score_input_value(row, "volume"),
                "avg_volume_5d": _score_input_value(row, "avg_volume_5d"),
                "avg_volume_10d": _score_input_value(row, "avg_volume_10d"),
            },
            "涨停": {"limit_up_count_20d": _score_input_value(row, "limit_up_count_20d")},
            "板块": {
                "sector_name": _score_input_value(row, "sector_name"),
                "sector_rank": _score_input_value(row, "sector_rank"),
            },
            "市场": {
                "sh_close": _score_input_value(row, "sh_close"),
                "sh_pct_chg": _score_input_value(row, "sh_pct_chg"),
                "sh_ma5": _score_input_value(row, "sh_ma5"),
                "sh_ma10": _score_input_value(row, "sh_ma10"),
                "up_count": _score_input_value(row, "up_count"),
                "down_count": _score_input_value(row, "down_count"),
                "market_amount": _score_input_value(row, "market_amount"),
                "market_amount_ma20": _score_input_value(row, "market_amount_ma20"),
            },
            "买点": {
                "high_20d": _score_input_value(row, "high_20d"),
                "prior_20d_high": _score_input_value(row, "prior_20d_high"),
                "distance_to_20d_high_pct": _score_input_value(row, "distance_to_20d_high_pct"),
                "volume_breakout": _score_input_value(row, "volume_breakout"),
                "recent_low_volume_pullback": _score_input_value(row, "recent_low_volume_pullback"),
            },
        }
        print("评分输入：" + json.dumps(payload, ensure_ascii=False, default=str))


def _score_input_value(row: pd.Series, column: str) -> Any:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def post_score_enrichment(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    market_environment: dict[str, Any],
    selection_trade_date: str | None = None,
) -> pd.DataFrame:
    """Apply modules that depend on score/level and broad market context."""
    output = candidates.copy()
    if output.empty:
        return output
    env_name = str(market_environment.get("market_environment", "未知") if market_environment else "未知")
    if feature_enabled(config, "enable_action_state"):
        output = enrich_action_state(output, env_name)
    if feature_enabled(config, "enable_t1_risk"):
        output = enrich_t1_risk(output, env_name)
        output["risk_summary"] = output["risk_summary"].fillna("").astype(str) + "；" + output["t1_risk_summary"].fillna("").astype(str)
    if feature_enabled(config, "enable_market_environment"):
        output = apply_market_level_cap(output, market_environment)
    if feature_enabled(config, "enable_tracking"):
        output = enrich_tracking(
            output,
            DATA_DIR / "quant_watch.db",
            selection_trade_date=selection_trade_date,
        )
    return output


def run_watch(config: dict[str, Any]) -> None:
    """Run scans repeatedly by configured interval."""
    interval_seconds = _scan_interval_seconds(config)
    while True:
        try:
            run_once(config)
        except KeyboardInterrupt:
            print("已停止循环运行。")
            raise
        except Exception:
            logging.exception("Scan cycle failed.")

        print(f"\n等待 {interval_seconds} 秒后进入下一轮扫描...\n")
        time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="A-share read-only watch scanner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run one scan")
    mode.add_argument("--watch", action="store_true", help="run scans in a loop")
    return parser.parse_args()


def main() -> None:
    """Program entry."""
    setup_logging()
    args = parse_args()
    config = load_config()

    if args.watch:
        run_watch(config)
    else:
        run_once(config)


def _scan_interval_seconds(config: dict[str, Any]) -> int:
    """Resolve scan interval from common config names."""
    runtime = config.get("runtime", {})
    return int(
        runtime.get("scan_interval_seconds")
        or runtime.get("full_market_scan_seconds")
        or runtime.get("refresh_seconds")
        or 180
    )


def _history_days(config: dict[str, Any]) -> int:
    """Resolve historical lookback length."""
    runtime = config.get("runtime", {})
    indicators = config.get("indicators", {})
    return int(runtime.get("history_days") or indicators.get("history_days") or 60)


def _history_calc_limit(config: dict[str, Any]) -> int:
    """Resolve how many active stocks should request historical K data."""
    runtime = config.get("runtime", {})
    return int(
        runtime.get("history_calc_limit")
        or runtime.get("max_history_stocks")
        or HISTORY_CALC_LIMIT
    )


def _active_watchlist_size(config: dict[str, Any]) -> int:
    """Resolve fallback active watchlist size."""
    runtime = config.get("runtime", {})
    return int(runtime.get("active_watchlist_size") or ACTIVE_WATCHLIST_SIZE)


def _limit_history_pool(active_pool: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Limit historical requests to the most liquid active stocks."""
    if active_pool.empty:
        return active_pool

    max_count = _history_calc_limit(config)
    if max_count <= 0 or len(active_pool) <= max_count:
        return active_pool

    if "amount" not in active_pool.columns:
        return active_pool.head(max_count).copy()

    limited = active_pool.copy()
    limited["amount"] = pd.to_numeric(limited["amount"], errors="coerce").fillna(0)
    return limited.sort_values("amount", ascending=False).head(max_count).reset_index(drop=True)


def _print_header(started_at: datetime) -> None:
    """Print scan header."""
    print("=" * 80)
    print(f"量化盯盘扫描开始: {started_at:%Y-%m-%d %H:%M:%S}")
    print("=" * 80)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
