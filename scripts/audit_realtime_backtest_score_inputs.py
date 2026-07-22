#!/usr/bin/env python3
"""Create a read-only realtime-vs-backtest V5.3 score-input audit report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.indicators import enrich_indicators_from_local_daily_cache
from core.market_environment import attach_market_score_fields, build_realtime_market_score_fields
from core.scoring import score_candidates
from core.sector_strength import build_realtime_sector_ranks, enrich_candidates_with_sector


DATA_DIR = PROJECT_ROOT / "data"
LATEST_CANDIDATES_PATH = DATA_DIR / "latest_candidates.csv"
LATEST_STATUS_PATH = DATA_DIR / "latest_market_status.json"
DAILY_DIR = DATA_DIR / "history" / "daily"
INDUSTRY_DIR = DATA_DIR / "history" / "industry"
MARKET_DIR = DATA_DIR / "history" / "market"
BACKTEST_SUMMARY_PATH = DATA_DIR / "reports" / "v5_4" / "score_v5_4_backtest_summary.json"
REPORT_PATH = DATA_DIR / "reports" / "realtime_backtest_score_input_audit.md"


MODULE_FIELD_GROUPS = {
    # Each tuple is one mandatory input group.  The capital-attack module may
    # legitimately use any one of its three equivalent float-cap inputs.
    "资金攻击": (("amount",), ("float_market_cap", "circ_mv", "float_share")),
    "换手": (("turnover",),),
    "趋势": (("price",), ("ma5",), ("ma10",), ("ma20",)),
    "量能": (("volume_ratio",), ("volume",), ("avg_volume_5d",), ("avg_volume_10d",)),
    "涨停": (("limit_up_count_20d",),),
    "板块": (("sector_rank",),),
    "市场": (("sh_close",), ("sh_pct_chg",), ("sh_ma5",), ("sh_ma10",), ("up_count",), ("down_count",), ("market_amount",), ("market_amount_ma20",)),
    "买点": (("price",), ("high_20d", "prior_20d_high"), ("recent_low_volume_pullback",)),
}


def main() -> int:
    candidates, candidate_source = _load_realtime_candidates()
    if candidates.empty:
        print("未找到可审计的实时候选股票，无法生成评分输入审计。")
        return 1

    enriched = enrich_indicators_from_local_daily_cache(candidates, DAILY_DIR, days=60)
    sector_ranks = build_realtime_sector_ranks(_latest_daily_as_spot(), INDUSTRY_DIR)
    enriched = enrich_candidates_with_sector(enriched, {}, sector_ranks)
    enriched = attach_market_score_fields(enriched, _market_score_fields())
    scored = score_candidates(enriched)

    coverage = _module_coverage(scored)
    missing_detail_counts = {
        "涨停活跃缺失": int(scored["score_detail"].str.contains("近20日涨停次数缺失", na=False).sum()),
        "板块热度缺失": int(scored["score_detail"].str.contains("所属板块涨幅排名缺失", na=False).sum()),
        "市场环境缺失": int(scored["score_detail"].str.contains("上证收盘/MA5缺失", na=False).sum()),
        "买点位置缺失": int(scored["score_detail"].str.contains("20日最高价数据不足", na=False).sum()),
    }
    historical = _load_historical_summary()
    _write_report(scored, coverage, missing_detail_counts, historical, candidate_source)
    print(f"实时评分输入审计完成：{REPORT_PATH}")
    print("；".join(f"{module}={item['complete']}/{item['total']}" for module, item in coverage.items()))
    return 0


def _load_realtime_candidates() -> tuple[pd.DataFrame, str]:
    if not LATEST_CANDIDATES_PATH.exists():
        return pd.DataFrame(), ""
    data = pd.read_csv(LATEST_CANDIDATES_PATH, dtype={"code": "string"})
    source_types = data.get("source_type", pd.Series(index=data.index, dtype="object"))
    strategy = data[source_types.eq("strategy_candidate")].copy()
    if not strategy.empty:
        data = strategy
        source = "策略候选池"
    else:
        # A downgraded source deliberately produces no formal strategy pool.
        # Audit the actual rows that will be scored and shown as active
        # watchlist instead of manufacturing a zero-row report.
        source = "活跃观察池（当前行情源降级，正式策略池为空）"
    data["code"] = data["code"].astype("string").str.zfill(6)
    return data, source


def _latest_daily_as_spot() -> pd.DataFrame:
    paths = sorted(DAILY_DIR.glob("daily_*.csv"))
    if not paths:
        return pd.DataFrame(columns=["code", "pct_chg"])
    raw = pd.read_csv(paths[-1], dtype={"ts_code": "string"})
    return pd.DataFrame(
        {
            "code": raw.get("ts_code", pd.Series(index=raw.index, dtype="string")).astype("string").str.extract(r"(\d{6})", expand=False),
            "pct_chg": pd.to_numeric(raw.get("pct_chg"), errors="coerce"),
        }
    )


def _market_score_fields() -> dict[str, object]:
    if not LATEST_STATUS_PATH.exists():
        return {}
    status = json.loads(LATEST_STATUS_PATH.read_text(encoding="utf-8"))
    environment = status.get("market_environment_detail") or {}
    index_quotes = pd.DataFrame(environment.get("index_quotes") or [])
    return build_realtime_market_score_fields(environment, index_quotes, MARKET_DIR)


def _module_coverage(scored: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for module, groups in MODULE_FIELD_GROUPS.items():
        group_coverage = []
        for alternatives in groups:
            available = pd.Series(False, index=scored.index)
            for field in alternatives:
                values = scored.get(field, pd.Series(index=scored.index, dtype="object"))
                available = available | values.notna()
            group_coverage.append(int(available.sum()))
        result[module] = {
            "complete": min(group_coverage) if group_coverage else 0,
            "total": len(scored),
        }
    return result


def _load_historical_summary() -> dict[str, object]:
    if not BACKTEST_SUMMARY_PATH.exists():
        return {}
    return json.loads(BACKTEST_SUMMARY_PATH.read_text(encoding="utf-8"))


def _write_report(
    scored: pd.DataFrame,
    coverage: dict[str, dict[str, int]],
    missing_detail_counts: dict[str, int],
    historical: dict[str, object],
    candidate_source: str,
) -> None:
    history_coverage = historical.get("coverage", {}) if historical else {}
    historical_enrichment = historical.get("historical_enrichment", {}) if historical else {}
    historical_market = historical.get("historical_market", {}) if historical else {}
    total = len(scored)
    lines = [
        "# 实时评分与历史回测评分输入一致性审计",
        "",
        "## 口径",
        "",
        f"- 实时侧：使用当前{candidate_source}快照、`data/history/daily/`、本地行业映射和本地市场缓存，重建传入 `core.scoring.score_stock()` 的 V5.3 输入。",
        "- 回测侧：使用 `data/reports/v5_4/score_v5_4_backtest_summary.json` 的全市场完整评分覆盖数据。",
        "- 评分权重、阈值和 `core/scoring.py` 均未修改。",
        "",
        "## 实时 score_detail 输入完整率",
        "",
        "| 模块 | 完整行数 | 完整率 |",
        "|---|---:|---:|",
    ]
    for module, item in coverage.items():
        rate = item["complete"] / item["total"] * 100 if item["total"] else 0
        lines.append(f"| {module} | {item['complete']}/{item['total']} | {rate:.1f}% |")
    lines.extend([
        "",
        "## score_detail 缺失提示",
        "",
    ])
    for label, count in missing_detail_counts.items():
        lines.append(f"- {label}：{count}/{total}。")
    lines.extend([
        "",
        "## 历史回测输入完整率",
        "",
        f"- V5.4 全市场完整评分行：{history_coverage.get('scored_rows', '--')}；数据不足跳过：{history_coverage.get('insufficient_rows', '--')}。",
        f"- daily_basic 缓存命中：{historical_enrichment.get('daily_basic_cache_hits', '--')}；行业缓存缺失日期：{historical_enrichment.get('industry_missing_dates', '--')}。",
        f"- 市场缓存命中：{historical_market.get('market_cache_hits', '--')}；指数完整行：{historical_market.get('rows_with_index', '--')}；市场成交额20日均值完整行：{historical_market.get('rows_with_amount_ma20', '--')}。",
        "",
        "## 结论",
        "",
        "- 本报告的实时完整率按 V5.3 各模块实际输入字段计算；数值为 0 的有效指标不会被误判为缺失。",
        "- 历史回测不持久化逐股票 `score_detail` 文本；历史侧完整率据其全市场成功评分行与缓存命中统计核验。",
        "- 历史回测对换手率、流通市值和行业排名缺失行直接跳过；实时侧保留缺失值并由 score_detail 明示“不评分”。",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
