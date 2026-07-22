#!/usr/bin/env python3
"""Apply optional V5.6 entry filters to the V5.5 core pool offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.buy_filter import BuyFilterConfig, evaluate_buy_filter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V5.6 entry-filter backtest offline")
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data" / "history" / "daily")
    parser.add_argument("--v5-5-trades", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_5" / "score_v5_5_core_pool_trades.csv")
    parser.add_argument("--v5-5-summary", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_5" / "score_v5_5_backtest_summary.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "reports" / "v5_6")
    parser.add_argument("--disable-signal-change", action="store_true")
    parser.add_argument("--disable-position-risk", action="store_true")
    parser.add_argument("--disable-volume-anomaly", action="store_true")
    parser.add_argument("--disable-intraday-reversal", action="store_true")
    parser.add_argument("--max-soft-risk", type=int, default=6)
    return parser.parse_args()


def _code(value: Any) -> str:
    return str(value).split(".")[0].zfill(6)


def _load_histories(daily_dir: Path, codes: set[str]) -> dict[str, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for path in sorted(daily_dir.glob("daily_*.csv")):
        raw = pd.read_csv(path, dtype={"ts_code": str})
        if raw.empty:
            continue
        data = pd.DataFrame(
            {
                "code": raw["ts_code"].map(_code),
                "date": pd.to_datetime(raw.get("trade_date"), format="%Y%m%d", errors="coerce"),
                "open": pd.to_numeric(raw.get("open"), errors="coerce"),
                "high": pd.to_numeric(raw.get("high"), errors="coerce"),
                "low": pd.to_numeric(raw.get("low"), errors="coerce"),
                "close": pd.to_numeric(raw.get("close"), errors="coerce"),
                "volume": pd.to_numeric(raw.get("vol"), errors="coerce"),
            }
        )
        data = data[data["code"].isin(codes)].dropna(subset=["date", "close"])
        if not data.empty:
            parts.append(data)
    merged = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    return {
        code: group.drop(columns="code").sort_values("date").reset_index(drop=True)
        for code, group in merged.groupby("code", sort=True)
    }


def _stats(data: pd.DataFrame, column: str, days: int) -> dict[str, Any]:
    subset = data[data["holding_days"] == days]
    values = pd.to_numeric(subset[column], errors="coerce").dropna()
    if values.empty:
        return {"sample_count": 0, "average_return_pct": None, "win_rate_pct": None, "profit_loss_ratio": None, "max_loss_pct": None}
    wins, losses = values[values > 0], values[values < 0]
    return {
        "sample_count": int(len(values)),
        "average_return_pct": float(values.mean()),
        "win_rate_pct": float((values > 0).mean() * 100),
        "profit_loss_ratio": float(wins.mean() / abs(losses.mean())) if not wins.empty and not losses.empty else None,
        "max_loss_pct": float(values.min()),
    }


def _pct(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}%"


def _ratio(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _overnight_stats(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"count": 0, "average_open_return_pct": None, "average_low_return_pct": None, "average_high_return_pct": None, "next_day_gap_down_probability_pct": None}
    return {
        "count": int(len(signals)),
        "average_open_return_pct": float(signals["next_open_return_pct"].mean()),
        "average_low_return_pct": float(signals["next_low_return_pct"].mean()),
        "average_high_return_pct": float(signals["next_high_return_pct"].mean()),
        "next_day_gap_down_probability_pct": float((signals["next_open_return_pct"] < 0).mean() * 100),
    }


def _write_report(output_dir: Path, report: dict[str, Any], trades: pd.DataFrame, signals: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score_v5_6_backtest_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    trades.to_csv(output_dir / "score_v5_6_core_pool_trades.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(output_dir / "score_v5_6_signals.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# V5.6 买入准入过滤全市场回测报告", "",
        "## 方法", "",
        "- 基础评分及V5.5交易质量层均不修改。",
        "- V5.6仅对V5.5核心池应用独立买入准入过滤：当日涨幅、20日位置、成交量异常、冲高回落。",
        "- 硬过滤：当日涨幅>9%。软过滤：位置/巨量/冲高回落风险累计达到阈值。",
        f"- 模块配置：{report['module_config']}；network_requests：0。", "",
        "## 信号数量", "",
        f"- V5.5核心池：{report['v5_5_signal_count']}。",
        f"- V5.6准入后核心池：{report['v5_6_signal_count']}。",
        f"- 硬过滤数量：{report['hard_reject_count']}；软过滤数量：{report['soft_reject_count']}。", "",
        "## 10日收益比较", "",
        "| 版本 | 样本数 | 平均收益 | 胜率 | 盈亏比 | 最大亏损 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| V5.5无止损 | {report['v5_5_stats']['sample_count']} | {_pct(report['v5_5_stats']['average_return_pct'])} | {_pct(report['v5_5_stats']['win_rate_pct'])} | {_ratio(report['v5_5_stats']['profit_loss_ratio'])} | {_pct(report['v5_5_stats']['max_loss_pct'])} |",
        f"| V5.6无止损 | {report['v5_6_raw_stats']['sample_count']} | {_pct(report['v5_6_raw_stats']['average_return_pct'])} | {_pct(report['v5_6_raw_stats']['win_rate_pct'])} | {_ratio(report['v5_6_raw_stats']['profit_loss_ratio'])} | {_pct(report['v5_6_raw_stats']['max_loss_pct'])} |",
        f"| V5.6 -8%止损 | {report['v5_6_stop_stats']['sample_count']} | {_pct(report['v5_6_stop_stats']['average_return_pct'])} | {_pct(report['v5_6_stop_stats']['win_rate_pct'])} | {_ratio(report['v5_6_stop_stats']['profit_loss_ratio'])} | {_pct(report['v5_6_stop_stats']['max_loss_pct'])} |",
        "",
        "## 隔夜风险（次日）", "",
        "| 版本 | 样本数 | 次日开盘平均收益 | 次日最低平均跌幅 | 次日最高平均涨幅 | 次日低开概率 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| V5.5核心池 | {report['v5_5_overnight']['count']} | {_pct(report['v5_5_overnight']['average_open_return_pct'])} | {_pct(report['v5_5_overnight']['average_low_return_pct'])} | {_pct(report['v5_5_overnight']['average_high_return_pct'])} | {_pct(report['v5_5_overnight']['next_day_gap_down_probability_pct'])} |",
        f"| V5.6准入后 | {report['v5_6_overnight']['count']} | {_pct(report['v5_6_overnight']['average_open_return_pct'])} | {_pct(report['v5_6_overnight']['average_low_return_pct'])} | {_pct(report['v5_6_overnight']['average_high_return_pct'])} | {_pct(report['v5_6_overnight']['next_day_gap_down_probability_pct'])} |",
        "",
    ]
    (output_dir / "score_v5_6_backtest_report.md").write_text("\n".join(lines), encoding="utf-8")
    comparison = [
        "# V5.5 vs V5.6 买入准入过滤对比", "",
        "| 指标 | V5.5 | V5.6 |",
        "|---|---:|---:|",
        f"| 85分以上/核心池样本 | {report['v5_5_signal_count']} | {report['v5_6_signal_count']} |",
        f"| 10日平均收益 | {_pct(report['v5_5_stats']['average_return_pct'])} | {_pct(report['v5_6_raw_stats']['average_return_pct'])} |",
        f"| 10日胜率 | {_pct(report['v5_5_stats']['win_rate_pct'])} | {_pct(report['v5_6_raw_stats']['win_rate_pct'])} |",
        f"| 10日盈亏比 | {_ratio(report['v5_5_stats']['profit_loss_ratio'])} | {_ratio(report['v5_6_raw_stats']['profit_loss_ratio'])} |",
        f"| 10日最大亏损 | {_pct(report['v5_5_stats']['max_loss_pct'])} | {_pct(report['v5_6_raw_stats']['max_loss_pct'])} |",
        f"| 次日低开概率 | {_pct(report['v5_5_overnight']['next_day_gap_down_probability_pct'])} | {_pct(report['v5_6_overnight']['next_day_gap_down_probability_pct'])} |",
        "",
    ]
    (output_dir / "V5.5_vs_V5.6_comparison.md").write_text("\n".join(comparison), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = BuyFilterConfig(
        signal_change_enabled=not args.disable_signal_change,
        position_risk_enabled=not args.disable_position_risk,
        volume_anomaly_enabled=not args.disable_volume_anomaly,
        intraday_reversal_enabled=not args.disable_intraday_reversal,
        max_soft_risk=args.max_soft_risk,
    )
    base_trades = pd.read_csv(args.v5_5_trades, encoding="utf-8-sig")
    signal_keys = base_trades[["code", "signal_date"]].drop_duplicates().copy()
    signal_keys["code"] = signal_keys["code"].map(_code)
    histories = _load_histories(args.daily_dir, set(signal_keys["code"]))
    evaluated: list[dict[str, Any]] = []
    for item in signal_keys.itertuples(index=False):
        history = histories.get(item.code)
        signal_date = pd.Timestamp(item.signal_date)
        if history is None:
            continue
        matches = history.index[history["date"] == signal_date].tolist()
        if not matches or matches[0] < 20 or matches[0] + 1 >= len(history):
            continue
        index = matches[0]
        current, prior, next_day = history.iloc[index], history.iloc[index - 1], history.iloc[index + 1]
        high20 = float(history.loc[index - 19 : index, "high"].max())
        ma20_volume = float(history.loc[index - 20 : index - 1, "volume"].mean())
        close_change = (float(current.close) / float(prior.close) - 1) * 100
        high_change = (float(current.high) / float(prior.close) - 1) * 100
        entry = float(current.close)
        values = {
            "signal_day_change": close_change,
            "distance_to_20d_high_pct": (high20 - entry) / high20 * 100 if high20 > 0 else float("nan"),
            "volume_to_ma20_ratio": float(current.volume) / ma20_volume if ma20_volume > 0 else float("nan"),
            "intraday_reversal_pct": high_change - close_change,
        }
        result = evaluate_buy_filter(values, config)
        evaluated.append(
            {
                "code": item.code,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                **values,
                **result,
                "next_open_return_pct": (float(next_day.open) / entry - 1) * 100,
                "next_low_return_pct": (float(next_day.low) / entry - 1) * 100,
                "next_high_return_pct": (float(next_day.high) / entry - 1) * 100,
            }
        )
    signals = pd.DataFrame(evaluated)
    selected = signals[signals["allowed"]].copy()
    merged = base_trades.copy()
    merged["code"] = merged["code"].map(_code)
    trades = merged.merge(selected[["code", "signal_date"]], on=["code", "signal_date"], how="inner")
    v5_5_stats = _stats(base_trades, "raw_return_pct", 10)
    report = {
        "module_config": {
            "signal_change": config.signal_change_enabled,
            "position_risk": config.position_risk_enabled,
            "volume_anomaly": config.volume_anomaly_enabled,
            "intraday_reversal": config.intraday_reversal_enabled,
            "max_soft_risk": config.max_soft_risk,
        },
        "network_requests": 0,
        "v5_5_signal_count": int(len(signal_keys)),
        "v5_6_signal_count": int(len(selected)),
        "hard_reject_count": int(signals["hard_reject"].sum()),
        "soft_reject_count": int((~signals["allowed"] & ~signals["hard_reject"]).sum()),
        "v5_5_stats": v5_5_stats,
        "v5_6_raw_stats": _stats(trades, "raw_return_pct", 10),
        "v5_6_stop_stats": _stats(trades, "stop_return_pct", 10),
        "v5_5_overnight": _overnight_stats(signals),
        "v5_6_overnight": _overnight_stats(selected),
    }
    _write_report(args.output_dir, report, trades, signals)
    print(f"V5.6完成：V5.5信号={len(signal_keys)}，V5.6准入={len(selected)}。")
    print(f"报告：{args.output_dir / 'score_v5_6_backtest_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
