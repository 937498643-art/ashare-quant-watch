"""Build resumable daily market-environment cache files from local daily data.

The builder never changes the existing daily, daily_basic, or industry caches.
It obtains Shanghai Composite history once from AKShare's Sina endpoint, while
market breadth is calculated only from a sufficiently complete local daily
universe.  A partial stock universe is never presented as all-market breadth.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DAILY_DIR = PROJECT_ROOT / "data" / "history" / "daily"
MARKET_DIR = PROJECT_ROOT / "data" / "history" / "market"
REQUIRED_FIELDS = (
    "trade_date",
    "sh_close",
    "sh_pct_chg",
    "sh_ma5",
    "sh_ma10",
    "up_count",
    "down_count",
    "up_ratio",
    "limit_up_count",
    "limit_down_count",
    "market_amount",
)
STATUS_FIELDS = ("breadth_universe_count", "breadth_complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local historical market-environment cache")
    parser.add_argument("--start-date", help="YYYY-MM-DD; defaults to the first local daily-cache date")
    parser.add_argument("--end-date", help="YYYY-MM-DD; defaults to the last local daily-cache date")
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--market-dir", type=Path, default=MARKET_DIR)
    parser.add_argument(
        "--min-breadth-universe",
        type=int,
        default=4000,
        help="Minimum local stock rows required before calculating all-market breadth",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even already-complete market files")
    return parser.parse_args()


def _daily_dates(daily_dir: Path) -> list[str]:
    return sorted(path.stem.removeprefix("daily_") for path in daily_dir.glob("daily_*.csv") if path.stem.removeprefix("daily_").isdigit())


def _compact_date(value: str | None) -> str | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.strftime("%Y%m%d")


def _fetch_shanghai_index(start: str, end: str) -> pd.DataFrame:
    """Fetch one Shanghai Composite history series and derive close-based fields."""
    import akshare as ak  # Imported lazily so cache inspection remains usable offline.

    raw = ak.stock_zh_index_daily(symbol="sh000001")
    if raw.empty:
        raise RuntimeError("AKShare 新浪上证指数历史接口返回空数据")
    data = raw.copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y%m%d")
    data["sh_close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["trade_date", "sh_close"]).sort_values("trade_date")
    data = data[(data["trade_date"] >= start) & (data["trade_date"] <= end)].copy()
    data["sh_pct_chg"] = data["sh_close"].pct_change(fill_method=None).mul(100)
    # Calculate MA values from the complete index series to retain valid values
    # when the requested range begins in the middle of a trading calendar.
    all_data = raw.copy()
    all_data["trade_date"] = pd.to_datetime(all_data["date"], errors="coerce").dt.strftime("%Y%m%d")
    all_data["sh_close"] = pd.to_numeric(all_data["close"], errors="coerce")
    all_data = all_data.dropna(subset=["trade_date", "sh_close"]).sort_values("trade_date")
    all_data["sh_pct_chg"] = all_data["sh_close"].pct_change(fill_method=None).mul(100)
    all_data["sh_ma5"] = all_data["sh_close"].rolling(5, min_periods=5).mean()
    all_data["sh_ma10"] = all_data["sh_close"].rolling(10, min_periods=10).mean()
    return all_data[(all_data["trade_date"] >= start) & (all_data["trade_date"] <= end)][
        ["trade_date", "sh_close", "sh_pct_chg", "sh_ma5", "sh_ma10"]
    ].copy()


def _limit_threshold(ts_code: str) -> float:
    code = "".join(ch for ch in str(ts_code) if ch.isdigit())[:6]
    if code.startswith(("300", "301", "688", "689")):
        return 19.5
    if code.startswith(("4", "8", "9")):
        return 29.5
    return 9.5


def _market_breadth(daily_path: Path, min_universe: int) -> dict[str, Any]:
    try:
        frame = pd.read_csv(daily_path, encoding="utf-8-sig", dtype={"ts_code": str})
    except Exception:
        frame = pd.DataFrame()
    required = {"ts_code", "pct_chg", "amount"}
    if not required.issubset(frame.columns):
        return _missing_breadth(0)
    frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    usable = frame.dropna(subset=["ts_code", "pct_chg"])
    count = int(len(usable))
    if count < min_universe:
        return _missing_breadth(count)
    thresholds = usable["ts_code"].map(_limit_threshold)
    up_count = int((usable["pct_chg"] > 0).sum())
    down_count = int((usable["pct_chg"] < 0).sum())
    directional_count = up_count + down_count
    return {
        "up_count": up_count,
        "down_count": down_count,
        "up_ratio": (
            float(up_count / directional_count)
            if directional_count
            else pd.NA
        ),
        "limit_up_count": int((usable["pct_chg"] >= thresholds).sum()),
        "limit_down_count": int((usable["pct_chg"] <= -thresholds).sum()),
        "market_amount": float(usable["amount"].sum(min_count=1)),
        "breadth_universe_count": count,
        "breadth_complete": True,
    }


def _missing_breadth(universe_count: int) -> dict[str, Any]:
    return {
        "up_count": pd.NA,
        "down_count": pd.NA,
        "up_ratio": pd.NA,
        "limit_up_count": pd.NA,
        "limit_down_count": pd.NA,
        "market_amount": pd.NA,
        "breadth_universe_count": universe_count,
        "breadth_complete": False,
    }


def _is_complete_market_file(path: Path) -> bool:
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return False
    if len(data) != 1 or not set(REQUIRED_FIELDS).issubset(data.columns):
        return False
    return bool(data.get("breadth_complete", pd.Series([False])).iloc[0])


def main() -> int:
    args = parse_args()
    local_dates = _daily_dates(args.daily_dir)
    if not local_dates:
        print(f"未找到本地日线缓存：{args.daily_dir}")
        return 1
    start = _compact_date(args.start_date) or local_dates[0]
    end = _compact_date(args.end_date) or local_dates[-1]
    if start > end:
        raise ValueError("start-date must not be later than end-date")
    args.market_dir.mkdir(parents=True, exist_ok=True)

    index = _fetch_shanghai_index(start, end).set_index("trade_date")
    expected_dates = [date for date in local_dates if start <= date <= end and date in index.index]
    created = 0
    skipped_complete = 0
    incomplete = 0
    errors: list[str] = []
    for trade_date in expected_dates:
        output_path = args.market_dir / f"market_{trade_date}.csv"
        if not args.force and _is_complete_market_file(output_path):
            skipped_complete += 1
            continue
        index_row = index.loc[trade_date]
        breadth = _market_breadth(args.daily_dir / f"daily_{trade_date}.csv", args.min_breadth_universe)
        output = {
            "trade_date": trade_date,
            "sh_close": index_row["sh_close"],
            "sh_pct_chg": index_row["sh_pct_chg"],
            "sh_ma5": index_row["sh_ma5"],
            "sh_ma10": index_row["sh_ma10"],
            **breadth,
        }
        pd.DataFrame([output], columns=[*REQUIRED_FIELDS, *STATUS_FIELDS]).to_csv(output_path, index=False, encoding="utf-8-sig")
        created += 1
        incomplete += int(not breadth["breadth_complete"])

    status = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "AKShare 新浪上证指数历史 + 本地 data/history/daily 市场宽度",
        "requested_range": [start, end],
        "expected_dates": len(expected_dates),
        "created_files": created,
        "skipped_complete_files": skipped_complete,
        "incomplete_breadth_files_rebuilt": incomplete,
        "min_breadth_universe": args.min_breadth_universe,
        "errors": errors,
        "limitation": "仅当本地 daily 单日覆盖达到门槛时计算全市场宽度；覆盖不足时宽度字段保持缺失，绝不以回测股票子集替代全市场。",
    }
    (args.market_dir / "market_cache_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
