"""Compatibility entry point for the offline history-cache builder.

Historical cache construction no longer calls Tushare or Eastmoney push2his.
Use ``scripts/build_history_cache.py`` directly for new workflows.
"""

from __future__ import annotations

from build_history_cache import main


if __name__ == "__main__":
    raise SystemExit(main())
