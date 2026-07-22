"""Risk boundary checks."""

from __future__ import annotations


FORBIDDEN_TRADING_KEYWORDS = (
    "买入",
    "卖出",
    "撤单",
    "委托",
    "银证转账",
    "交易密码",
    "验证码",
)


def assert_read_only_purpose(action_text: str) -> None:
    """Raise if a future module tries to cross the no-trading boundary."""
    if any(keyword in action_text for keyword in FORBIDDEN_TRADING_KEYWORDS):
        raise ValueError(f"Forbidden trading-related action: {action_text}")
