"""Email alerts for candidate stocks.

The module only sends read-only watchlist notifications. It does not contain
any trading, account, order, or broker operation.
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


class EmailAlert:
    """Send candidate stock alerts by email when enabled."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_email_config()
        self.logger = logging.getLogger(self.__class__.__name__)

    def send_candidates(
        self,
        candidates: pd.DataFrame,
        scan_time: datetime | None = None,
    ) -> bool:
        """Send candidate stock email and return whether an email was sent."""
        scan_time = scan_time or datetime.now()

        if not self.config.get("enabled", False):
            message = "邮件提醒未启用：email.enabled = false，本轮只在控制台输出。"
            print(message)
            self.logger.info(message)
            return False

        receivers = _as_list(
            self.config.get("receivers")
            or self.config.get("recipients")
            or self.config.get("to")
        )
        if not receivers:
            self.logger.warning("Email alert enabled but receivers is empty.")
            return False

        password = _resolve_password(self.config)
        if not password:
            self.logger.warning("Email alert enabled but password is missing.")
            return False

        subject = f"【A股个股盯盘提醒】发现 {len(candidates)} 只候选股"
        body = build_email_body(candidates, scan_time)

        email_message = EmailMessage()
        email_message["Subject"] = subject
        email_message["From"] = str(self.config.get("sender") or self.config.get("username") or "")
        email_message["To"] = ", ".join(receivers)
        email_message.set_content(body)

        self._send(email_message, password)
        self.logger.info("Email alert sent to %s receivers.", len(receivers))
        return True

    def _send(self, email_message: EmailMessage, password: str) -> None:
        """Send one message through configured SMTP server."""
        smtp_host = str(self.config.get("smtp_host", ""))
        smtp_port = int(self.config.get("smtp_port", 465))
        username = str(self.config.get("username", ""))
        use_ssl = bool(self.config.get("use_ssl", smtp_port == 465))

        if not smtp_host:
            raise ValueError("email.smtp_host is required when email is enabled.")
        if not username:
            raise ValueError("email.username is required when email is enabled.")

        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as smtp:
                smtp.login(username, password)
                smtp.send_message(email_message)
            return

        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(email_message)


def load_email_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load email configuration from ``config.yaml``."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing PyYAML. Run `pip install -r requirements.txt`.") from exc

    if not config_path.exists():
        return {"enabled": False}

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    alerts_config = config.get("alerts", {})
    return alerts_config.get("email", config.get("email", {"enabled": False}))


def build_email_body(candidates: pd.DataFrame, scan_time: datetime | None = None) -> str:
    """Build the plain-text candidate email body."""
    scan_time = scan_time or datetime.now()
    lines = [
        f"扫描时间：{scan_time:%Y-%m-%d %H:%M:%S}",
        "",
        "本邮件仅用于行情盯盘提醒，不构成任何交易指令。",
        "",
    ]

    if candidates.empty:
        lines.append("本轮未发现候选股。")
        return "\n".join(lines)

    fields = [
        ("股票代码", ("code", "代码")),
        ("股票名称", ("name", "名称")),
        ("最新价", ("price", "最新价")),
        ("涨跌幅", ("pct_chg", "涨跌幅")),
        ("成交额", ("amount", "成交额")),
        ("换手率", ("turnover", "换手率")),
        ("触发策略", ("strategy_name", "策略名称")),
        ("评分", ("score",)),
        ("等级", ("level",)),
        ("触发原因", ("reason", "触发原因")),
        ("风险提示", ("risk_summary", "risk")),
    ]

    for index, (_, row) in enumerate(candidates.iterrows(), start=1):
        lines.append(f"候选股 {index}")
        for label, keys in fields:
            value = _value(row, *keys)
            lines.append(f"{label}：{_format_value(value)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _resolve_password(config: dict[str, Any]) -> str:
    """Read email password from config or environment variables."""
    _load_dotenv_if_available()

    password = str(config.get("password") or "")
    if password:
        return password

    password_env = str(config.get("password_env") or config.get("password_env_var") or "")
    if password_env:
        return os.getenv(password_env, "")

    return os.getenv("QUANT_STOCK_WATCH_EMAIL_PASSWORD", "")


def _load_dotenv_if_available() -> None:
    """Load ``.env`` when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(DEFAULT_DOTENV_PATH)


def _as_list(value: Any) -> list[str]:
    """Normalize receiver config to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _value(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    """Get the first non-empty value from a row."""
    for key in keys:
        value = row.get(key) if hasattr(row, "get") else None
        if value is not None and not pd.isna(value):
            return value
    return ""


def _format_value(value: Any) -> str:
    """Format scalar values for plain-text email."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
