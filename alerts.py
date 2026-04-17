"""
alerts.py
Thin webhook alerting for Discord / Slack / Pushover / any URL.

Pick one channel via env: ALERT_WEBHOOK_URL points at Discord or Slack
incoming-webhook. Disabled when unset — callers never need to branch.

Used by risk_manager kill-switch trips, executor skip-on-anomaly, and
the dashboard's nightly PnL summary.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0


class Alerts:
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "kronos-sniper",
        session: Optional[requests.Session] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.webhook_url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
        self.username = username
        self.session = session or requests.Session()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, message: str, *, level: str = "info") -> bool:
        """Fire-and-forget send. Returns True on success, False otherwise.

        Any failure is logged but not raised — alerts must never break
        the hot path of a trading system.
        """
        if not self.enabled:
            return False
        payload = self._payload_for_url(self.webhook_url, message, level)
        try:
            resp = self.session.post(self.webhook_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.warning("Alert webhook failed: %s", exc)
            return False

    @staticmethod
    def _payload_for_url(url: str, message: str, level: str) -> dict:
        prefix = {"info": "ℹ️", "warn": "⚠️", "error": "🛑", "success": "✅"}.get(level, "")
        text = f"{prefix} {message}".strip()
        if "slack.com" in url:
            return {"text": text}
        if "discord.com" in url or "discordapp.com" in url:
            return {"content": text, "username": "kronos-sniper"}
        # Fallback: generic content + text keys so most webhook receivers work
        return {"content": text, "text": text}
