"""Tests for alerts.py — fully mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from alerts import Alerts


def _session(status_ok: bool = True, raise_exc: Exception | None = None):
    session = MagicMock(spec=requests.Session)
    if raise_exc is not None:
        session.post.side_effect = raise_exc
        return session
    resp = MagicMock()
    resp.raise_for_status.return_value = None if status_ok else MagicMock()
    session.post.return_value = resp
    return session


def test_disabled_when_no_webhook_url(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    alerts = Alerts(webhook_url=None)
    assert not alerts.enabled
    assert alerts.send("hello") is False


def test_slack_payload_uses_text_key():
    session = _session()
    a = Alerts(webhook_url="https://hooks.slack.com/services/xxx", session=session)
    assert a.send("hi")
    _, kwargs = session.post.call_args
    assert kwargs["json"] == {"text": "ℹ️ hi"}


def test_discord_payload_uses_content_key():
    session = _session()
    a = Alerts(webhook_url="https://discord.com/api/webhooks/xxx", session=session)
    a.send("warning", level="warn")
    _, kwargs = session.post.call_args
    assert kwargs["json"]["content"].startswith("⚠️")


def test_generic_url_uses_both_keys():
    session = _session()
    a = Alerts(webhook_url="https://example.com/hook", session=session)
    a.send("hello")
    _, kwargs = session.post.call_args
    assert "content" in kwargs["json"] and "text" in kwargs["json"]


def test_network_failure_returns_false_and_does_not_raise():
    a = Alerts(
        webhook_url="https://example.com/hook",
        session=_session(raise_exc=requests.ConnectionError("boom")),
    )
    assert a.send("test") is False
