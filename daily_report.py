#!/usr/bin/env python3
"""
daily_report.py
Generates a daily summary of backtest/paper trading results and
sends it to Discord via OpenClaw.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE = Path(__file__).parent


def read_optimizer_log():
    log_path = BASE / "optimizer_history.jsonl"
    if not log_path.exists():
        return []
    entries = []
    with open(log_path) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                pass
    return entries


def run_backtest_summary():
    """Run a quick backtest on last 30 days and return key metrics."""
    import yaml
    config_path = BASE / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    config["backtest"]["start_date"] = start.strftime("%Y-%m-%d")
    config["backtest"]["end_date"] = end.strftime("%Y-%m-%d")

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        import yaml
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    try:
        venv_python = BASE / "venv" / "bin" / "python"
        result = subprocess.run(
            [str(venv_python), str(BASE / "backtest.py"), "--config", tmp_path, "--plot", "/tmp/daily_equity.png"],
            capture_output=True, text=True, timeout=180, cwd=str(BASE)
        )
        output = result.stdout + result.stderr
        # Extract key lines
        lines = [l for l in output.split("\n") if any(k in l for k in [
            "Total return", "Win rate", "Total trades", "Profit factor",
            "Max drawdown", "Sharpe", "BACKTEST REPORT"
        ])]
        return "\n".join(lines)
    except Exception as e:
        return f"Backtest failed: {e}"
    finally:
        os.unlink(tmp_path)


def build_message():
    now = datetime.now(timezone.utc)
    lines = [
        f"**Kronos Trader — Daily Report** {now.strftime('%Y-%m-%d')}",
        "",
        "**30-Day Rolling Backtest (stat fallback models):**",
    ]

    summary = run_backtest_summary()
    lines.append(f"```\n{summary}\n```")

    # Optimizer history
    opt_entries = read_optimizer_log()
    recent = [e for e in opt_entries
              if (now - datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))).days < 7]
    if recent:
        lines.append(f"\n**Optimizer retunes (last 7 days): {len(recent)}**")
        for e in recent[-3:]:
            status = "✅ applied" if e["applied"] else "⏭️ skipped (OOS PF too low)"
            lines.append(
                f"- {e['timestamp'][:10]} | trigger={e['trigger']} | "
                f"IS_PF={e['in_sample_pf']:.2f} OOS_PF={e['out_of_sample_pf']:.2f} | {status}"
            )
    else:
        lines.append("\n**Optimizer:** No retunes in last 7 days")

    lines.append("\n_Note: Install Kronos + TimesFM for real ML predictions. Stat fallback only._")
    return "\n".join(lines)


if __name__ == "__main__":
    msg = build_message()
    print(msg)

    # Send to Discord via openclaw CLI
    channel = os.getenv("REPORT_DISCORD_CHANNEL", "jerry")
    try:
        subprocess.run(
            ["openclaw", "message", "send", "--channel", channel, "--message", msg],
            check=True, timeout=15
        )
        print(f"Sent to #{channel}")
    except Exception as e:
        print(f"Discord send failed: {e} — report printed above")
