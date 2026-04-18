"""Signal validation framework.

Answers the question "does a setup detector catch real pops?" without
needing a full P&L backtest. Four stages:

    labeler.py   -> identifies historical pops (20%+ moves) as ground truth
    matcher.py   -> matches trigger stream against labeled pops
    capture.py   -> simulates single-trade outcome per match
    report.py    -> aggregates recall/precision/capture per setup/direction

See tasks/shiller_bot_architecture.md §9 for detailed design.
"""
