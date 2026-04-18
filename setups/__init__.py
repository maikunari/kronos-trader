"""Setup detectors for the Shiller-style engine.

Each setup is an isolated module implementing the `SetupDetector` protocol
(see setups.base). The scanner (Phase E) runs all registered setups in
parallel against each ticker on each bar close.

Phase B ships: divergence reversal.
Phase C adds: two-bar momentum, diagonal breakout, V-reversal,
consolidation-under-resistance, weak-bounce.
"""
