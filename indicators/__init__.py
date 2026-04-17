"""Indicator primitives for the Shiller-style engine.

Each module exposes pure functions taking pd.Series inputs and returning
pd.Series outputs. Helpers for setup-specific logic (divergences,
two-bar rules, etc.) live here when they're indicator-local; composite
rules belong to the respective setup modules.
"""
