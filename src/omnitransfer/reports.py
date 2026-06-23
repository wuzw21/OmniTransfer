"""Report helpers."""

from __future__ import annotations


def format_percent(value: float) -> str:
    """Format a ratio as a two-decimal percentage."""

    return f"{100.0 * float(value):.2f}%"
