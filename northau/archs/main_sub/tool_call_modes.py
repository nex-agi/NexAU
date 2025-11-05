"""Shared constants and helpers for tool call modes."""

from __future__ import annotations

VALID_TOOL_CALL_MODES: set[str] = {"xml", "openai", "anthorpic"}
STRUCTURED_TOOL_CALL_MODES: set[str] = {"openai", "anthorpic"}
DEFAULT_TOOL_CALL_MODE = "openai"


def normalize_tool_call_mode(mode: str | None) -> str:
    """Normalize a tool_call_mode string and validate it."""
    normalized = (mode or DEFAULT_TOOL_CALL_MODE).lower()
    if normalized not in VALID_TOOL_CALL_MODES:
        raise ValueError(
            "tool_call_mode must be one of 'xml', 'openai', or 'anthorpic'",
        )
    return normalized
