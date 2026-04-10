"""Agent tool specific formatter.

RFC-0017: Tool output flattening

Formats sub-agent results into a concise summary plus the sub-agent body, so the
parent LLM sees the useful answer directly instead of a nested dict shell.
"""

from __future__ import annotations

import json
from typing import cast

from . import ToolFormatterContext


def format_agent_tool_output(context: ToolFormatterContext) -> object:
    """Flatten Agent tool results into a readable text payload."""

    raw_output = context.tool_output
    if not isinstance(raw_output, dict):
        return str(raw_output)

    raw_output_dict = cast(dict[object, object], raw_output)
    output = {str(key): value for key, value in raw_output_dict.items()}
    status = output.get("status")
    is_error = context.is_error or status == "error" or "error" in output

    summary = _build_summary(output, is_error=is_error)
    body = _pick_body(output, is_error=is_error)
    if body:
        return f"{summary}\n\n{body}"
    return summary


def _build_summary(output: dict[str, object], *, is_error: bool) -> str:
    parts: list[str] = []
    sub_agent_name = output.get("sub_agent_name")
    sub_agent_id = output.get("sub_agent_id")
    if isinstance(sub_agent_name, str) and sub_agent_name:
        parts.append(f"sub_agent_name: {sub_agent_name}")
    if isinstance(sub_agent_id, str) and sub_agent_id:
        parts.append(f"sub_agent_id: {sub_agent_id}")

    prefix = "Sub-agent failed" if is_error else "Sub-agent finished"
    if parts:
        return f"{prefix} ({', '.join(parts)})."
    return f"{prefix}."


def _pick_body(output: dict[str, object], *, is_error: bool) -> str:
    keys = ("error", "result", "message", "content") if is_error else ("result", "content", "message")
    for key in keys:
        value = output.get(key)
        rendered = _render_body_value(value)
        if rendered:
            return rendered
    return ""


def _render_body_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


__all__ = ["format_agent_tool_output"]
