"""Tool output formatter registry and shared helpers.

RFC-0017: Tool output flattening

Provides the formatter context, builtin formatter resolution, and image-like
output detection used by the tool execution pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from nexau.archs.main_sub.utils import import_from_string
from nexau.core.messages import ToolOutputImage


@dataclass(frozen=True)
class ToolFormatterContext:
    """Execution context passed to tool formatters.

    RFC-0017: formatter 在 after_tool middleware 之前执行

    tool_output 保留原始 runtime 结果；formatter 负责将其转换为适合 LLM
    消费的 llm-facing output。
    """

    tool_name: str
    tool_input: dict[str, object]
    tool_output: object
    tool_call_id: str | None
    is_error: bool


ToolFormatter = Callable[[ToolFormatterContext], object]


def resolve_tool_formatter(formatter: str | ToolFormatter | None) -> ToolFormatter:
    """Resolve a formatter spec to a callable.

    RFC-0017: 支持 builtin alias（首期仅 xml）与 import path
    """

    if formatter is None:
        from .xml import format_tool_output_as_xml

        return format_tool_output_as_xml

    if callable(formatter):
        return formatter

    if formatter == "xml":
        from .xml import format_tool_output_as_xml

        return format_tool_output_as_xml

    imported = import_from_string(formatter)
    if not callable(imported):
        raise ValueError(f"Tool formatter '{formatter}' did not resolve to a callable")
    return cast(ToolFormatter, imported)


def is_image_like_tool_output(value: object) -> bool:
    """Return whether a tool output should stay on the multimodal path."""

    if isinstance(value, ToolOutputImage):
        return True

    if isinstance(value, list):
        value_list = cast(list[object], value)
        return any(is_image_like_tool_output(item) for item in value_list)

    if isinstance(value, dict):
        value_dict = cast(dict[str, object], value)
        value_type = value_dict.get("type")
        if isinstance(value_type, str) and value_type in {"input_image", "image"}:
            image_url = value_dict.get("image_url")
            url = value_dict.get("url")
            base64_data = value_dict.get("base64")
            return isinstance(image_url, str) or isinstance(url, str) or isinstance(base64_data, str)

        nested_content = value_dict.get("content")
        if isinstance(nested_content, (dict, list, ToolOutputImage)) and is_image_like_tool_output(cast(object, nested_content)):
            return True

        nested_result = value_dict.get("result")
        if isinstance(nested_result, (dict, list, ToolOutputImage)) and is_image_like_tool_output(cast(object, nested_result)):
            return True

    return False


__all__ = [
    "ToolFormatter",
    "ToolFormatterContext",
    "is_image_like_tool_output",
    "resolve_tool_formatter",
]
