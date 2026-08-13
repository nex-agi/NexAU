# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lossless MCP tool discovery and result adaptation.

The official MCP SDK owns wire parsing and validation. This module only
bridges its public Tool and CallToolResult values into NexAU's two tool result
views:

* raw_output keeps every content block and top-level field JSON-compatible;
* llm_output maps text and image blocks natively and summarizes content types
  that NexAU's current message model cannot represent.

Future SDK content-block models are deliberately handled through model_dump
rather than a closed type union, so an SDK upgrade cannot make the adapter fail
merely because a new block type was added.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, cast

from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool

from nexau.archs.tool.formatters import ToolFormatterContext

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
LLMContentPart = dict[str, JsonValue]
LLMOutput = str | list[LLMContentPart]


class MCPToolListClient(Protocol):
    """The public subset of the SDK Client used for tool discovery."""

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult: ...


@dataclass(frozen=True)
class MCPAdaptedResult:
    """The raw and model-facing views of one official SDK tool result."""

    raw_output: dict[str, JsonValue]
    llm_output: LLMOutput
    is_error: bool

    @property
    def tool_output(self) -> dict[str, JsonValue]:
        """Return the lossless result in NexAU pipeline-compatible form.

        MCP tool failures are ordinary CallToolResult values rather than
        protocol exceptions. NexAU's current executor recognizes status/error,
        so this view adds those derived keys without removing or rewriting any
        SDK field. Consumers that only need the exact result use raw_output.
        """
        output = dict(self.raw_output)
        if self.is_error:
            output["status"] = "error"
            output["error"] = _error_summary(self.llm_output)
        return output


async def list_all_tools(client: MCPToolListClient) -> list[MCPTool]:
    """Drain every tools/list page returned by an SDK client.

    Cursors are passed back verbatim. Tool names are the protocol identity, so
    a malformed server repeating a name does not create duplicate NexAU tools;
    the latest definition replaces the earlier one while retaining its
    first-seen position.

    A repeated cursor is rejected instead of spinning forever.
    """
    cursor: str | None = None
    seen_cursors: set[str] = set()
    tools_by_name: dict[str, MCPTool] = {}

    while True:
        page = await client.list_tools(cursor=cursor)
        for tool in page.tools:
            tools_by_name[tool.name] = tool

        next_cursor = page.next_cursor
        if next_cursor is None:
            return list(tools_by_name.values())
        if next_cursor in seen_cursors:
            raise RuntimeError(f"MCP tools/list returned a repeated cursor: {next_cursor!r}")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def adapt_call_tool_result(result: object) -> MCPAdaptedResult:
    """Adapt an SDK CallToolResult without discarding protocol data.

    The result is intentionally accepted as object. A future SDK version may
    add a content-block model that today's static union cannot name; as long as
    that value provides the SDK/Pydantic model_dump contract, its fields remain
    present in raw_output and it receives a safe text summary in llm_output.
    """
    dumped = serialize_mcp_value(result)
    raw_output = cast(dict[str, JsonValue], dumped) if isinstance(dumped, dict) else {}

    content_value = _read_result_field(result, raw_output, "content", "content", default=[])
    content = serialize_mcp_value(content_value)
    if not isinstance(content, list):
        content = []

    structured_value = _read_result_field(
        result,
        raw_output,
        "structured_content",
        "structuredContent",
        default=None,
    )
    structured_content = serialize_mcp_value(structured_value)
    is_error_value = _read_result_field(result, raw_output, "is_error", "isError", default=False)
    is_error = bool(is_error_value)

    # Preserve official wire aliases and all future SDK fields. Values read
    # through attributes above are written back only to support result-like
    # compatibility objects whose model dump omitted a standard field.
    raw_output["content"] = content
    raw_output["structuredContent"] = structured_content
    raw_output["isError"] = is_error

    return MCPAdaptedResult(
        raw_output=raw_output,
        llm_output=project_mcp_content_for_llm(content, structured_content=structured_content, is_error=is_error),
        is_error=is_error,
    )


def serialize_mcp_value(value: object) -> JsonValue:
    """Recursively serialize SDK and future SDK values without stringifying.

    Pydantic aliases are requested so nested MCP blocks retain their wire field
    names (for example mimeType and _meta). Non-model mappings and dataclasses
    are handled as a compatibility path for tests and future SDK wrappers.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Enum):
        return serialize_mcp_value(value.value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(by_alias=True, mode="json", exclude_none=False)
        except TypeError:
            dumped = model_dump()
        return serialize_mcp_value(dumped)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return serialize_mcp_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        value_mapping = cast(Mapping[object, object], value)
        return {str(key): serialize_mcp_value(item) for key, item in value_mapping.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value_sequence = cast(Sequence[object], value)
        return [serialize_mcp_value(item) for item in value_sequence]

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        attributes_dict = cast(dict[object, object], attributes)
        public_attributes = {str(key): item for key, item in attributes_dict.items() if not str(key).startswith("_")}
        if public_attributes:
            return serialize_mcp_value(public_attributes)
    return str(value)


def project_mcp_content_for_llm(
    content: Sequence[JsonValue],
    *,
    structured_content: JsonValue = None,
    is_error: bool = False,
) -> LLMOutput:
    """Project raw MCP blocks into NexAU's current text/image content model."""
    parts: list[LLMContentPart] = []
    contains_image = False

    for block in content:
        if not isinstance(block, dict):
            parts.append(_text_part(f"[MCP content block: {type(block).__name__}]"))
            continue

        block_type = str(block.get("type") or "unknown")
        if block_type == "text":
            parts.append(_text_part(str(block.get("text") or "")))
        elif block_type == "image":
            contains_image = True
            parts.append(
                {
                    "type": "image",
                    "base64": str(block.get("data") or ""),
                    "media_type": str(block.get("mimeType") or block.get("mime_type") or "application/octet-stream"),
                }
            )
        elif block_type == "audio":
            mime_type = str(block.get("mimeType") or block.get("mime_type") or "application/octet-stream")
            parts.append(_text_part(f"[MCP audio: {mime_type}]"))
        elif block_type == "resource_link":
            parts.append(_text_part(_summarize_resource_link(block)))
        elif block_type == "resource":
            parts.append(_text_part(_summarize_embedded_resource(block)))
        else:
            parts.append(_text_part(f"[MCP content block: {block_type}]"))

    if not parts and structured_content is not None:
        parts.append(_text_part(_stable_json(structured_content)))
    if not parts and is_error:
        parts.append(_text_part("MCP tool returned an error."))

    if contains_image:
        return parts
    return "\n".join(str(part.get("text") or "") for part in parts)


def format_mcp_tool_output_for_llm(context: ToolFormatterContext) -> LLMOutput:
    """Tool formatter for an MCPTool returning MCPAdaptedResult.tool_output."""
    output = context.tool_output
    if not isinstance(output, Mapping):
        return str(output)
    output_mapping = cast(Mapping[str, object], output)
    content = serialize_mcp_value(output_mapping.get("content", []))
    if not isinstance(content, list):
        content = []
    structured_content = serialize_mcp_value(output_mapping.get("structuredContent", output_mapping.get("structured_content")))
    is_error = bool(output_mapping.get("isError", output_mapping.get("is_error", context.is_error)))
    return project_mcp_content_for_llm(
        content,
        structured_content=structured_content,
        is_error=is_error,
    )


def _read_result_field(
    result: object,
    dumped: Mapping[str, JsonValue],
    attribute_name: str,
    alias: str,
    *,
    default: object,
) -> object:
    if hasattr(result, attribute_name):
        return getattr(result, attribute_name)
    if attribute_name in dumped:
        return dumped[attribute_name]
    return dumped.get(alias, default)


def _text_part(text: str) -> LLMContentPart:
    return {"type": "text", "text": text}


def _summarize_resource_link(block: Mapping[str, JsonValue]) -> str:
    name = str(block.get("name") or block.get("title") or "resource")
    uri = str(block.get("uri") or "unknown URI")
    mime_type = block.get("mimeType") or block.get("mime_type")
    suffix = f"; {mime_type}" if mime_type else ""
    return f"[MCP resource link: {name} ({uri}{suffix})]"


def _summarize_embedded_resource(block: Mapping[str, JsonValue]) -> str:
    resource = block.get("resource")
    if not isinstance(resource, dict):
        return "[MCP embedded resource]"
    uri = str(resource.get("uri") or "unknown URI")
    mime_type = resource.get("mimeType") or resource.get("mime_type")
    kind = "binary" if "blob" in resource else "text" if "text" in resource else "content"
    media = str(mime_type) if mime_type else "unknown media type"
    return f"[MCP embedded resource: {uri} ({media}; {kind})]"


def _stable_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _error_summary(llm_output: LLMOutput) -> str:
    if isinstance(llm_output, str):
        return llm_output or "MCP tool returned an error."
    text = "\n".join(str(part.get("text") or "") for part in llm_output if part.get("type") == "text").strip()
    return text or "MCP tool returned an error."


__all__ = [
    "LLMOutput",
    "MCPAdaptedResult",
    "MCPToolListClient",
    "adapt_call_tool_result",
    "format_mcp_tool_output_for_llm",
    "list_all_tools",
    "project_mcp_content_for_llm",
    "serialize_mcp_value",
]
