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

"""RFC-0029 tests for MCP pagination, result adaptation, and registry refresh."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ListToolsResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool

from nexau.archs.tool.builtin.mcp_result import (
    adapt_call_tool_result,
    format_mcp_tool_output_for_llm,
    list_all_tools,
    serialize_mcp_value,
)
from nexau.archs.tool.formatters import ToolFormatterContext
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry


class _PagedClient:
    def __init__(self, pages: dict[str | None, ListToolsResult]) -> None:
        self.pages = pages
        self.seen_cursors: list[str | None] = []

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
        self.seen_cursors.append(cursor)
        return self.pages[cursor]


def _mcp_tool(name: str, *, description: str = "") -> MCPTool:
    return MCPTool(name=name, description=description, input_schema={"type": "object"})


def _nexau_tool(name: str, *, defer: bool = False) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": name},
        defer_loading=defer,
    )


@pytest.mark.anyio
async def test_list_all_tools_follows_opaque_cursors_and_deduplicates_by_name() -> None:
    first_cursor = "YWxwaGE+YnJhdm8/Y2hhcmxpZQ=="
    second_cursor = "page/2?opaque=yes"
    client = _PagedClient(
        {
            None: ListToolsResult(
                tools=[_mcp_tool("alpha"), _mcp_tool("shared", description="old")],
                next_cursor=first_cursor,
            ),
            first_cursor: ListToolsResult(
                tools=[_mcp_tool("beta")],
                next_cursor=second_cursor,
            ),
            second_cursor: ListToolsResult(
                tools=[_mcp_tool("shared", description="new"), _mcp_tool("gamma")],
                next_cursor=None,
            ),
        }
    )

    tools = await list_all_tools(client)

    assert client.seen_cursors == [None, first_cursor, second_cursor]
    assert [tool.name for tool in tools] == ["alpha", "shared", "beta", "gamma"]
    assert tools[1].description == "new"


@pytest.mark.anyio
async def test_list_all_tools_rejects_a_repeated_cursor() -> None:
    client = _PagedClient(
        {
            None: ListToolsResult(tools=[], next_cursor="same"),
            "same": ListToolsResult(tools=[], next_cursor="same"),
        }
    )

    with pytest.raises(RuntimeError, match="repeated cursor"):
        await list_all_tools(client)


def test_adapt_call_tool_result_preserves_all_standard_content_blocks() -> None:
    result = CallToolResult(
        content=[
            TextContent(text="hello"),
            ImageContent(data="aW1n", mime_type="image/png"),
            AudioContent(data="YXVk", mime_type="audio/wav"),
            ResourceLink(
                name="report",
                uri="resource://reports/1",
                description="Full report",
                mime_type="text/plain",
            ),
            EmbeddedResource(
                resource=TextResourceContents(
                    uri="resource://reports/1",
                    mime_type="text/plain",
                    text="embedded text body",
                )
            ),
            EmbeddedResource(
                resource=BlobResourceContents(
                    uri="resource://audio/1",
                    mime_type="application/octet-stream",
                    blob="YmluYXJ5",
                )
            ),
        ],
        structured_content={"answer": 42},
        _meta={"trace": "mcp-meta"},
    )

    adapted = adapt_call_tool_result(result)

    assert adapted.is_error is False
    assert adapted.raw_output["structuredContent"] == {"answer": 42}
    assert adapted.raw_output["isError"] is False
    assert adapted.raw_output["_meta"] == {"trace": "mcp-meta"}
    raw_content = adapted.raw_output["content"]
    assert isinstance(raw_content, list)
    assert [block["type"] for block in raw_content if isinstance(block, dict)] == [
        "text",
        "image",
        "audio",
        "resource_link",
        "resource",
        "resource",
    ]
    assert raw_content[1] == {
        "type": "image",
        "data": "aW1n",
        "mimeType": "image/png",
        "annotations": None,
        "_meta": None,
    }
    assert isinstance(raw_content[2], dict)
    assert raw_content[2]["data"] == "YXVk"
    assert isinstance(raw_content[4], dict)
    text_resource = raw_content[4]["resource"]
    assert isinstance(text_resource, dict)
    assert text_resource["text"] == "embedded text body"
    assert isinstance(raw_content[5], dict)
    blob_resource = raw_content[5]["resource"]
    assert isinstance(blob_resource, dict)
    assert blob_resource["blob"] == "YmluYXJ5"

    assert isinstance(adapted.llm_output, list)
    assert adapted.llm_output[0] == {"type": "text", "text": "hello"}
    assert adapted.llm_output[1] == {
        "type": "image",
        "base64": "aW1n",
        "media_type": "image/png",
    }
    llm_text = "\n".join(str(part.get("text") or "") for part in adapted.llm_output)
    assert "[MCP audio: audio/wav]" in llm_text
    assert "[MCP resource link: report (resource://reports/1; text/plain)]" in llm_text
    assert "[MCP embedded resource: resource://reports/1 (text/plain; text)]" in llm_text
    assert "[MCP embedded resource: resource://audio/1 (application/octet-stream; binary)]" in llm_text
    assert "YXVk" not in llm_text
    assert "YmluYXJ5" not in llm_text
    assert "embedded text body" not in llm_text


def test_adapt_call_tool_result_uses_structured_content_when_content_is_empty() -> None:
    adapted = adapt_call_tool_result(
        CallToolResult(
            content=[],
            structured_content={"z": 1, "message": "你好"},
        )
    )

    assert isinstance(adapted.llm_output, str)
    assert adapted.llm_output == '{\n  "message": "你好",\n  "z": 1\n}'
    assert adapted.raw_output["structuredContent"] == {"z": 1, "message": "你好"}


def test_adapt_call_tool_result_preserves_tool_error_semantics() -> None:
    adapted = adapt_call_tool_result(
        CallToolResult(
            content=[TextContent(text="the upstream tool failed")],
            is_error=True,
        )
    )

    assert adapted.is_error is True
    assert adapted.raw_output["isError"] is True
    assert adapted.llm_output == "the upstream tool failed"
    assert adapted.tool_output["content"] == adapted.raw_output["content"]
    assert adapted.tool_output["structuredContent"] is None
    assert adapted.tool_output["isError"] is True
    assert adapted.tool_output["status"] == "error"
    assert adapted.tool_output["error"] == "the upstream tool failed"


def test_mcp_tool_formatter_recovers_llm_projection_from_pipeline_output() -> None:
    adapted = adapt_call_tool_result(
        CallToolResult(
            content=[
                TextContent(text="chart"),
                ImageContent(data="aW1n", mime_type="image/png"),
                AudioContent(data="YXVk", mime_type="audio/wav"),
            ]
        )
    )

    formatted = format_mcp_tool_output_for_llm(
        ToolFormatterContext(
            tool_name="mcp__charts__render",
            tool_input={},
            tool_output=adapted.tool_output,
            tool_call_id="call-1",
            is_error=False,
        )
    )

    assert formatted == adapted.llm_output
    assert isinstance(formatted, list)
    assert formatted[-1] == {"type": "text", "text": "[MCP audio: audio/wav]"}


@dataclass
class _FutureContentBlock:
    type: str
    payload: dict[str, object]
    priority: int

    def model_dump(self, **_kwargs: object) -> dict[str, object]:
        return {
            "type": self.type,
            "payload": self.payload,
            "priority": self.priority,
        }


class _FutureCallToolResult:
    def __init__(self, content: Sequence[object]) -> None:
        self.content = list(content)
        self.structured_content = {"future": True}
        self.is_error = False

    def model_dump(self, **_kwargs: object) -> dict[str, Any]:
        return {
            "content": self.content,
            "structuredContent": self.structured_content,
            "isError": self.is_error,
            "resultType": "future_complete",
            "futureTopLevel": {"version": 3},
        }


def test_adapt_call_tool_result_preserves_unknown_future_blocks_and_fields() -> None:
    adapted = adapt_call_tool_result(
        _FutureCallToolResult(
            [
                _FutureContentBlock(
                    type="hologram",
                    payload={"frames": ["one", "two"], "codec": "future/v1"},
                    priority=7,
                )
            ]
        )
    )

    assert adapted.raw_output["content"] == [
        {
            "type": "hologram",
            "payload": {"frames": ["one", "two"], "codec": "future/v1"},
            "priority": 7,
        }
    ]
    assert adapted.raw_output["futureTopLevel"] == {"version": 3}
    assert adapted.raw_output["resultType"] == "future_complete"
    assert adapted.llm_output == "[MCP content block: hologram]"


def test_serialize_mcp_value_keeps_bytes_json_compatible() -> None:
    assert serialize_mcp_value({"payload": b"binary"}) == {"payload": "YmluYXJ5"}


class TestToolRegistryReplaceSource:
    def test_replace_source_removes_stale_tools_without_accumulating_duplicates(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_nexau_tool("configured")])
        registry.add_source("mcp", [_nexau_tool("mcp__one__old"), _nexau_tool("mcp__two__same")])

        new_tool = _nexau_tool("mcp__one__new")
        replacement_same = _nexau_tool("mcp__two__same")
        registry.replace_source("mcp", [new_tool, replacement_same])
        registry.replace_source("mcp", [new_tool, replacement_same])

        assert list(registry.get_all()) == ["configured", "mcp__one__new", "mcp__two__same"]
        assert registry.get_tool("mcp__one__old") is None
        assert registry.get_tool("mcp__two__same") is replacement_same

    def test_replace_source_preserves_only_valid_deferred_injections(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "mcp",
            [
                _nexau_tool("kept", defer=True),
                _nexau_tool("removed", defer=True),
                _nexau_tool("became_eager", defer=True),
            ],
        )
        assert registry.inject("kept")
        assert registry.inject("removed")
        assert registry.inject("became_eager")

        registry.replace_source(
            "mcp",
            [
                _nexau_tool("kept", defer=True),
                _nexau_tool("became_eager"),
                _nexau_tool("new", defer=True),
            ],
        )

        assert registry.injected_count == 1
        assert [tool.name for tool in registry.compute_eager_tools()] == ["kept", "became_eager"]
        assert [tool.name for tool in registry.compute_deferred_tools()] == ["new"]

    def test_empty_replacement_clears_source_but_not_other_sources(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_nexau_tool("configured")])
        registry.add_source("mcp", [_nexau_tool("remote", defer=True)])
        assert registry.inject("remote")

        registry.replace_source("mcp", [])

        assert list(registry.get_all()) == ["configured"]
        assert registry.injected_count == 0

    def test_injection_is_retained_when_same_named_deferred_tool_exists_elsewhere(self) -> None:
        registry = ToolRegistry()
        registry.add_source("runtime", [_nexau_tool("shared", defer=True)])
        registry.add_source("mcp", [_nexau_tool("shared", defer=True)])
        assert registry.inject("shared")

        registry.replace_source("mcp", [])

        assert registry.injected_count == 1
        assert [tool.name for tool in registry.compute_eager_tools()] == ["shared"]
