# Copyright (c) Nex-AGI. All rights reserved.

from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool.builtin.tool_search import tool_search
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry


def _make_tool(name: str, *, defer: bool = True, description: str = "") -> Tool:
    return Tool(
        name=name,
        description=description or f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": name},
        defer_loading=defer,
    )


def _make_ctx(registry: ToolRegistry) -> FrameworkContext:
    import threading

    return FrameworkContext(
        agent_name="agent",
        agent_id="agent_123",
        run_id="run_123",
        root_run_id="root_123",
        _tool_registry=registry,
        _shutdown_event=threading.Event(),
    )


class TestToolSearchBuiltin:
    def test_tool_search_returns_matches(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("GetWeather", description="Get the current weather")])

        result = tool_search("weather", _make_ctx(registry), max_results=1)

        assert result == "Found 1 tool(s): GetWeather. They are now available for use."

    def test_tool_search_returns_no_match(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("GetWeather")])

        result = tool_search("stocks", _make_ctx(registry))

        assert result == "No matching tools found."

    def test_tool_search_empty_query(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("GetWeather")])

        result = tool_search("", _make_ctx(registry))

        assert result == "No matching tools found."

    def test_tool_search_multiple_results_lists_all_names(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("WebSearch", description="Search the web"),
                _make_tool("WebFetch", description="Fetch a web page"),
            ],
        )

        result = tool_search("web", _make_ctx(registry))

        assert result == "Found 2 tool(s): WebSearch, WebFetch. They are now available for use."

    def test_tool_search_max_results_zero(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", description="test")])

        result = tool_search("test", _make_ctx(registry), max_results=0)

        assert result == "No matching tools found."
