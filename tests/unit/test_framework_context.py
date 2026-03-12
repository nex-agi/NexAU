# Copyright (c) Nex-AGI. All rights reserved.

import threading

from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry


def _make_tool(name: str, *, defer: bool = False) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": name},
        defer_loading=defer,
    )


class TestFrameworkContext:
    def test_tools_api_search_get_and_repr(self) -> None:
        registry = ToolRegistry()
        weather = _make_tool("WeatherSearch", defer=True)
        notes = _make_tool("Notes")
        registry.add_source("config", [weather, notes])

        ctx = FrameworkContext(
            agent_name="agent",
            agent_id="agent_123",
            run_id="run_123",
            root_run_id="root_123",
            _tool_registry=registry,
            _shutdown_event=threading.Event(),
        )

        matched = ctx.tools.search(query="weather", max_results=1)

        assert [tool.name for tool in matched] == ["WeatherSearch"]
        assert ctx.tools.get(name="WeatherSearch") is weather
        assert "agent_123" in repr(ctx)

    def test_tools_api_add_writes_to_registry(self) -> None:
        registry = ToolRegistry()
        ctx = FrameworkContext(
            agent_name="agent",
            agent_id="agent_123",
            run_id="run_123",
            root_run_id="root_123",
            _tool_registry=registry,
            _shutdown_event=threading.Event(),
        )
        tool = _make_tool("DynamicTool")

        ctx.tools.add(tool=tool)

        assert registry.get_tool("DynamicTool") is tool

    def test_execution_api_is_shutting_down(self) -> None:
        event = threading.Event()
        ctx = FrameworkContext.for_testing(shutdown_event=event)

        assert not ctx.execution.is_shutting_down()
        event.set()
        assert ctx.execution.is_shutting_down()

    def test_for_testing_defaults(self) -> None:
        ctx = FrameworkContext.for_testing()

        assert ctx.agent_name == "test_agent"
        assert not ctx.execution.is_shutting_down()
        assert ctx.tools.get(name="nonexistent") is None
