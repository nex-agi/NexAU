# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ToolRegistry (RFC-0005)."""

from __future__ import annotations

import threading

from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry


def _make_tool(
    name: str,
    *,
    defer: bool = False,
    description: str = "",
    search_hint: str | None = None,
    disable_parallel: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description=description or f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": "ok"},
        defer_loading=defer,
        search_hint=search_hint,
        disable_parallel=disable_parallel,
    )


class TestToolRegistryBasic:
    def test_add_source_and_get_all(self) -> None:
        registry = ToolRegistry()
        t1 = _make_tool("A")
        t2 = _make_tool("B")
        registry.add_source("config", [t1, t2])
        all_tools = registry.get_all()
        assert set(all_tools.keys()) == {"A", "B"}

    def test_eager_excludes_deferred(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A"), _make_tool("B", defer=True)])
        eager = registry.compute_eager_tools()
        assert [t.name for t in eager] == ["A"]

    def test_deferred_only_uninjected(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A"), _make_tool("B", defer=True), _make_tool("C", defer=True)])
        deferred = registry.compute_deferred_tools()
        assert {t.name for t in deferred} == {"B", "C"}

    def test_inject_makes_eager(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A"), _make_tool("B", defer=True)])
        assert registry.inject("B")
        eager = registry.compute_eager_tools()
        assert {t.name for t in eager} == {"A", "B"}
        assert registry.deferred_count == 0

    def test_compute_serial_tool_names(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("A"),
                _make_tool("B", disable_parallel=True),
                _make_tool("C", defer=True, disable_parallel=True),
            ],
        )
        assert registry.compute_serial_tool_names() == ["B", "C"]

    def test_inject_nonexistent_returns_false(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A")])
        assert not registry.inject("Z")

    def test_inject_idempotent(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("B", defer=True)])
        assert registry.inject("B")
        assert registry.inject("B")
        assert registry.injected_count == 1


class TestToolRegistrySearch:
    def test_search_keyword_name_match(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("WebSearch", defer=True, description="Search the web"),
                _make_tool("WebFetch", defer=True, description="Fetch a URL"),
                _make_tool("NotebookEdit", defer=True, description="Edit notebooks"),
            ],
        )
        results = registry.search("web")
        names = {t.name for t in results}
        assert "WebSearch" in names
        assert "WebFetch" in names

    def test_search_keyword_description_match(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("MyTool", defer=True, description="Search the internet for information"),
            ],
        )
        results = registry.search("internet")
        assert len(results) == 1
        assert results[0].name == "MyTool"

    def test_search_keyword_hint_match(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("SlackPost", defer=True, description="Post message", search_hint="slack messaging"),
            ],
        )
        results = registry.search("messaging")
        assert len(results) == 1

    def test_search_required_token(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("slack_read", defer=True, description="Read slack"),
                _make_tool("slack_post", defer=True, description="Post to slack"),
                _make_tool("email_send", defer=True, description="Send email"),
            ],
        )
        results = registry.search("+slack send")
        # Only slack tools should match (required +slack), ranked by "send"
        names = {t.name for t in results}
        assert "email_send" not in names

    def test_search_max_results(self) -> None:
        registry = ToolRegistry()
        tools = [_make_tool(f"Tool{i}", defer=True, description="test tool") for i in range(10)]
        registry.add_source("config", tools)
        results = registry.search("test", max_results=2)
        assert len(results) <= 2

    def test_search_max_results_default_is_five(self) -> None:
        registry = ToolRegistry()
        tools = [_make_tool(f"Tool{i}", defer=True, description="test tool") for i in range(10)]
        registry.add_source("config", tools)
        results = registry.search("test")
        assert len(results) == 5

    def test_search_no_match(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=True)])
        results = registry.search("zzzzz_nonexistent")
        assert len(results) == 0

    def test_search_ignores_already_injected(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=True, description="test")])
        registry.inject("A")
        results = registry.search("test")
        assert len(results) == 0

    def test_search_ignores_eager_tools(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=False, description="test")])
        results = registry.search("test")
        assert len(results) == 0


class TestToolRegistryIndex:
    def test_build_deferred_index(self) -> None:
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("WebSearch", defer=True, description="Search the web"),
                _make_tool("Read", defer=False),
            ],
        )
        index = registry.build_deferred_index()
        assert "WebSearch" in index
        assert "Read" not in index

    def test_build_deferred_index_empty(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("Read")])
        assert registry.build_deferred_index() == ""


class TestToolRegistryThreadSafety:
    def test_concurrent_inject_and_compute(self) -> None:
        registry = ToolRegistry()
        tools = [_make_tool(f"T{i}", defer=True) for i in range(50)]
        registry.add_source("config", tools)

        errors: list[Exception] = []

        def inject_all() -> None:
            try:
                for i in range(50):
                    registry.inject(f"T{i}")
            except Exception as e:
                errors.append(e)

        def compute_loop() -> None:
            try:
                for _ in range(100):
                    registry.compute_eager_tools()
                    registry.compute_deferred_tools()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inject_all), threading.Thread(target=compute_loop)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert registry.injected_count == 50


class TestToolRegistryMultiSource:
    def test_multiple_sources(self) -> None:
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A")])
        registry.add_source("mcp", [_make_tool("B", defer=True)])
        registry.add_source("builtin", [_make_tool("ToolSearch")])

        assert registry.eager_count == 2  # A + ToolSearch
        assert registry.deferred_count == 1  # B
        assert len(registry.get_all()) == 3

    def test_add_source_appends(self) -> None:
        registry = ToolRegistry()
        registry.add_source("mcp", [_make_tool("X")])
        registry.add_source("mcp", [_make_tool("Y")])
        assert len(registry.get_all()) == 2


class TestToolRegistryCamelCase:
    """Phase 1.5: CamelCase tokenization edge cases."""

    def test_camel_case_name_tokenized(self) -> None:
        """WebSearch should match keyword 'web' via CamelCase splitting."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("WebSearch", defer=True, description="Search")])
        results = registry.search("web")
        assert len(results) == 1
        assert results[0].name == "WebSearch"

    def test_camel_case_multi_part(self) -> None:
        """GetCurrentWeather should match 'current' and 'weather'."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("GetCurrentWeather", defer=True, description="Weather info")])
        results = registry.search("current")
        assert len(results) == 1

    def test_snake_case_still_works(self) -> None:
        """snake_case names should still tokenize correctly."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("read_file", defer=True, description="Read a file")])
        results = registry.search("read")
        assert len(results) == 1

    def test_mixed_case_and_underscore(self) -> None:
        """Names like My_CamelTool should tokenize into [my, camel, tool]."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("My_CamelTool", defer=True, description="A tool")])
        results = registry.search("camel")
        assert len(results) == 1

    def test_single_word_name(self) -> None:
        """Single-word names should still be searchable."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("Calculator", defer=True, description="Math")])
        results = registry.search("calculator")
        assert len(results) == 1

    def test_all_uppercase_name(self) -> None:
        """All-uppercase like 'HTTPClient' should tokenize as ['http', 'client']."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("HTTPClient", defer=True, description="HTTP client")])
        results = registry.search("client")
        assert len(results) == 1

    def test_consecutive_uppercase_prefix_match(self) -> None:
        """'http' should get +10 exact match against HTTPClient, not just +5 substring."""
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("HTTPClient", defer=True, description="An HTTP client"),
                _make_tool("SimpleClient", defer=True, description="A simple client"),
            ],
        )
        results = registry.search("http")
        # HTTPClient should rank first (name part exact match +10 vs description-only +2)
        assert results[0].name == "HTTPClient"


class TestToolRegistryEdgeCases:
    """Phase 1.5: Additional edge case coverage."""

    def test_empty_query(self) -> None:
        """Empty string query should return no results."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=True)])
        results = registry.search("")
        assert len(results) == 0

    def test_whitespace_only_query(self) -> None:
        """Whitespace-only query should return no results."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=True)])
        results = registry.search("   ")
        assert len(results) == 0

    def test_no_deferred_tools_returns_empty(self) -> None:
        """Search on registry with only eager tools returns nothing."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=False)])
        results = registry.search("A")
        assert len(results) == 0

    def test_search_case_insensitive(self) -> None:
        """Search should be case-insensitive."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("WebSearch", defer=True, description="Search the web")])
        results = registry.search("WEBSEARCH")
        assert len(results) == 1

    def test_inject_after_search_makes_tool_eager(self) -> None:
        """After search injects a tool, it should appear in eager list."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=True, description="test")])
        registry.search("test")
        eager = registry.compute_eager_tools()
        assert any(t.name == "A" for t in eager)

    def test_search_does_not_return_duplicates(self) -> None:
        """Multiple search calls should not return already-injected tools."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=True, description="test")])
        first = registry.search("test")
        second = registry.search("test")
        assert len(first) == 1
        assert len(second) == 0  # already injected

    def test_empty_description_tool_searchable_by_name(self) -> None:
        """Tool with empty description should still be found by name match."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("WebSearch", defer=True, description="")])
        results = registry.search("web")
        assert len(results) == 1

    def test_name_match_ranks_higher_than_description(self) -> None:
        """Tool matching by name should rank higher than tool matching by description only."""
        registry = ToolRegistry()
        registry.add_source(
            "config",
            [
                _make_tool("Calculator", defer=True, description="Do math"),
                _make_tool("MathHelper", defer=True, description="A calculator for math"),
            ],
        )
        results = registry.search("calculator")
        # Calculator should rank first (name match > description match)
        assert results[0].name == "Calculator"

    def test_special_characters_in_query(self) -> None:
        """Special characters in query should not cause errors."""
        registry = ToolRegistry()
        registry.add_source("config", [_make_tool("A", defer=True)])
        # Should not raise
        results = registry.search("hello@#$%^&*()")
        assert isinstance(results, list)
