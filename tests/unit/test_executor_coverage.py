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

"""Coverage improvement tests for Executor class.

Targets uncovered paths in:
- nexau/archs/main_sub/execution/executor.py
"""

import threading
from unittest.mock import Mock

from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.tool.tool import Tool, build_structured_tool_definition
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.core.messages import Role


def make_tool_registry(tools: dict[str, Tool] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    if tools:
        registry.add_source("test", list(tools.values()))
    return registry


def make_tool(name: str, *, disable_parallel: bool = False, defer_loading: bool = False) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": name},
        disable_parallel=disable_parallel,
        defer_loading=defer_loading,
    )


# ---------------------------------------------------------------------------
# Executor properties
# ---------------------------------------------------------------------------


class TestExecutorProperties:
    def test_shutdown_event(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        assert isinstance(executor.shutdown_event, threading.Event)
        assert not executor.shutdown_event.is_set()

    def test_has_running_executors(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        assert executor.has_running_executors is False

    def test_is_executing(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        assert executor.is_executing is False

    def test_is_idle(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        assert executor.is_idle is False

    def test_is_waiting_for_user(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        assert executor.is_waiting_for_user is False

    def test_execution_done_event(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        assert executor.execution_done_event.is_set()


# ---------------------------------------------------------------------------
# Executor enqueue_message
# ---------------------------------------------------------------------------


class TestExecutorEnqueueMessage:
    def test_enqueue_basic_message(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        executor.enqueue_message({"role": "user", "content": "hello"})
        assert len(executor.queued_messages) == 1
        assert executor.queued_messages[0].role == Role.USER
        assert executor.queued_messages[0].get_text_content() == "hello"

    def test_enqueue_multiple_messages(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        executor.enqueue_message({"role": "user", "content": "msg1"})
        executor.enqueue_message({"role": "user", "content": "msg2"})
        assert len(executor.queued_messages) == 2


# ---------------------------------------------------------------------------
# _mark_waiting_for_user
# ---------------------------------------------------------------------------


class TestMarkWaitingForUser:
    def test_marks_when_ask_user(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        executor._last_stop_tool_name = "ask_user"
        executor._mark_waiting_for_user()
        assert executor.is_waiting_for_user is True

    def test_does_not_mark_for_other_tools(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        executor._last_stop_tool_name = "other_tool"
        executor._mark_waiting_for_user()
        assert executor.is_waiting_for_user is False


# ---------------------------------------------------------------------------
# structured_tool_payload
# ---------------------------------------------------------------------------


class TestStructuredToolPayload:
    def test_returns_empty_for_xml_mode(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            tool_call_mode="xml",
        )
        assert executor.structured_tool_payload == []

    def test_returns_definitions_for_structured_mode(self, mock_llm_config):
        tool = make_tool("test_tool")
        registry = make_tool_registry({"test_tool": tool})
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=registry,
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            tool_call_mode="structured",
        )
        payload = executor.structured_tool_payload
        assert len(payload) >= 1
        assert payload[0]["name"] == "test_tool"


# ---------------------------------------------------------------------------
# update_structured_tools
# ---------------------------------------------------------------------------


class TestUpdateStructuredTools:
    def test_replaces_definitions(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            tool_call_mode="structured",
        )
        new_defs = [
            build_structured_tool_definition(
                name="new_tool",
                description="new desc",
                input_schema={"type": "object", "properties": {}},
                kind="tool",
            ),
        ]
        executor.update_structured_tools(new_defs)
        assert len(executor.structured_tool_definitions) == 1
        assert executor.structured_tool_definitions[0]["name"] == "new_tool"


# ---------------------------------------------------------------------------
# Middleware wiring
# ---------------------------------------------------------------------------


class TestMiddlewareWiring:
    def test_wire_event_emitters_no_middleware(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        # Should not raise even without middleware
        executor._wire_middleware_event_emitters()

    def test_wire_llm_runtime_no_middleware(self, mock_llm_config):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )
        # Should not raise
        executor._wire_middleware_llm_runtime(mock_llm_config, Mock())
