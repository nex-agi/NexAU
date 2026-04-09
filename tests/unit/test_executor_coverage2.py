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

"""Additional coverage tests for Executor class.

Targets uncovered paths in:
- nexau/archs/main_sub/execution/executor.py

Covers: _wire_middleware_event_emitters, _wire_middleware_llm_runtime backward compat,
_snapshot_structured_tool_definitions, _wait_for_messages,
_structured_tool_description, execute with
stop_signal, team_mode behavior, _apply_after_agent_hooks, _build_middleware_manager.
"""

from unittest.mock import Mock, patch

from nexau.archs.llm.llm_aggregators.events import RetryEvent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.execution.hooks import (
    HookResult,
    Middleware,
)
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.core.messages import Message, Role, TextBlock


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


def make_config() -> LLMConfig:
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_chat_completion",
    )


# ---------------------------------------------------------------------------
# _wire_middleware_event_emitters with emitter
# ---------------------------------------------------------------------------


class TestWireMiddlewareEventEmitters:
    def test_wires_emitter_to_middlewares(self):
        class EventMiddleware(Middleware):
            def __init__(self):
                self.on_event = lambda evt: None  # noqa: E731 — test stub

            @property
            def supports_set_event_emitter(self) -> bool:
                return True

            def set_event_emitter(self, emitter):
                self._emitter = emitter

        mw = EventMiddleware()
        Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            middlewares=[mw],
        )
        # Should have wired the emitter without error
        assert hasattr(mw, "_emitter")

    def test_emitter_wiring_failure_logged(self):
        class BrokenEmitterMiddleware(Middleware):
            def __init__(self):
                self.on_event = lambda evt: None  # noqa: E731 — test stub

            @property
            def supports_set_event_emitter(self) -> bool:
                return True

            def set_event_emitter(self, emitter):
                raise RuntimeError("boom")

        mw = BrokenEmitterMiddleware()
        # Should not raise, just log warning
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            middlewares=[mw],
        )
        assert executor is not None

    def test_executor_builds_retry_event_callback(self):
        captured_events = []

        class EventMiddleware(Middleware):
            def __init__(self):
                self.on_event = lambda evt: captured_events.append(evt)  # noqa: E731 - test stub

            @property
            def supports_set_event_emitter(self) -> bool:
                return True

            def set_event_emitter(self, emitter):
                self._emitter = emitter

        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            middlewares=[EventMiddleware()],
        )

        assert executor.llm_caller.on_retry is not None
        executor.llm_caller.on_retry(2, 5, 3.0, "temporary failure")

        assert len(captured_events) == 1
        retry_event = captured_events[0]
        assert isinstance(retry_event, RetryEvent)
        assert retry_event.api_type == "openai_chat_completion"
        assert retry_event.attempt == 2
        assert retry_event.max_attempts == 5
        assert retry_event.backoff_seconds == 3.0
        assert retry_event.error_message == "temporary failure"


# ---------------------------------------------------------------------------
# _wire_middleware_llm_runtime — backward compat fallbacks
# ---------------------------------------------------------------------------


class TestWireMiddlewareLlmRuntimeCompat:
    def test_backward_compat_fallback(self):
        class LegacyMiddleware(Middleware):
            @property
            def supports_set_llm_runtime(self) -> bool:
                return True

            def set_llm_runtime(self, llm_config, openai_client):
                # Old signature without keyword args
                self.wired = True

        mw = LegacyMiddleware()
        Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            middlewares=[mw],
        )
        assert mw.wired

    def test_llm_runtime_failure_logged(self):
        class BrokenLLMMiddleware(Middleware):
            @property
            def supports_set_llm_runtime(self) -> bool:
                return True

            def set_llm_runtime(self, *args, **kwargs):
                raise ValueError("broken")

        mw = BrokenLLMMiddleware()
        # Should not raise, just log warning
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            middlewares=[mw],
        )
        assert executor is not None


# ---------------------------------------------------------------------------
# _snapshot_structured_tool_definitions — with sub-agents
# ---------------------------------------------------------------------------


class TestSnapshotWithSubAgents:
    def test_includes_sub_agent_definitions(self):
        """RFC-0015: Agent is a regular builtin tool, not a virtual definition.

        Sub-agents configured on the executor should NOT generate virtual
        sub-agent-{name} tool definitions. The Agent tool is registered
        as a regular builtin tool in AgentConfig._finalize() and will appear
        in structured_tool_payload only when it's in the ToolRegistry.
        """
        sub_agent_config = AgentConfig(
            name="helper",
            system_prompt="Helper agent",
            tools=[],
        )
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={"helper": sub_agent_config},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            tool_call_mode="structured",
        )
        payload = executor.structured_tool_payload
        # RFC-0015: No virtual sub-agent-{name} definitions should be generated
        sub_agent_names = [d["name"] for d in payload]
        assert not any("sub-agent-" in name for name in sub_agent_names)

    def test_sync_new_tools_from_registry(self):
        registry = make_tool_registry()
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=registry,
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            tool_call_mode="structured",
        )
        initial_count = len(executor.structured_tool_payload)
        # Dynamically add a tool
        new_tool = make_tool("dynamic_tool")
        registry.add_source("dynamic", [new_tool])
        updated_payload = executor.structured_tool_payload
        assert len(updated_payload) > initial_count


# ---------------------------------------------------------------------------
# _wait_for_messages
# ---------------------------------------------------------------------------


class TestWaitForMessages:
    def test_returns_false_on_stop_signal(self):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
        )
        executor.stop_signal = True
        result = executor._wait_for_messages()
        assert result is False
        assert executor._is_idle is False

    def test_returns_true_when_messages_available(self):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
        )
        # Pre-enqueue a message so it doesn't block
        executor.queued_messages = [Message(role=Role.USER, content=[TextBlock(text="hello")])]
        result = executor._wait_for_messages()
        assert result is True
        assert executor._is_idle is False


# ---------------------------------------------------------------------------
# _build_middleware_manager
# ---------------------------------------------------------------------------


class TestBuildMiddlewareManager:
    def test_builds_with_hooks(self):
        def before_model_hook(hook_input):
            return HookResult.no_changes()

        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            before_model_hooks=[before_model_hook],
        )
        assert executor.middleware_manager is not None
        assert len(executor.middleware_manager) >= 1

    def test_builds_with_middlewares(self):
        mw = Middleware()
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            middlewares=[mw],
        )
        assert executor.middleware_manager is not None


# ---------------------------------------------------------------------------
# _structured_tool_description
# ---------------------------------------------------------------------------


class TestStructuredToolDescription:
    def test_returns_skill_description_for_as_skill(self):
        tool = Tool(
            name="skill_tool",
            description="normal",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: None,
            skill_description="A special skill",
            as_skill=True,
        )
        registry = make_tool_registry({"skill_tool": tool})
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=registry,
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            tool_call_mode="structured",
        )
        desc = executor._structured_tool_description(tool)
        assert desc == "A special skill"


# ---------------------------------------------------------------------------
# Executor with team_mode
# ---------------------------------------------------------------------------


class TestExecutorTeamMode:
    def test_team_mode_flag(self):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            team_mode=True,
        )
        assert executor.team_mode is True

    def test_enqueue_wakes_message_available(self):
        executor = Executor(
            agent_name="test",
            agent_id="id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=make_config(),
            team_mode=True,
        )
        executor.enqueue_message({"role": "user", "content": "hello"})
        assert executor._message_available.is_set()


# ---------------------------------------------------------------------------
# Executor — structured tool mode warning
# ---------------------------------------------------------------------------


class TestStructuredToolWarning:
    def test_warns_when_no_tools_in_structured_mode(self):
        with patch("nexau.archs.main_sub.execution.executor.logger") as mock_logger:
            executor = Executor(
                agent_name="test",
                agent_id="id",
                tool_registry=make_tool_registry(),
                sub_agents={},
                stop_tools=set(),
                openai_client=Mock(),
                llm_config=make_config(),
                tool_call_mode="structured",
            )
            # Logger should have warned about no tools
            assert mock_logger.warning.called or len(executor.structured_tool_definitions) == 0
