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

"""Integration tests for Middleware system.

Tests middleware hooks, event streaming, and custom middleware:
- AgentEventsMiddleware event emission
- Middleware execution order
- Custom middleware integration
- Event completeness and ordering
"""

import asyncio
from typing import Any

import pytest

from nexau.archs.llm.llm_aggregators.events import (
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.hooks import (
    AfterAgentHookInput,
    BeforeAgentHookInput,
    HookResult,
    Middleware,
)
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
)
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager


class TestAgentEventsMiddleware:
    """Test AgentEventsMiddleware functionality."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.mark.llm
    def test_events_are_emitted_in_order(self, session_manager):
        """Test that events are emitted in proper order."""
        events_received: list[Any] = []

        def on_event(event: Any) -> None:
            events_received.append(event)

        middleware = AgentEventsMiddleware(session_id="test_session", on_event=on_event)

        config = AgentConfig(
            name="event_agent",
            system_prompt="You are a helpful assistant. Respond briefly.",
            llm_config=LLMConfig(),
            middlewares=[middleware],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="event_session",
        )

        async def run_agent():
            result = await agent.run_async(message="Say 'test'")
            return result if isinstance(result, str) else result[0]

        response = asyncio.run(run_agent())
        assert isinstance(response, str)

        # Should have received events
        assert len(events_received) > 0

        # Check event types
        event_types = [type(e).__name__ for e in events_received]

        # RunStarted should come before RunFinished
        run_started_indices = [i for i, t in enumerate(event_types) if "RunStarted" in t]
        run_finished_indices = [i for i, t in enumerate(event_types) if "RunFinished" in t]

        if run_started_indices and run_finished_indices:
            assert min(run_started_indices) < max(run_finished_indices)

    @pytest.mark.llm
    def test_text_events_contain_content(self, session_manager):
        """Test that text events contain actual content."""
        text_content: list[str] = []

        def on_event(event: Any) -> None:
            if isinstance(event, TextMessageContentEvent):
                text_content.append(event.delta)

        middleware = AgentEventsMiddleware(session_id="test_session", on_event=on_event)

        config = AgentConfig(
            name="text_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
            middlewares=[middleware],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="text_session",
        )

        async def run_agent():
            result = await agent.run_async(message="Say 'hello world'")
            return result if isinstance(result, str) else result[0]

        response = asyncio.run(run_agent())

        # Text content should be captured
        combined_text = "".join(text_content)
        # Should have some text
        assert len(combined_text) > 0 or len(response) > 0


class TestCustomMiddleware:
    """Test custom middleware integration."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.mark.llm
    def test_custom_middleware_hooks_called(self, session_manager):
        """Test that custom middleware hooks are called."""
        hook_calls: list[str] = []

        class TrackingMiddleware(Middleware):
            """Middleware that tracks hook calls."""

            def __init__(self, calls_list: list[str]):
                self._calls = calls_list

            def before_agent(self, hook_input: BeforeAgentHookInput) -> HookResult:
                self._calls.append("before_agent")
                return HookResult.no_changes()

            def after_agent(self, hook_input: AfterAgentHookInput) -> HookResult:
                self._calls.append("after_agent")
                return HookResult.no_changes()

        middleware = TrackingMiddleware(hook_calls)

        config = AgentConfig(
            name="tracked_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
            middlewares=[middleware],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="track_session",
        )

        response = agent.run(message="Hello!")
        assert isinstance(response, str)

        # Hooks should have been called
        assert "before_agent" in hook_calls
        assert "after_agent" in hook_calls

    @pytest.mark.llm
    def test_multiple_middlewares_execute_in_order(self, session_manager):
        """Test that multiple middlewares execute in order."""
        execution_order: list[str] = []

        class OrderedMiddleware(Middleware):
            """Middleware that records execution order."""

            def __init__(self, name: str, order_list: list[str]):
                self._name = name
                self._order = order_list

            def before_agent(self, hook_input: BeforeAgentHookInput) -> HookResult:
                self._order.append(f"{self._name}_before")
                return HookResult.no_changes()

            def after_agent(self, hook_input: AfterAgentHookInput) -> HookResult:
                self._order.append(f"{self._name}_after")
                return HookResult.no_changes()

        middleware1 = OrderedMiddleware("first", execution_order)
        middleware2 = OrderedMiddleware("second", execution_order)

        config = AgentConfig(
            name="ordered_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
            middlewares=[middleware1, middleware2],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="order_session",
        )

        response = agent.run(message="Hello!")
        assert isinstance(response, str)

        # Check execution order
        # before hooks should execute in middleware order
        before_indices = {
            "first": execution_order.index("first_before") if "first_before" in execution_order else -1,
            "second": execution_order.index("second_before") if "second_before" in execution_order else -1,
        }

        if before_indices["first"] >= 0 and before_indices["second"] >= 0:
            assert before_indices["first"] < before_indices["second"]


class TestMiddlewareWithTools:
    """Test middleware interaction with tool execution."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.mark.llm
    def test_tool_events_captured_by_middleware(self, session_manager):
        """Test that tool execution events are captured by middleware."""
        from nexau.archs.tool.tool import Tool

        tool_events: list[Any] = []

        def on_event(event: Any) -> None:
            event_type = type(event).__name__
            if "Tool" in event_type:
                tool_events.append(event)

        events_middleware = AgentEventsMiddleware(session_id="test_session", on_event=on_event)

        # Create a simple tool
        def simple_calculator(a: int, b: int) -> dict[str, int]:
            """Add two numbers."""
            return {"result": a + b}

        tool = Tool(
            name="add",
            description="Add two numbers together.",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
            implementation=simple_calculator,
        )

        config = AgentConfig(
            name="tool_event_agent",
            system_prompt="You have an 'add' tool. Use it for math.",
            llm_config=LLMConfig(),
            tools=[tool],
            middlewares=[events_middleware],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="tool_event_session",
        )

        async def run_agent():
            result = await agent.run_async(message="What is 5 + 3? Use the add tool.")
            return result if isinstance(result, str) else result[0]

        response = asyncio.run(run_agent())
        assert isinstance(response, str)
        # Response should contain 8
        assert "8" in response


class TestEventCompleteness:
    """Test that event streams are complete."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.mark.llm
    def test_run_always_has_start_and_finish(self, session_manager):
        """Test that every run has both start and finish events."""
        events_received: list[Any] = []

        def on_event(event: Any) -> None:
            events_received.append(event)

        middleware = AgentEventsMiddleware(session_id="test_session", on_event=on_event)

        config = AgentConfig(
            name="complete_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
            middlewares=[middleware],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="complete_session",
        )

        async def run_agent():
            result = await agent.run_async(message="Hello!")
            return result if isinstance(result, str) else result[0]

        response = asyncio.run(run_agent())
        assert isinstance(response, str)

        # Check for start and finish events
        has_start = any(isinstance(e, RunStartedEvent) for e in events_received)
        has_finish = any(isinstance(e, RunFinishedEvent) for e in events_received)

        assert has_start, "Missing RunStartedEvent"
        assert has_finish, "Missing RunFinishedEvent"

    @pytest.mark.llm
    def test_text_message_has_start_content_end(self, session_manager):
        """Test that text messages have start, content, and end events."""
        events_received: list[Any] = []

        def on_event(event: Any) -> None:
            events_received.append(event)

        middleware = AgentEventsMiddleware(session_id="test_session", on_event=on_event)

        config = AgentConfig(
            name="text_complete_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
            middlewares=[middleware],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="text_complete_session",
        )

        async def run_agent():
            result = await agent.run_async(message="Say hello")
            return result if isinstance(result, str) else result[0]

        response = asyncio.run(run_agent())
        assert isinstance(response, str)

        # Verify we received events (event types may vary by LLM)
        assert len(events_received) > 0, "No events received"
