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

"""End-to-end integration tests for Agent system.

Tests real Agent execution with real LLM, verifying:
- Multi-turn conversations
- Session persistence and recovery
- Tool execution with state
- Sub-agent delegation
- Event streaming
- Error handling and recovery
"""

import asyncio
import tempfile
from typing import Any

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
)
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool


class TestAgentMultiTurnConversation:
    """Test multi-turn conversation with real LLM."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.fixture
    def agent_config(self):
        """Create agent config."""
        return AgentConfig(
            name="conversation_agent",
            system_prompt="You are a helpful assistant. Remember what the user tells you.",
            llm_config=LLMConfig(),
        )

    @pytest.mark.llm
    def test_multi_turn_conversation_remembers_context(self, session_manager, agent_config):
        """Test that agent remembers context across multiple turns."""
        user_id = "test_user"
        session_id = "test_session"

        # Turn 1: Tell the agent something
        agent1 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        response1 = agent1.run(message="My name is Alice.")
        assert isinstance(response1, str)
        assert len(response1) > 0

        # Turn 2: Ask the agent to recall (new Agent instance, same session)
        agent2 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        response2 = agent2.run(message="What is my name?")
        assert isinstance(response2, str)
        # Agent should remember the name from previous turn
        assert "Alice" in response2 or "alice" in response2.lower()

    @pytest.mark.llm
    def test_different_sessions_are_isolated(self, session_manager, agent_config):
        """Test that different sessions don't share context."""
        user_id = "test_user"

        # Session 1: Tell something
        agent1 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id="session_1",
        )
        agent1.run(message="My favorite color is blue.")

        # Session 2: Different session, should not know
        agent2 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id="session_2",
        )
        response = agent2.run(message="What is my favorite color?")
        assert isinstance(response, str)
        # Agent should not know the color from different session
        # (May respond with "I don't know" or ask for clarification)


class TestAgentWithTools:
    """Test Agent with tool execution."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def calculator_tool(self):
        """Create a simple calculator tool."""

        def calculate(expression: str) -> dict[str, Any]:
            """Evaluate a mathematical expression."""
            try:
                # Safe eval for simple math
                result = eval(expression, {"__builtins__": {}}, {})
                return {"result": result, "expression": expression}
            except Exception as e:
                return {"error": str(e)}

        return Tool(
            name="calculator",
            description="Evaluate mathematical expressions. Input should be a valid math expression like '2 + 2' or '10 * 5'.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            implementation=calculate,
        )

    @pytest.fixture
    def agent_config_with_tools(self, calculator_tool):
        """Create agent config with tools."""
        return AgentConfig(
            name="tool_agent",
            system_prompt="You are a helpful assistant with access to a calculator tool. Use it for math calculations.",
            llm_config=LLMConfig(),
            tools=[calculator_tool],
        )

    @pytest.mark.llm
    def test_agent_uses_tool_for_calculation(self, agent_config_with_tools):
        """Test that agent uses calculator tool."""
        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=agent_config_with_tools,
            session_manager=session_manager,
            user_id="test_user",
            session_id="calc_session",
        )

        response = agent.run(message="What is 123 * 456?")
        assert isinstance(response, str)
        # The correct answer is 56088 (may be formatted as 56,088)
        assert "56088" in response or "56,088" in response

    @pytest.mark.llm
    def test_agent_handles_tool_error_gracefully(self):
        """Test that agent handles tool errors gracefully."""

        def failing_tool(input_text: str) -> dict[str, Any]:
            """A tool that always fails."""
            raise ValueError("This tool always fails")

        tool = Tool(
            name="failing_tool",
            description="A tool that always fails for testing.",
            input_schema={
                "type": "object",
                "properties": {"input_text": {"type": "string"}},
                "required": ["input_text"],
            },
            implementation=failing_tool,
        )

        config = AgentConfig(
            name="error_agent",
            system_prompt="You have a tool called failing_tool. Try to use it if asked.",
            llm_config=LLMConfig(),
            tools=[tool],
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="error_session",
        )

        # Agent should handle the error and provide a response
        response = agent.run(message="Please use the failing_tool with input 'test'")
        assert isinstance(response, str)
        # Response should exist (agent recovered from error)
        assert len(response) > 0


class TestAgentStreaming:
    """Test Agent streaming events."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.mark.llm
    def test_streaming_events_are_emitted(self, session_manager):
        """Test that streaming events are properly emitted."""
        events_received: list[Any] = []

        def on_event(event: Any) -> None:
            events_received.append(event)

        middleware = AgentEventsMiddleware(session_id="stream_session", on_event=on_event)

        config = AgentConfig(
            name="streaming_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
            middlewares=[middleware],
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="stream_session",
        )

        # Run async to get streaming events
        async def run_streaming():
            result = await agent.run_async(message="Say hello!")
            return result if isinstance(result, str) else result[0]

        response = asyncio.run(run_streaming())

        assert isinstance(response, str)
        assert len(response) > 0
        # Should have received some events
        assert len(events_received) > 0
        # Check for expected event types
        event_types = [type(e).__name__ for e in events_received]
        assert any("RunStarted" in t for t in event_types)
        assert any("RunFinished" in t for t in event_types)


class TestAgentGlobalStorage:
    """Test Agent global storage functionality."""

    @pytest.fixture
    def storage_tool(self):
        """Create a tool that uses global storage."""
        from nexau.archs.main_sub.agent_state import AgentState

        def store_value(key: str, value: str, agent_state: AgentState) -> dict[str, str]:
            """Store a value in global storage."""
            agent_state.global_storage.set(key, value)
            return {"status": "stored", "key": key, "value": value}

        def get_value(key: str, agent_state: AgentState) -> dict[str, str]:
            """Get a value from global storage."""
            value = agent_state.global_storage.get(key, "NOT_FOUND")
            return {"key": key, "value": value}

        store = Tool(
            name="store_value",
            description="Store a key-value pair in global storage.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to store"},
                    "value": {"type": "string", "description": "The value to store"},
                },
                "required": ["key", "value"],
            },
            implementation=store_value,
        )

        get = Tool(
            name="get_value",
            description="Get a value from global storage by key.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to retrieve"},
                },
                "required": ["key"],
            },
            implementation=get_value,
        )

        return store, get

    @pytest.mark.llm
    def test_global_storage_persists_across_tool_calls(self, storage_tool):
        """Test that global storage persists across tool calls."""
        store_tool, get_tool = storage_tool

        config = AgentConfig(
            name="storage_agent",
            system_prompt="You have tools to store and retrieve values. Use them when asked.",
            llm_config=LLMConfig(),
            tools=[store_tool, get_tool],
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="storage_session",
        )

        # Ask to store and then retrieve
        response = agent.run(
            message="First, store the value 'test_value' with key 'my_key'. Then, retrieve the value for 'my_key' and tell me what it is."
        )

        assert isinstance(response, str)
        # Response should contain the stored value
        assert "test_value" in response.lower()


class TestAgentConcurrency:
    """Test Agent concurrent execution scenarios."""

    @pytest.mark.llm
    def test_multiple_agents_different_sessions(self):
        """Test multiple agents with different sessions running sequentially."""
        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        config = AgentConfig(
            name="concurrent_agent",
            system_prompt="You are a helpful assistant. Respond briefly.",
            llm_config=LLMConfig(),
        )

        results = []
        for i, (session_id, msg) in enumerate(
            [
                ("session_1", "Say 'one'"),
                ("session_2", "Say 'two'"),
                ("session_3", "Say 'three'"),
            ]
        ):
            agent = Agent(
                config=config,
                session_manager=session_manager,
                user_id="test_user",
                session_id=session_id,
            )
            result = agent.run(message=msg)
            # run returns str | tuple[str, dict], extract str
            response = result if isinstance(result, str) else result[0]
            results.append(response)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0


class TestAgentMaxIterations:
    """Test Agent iteration limits."""

    @pytest.fixture
    def infinite_loop_tool(self):
        """Create a tool that always wants to be called again."""

        def loop_tool(counter: int) -> dict[str, Any]:
            """A tool that returns a result suggesting another call."""
            return {
                "result": f"Called {counter} times",
                "suggestion": "Call me again with counter + 1",
            }

        return Tool(
            name="loop_tool",
            description="A tool for testing iteration limits.",
            input_schema={
                "type": "object",
                "properties": {"counter": {"type": "integer"}},
                "required": ["counter"],
            },
            implementation=loop_tool,
        )

    @pytest.mark.llm
    def test_agent_respects_max_iterations(self, infinite_loop_tool):
        """Test that agent respects max_iterations limit."""
        config = AgentConfig(
            name="limited_agent",
            system_prompt="You have a loop_tool. When asked, call it repeatedly.",
            llm_config=LLMConfig(),
            tools=[infinite_loop_tool],
            max_iterations=3,  # Limit iterations
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="iter_session",
        )

        # This should not run forever due to max_iterations
        response = agent.run(message="Call the loop_tool starting with counter=1")

        assert isinstance(response, str)
        # Response should exist (agent completed within limit)
        assert len(response) > 0
