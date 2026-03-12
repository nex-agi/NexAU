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

"""
Integration tests for agent execution with real LLM.

Tests tool execution flow, error handling, and sub-agent delegation.
Previously these tests used mocks; now they use real LLM calls (@pytest.mark.llm).
"""

from typing import Any

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool


class TestAgentToolExecution:
    """Integration tests for agent tool execution with real LLM."""

    @pytest.fixture
    def stateful_tool(self):
        """Create a stateful counter tool."""
        state = {"counter": 0}

        def increment(amount: int = 1) -> dict[str, Any]:
            """Increment the counter by amount."""
            state["counter"] += amount
            return {"counter": state["counter"], "amount": amount}

        return Tool(
            name="increment_counter",
            description="Increment a counter by a given amount. Returns the new counter value.",
            input_schema={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "description": "Amount to increment by",
                    }
                },
                "required": ["amount"],
            },
            implementation=increment,
        )

    @pytest.mark.llm
    def test_agent_calls_tool_and_uses_result(self, stateful_tool):
        """Test agent calls a tool and incorporates the result into its response."""
        config = AgentConfig(
            name="tool_agent",
            system_prompt="You have an increment_counter tool. Use it when asked to increment.",
            llm_config=LLMConfig(),
            tools=[stateful_tool],
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="tool_exec_session",
        )

        response = agent.run(message="Increment the counter by 10. Tell me the new value.")
        assert isinstance(response, str)
        assert "10" in response

    @pytest.mark.llm
    def test_agent_handles_tool_error(self):
        """Test agent gracefully handles tool execution errors."""

        def failing_tool(input_text: str) -> dict[str, Any]:
            """A tool that always fails."""
            raise ValueError("Intentional failure for testing")

        tool = Tool(
            name="failing_tool",
            description="A tool that always fails. Use it when asked to test error handling.",
            input_schema={
                "type": "object",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "Input text",
                    }
                },
                "required": ["input_text"],
            },
            implementation=failing_tool,
        )

        config = AgentConfig(
            name="error_agent",
            system_prompt="You have a failing_tool. Use it when asked, and report errors to the user.",
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

        response = agent.run(message="Use the failing_tool with input 'test'.")
        assert isinstance(response, str)
        # Agent should recover from error and provide a response
        assert len(response) > 0

    @pytest.mark.llm
    def test_agent_multiple_tool_calls_in_sequence(self):
        """Test agent making multiple sequential tool calls."""

        def string_tool(text: str, operation: str) -> dict[str, str]:
            """Transform a string."""
            if operation == "upper":
                return {"result": text.upper()}
            if operation == "reverse":
                return {"result": text[::-1]}
            return {"result": text}

        tool = Tool(
            name="string_transform",
            description="Transform a string. Operations: 'upper' (uppercase), 'reverse' (reverse).",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Input text"},
                    "operation": {
                        "type": "string",
                        "description": "Operation: 'upper' or 'reverse'",
                        "enum": ["upper", "reverse"],
                    },
                },
                "required": ["text", "operation"],
            },
            implementation=string_tool,
        )

        config = AgentConfig(
            name="multi_tool_agent",
            system_prompt="You have a string_transform tool. Use it as asked.",
            llm_config=LLMConfig(),
            tools=[tool],
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="multi_tool_session",
        )

        response = agent.run(message="First uppercase 'hello', then reverse the result. Tell me the final string.")
        assert isinstance(response, str)
        # "hello" → "HELLO" → "OLLEH"
        assert "OLLEH" in response
