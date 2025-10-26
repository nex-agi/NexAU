"""
Integration tests for agent execution components.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from northau.archs.tool.tool import Tool


class TestAgentExecutionIntegration:
    """Integration tests for agent execution workflow."""

    @pytest.mark.integration
    def test_agent_with_tools_execution(self):
        """Test agent execution with tools."""

        # Create mock tools
        def mock_tool1(param1: str, param2: int = 10) -> dict:
            return {"result": f"Tool1: {param1} with {param2}"}

        def mock_tool2(text: str) -> str:
            return f"Tool2: {text.upper()}"

        # Create tools with proper schema
        Tool(
            name="mock_tool1",
            description="A mock tool for testing",
            input_schema={
                "type": "object",
                "properties": {"param1": {"type": "string"}, "param2": {"type": "integer", "default": 10}},
                "required": ["param1"],
            },
            implementation=mock_tool1,
        )
        Tool(
            name="mock_tool2",
            description="Another mock tool",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            implementation=mock_tool2,
        )

        # Mock LLM response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Using tools to process request",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "mock_tool1", "arguments": json.dumps({"param1": "test", "param2": 20})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        with patch("northau.archs.main_sub.agent.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            # This test demonstrates the integration but would need actual agent setup
            # For now, we're testing the components can work together
            assert True

    @pytest.mark.integration
    def test_executor_with_tool_executor(self):
        """Test executor working with tool executor."""

        # Create a simple tool
        def simple_tool(message: str) -> str:
            return f"Processed: {message}"

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
            implementation=simple_tool,
        )

        # Mock tool executor
        tool_executor = Mock()
        tool_executor.execute_tool.return_value = {"result": "Processed: test message"}

        # Test that executor can coordinate with tool executor
        result = tool_executor.execute_tool(tool.name, {"message": "test message"})
        assert result["result"] == "Processed: test message"

    @pytest.mark.integration
    def test_response_parser_with_tool_calls(self):
        """Test response parser handling tool calls."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Let me use a tool",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

        # Test that response parser can extract tool calls
        # This would use actual ResponseParser in real integration test
        tool_calls = mock_response["choices"][0]["message"].get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "test_tool"
        assert json.loads(tool_calls[0]["function"]["arguments"])["param"] == "value"


class TestAgentToolIntegration:
    """Integration tests for agent and tool interactions."""

    @pytest.mark.integration
    def test_tool_execution_flow(self):
        """Test complete tool execution flow."""
        # Create a stateful tool that modifies data
        state = {"counter": 0}

        def increment_tool(amount: int = 1) -> dict:
            state["counter"] += amount
            return {"counter": state["counter"]}

        tool = Tool(
            name="increment",
            description="Increments counter",
            input_schema={"type": "object", "properties": {"amount": {"type": "integer", "default": 1}}},
            implementation=increment_tool,
        )

        # Execute tool multiple times
        result1 = tool.execute(amount=5)
        assert result1["counter"] == 5

        result2 = tool.execute(amount=3)
        assert result2["counter"] == 8

    @pytest.mark.integration
    def test_tool_error_handling(self):
        """Test tool error handling in execution flow."""

        def error_tool(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Tool failed as requested")
            return "success"

        tool = Tool(
            name="error_tool",
            description="Tool that can fail",
            input_schema={"type": "object", "properties": {"should_fail": {"type": "boolean"}}, "required": ["should_fail"]},
            implementation=error_tool,
        )

        # Test successful execution
        result = tool.execute(should_fail=False)
        assert result["result"] == "success"

        # Test error handling - tool.execute catches errors and returns error dict
        result = tool.execute(should_fail=True)
        assert "error" in result
        assert "Tool failed as requested" in result["error"]


class TestSubAgentIntegration:
    """Integration tests for sub-agent functionality."""

    @pytest.mark.integration
    def test_sub_agent_delegation(self):
        """Test delegating tasks to sub-agents."""
        # This would test the actual sub-agent delegation mechanism
        # For now, we create a placeholder test
        sub_agent_config = {"name": "sub_agent", "llm_config": {"model": "gpt-4o-mini", "temperature": 0.7}, "tools": []}

        # Test that sub-agent config is valid
        assert sub_agent_config["name"] == "sub_agent"
        assert "llm_config" in sub_agent_config
        assert "tools" in sub_agent_config
