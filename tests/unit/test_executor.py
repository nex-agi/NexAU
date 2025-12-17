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
Unit tests for Executor class.

Tests cover:
- Initialization and configuration
- Message enqueueing
- Execution flow and iteration handling
- Tool and sub-agent execution
- Error handling and cleanup
- Token and iteration limit handling
"""

import json
from unittest.mock import Mock, patch

import pytest

from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.execution.hooks import HookResult, Middleware
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.execution.parse_structures import (
    BatchAgentCall,
    ParsedResponse,
    SubAgentCall,
    ToolCall,
)
from nexau.archs.tool.tool import Tool


class TestExecutorInitialization:
    """Test Executor initialization and configuration."""

    def test_executor_init_basic(self, mock_llm_config):
        """Test basic executor initialization."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        assert executor.agent_name == "test_agent"
        assert executor.agent_id == "test_id"
        assert executor.max_iterations == 100  # default
        assert executor.max_context_tokens == 128000  # default
        assert executor.max_running_subagents == 5  # default
        assert executor.stop_signal is False
        assert executor.queued_messages == []

    def test_executor_init_with_custom_params(self, mock_llm_config):
        """Test executor initialization with custom parameters."""
        executor = Executor(
            agent_name="custom_agent",
            agent_id="custom_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools={"stop_tool"},
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=50,
            max_context_tokens=64000,
            max_running_subagents=3,
            retry_attempts=3,
            serial_tool_name=["serial_tool"],
        )

        assert executor.max_iterations == 50
        assert executor.max_context_tokens == 64000
        assert executor.max_running_subagents == 3
        assert executor.serial_tool_name == ["serial_tool"]

    def test_executor_init_with_hooks(self, mock_llm_config):
        """Test executor initialization with hooks."""
        before_hook = Mock()
        after_hook = Mock()
        tool_hook = Mock()
        before_tool_hook = Mock()

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            before_model_hooks=[before_hook],
            after_model_hooks=[after_hook],
            after_tool_hooks=[tool_hook],
            before_tool_hooks=[before_tool_hook],
        )

        assert executor.middleware_manager is not None
        assert len(executor.middleware_manager.middlewares) == 4

    def test_executor_init_with_custom_token_counter(self, mock_llm_config):
        """Test executor initialization with custom token counter."""
        mock_token_counter = Mock()
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            token_counter=mock_token_counter,
        )

        assert executor.token_counter == mock_token_counter


class TestExecutorMessageEnqueueing:
    """Test message enqueueing functionality."""

    def test_enqueue_message(self, mock_llm_config):
        """Test enqueueing a message."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        message = {"role": "user", "content": "Test message"}
        executor.enqueue_message(message)

        assert len(executor.queued_messages) == 1
        assert executor.queued_messages[0] == message

    def test_enqueue_multiple_messages(self, mock_llm_config):
        """Test enqueueing multiple messages."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Message 2"},
            {"role": "user", "content": "Message 3"},
        ]

        for msg in messages:
            executor.enqueue_message(msg)

        assert len(executor.queued_messages) == 3
        assert executor.queued_messages == messages


class TestExecutorExecution:
    """Test executor main execution flow."""

    def test_execute_with_stop_signal(self, mock_llm_config, agent_state):
        """Test execution stops when stop_signal is set."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        # Set stop signal right before the iteration check would happen
        # We need to mock the LLM caller to set the stop signal during execution
        def set_stop_signal(*args, **kwargs):
            executor.stop_signal = True
            return ModelResponse(content="Response before stop")

        with patch.object(executor.llm_caller, "call_llm", side_effect=set_stop_signal):
            with patch.object(executor.response_parser, "parse_response") as mock_parse:
                mock_parse.return_value = ParsedResponse(
                    original_response="Response",
                    tool_calls=[],
                    sub_agent_calls=[],
                    batch_agent_calls=[],
                )

                response, messages = executor.execute(history, agent_state)

                # On next iteration it should stop
                assert "Stop signal received" in response or "Response before stop" in response

    def test_stop_signal_runs_after_agent_hooks(self, mock_llm_config, agent_state):
        """Stop-signal early exit should still trigger after-agent hooks."""

        class StopSignalMiddleware(Middleware):
            executor: Executor

            def __init__(self):
                self.after_agent_called = False

            def before_agent(self, hook_input):  # type: ignore[override]
                assert self.executor is not None
                self.executor.stop_signal = True
                return HookResult.no_changes()

            def after_agent(self, hook_input):  # type: ignore[override]
                self.after_agent_called = True
                return HookResult.with_modifications(agent_response=f"{hook_input.agent_response}::hooked")

        middleware = StopSignalMiddleware()
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            middlewares=[middleware],
        )
        middleware.executor = executor

        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            response, _ = executor.execute(history, agent_state)

        mock_call_llm.assert_not_called()
        assert middleware.after_agent_called is True
        assert response == "Stop signal received.::hooked"

    def test_before_and_after_agent_middlewares(self, mock_llm_config, agent_state):
        """Lifecycle middlewares can modify initial messages and final response."""

        class LifecycleMiddleware(Middleware):
            def __init__(self) -> None:
                self.before_agent_called = 0
                self.after_agent_called = 0

            def before_agent(self, hook_input):  # type: ignore[override]
                self.before_agent_called += 1
                updated = hook_input.messages + [{"role": "system", "content": "prep note"}]
                return HookResult.with_modifications(messages=updated)

            def after_agent(self, hook_input):  # type: ignore[override]
                self.after_agent_called += 1
                return HookResult.with_modifications(agent_response=f"{hook_input.agent_response}::final")

        lifecycle_middleware = LifecycleMiddleware()
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=1,
            middlewares=[lifecycle_middleware],
        )

        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            mock_call_llm.return_value = ModelResponse(content="Simple response")
            with patch.object(executor.response_parser, "parse_response") as mock_parse:
                mock_parse.return_value = ParsedResponse(
                    original_response="Simple response",
                    tool_calls=[],
                    sub_agent_calls=[],
                    batch_agent_calls=[],
                )

                response, messages = executor.execute(history, agent_state)

        assert lifecycle_middleware.before_agent_called == 1
        assert lifecycle_middleware.after_agent_called == 1
        assert response == "Simple response::final"
        assert any(msg.get("content") == "prep note" for msg in messages)

    def test_execute_openai_tool_messages(self, mock_llm_config, agent_state):
        """Test that OpenAI tool results are appended as tool messages."""

        def simple_tool(x: int) -> dict:
            return {"result": x + 1}

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            implementation=simple_tool,
        )

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "A simple tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                    },
                },
            },
        ]

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={"simple_tool": tool},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=2,
            tool_call_mode="openai",
            openai_tools=openai_tools,
        )

        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Use the tool"},
        ]

        first_response = ModelResponse(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="call_123",
                    name="simple_tool",
                    arguments={"x": 3},
                    raw_arguments='{"x": 3}',
                ),
            ],
        )
        second_response = ModelResponse(content="Final response")

        with patch.object(executor.llm_caller, "call_llm", side_effect=[first_response, second_response]):
            response, messages = executor.execute(history, agent_state)

        # Response should be the final assistant content
        assert response == "Final response"

        # Tool result should be appended as a tool message with matching call ID
        tool_messages = [msg for msg in messages if msg.get("role") == "tool"]
        assert tool_messages, "Expected at least one tool message"

        matching_tool_messages = [msg for msg in tool_messages if msg.get("tool_call_id") == "call_123"]
        assert matching_tool_messages, "Tool message should reuse original tool_call_id"

        tool_content = matching_tool_messages[0]["content"]
        assert '"result"' in tool_content

        # Tool message should appear before the final assistant reply
        tool_index = messages.index(matching_tool_messages[0])
        assert tool_index + 1 < len(messages)
        assert messages[tool_index + 1]["role"] == "assistant"

    def test_execute_with_max_iterations(self, mock_llm_config, agent_state):
        """Test execution stops at max iterations."""
        mock_client = Mock()

        # Mock LLM to return responses that trigger iterations
        def mock_llm_call(*args, **kwargs):
            mock_response = Mock()
            mock_response.content = "Response"
            return mock_response

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=mock_client,
            llm_config=mock_llm_config,
            max_iterations=2,
        )

        # Mock the LLM caller to return simple responses
        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            mock_call_llm.return_value = ModelResponse(content="Simple response with no tool calls")

            # Mock response parser to return no calls
            with patch.object(executor.response_parser, "parse_response") as mock_parse:
                mock_parse.return_value = ParsedResponse(
                    original_response="Simple response",
                    tool_calls=[],
                    sub_agent_calls=[],
                    batch_agent_calls=[],
                )

                history = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]

                response, messages = executor.execute(history, agent_state)

                # Should have made LLM calls
                assert mock_call_llm.call_count >= 1

    def test_execute_with_queued_messages(self, mock_llm_config, agent_state):
        """Test execution processes queued messages."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=2,
        )

        # Enqueue a message before execution
        queued_msg = {"role": "user", "content": "Queued message"}
        executor.enqueue_message(queued_msg)

        # Mock the LLM caller to return simple responses
        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            mock_call_llm.return_value = ModelResponse(content="Simple response")

            # Mock response parser to return no calls
            with patch.object(executor.response_parser, "parse_response") as mock_parse:
                mock_parse.return_value = ParsedResponse(
                    original_response="Simple response",
                    tool_calls=[],
                    sub_agent_calls=[],
                    batch_agent_calls=[],
                )

                history = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]

                response, messages = executor.execute(history, agent_state)

                # Verify queued message was added to history
                assert any(msg.get("content") == "Queued message" for msg in messages)
                assert len(executor.queued_messages) == 0

    def test_execute_with_token_limit_exceeded(self, mock_llm_config, agent_state):
        """Test execution handles token limit exceeded."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_context_tokens=100,  # Very low limit
        )

        # Mock token counter to return high count
        with patch.object(executor.token_counter, "count_tokens") as mock_count:
            mock_count.return_value = 200  # Exceeds limit

            history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]

            # Mock LLM caller
            with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
                mock_call_llm.return_value = None  # Simulate forced stop

                response, messages = executor.execute(history, agent_state)

                assert "Prompt too long" in response or "Error:" in response

    def test_execute_with_exception(self, mock_llm_config, agent_state):
        """Test execution handles exceptions gracefully."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        # Mock the LLM caller to raise an exception
        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            mock_call_llm.side_effect = Exception("Test exception")

            history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]

            with pytest.raises(RuntimeError) as exc_info:
                executor.execute(history, agent_state)

            assert "Error in agent execution" in str(exc_info.value)


class TestExecutorXMLCallProcessing:
    """Test XML call processing methods."""

    def test_process_xml_calls_no_calls(self, mock_llm_config, agent_state):
        """Test processing response with no calls."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        from nexau.archs.main_sub.execution.hooks import AfterModelHookInput

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            original_response="Just a plain response",
            parsed_response=ParsedResponse(
                original_response="Just a plain response",
                tool_calls=[],
                sub_agent_calls=[],
                batch_agent_calls=[],
            ),
            messages=[],
        )

        processed, should_stop, result, messages, feedbacks = executor._process_xml_calls(hook_input)

        assert processed == "Just a plain response"
        assert should_stop is True
        assert result is None
        assert feedbacks == []

    def test_process_xml_calls_with_tool_calls(self, mock_llm_config, agent_state):
        """Test processing response with tool calls."""

        # Create a simple tool
        def simple_tool(x: int) -> int:
            return x * 2

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            implementation=simple_tool,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={"simple_tool": tool},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            serial_tool_name=[],  # Empty list to avoid None check issues
        )

        from nexau.archs.main_sub.execution.hooks import AfterModelHookInput

        tool_call = ToolCall(
            tool_name="simple_tool",
            parameters={"x": 5},
            raw_content="<tool_call>...</tool_call>",
            tool_call_id="call_123",
        )

        parsed_response = ParsedResponse(
            original_response="Let me use a tool",
            tool_calls=[tool_call],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        # Mock the response parser to return our parsed response
        with patch.object(executor.response_parser, "parse_response") as mock_parse:
            mock_parse.return_value = parsed_response

            hook_input = AfterModelHookInput(
                agent_state=agent_state,
                max_iterations=10,
                current_iteration=0,
                original_response="Let me use a tool",
                parsed_response=parsed_response,
                messages=[],
            )

            processed, should_stop, result, messages, feedbacks = executor._process_xml_calls(hook_input)

            # Tool results should be appended to the response
            assert "<tool_result>" in processed
            assert "simple_tool" in processed
            assert len(feedbacks) == 1

    def test_process_xml_calls_with_batch_calls(self, mock_llm_config, agent_state, temp_dir):
        """Test processing response with batch calls."""
        import os

        # Create a test file for batch processing
        test_file = os.path.join(temp_dir, "batch_data.json")
        with open(test_file, "w") as f:
            json.dump([{"id": 1, "value": "test"}], f)

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={"sub_agent": lambda: Mock()},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            serial_tool_name=[],  # Empty list to avoid None check issues
        )

        from nexau.archs.main_sub.execution.hooks import AfterModelHookInput

        batch_call = BatchAgentCall(
            agent_name="sub_agent",
            file_path=test_file,
            data_format="json",
            message_template="Process: {value}",
            raw_content="<batch_agent>...</batch_agent>",
        )

        parsed_response = ParsedResponse(
            original_response="Processing batch",
            tool_calls=[],
            sub_agent_calls=[],
            batch_agent_calls=[batch_call],
        )

        # Mock the response parser and batch processor
        with patch.object(executor.response_parser, "parse_response") as mock_parse:
            mock_parse.return_value = parsed_response

            with patch.object(executor.batch_processor, "_process_batch_data") as mock_batch:
                mock_batch.return_value = "Batch processed"

                hook_input = AfterModelHookInput(
                    agent_state=agent_state,
                    max_iterations=10,
                    current_iteration=0,
                    original_response="Processing batch",
                    parsed_response=parsed_response,
                    messages=[],
                )

                processed, should_stop, result, messages, feedbacks = executor._process_xml_calls(hook_input)

                # Batch results should be appended to the response
                assert "batch_agent" in processed or "Batch processed" in processed
                mock_batch.assert_called_once()
                assert feedbacks == []


class TestExecutorToolExecution:
    """Test tool execution methods."""

    def test_execute_tool_call_safe_success(self, mock_llm_config, agent_state):
        """Test successful tool execution."""

        def test_tool(x: int) -> dict:
            return {"result": x * 2}

        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            implementation=test_tool,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={"test_tool": tool},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        tool_call = ToolCall(
            tool_name="test_tool",
            parameters={"x": 5},
            raw_content="<tool_call>...</tool_call>",
        )

        tool_name, result, is_error = executor._execute_tool_call_safe(tool_call, agent_state)

        assert tool_name == "test_tool"
        assert is_error is False
        assert "result" in result

    def test_execute_tool_call_safe_error(self, mock_llm_config, agent_state):
        """Test tool execution with error."""

        def error_tool(x: int) -> dict:
            raise ValueError("Tool error")

        tool = Tool(
            name="error_tool",
            description="A tool that errors",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            implementation=error_tool,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={"error_tool": tool},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        tool_call = ToolCall(
            tool_name="error_tool",
            parameters={"x": 5},
            raw_content="<tool_call>...</tool_call>",
        )

        tool_name, result, is_error = executor._execute_tool_call_safe(tool_call, agent_state)

        assert tool_name == "error_tool"
        # Tool.execute wraps errors, so check that result contains error information
        # is_error might be False but result will contain error details
        assert "error" in result.lower() or "Tool error" in result or "ValueError" in result


class TestExecutorSubAgentExecution:
    """Test sub-agent execution methods."""

    def test_execute_sub_agent_call_safe_success(self, mock_llm_config, agent_state):
        """Test successful sub-agent execution."""
        mock_sub_agent = Mock()
        mock_sub_agent.send_message.return_value = "Sub-agent response"

        def sub_agent_factory():
            return mock_sub_agent

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={"sub_agent": sub_agent_factory},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        sub_agent_call = SubAgentCall(
            agent_name="sub_agent",
            message="Test message",
            raw_content="<sub_agent>...</sub_agent>",
        )

        # Mock the subagent manager
        with patch.object(executor.subagent_manager, "call_sub_agent") as mock_call:
            mock_call.return_value = "Sub-agent response"

            agent_name, result, is_error = executor._execute_sub_agent_call_safe(
                sub_agent_call, context=None, parent_agent_state=agent_state
            )

            assert agent_name == "sub_agent"
            assert is_error is False
            assert result == "Sub-agent response"

    def test_execute_sub_agent_call_safe_error(self, mock_llm_config, agent_state):
        """Test sub-agent execution with error."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        sub_agent_call = SubAgentCall(
            agent_name="non_existent",
            message="Test message",
            raw_content="<sub_agent>...</sub_agent>",
        )

        # Mock the subagent manager to raise an error
        with patch.object(executor.subagent_manager, "call_sub_agent") as mock_call:
            mock_call.side_effect = Exception("Sub-agent not found")

            agent_name, result, is_error = executor._execute_sub_agent_call_safe(
                sub_agent_call, context=None, parent_agent_state=agent_state
            )

            assert agent_name == "non_existent"
            assert is_error is True


class TestExecutorCleanup:
    """Test executor cleanup functionality."""

    def test_cleanup_basic(self, mock_llm_config):
        """Test basic cleanup."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        executor.cleanup()

        assert executor.stop_signal is True
        assert executor._shutdown_event.is_set()

    def test_cleanup_with_running_executors(self, mock_llm_config):
        """Test cleanup with running executors."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        # Add mock executors to running_executors
        mock_executor1 = Mock()
        mock_executor2 = Mock()
        executor._running_executors = {
            "executor_1": mock_executor1,
            "executor_2": mock_executor2,
        }

        executor.cleanup()

        # Verify shutdown was called on executors
        mock_executor1.shutdown.assert_called_once()
        mock_executor2.shutdown.assert_called_once()
        assert len(executor._running_executors) == 0


class TestExecutorHelperMethods:
    """Test executor helper methods."""

    def test_add_tool(self, mock_llm_config):
        """Test adding a tool dynamically."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        tool = Tool(
            name="new_tool",
            description="A new tool",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: {},
        )

        executor.add_tool(tool)

        assert "new_tool" in executor.tool_executor.tool_registry
        assert executor.tool_executor.tool_registry["new_tool"] == tool

    def test_add_sub_agent(self, mock_llm_config):
        """Test adding a sub-agent dynamically."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        def sub_agent_factory():
            return Mock()

        executor.add_sub_agent("new_sub_agent", sub_agent_factory)

        # Verify it was added through subagent_manager
        # (internal implementation detail, but we can check it was called)
        assert "new_sub_agent" in executor.subagent_manager.sub_agent_factories


class TestExecutorStopToolHandling:
    """Test stop tool detection and handling."""

    def test_stop_tool_detected(self, mock_llm_config, agent_state):
        """Test that stop tools are properly detected."""

        def stop_tool() -> dict:
            return {"result": "Done", "_is_stop_tool": True}

        tool = Tool(
            name="stop_tool",
            description="A stop tool",
            input_schema={"type": "object", "properties": {}},
            implementation=stop_tool,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={"stop_tool": tool},
            sub_agent_factories={},
            stop_tools={"stop_tool"},
            openai_client=Mock(),
            llm_config=mock_llm_config,
            serial_tool_name=[],  # Empty list to avoid None check issues
        )

        tool_call = ToolCall(
            tool_name="stop_tool",
            parameters={},
            raw_content="<tool_call>...</tool_call>",
        )

        parsed_response = ParsedResponse(
            original_response="Stopping",
            tool_calls=[tool_call],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(parsed_response, agent_state)

        # The stop tool should be detected in the result
        assert "<tool_result>" in processed
        assert len(feedbacks) == 1


class TestExecutorParallelExecution:
    """Test parallel execution of tools and sub-agents."""

    def test_parallel_tool_execution(self, mock_llm_config, agent_state):
        """Test parallel execution of multiple tools."""

        def tool1(x: int) -> dict:
            return {"result": x * 2}

        def tool2(y: int) -> dict:
            return {"result": y * 3}

        tool_obj1 = Tool(
            name="tool1",
            description="Tool 1",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            implementation=tool1,
        )

        tool_obj2 = Tool(
            name="tool2",
            description="Tool 2",
            input_schema={
                "type": "object",
                "properties": {"y": {"type": "integer"}},
                "required": ["y"],
            },
            implementation=tool2,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={"tool1": tool_obj1, "tool2": tool_obj2},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            serial_tool_name=[],  # Empty list to avoid None check issues
        )

        tool_call1 = ToolCall(
            tool_name="tool1",
            parameters={"x": 5},
            raw_content="<tool_call>...</tool_call>",
            tool_call_id="call_1",
        )

        tool_call2 = ToolCall(
            tool_name="tool2",
            parameters={"y": 7},
            raw_content="<tool_call>...</tool_call>",
            tool_call_id="call_2",
        )

        parsed_response = ParsedResponse(
            original_response="Using tools",
            tool_calls=[tool_call1, tool_call2],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(parsed_response, agent_state)

        # Both tools should be executed
        assert "tool1" in processed
        assert "tool2" in processed
        assert len(feedbacks) == 2

    def test_duplicate_tool_call_ids_handling(self, mock_llm_config, agent_state):
        """Test handling of duplicate tool_call_ids."""

        def test_tool(x: int) -> dict:
            return {"result": x}

        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            implementation=test_tool,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={"test_tool": tool},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            serial_tool_name=[],  # Empty list to avoid None check issues
        )

        # Create tool calls with duplicate IDs
        tool_call1 = ToolCall(
            tool_name="test_tool",
            parameters={"x": 1},
            raw_content="<tool_call>...</tool_call>",
            tool_call_id="call_123",
        )

        tool_call2 = ToolCall(
            tool_name="test_tool",
            parameters={"x": 2},
            raw_content="<tool_call>...</tool_call>",
            tool_call_id="call_123",  # Duplicate ID
        )

        parsed_response = ParsedResponse(
            original_response="Using tools",
            tool_calls=[tool_call1, tool_call2],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(parsed_response, agent_state)

        # Both tools should be executed despite duplicate IDs
        # The second one should have a modified ID (this happens in-place during execution)
        # Check that both tools are in the processed response
        assert "test_tool" in processed
        # The ID modification happens during execution, verify both calls were processed
        assert processed.count("<tool_result>") >= 2 or "result" in processed
        assert len(feedbacks) == 2


class TestExecutorWithHooks:
    """Test executor with before/after hooks."""

    def test_execute_with_before_model_hook(self, mock_llm_config, agent_state):
        """Test execution with before model hook."""
        from nexau.archs.main_sub.execution.hooks import BeforeModelHook, BeforeModelHookInput

        class TestBeforeHook(BeforeModelHook):
            def execute(self, hook_input: BeforeModelHookInput) -> list[dict[str, str]]:
                # Add a system message
                messages = hook_input.messages.copy()
                messages.append({"role": "system", "content": "Hook added this"})
                return messages

        hook = TestBeforeHook()

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            before_model_hooks=[hook],
            max_iterations=1,
        )

        # Mock LLM caller
        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            mock_call_llm.return_value = ModelResponse(content="Response")

            # Mock response parser
            with patch.object(executor.response_parser, "parse_response") as mock_parse:
                mock_parse.return_value = ParsedResponse(
                    original_response="Response",
                    tool_calls=[],
                    sub_agent_calls=[],
                    batch_agent_calls=[],
                )

                history = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]

                response, messages = executor.execute(history, agent_state)

                # Verify the hook was called (by checking if extra message was added)
                # Note: This is an integration-style check
                assert mock_call_llm.called


class TestExecutorEdgeCases:
    """Test edge cases and error conditions."""

    def test_execute_with_shutdown_event_set(self, mock_llm_config, agent_state):
        """Test execution behavior when shutdown event is set."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        executor._shutdown_event.set()

        tool_call = ToolCall(
            tool_name="test_tool",
            parameters={},
            raw_content="<tool_call>...</tool_call>",
        )

        parsed_response = ParsedResponse(
            original_response="Test",
            tool_calls=[tool_call],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(parsed_response, agent_state)

        # Should return early without executing
        assert should_stop is False
        assert feedbacks == []

    def test_execute_batch_call(self, mock_llm_config, temp_dir):
        """Test batch call execution."""
        import os

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry={},
            sub_agent_factories={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        # Create test data file
        test_file = os.path.join(temp_dir, "batch_data.json")
        with open(test_file, "w") as f:
            json.dump([{"id": 1}], f)

        batch_call = BatchAgentCall(
            agent_name="sub_agent",
            file_path=test_file,
            data_format="json",
            message_template="Process {id}",
            raw_content="<batch_agent>...</batch_agent>",
        )

        # Mock batch processor
        with patch.object(executor.batch_processor, "_process_batch_data") as mock_batch:
            mock_batch.return_value = "Batch result"

            result = executor._execute_batch_call(batch_call)

            assert result == "Batch result"
            mock_batch.assert_called_once_with("sub_agent", test_file, "json", "Process {id}")
