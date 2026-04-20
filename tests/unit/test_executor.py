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

from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.execution.hooks import HookResult, Middleware
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.execution.parse_structures import (
    ParsedResponse,
    ToolCall,
)
from nexau.archs.main_sub.execution.tool_executor import ToolExecutionResult
from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool.tool import Tool, build_structured_tool_definition
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolOutputImage, ToolResultBlock


def make_history(system_text: str, user_text: str) -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content=[TextBlock(text=system_text)]),
        Message.user(user_text),
    ]


def make_tool_registry(tools: dict[str, Tool] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    if tools:
        registry.add_source("test", list(tools.values()))
    return registry


def make_tool(
    name: str,
    *,
    disable_parallel: bool = False,
    defer_loading: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": name},
        disable_parallel=disable_parallel,
        defer_loading=defer_loading,
    )


def make_framework_context(executor: Executor) -> FrameworkContext:
    return FrameworkContext(
        agent_name=executor.agent_name,
        agent_id=executor.agent_id,
        run_id="run_123",
        root_run_id="root_123",
        _tool_registry=executor._tool_registry,
        _shutdown_event=executor._shutdown_event,
    )


class TestExecutorInitialization:
    """Test Executor initialization and configuration."""

    def test_executor_init_basic(self, mock_llm_config):
        """Test basic executor initialization."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
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
        serial_tool = make_tool("serial_tool", disable_parallel=True)
        registry = make_tool_registry({"serial_tool": serial_tool})
        executor = Executor(
            agent_name="custom_agent",
            agent_id="custom_id",
            tool_registry=registry,
            sub_agents={},
            stop_tools={"stop_tool"},
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=50,
            max_context_tokens=64000,
            max_running_subagents=3,
            retry_attempts=3,
        )

        assert executor.max_iterations == 50
        assert executor.max_context_tokens == 64000
        assert executor.max_running_subagents == 3
        assert executor._tool_registry.get_all()["serial_tool"].disable_parallel is True
        assert registry.compute_serial_tool_names() == ["serial_tool"]

    def test_executor_init_with_hooks(self, mock_llm_config):
        """Test executor initialization with hooks."""
        before_hook = Mock()
        after_hook = Mock()
        tool_hook = Mock()
        before_tool_hook = Mock()

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
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
            tool_registry=make_tool_registry(),
            sub_agents={},
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
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        message = {"role": "user", "content": "Test message"}
        executor.enqueue_message(message)

        assert len(executor.queued_messages) == 1
        assert executor.queued_messages[0].role == Role.USER
        assert executor.queued_messages[0].get_text_content() == "Test message"

    def test_enqueue_multiple_messages(self, mock_llm_config):
        """Test enqueueing multiple messages."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
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
        assert [m.role.value for m in executor.queued_messages] == ["user", "assistant", "user"]
        assert [m.get_text_content() for m in executor.queued_messages] == ["Message 1", "Message 2", "Message 3"]


class TestExecutorExecution:
    """Test executor main execution flow."""

    def test_execute_with_stop_signal(self, mock_llm_config, agent_state):
        """Test execution stops when stop_signal is set."""

        class StopSignalMiddleware(Middleware):
            executor: Executor

            def before_agent(self, hook_input):  # type: ignore[override]
                self.executor.stop_signal = True
                return HookResult.no_changes()

        middleware = StopSignalMiddleware()
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            middlewares=[middleware],
        )
        middleware.executor = executor

        history = make_history("You are a helpful assistant.", "Hello")

        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            response, _ = executor.execute(history, agent_state)

        mock_call_llm.assert_not_called()
        assert response == "Stop signal received."

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
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            middlewares=[middleware],
        )
        middleware.executor = executor

        history = make_history("You are a helpful assistant.", "Hello")

        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            response, _ = executor.execute(history, agent_state)

        mock_call_llm.assert_not_called()
        assert middleware.after_agent_called is True
        assert response == "Stop signal received.::hooked"

    def test_execute_generate_with_token_stores_trace_memory(self, agent_state, global_storage):
        """generate_with_token execution should write token trace into shared trace memory."""
        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=None,
            llm_config=llm_config,
            global_storage=global_storage,
        )
        agent_state.global_storage = global_storage

        history = make_history("You are a helpful assistant.", "Hello")

        with patch("nexau.archs.main_sub.execution.executor.TokenTraceSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.export_trace.return_value = {
                "final_token_list": [1, 2, 3, 4],
                "response_mask": [0, 0, 1, 1],
                "round_traces": [{"request_tokens": [1, 2], "response_tokens": [3, 4]}],
                "token_provider_usage": [{"total_tokens": 4}],
            }

            with patch.object(
                executor.llm_caller,
                "call_llm",
                return_value=ModelResponse(content="Done", output_token_ids=[3, 4]),
            ):
                response, _ = executor.execute(history, agent_state)

        assert response == "Done"
        trace_memory = global_storage.get("trace_memory", {})
        assert trace_memory["final_token_list"] == [1, 2, 3, 4]
        assert trace_memory["response_mask"] == [0, 0, 1, 1]
        mock_session.initialize_from_messages.assert_called_once()
        mock_session.append_model_response.assert_called_once()

    def test_execute_generate_with_token_initializes_session_with_tools(self, agent_state, global_storage):
        """generate_with_token should include structured tools during initial tokenization."""
        structured_tools = [
            build_structured_tool_definition(
                name="simple_tool",
                description="A simple tool",
                input_schema={"type": "object"},
            )
        ]
        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=None,
            llm_config=llm_config,
            global_storage=global_storage,
            tool_call_mode="openai",
            structured_tools=structured_tools,
        )
        agent_state.global_storage = global_storage

        history = make_history("You are a helpful assistant.", "Hello")

        with patch("nexau.archs.main_sub.execution.executor.TokenTraceSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.export_trace.return_value = {
                "final_token_list": [1, 2, 3],
                "response_mask": [0, 0, 1],
                "round_traces": [],
                "token_provider_usage": [],
            }

            with patch.object(
                executor.llm_caller,
                "call_llm",
                return_value=ModelResponse(content="Done", output_token_ids=[3]),
            ):
                response, _ = executor.execute(history, agent_state)

        assert response == "Done"
        mock_session.initialize_from_messages.assert_called_once()
        init_args = mock_session.initialize_from_messages.call_args
        initialized_messages = init_args.args[0]
        assert [message.role for message in initialized_messages[:2]] == [Role.SYSTEM, Role.USER]
        assert [message.get_text_content() for message in initialized_messages[:2]] == [
            "You are a helpful assistant.",
            "Hello",
        ]
        assert init_args.kwargs["tools"] == structured_tools

    def test_execute_generate_with_token_multi_turn_tool_trace(self, agent_state, global_storage):
        """generate_with_token should keep token trace state across tool-call rounds."""

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

        structured_tools = [
            build_structured_tool_definition(
                name="simple_tool",
                description="A simple tool",
                input_schema={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            )
        ]

        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"simple_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=None,
            llm_config=llm_config,
            global_storage=global_storage,
            max_iterations=2,
            tool_call_mode="openai",
            structured_tools=structured_tools,
        )
        agent_state.global_storage = global_storage

        history = make_history("You are a helpful assistant.", "Use the tool and finish.")

        first_response = ModelResponse(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="call_123",
                    name="simple_tool",
                    arguments={"x": 3},
                    raw_arguments='{"x": 3}',
                )
            ],
            output_token_ids=[101, 102],
        )
        second_response = ModelResponse(content="Final response", output_token_ids=[201])

        with patch("nexau.archs.main_sub.execution.executor.TokenTraceSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.export_trace.return_value = {
                "final_token_list": [1, 2, 101, 102, 301, 302, 201],
                "response_mask": [0, 0, 1, 1, 0, 0, 1],
                "round_traces": [
                    {"request_tokens": [1, 2], "response_tokens": [101, 102]},
                    {"request_tokens": [1, 2, 101, 102, 301, 302], "response_tokens": [201]},
                ],
                "token_provider_usage": [{"total_tokens": 4}, {"total_tokens": 7}],
            }

            call_messages: list[list[Message]] = []

            def llm_side_effect(messages, **kwargs):
                call_messages.append(list(messages))
                assert kwargs["token_trace_session"] is mock_session
                if len(call_messages) == 1:
                    return first_response

                assert len(call_messages) == 2
                tool_message = next(msg for msg in messages if msg.role == Role.TOOL)
                tool_block = next(block for block in tool_message.content if isinstance(block, ToolResultBlock))
                assert tool_block.tool_use_id == "call_123"
                assert isinstance(tool_block.content, str)
                assert tool_block.content == "4"
                return second_response

            with patch.object(executor.llm_caller, "call_llm", side_effect=llm_side_effect):
                response, messages = executor.execute(history, agent_state)

        assert response == "Final response"
        assert len(call_messages) == 2
        assert mock_session.initialize_from_messages.call_count == 1
        assert mock_session.append_model_response.call_count == 2

        first_append = mock_session.append_model_response.call_args_list[0].kwargs
        second_append = mock_session.append_model_response.call_args_list[1].kwargs
        assert first_append["output_token_ids"] == [101, 102]
        assert second_append["output_token_ids"] == [201]

        assert mock_session.append_messages.call_count == 1
        append_messages_args = mock_session.append_messages.call_args.args
        append_messages_kwargs = mock_session.append_messages.call_args.kwargs
        appended_tool_messages = append_messages_args[0]
        assert append_messages_kwargs["mask_value"] == 0
        assert len(appended_tool_messages) == 1
        appended_tool_message = appended_tool_messages[0]
        assert appended_tool_message.role == Role.TOOL
        appended_tool_block = next(block for block in appended_tool_message.content if isinstance(block, ToolResultBlock))
        assert appended_tool_block.tool_use_id == "call_123"

        trace_memory = global_storage.get("trace_memory", {})
        assert trace_memory["final_token_list"] == [1, 2, 101, 102, 301, 302, 201]
        assert trace_memory["response_mask"] == [0, 0, 1, 1, 0, 0, 1]
        assert messages[-1].role == Role.ASSISTANT

    def test_before_and_after_agent_middlewares(self, mock_llm_config, agent_state):
        """Lifecycle middlewares can modify initial messages and final response."""

        class LifecycleMiddleware(Middleware):
            def __init__(self) -> None:
                self.before_agent_called = 0
                self.after_agent_called = 0

            def before_agent(self, hook_input):  # type: ignore[override]
                self.before_agent_called += 1
                from nexau.core.messages import Message, Role, TextBlock

                updated = hook_input.messages + [Message(role=Role.SYSTEM, content=[TextBlock(text="prep note")])]
                return HookResult.with_modifications(messages=updated)

            def after_agent(self, hook_input):  # type: ignore[override]
                self.after_agent_called += 1
                return HookResult.with_modifications(agent_response=f"{hook_input.agent_response}::final")

        lifecycle_middleware = LifecycleMiddleware()
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=1,
            middlewares=[lifecycle_middleware],
        )

        history = make_history("You are a helpful assistant.", "Hello")

        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            mock_call_llm.return_value = ModelResponse(content="Simple response")
            with patch.object(executor.response_parser, "parse_response") as mock_parse:
                mock_parse.return_value = ParsedResponse(
                    original_response="Simple response",
                    tool_calls=[],
                )

                response, messages = executor.execute(history, agent_state)

        assert lifecycle_middleware.before_agent_called == 1
        assert lifecycle_middleware.after_agent_called == 1
        assert response == "Simple response::final"
        assert any(msg.get_text_content() == "prep note" for msg in messages)

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

        structured_tools = [tool.to_structured_definition()]

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"simple_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=2,
            tool_call_mode="openai",
            structured_tools=structured_tools,
        )

        history = make_history("You are a helpful assistant.", "Use the tool")

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
        tool_messages = [msg for msg in messages if msg.role == Role.TOOL]
        assert tool_messages, "Expected at least one tool message"

        matching_tool_messages = []
        for msg in tool_messages:
            for block in msg.content:
                if isinstance(block, ToolResultBlock) and block.tool_use_id == "call_123":
                    matching_tool_messages.append(msg)
                    break
        assert matching_tool_messages, "Tool message should reuse original tool_call_id"

        tool_block = next(block for block in matching_tool_messages[0].content if isinstance(block, ToolResultBlock))
        tool_content = tool_block.content
        assert isinstance(tool_content, str)
        assert tool_content == "4"

        # Tool message should appear before the final assistant reply
        tool_index = messages.index(matching_tool_messages[0])
        assert tool_index + 1 < len(messages)
        assert messages[tool_index + 1].role == Role.ASSISTANT

    def test_execute_openai_tool_messages_support_image_results(self, mock_llm_config, agent_state):
        """Tool results can include images (ToolOutputImage or dict form) in tool messages."""

        def image_tool() -> list[object]:
            return [
                {"type": "text", "text": "Here is the image:"},
                ToolOutputImage(image_url="data:image/png;base64,AAAA", detail="high"),
            ]

        tool = Tool(
            name="image_tool",
            description="Returns an image",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=image_tool,
        )

        structured_tools = [tool.to_structured_definition()]

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"image_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=2,
            tool_call_mode="openai",
            structured_tools=structured_tools,
        )

        history = make_history("You are a helpful assistant.", "Use the tool")

        first_response = ModelResponse(
            content="",
            tool_calls=[ModelToolCall(call_id="call_img", name="image_tool", arguments={}, raw_arguments="{}")],
        )
        second_response = ModelResponse(content="Final response")

        with patch.object(executor.llm_caller, "call_llm", side_effect=[first_response, second_response]):
            response, messages = executor.execute(history, agent_state)

        assert response == "Final response"

        tool_messages = [msg for msg in messages if msg.role == Role.TOOL]
        assert tool_messages, "Expected a tool message"

        tool_block = next(
            block
            for msg in tool_messages
            for block in msg.content
            if isinstance(block, ToolResultBlock) and block.tool_use_id == "call_img"
        )

        assert isinstance(tool_block.content, list)
        assert any(isinstance(p, TextBlock) and "Here is the image" in p.text for p in tool_block.content)
        img = next(p for p in tool_block.content if isinstance(p, ImageBlock))
        assert img.detail == "high"

    def test_execute_openai_tool_messages_input_image_dict_with_detail(self, mock_llm_config, agent_state):
        """Dict form: tools can return type='input_image' with detail."""

        def image_tool() -> dict:
            return {
                "type": "input_image",
                "image_url": "https://example.com/image.jpg",
                "detail": "high",
            }

        tool = Tool(
            name="image_tool",
            description="Returns an image dict",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=image_tool,
        )

        structured_tools = [tool.to_structured_definition()]

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"image_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_iterations=2,
            tool_call_mode="openai",
            structured_tools=structured_tools,
        )

        history = make_history("You are a helpful assistant.", "Use the tool")

        first_response = ModelResponse(
            content="",
            tool_calls=[ModelToolCall(call_id="call_img", name="image_tool", arguments={}, raw_arguments="{}")],
        )
        second_response = ModelResponse(content="Final response")

        with patch.object(executor.llm_caller, "call_llm", side_effect=[first_response, second_response]):
            response, messages = executor.execute(history, agent_state)

        assert response == "Final response"
        tool_block = next(
            block
            for msg in messages
            if msg.role == Role.TOOL
            for block in msg.content
            if isinstance(block, ToolResultBlock) and block.tool_use_id == "call_img"
        )
        assert isinstance(tool_block.content, list)
        img = next(p for p in tool_block.content if isinstance(p, ImageBlock))
        assert img.url == "https://example.com/image.jpg"
        assert img.detail == "high"

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
            tool_registry=make_tool_registry(),
            sub_agents={},
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
                )

                history = make_history("You are a helpful assistant.", "Hello")

                response, messages = executor.execute(history, agent_state)

                # Should have made LLM calls
                assert mock_call_llm.call_count >= 1

    def test_execute_with_queued_messages(self, mock_llm_config, agent_state):
        """Test execution processes queued messages."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
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
                )

                history = make_history("You are a helpful assistant.", "Hello")

                response, messages = executor.execute(history, agent_state)

                # Verify queued message was added to history
                assert any(msg.role == Role.USER and msg.get_text_content() == "Queued message" for msg in messages)
                assert len(executor.queued_messages) == 0

    def test_execute_with_token_limit_exceeded(self, mock_llm_config, agent_state):
        """Test execution continues even when local token budget is exceeded."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            max_context_tokens=100,  # Very low limit
        )

        # Mock token counter to return high count
        with patch.object(executor.token_counter, "count_tokens") as mock_count:
            mock_count.return_value = 200  # Exceeds limit

            history = make_history("You are a helpful assistant.", "Hello")

            # Mock LLM caller
            with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
                mock_call_llm.return_value = ModelResponse(content="Simple response")

                with patch.object(executor.response_parser, "parse_response") as mock_parse:
                    mock_parse.return_value = ParsedResponse(
                        original_response="Simple response",
                        tool_calls=[],
                    )

                    response, messages = executor.execute(history, agent_state)

                mock_call_llm.assert_called()
                assert response == "Simple response"

    def test_execute_with_exception(self, mock_llm_config, agent_state):
        """Test execution handles exceptions gracefully."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        # Mock the LLM caller to raise an exception
        with patch.object(executor.llm_caller, "call_llm") as mock_call_llm:
            mock_call_llm.side_effect = Exception("Test exception")

            history = make_history("You are a helpful assistant.", "Hello")

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
            tool_registry=make_tool_registry(),
            sub_agents={},
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
            ),
            messages=[],
        )

        ctx = make_framework_context(executor)
        processed, should_stop, result, messages, feedbacks = executor._process_xml_calls(hook_input, framework_context=ctx)

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
            tool_registry=make_tool_registry({"simple_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
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

            ctx = make_framework_context(executor)
            processed, should_stop, result, messages, feedbacks = executor._process_xml_calls(hook_input, framework_context=ctx)

            # Tool results should be appended to the response
            assert "<tool_result>" in processed
            assert "simple_tool" in processed
            assert len(feedbacks) == 1


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
            tool_registry=make_tool_registry({"test_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        tool_call = ToolCall(
            tool_name="test_tool",
            parameters={"x": 5},
            raw_content="<tool_call>...</tool_call>",
        )

        ctx = make_framework_context(executor)
        tool_name, result, is_error = executor._execute_tool_call_safe(tool_call, agent_state, ctx)

        assert tool_name == "test_tool"
        assert is_error is False
        assert isinstance(result, ToolExecutionResult)
        assert result.raw_output["result"] == 10
        assert result.llm_tool_output == 10

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
            tool_registry=make_tool_registry({"error_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        tool_call = ToolCall(
            tool_name="error_tool",
            parameters={"x": 5},
            raw_content="<tool_call>...</tool_call>",
        )

        ctx = make_framework_context(executor)
        tool_name, result, is_error = executor._execute_tool_call_safe(tool_call, agent_state, ctx)

        assert tool_name == "error_tool"
        assert is_error is False
        assert isinstance(result, ToolExecutionResult)
        assert result.raw_output["error"] == "Tool error"


class TestExecutorCleanup:
    """Test executor cleanup functionality."""

    def test_cleanup_basic(self, mock_llm_config):
        """Test basic cleanup."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
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
            tool_registry=make_tool_registry(),
            sub_agents={},
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

    def test_structured_tool_payload_includes_sub_agents(self, mock_llm_config, agent_config):
        """RFC-0015: Agent is a regular builtin tool, not a virtual definition.

        Sub-agents configured on the executor should NOT generate virtual
        sub-agent-{name} tool definitions. The Agent tool is registered
        as a regular builtin tool in AgentConfig._finalize() and will appear
        in structured_tool_payload only when it's in the ToolRegistry.
        """
        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=lambda: {"result": "ok"},
        )
        child_config = agent_config.model_copy(update={"name": "child", "description": "Delegate to child"})
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"simple_tool": tool}),
            sub_agents={"child": child_config},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            tool_call_mode="openai",
        )

        payload_by_name = {spec["name"]: spec for spec in executor.structured_tool_payload}

        assert "simple_tool" in payload_by_name
        # RFC-0015: No virtual sub-agent-{name} definitions should be generated
        assert "sub-agent-child" not in payload_by_name

    def test_structured_tool_payload_uses_skill_description(self, mock_llm_config):
        """Structured payload should use brief skill descriptions for as_skill tools."""
        skill_tool = Tool(
            name="web_search",
            description="FULL DESCRIPTION: search the web with examples and workflow guidance.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            implementation=lambda query: {"query": query},
            as_skill=True,
            skill_description="BRIEF SKILL DESCRIPTION",
        )
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"web_search": skill_tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            tool_call_mode="anthropic",
        )

        payload_by_name = {spec["name"]: spec for spec in executor.structured_tool_payload}

        assert payload_by_name["web_search"]["description"] == "BRIEF SKILL DESCRIPTION"
        assert payload_by_name["web_search"]["input_schema"]["properties"]["query"]["type"] == "string"

    def test_add_sub_agent(self, mock_llm_config, agent_config):
        """Test adding a sub-agent config dynamically."""
        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        new_config = agent_config.model_copy(update={"name": "new_sub_agent", "agent_id": "new_sub_agent_123"})
        executor.add_sub_agent("new_sub_agent", new_config)

        # Verify it was added through subagent_manager
        # (internal implementation detail, but we can check it was called)
        assert "new_sub_agent" in executor.subagent_manager.sub_agents


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
            tool_registry=make_tool_registry({"stop_tool": tool}),
            sub_agents={},
            stop_tools={"stop_tool"},
            openai_client=Mock(),
            llm_config=mock_llm_config,
        )

        tool_call = ToolCall(
            tool_name="stop_tool",
            parameters={},
            raw_content="<tool_call>...</tool_call>",
        )

        parsed_response = ParsedResponse(
            original_response="Stopping",
            tool_calls=[tool_call],
        )

        ctx = make_framework_context(executor)
        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(parsed_response, agent_state, framework_context=ctx)

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
            tool_registry=make_tool_registry({"tool1": tool_obj1, "tool2": tool_obj2}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
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
        )

        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(
            parsed_response, agent_state, framework_context=make_framework_context(executor)
        )

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
            tool_registry=make_tool_registry({"test_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
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
        )

        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(
            parsed_response, agent_state, framework_context=make_framework_context(executor)
        )

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
        from nexau.archs.main_sub.execution.hooks import BeforeModelHookInput, HookResult
        from nexau.core.messages import Message, Role, TextBlock

        class TestBeforeHook:
            def __call__(self, hook_input: BeforeModelHookInput) -> HookResult:
                # Add a system message
                messages = hook_input.messages.copy()
                messages.append(Message(role=Role.SYSTEM, content=[TextBlock(text="Hook added this")]))
                return HookResult.with_modifications(messages=messages)

        hook = TestBeforeHook()

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry(),
            sub_agents={},
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
                )

                history = make_history("You are a helpful assistant.", "Hello")

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
            tool_registry=make_tool_registry(),
            sub_agents={},
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
        )

        processed, should_stop, result, feedbacks = executor._execute_parsed_calls(
            parsed_response, agent_state, framework_context=make_framework_context(executor)
        )

        # Should return early without executing
        assert should_stop is False
        assert feedbacks == []


class TestExecutorParallelExecutionId:
    """Test parallel_execution_id generation and propagation in Executor."""

    def test_executor_assigns_same_parallel_execution_id_to_all_tool_calls(self, mock_llm_config, agent_state):
        """Test that all tool calls in the same batch get the same parallel_execution_id."""
        from nexau.archs.main_sub.execution.hooks import FunctionMiddleware

        captured_parallel_execution_ids = []

        def mock_tool1() -> str:
            return "result1"

        def mock_tool2() -> str:
            return "result2"

        def before_tool_hook(hook_input):
            captured_parallel_execution_ids.append(hook_input.parallel_execution_id)
            return HookResult()

        middleware = FunctionMiddleware(before_tool_hook=before_tool_hook)

        tool1 = Tool(
            name="tool1",
            description="Test tool 1",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=mock_tool1,
        )

        tool2 = Tool(
            name="tool2",
            description="Test tool 2",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=mock_tool2,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"tool1": tool1, "tool2": tool2}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            middlewares=[middleware],
        )

        # Create parsed response with multiple tool calls
        parsed_response = ParsedResponse(
            original_response="Test",
            tool_calls=[
                ToolCall(tool_name="tool1", parameters={}),
                ToolCall(tool_name="tool2", parameters={}),
            ],
        )

        # Execute the parsed response
        executor._execute_parsed_calls(parsed_response, agent_state, framework_context=make_framework_context(executor))

        # Verify all tool calls received the same parallel_execution_id
        assert len(captured_parallel_execution_ids) == 2
        assert captured_parallel_execution_ids[0] == captured_parallel_execution_ids[1]
        assert captured_parallel_execution_ids[0] is not None

    def test_executor_assigns_different_parallel_execution_ids_to_different_batches(self, mock_llm_config, agent_state):
        """Test that different execution batches get different parallel_execution_ids."""
        from nexau.archs.main_sub.execution.hooks import FunctionMiddleware

        captured_parallel_execution_ids = []

        def mock_tool() -> str:
            return "result"

        def before_tool_hook(hook_input):
            captured_parallel_execution_ids.append(hook_input.parallel_execution_id)
            return HookResult()

        middleware = FunctionMiddleware(before_tool_hook=before_tool_hook)

        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=mock_tool,
        )

        executor = Executor(
            agent_name="test_agent",
            agent_id="test_id",
            tool_registry=make_tool_registry({"test_tool": tool}),
            sub_agents={},
            stop_tools=set(),
            openai_client=Mock(),
            llm_config=mock_llm_config,
            middlewares=[middleware],
        )

        # Execute first batch
        parsed_response1 = ParsedResponse(
            original_response="Test 1",
            tool_calls=[
                ToolCall(tool_name="test_tool", parameters={}),
                ToolCall(tool_name="test_tool", parameters={}),
            ],
        )
        ctx = make_framework_context(executor)
        executor._execute_parsed_calls(parsed_response1, agent_state, framework_context=ctx)

        # Execute second batch
        parsed_response2 = ParsedResponse(
            original_response="Test 2",
            tool_calls=[
                ToolCall(tool_name="test_tool", parameters={}),
            ],
        )
        executor._execute_parsed_calls(parsed_response2, agent_state, framework_context=ctx)

        # Verify: first 2 calls have same ID, third call has different ID
        assert len(captured_parallel_execution_ids) == 3
        assert captured_parallel_execution_ids[0] == captured_parallel_execution_ids[1]
        assert captured_parallel_execution_ids[0] != captured_parallel_execution_ids[2]

    def test_executor_assigns_parallel_execution_id_to_mixed_calls(self, mock_llm_config, agent_state):
        """Test that both tool calls and sub-agent calls in the same batch get the same parallel_execution_id."""
        from nexau.archs.main_sub.config.config import AgentConfig
        from nexau.archs.main_sub.execution.hooks import FunctionMiddleware

        captured_tool_parallel_id = None
        captured_subagent_parallel_id = None

        def mock_tool() -> str:
            return "tool result"

        def before_tool_hook(hook_input):
            nonlocal captured_tool_parallel_id
            captured_tool_parallel_id = hook_input.parallel_execution_id
            return HookResult()

        middleware = FunctionMiddleware(before_tool_hook=before_tool_hook)

        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=mock_tool,
        )

        # Create sub-agent config
        sub_agent_config = AgentConfig(
            name="test_sub_agent",
            system_prompt="You are a test sub-agent",
            llm_config=mock_llm_config,
        )

        # Mock the sub-agent manager to capture parallel_execution_id
        with patch("nexau.archs.main_sub.execution.executor.SubAgentManager") as mock_subagent_manager_cls:
            mock_manager = Mock()

            def mock_call_sub_agent(*args, **kwargs):
                nonlocal captured_subagent_parallel_id
                captured_subagent_parallel_id = kwargs.get("parallel_execution_id")
                return "sub agent result"

            mock_manager.call_sub_agent = mock_call_sub_agent
            mock_subagent_manager_cls.return_value = mock_manager

            executor = Executor(
                agent_name="test_agent",
                agent_id="test_id",
                tool_registry=make_tool_registry({"test_tool": tool}),
                sub_agents={"test_sub_agent": sub_agent_config},
                stop_tools=set(),
                openai_client=Mock(),
                llm_config=mock_llm_config,
                middlewares=[middleware],
            )

            # Create parsed response with both tool calls and sub-agent calls
            parsed_response = ParsedResponse(
                original_response="Test",
                tool_calls=[
                    ToolCall(tool_name="test_tool", parameters={}),
                    ToolCall(tool_name="Agent", parameters={"sub_agent_name": "test_sub_agent", "message": "task"}),
                ],
            )

            # Execute the parsed response
            executor._execute_parsed_calls(parsed_response, agent_state, framework_context=make_framework_context(executor))

            # Verify tool call received parallel_execution_id
            assert captured_tool_parallel_id is not None
