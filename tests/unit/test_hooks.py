"""Comprehensive tests for the hooks module."""

import logging
from unittest.mock import patch

import pytest

from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.hooks import (
    AfterModelHookInput,
    AfterModelHookResult,
    AfterToolHookInput,
    AfterToolHookResult,
    BeforeModelHookInput,
    BeforeModelHookResult,
    BeforeToolHookInput,
    FunctionMiddleware,
    HookResult,
    LoggingMiddleware,
    Middleware,
    MiddlewareManager,
    ModelCallParams,
    ToolCallParams,
    create_logging_hook,
    create_remaining_reminder_hook,
    create_tool_after_approve_hook,
    create_tool_logging_hook,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.parse_structures import (
    BatchAgentCall,
    ParsedResponse,
    SubAgentCall,
    ToolCall,
)


@pytest.fixture
def agent_state():
    """Create a mock agent state for testing."""
    context = AgentContext()
    global_storage = GlobalStorage()
    return AgentState(
        agent_name="test_agent",
        agent_id="test_id_123",
        context=context,
        global_storage=global_storage,
    )


@pytest.fixture
def messages():
    """Create sample messages for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]


@pytest.fixture
def parsed_response():
    """Create a sample parsed response for testing."""
    tool_call = ToolCall(
        tool_name="test_tool",
        parameters={"param1": "value1"},
        raw_content="<tool>test</tool>",
    )
    sub_agent_call = SubAgentCall(
        agent_name="sub_agent",
        message="test message",
        raw_content="<sub_agent>test</sub_agent>",
    )
    batch_call = BatchAgentCall(
        agent_name="batch_agent",
        file_path="test.json",
        data_format="json",
        message_template="Process: {{item}}",
        raw_content="<batch>test</batch>",
    )
    return ParsedResponse(
        original_response="test response",
        tool_calls=[tool_call],
        sub_agent_calls=[sub_agent_call],
        batch_agent_calls=[batch_call],
        is_parallel_tools=True,
        is_parallel_sub_agents=False,
    )


class TestBeforeModelHookInput:
    """Tests for BeforeModelHookInput dataclass."""

    def test_initialization(self, agent_state, messages):
        """Test initialization of BeforeModelHookInput."""
        hook_input = BeforeModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=messages,
        )

        assert hook_input.agent_state == agent_state
        assert hook_input.max_iterations == 10
        assert hook_input.current_iteration == 3
        assert hook_input.messages == messages


class TestAfterModelHookInput:
    """Tests for AfterModelHookInput dataclass."""

    def test_initialization(self, agent_state, messages, parsed_response):
        """Test initialization of AfterModelHookInput."""
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=messages,
            original_response="test response",
            parsed_response=parsed_response,
        )

        assert hook_input.agent_state == agent_state
        assert hook_input.max_iterations == 10
        assert hook_input.current_iteration == 3
        assert hook_input.messages == messages
        assert hook_input.original_response == "test response"
        assert hook_input.parsed_response == parsed_response

    def test_initialization_with_none_parsed_response(self, agent_state, messages):
        """Test initialization with None parsed_response."""
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=messages,
            original_response="test response",
            parsed_response=None,
        )

        assert hook_input.parsed_response is None


class TestBeforeModelHookResult:
    """Tests for BeforeModelHookResult dataclass."""

    def test_has_modifications_true(self):
        """Test has_modifications returns True when messages are modified."""
        result = BeforeModelHookResult(messages=[{"role": "user", "content": "test"}])
        assert result.has_modifications() is True

    def test_has_modifications_false(self):
        """Test has_modifications returns False when no modifications."""
        result = BeforeModelHookResult(messages=None)
        assert result.has_modifications() is False

    def test_no_changes_classmethod(self):
        """Test no_changes class method."""
        result = BeforeModelHookResult.no_changes()
        assert result.messages is None
        assert result.has_modifications() is False

    def test_with_modifications_messages(self):
        """Test with_modifications with messages."""
        messages = [{"role": "user", "content": "test"}]
        result = BeforeModelHookResult.with_modifications(messages=messages)
        assert result.messages == messages
        assert result.has_modifications() is True

    def test_with_modifications_none(self):
        """Test with_modifications with None."""
        result = BeforeModelHookResult.with_modifications(messages=None)
        assert result.messages is None
        assert result.has_modifications() is False


class TestAfterModelHookResult:
    """Tests for AfterModelHookResult dataclass."""

    def test_has_modifications_parsed_response(self, parsed_response):
        """Test has_modifications with parsed_response."""
        result = AfterModelHookResult(parsed_response=parsed_response)
        assert result.has_modifications() is True

    def test_has_modifications_messages(self):
        """Test has_modifications with messages."""
        result = AfterModelHookResult(messages=[{"role": "user", "content": "test"}])
        assert result.has_modifications() is True

    def test_has_modifications_both(self, parsed_response):
        """Test has_modifications with both modifications."""
        result = AfterModelHookResult(
            parsed_response=parsed_response,
            messages=[{"role": "user", "content": "test"}],
        )
        assert result.has_modifications() is True

    def test_has_modifications_false(self):
        """Test has_modifications returns False when no modifications."""
        result = AfterModelHookResult()
        assert result.has_modifications() is False

    def test_no_changes_classmethod(self):
        """Test no_changes class method."""
        result = AfterModelHookResult.no_changes()
        assert result.parsed_response is None
        assert result.messages is None
        assert result.has_modifications() is False

    def test_with_modifications_parsed_response(self, parsed_response):
        """Test with_modifications with parsed_response."""
        result = AfterModelHookResult.with_modifications(parsed_response=parsed_response)
        assert result.parsed_response == parsed_response
        assert result.messages is None
        assert result.has_modifications() is True

    def test_with_modifications_messages(self):
        """Test with_modifications with messages."""
        messages = [{"role": "user", "content": "test"}]
        result = AfterModelHookResult.with_modifications(messages=messages)
        assert result.parsed_response is None
        assert result.messages == messages
        assert result.has_modifications() is True

    def test_with_modifications_both(self, parsed_response):
        """Test with_modifications with both."""
        messages = [{"role": "user", "content": "test"}]
        result = AfterModelHookResult.with_modifications(
            parsed_response=parsed_response,
            messages=messages,
        )
        assert result.parsed_response == parsed_response
        assert result.messages == messages
        assert result.has_modifications() is True

    def test_force_continue_default_false(self):
        """Test force_continue defaults to False."""
        result = AfterModelHookResult()
        assert result.force_continue is False
        assert result.has_modifications() is False

    def test_force_continue_set_true(self):
        """Test force_continue can be set to True."""
        result = AfterModelHookResult(force_continue=True)
        assert result.force_continue is True
        assert result.has_modifications() is True

    def test_no_changes_force_continue_false(self):
        """Test no_changes returns force_continue=False."""
        result = AfterModelHookResult.no_changes()
        assert result.force_continue is False
        assert result.has_modifications() is False

    def test_with_modifications_force_continue_true(self):
        """Test with_modifications with force_continue=True."""
        messages = [{"role": "user", "content": "feedback"}]
        result = AfterModelHookResult.with_modifications(
            messages=messages,
            force_continue=True,
        )
        assert result.messages == messages
        assert result.force_continue is True
        assert result.has_modifications() is True

    def test_with_modifications_force_continue_default(self):
        """Test with_modifications force_continue defaults to False."""
        messages = [{"role": "user", "content": "test"}]
        result = AfterModelHookResult.with_modifications(messages=messages)
        assert result.force_continue is False

    def test_has_modifications_with_force_continue_only(self):
        """Test has_modifications returns True when only force_continue is set."""
        result = AfterModelHookResult(force_continue=True)
        assert result.parsed_response is None
        assert result.messages is None
        assert result.force_continue is True
        assert result.has_modifications() is True


class TestAfterToolHookInput:
    """Tests for AfterToolHookInput dataclass."""

    def test_initialization(self, agent_state):
        """Test initialization of AfterToolHookInput."""
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="result",
        )

        assert hook_input.agent_state == agent_state
        assert hook_input.tool_name == "test_tool"
        assert hook_input.tool_call_id == "call_123"
        assert hook_input.tool_input == {"param": "value"}
        assert hook_input.tool_output == "result"


class TestAfterToolHookResult:
    """Tests for AfterToolHookResult dataclass."""

    def test_has_modifications_true(self):
        """Test has_modifications returns True when tool_output is set."""
        result = AfterToolHookResult(tool_output="modified")
        assert result.has_modifications() is True

    def test_has_modifications_false(self):
        """Test has_modifications returns False when no modifications."""
        result = AfterToolHookResult()
        assert result.has_modifications() is False

    def test_no_changes_classmethod(self):
        """Test no_changes class method."""
        result = AfterToolHookResult.no_changes()
        assert result.tool_output is None
        assert result.has_modifications() is False

    def test_with_modifications(self):
        """Test with_modifications class method."""
        result = AfterToolHookResult.with_modifications(tool_output="modified")
        assert result.tool_output == "modified"
        assert result.has_modifications() is True


class TestCreateLoggingHook:
    """Tests for create_logging_hook function."""

    def test_create_logging_hook_with_parsed_response(self, agent_state, messages, parsed_response, caplog):
        """Test logging hook with parsed response."""
        caplog.set_level(logging.INFO)

        hook = create_logging_hook(logger_name="test_logger")
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=parsed_response,
        )

        result = hook(hook_input)

        assert isinstance(result, AfterModelHookResult)
        assert result.has_modifications() is False
        assert "AFTER MODEL HOOK TRIGGERED" in caplog.text
        assert "test_agent" in caplog.text
        assert "test_id_123" in caplog.text

    def test_create_logging_hook_without_parsed_response(self, agent_state, messages, caplog):
        """Test logging hook without parsed response."""
        caplog.set_level(logging.INFO)

        hook = create_logging_hook(logger_name="test_logger")
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=None,
        )

        result = hook(hook_input)

        assert isinstance(result, AfterModelHookResult)
        assert result.has_modifications() is False
        assert "No parsed response available" in caplog.text


class TestCreateRemainingReminderHook:
    """Tests for create_remaining_reminder_hook function."""

    def test_create_remaining_reminder_hook(self, agent_state, messages, caplog):
        """Test remaining reminder hook."""
        caplog.set_level(logging.INFO)

        hook = create_remaining_reminder_hook(logger_name="test_logger")
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=messages.copy(),
            original_response="test response",
            parsed_response=None,
        )

        result = hook(hook_input)

        assert isinstance(result, AfterModelHookResult)
        assert result.has_modifications() is True
        assert result.messages is not None
        assert len(result.messages) == len(messages) + 1
        assert "Remaining iterations: 7" in result.messages[-1]["content"]
        assert "Remaining iterations: 7" in caplog.text


class TestCreateToolAfterApproveHook:
    """Tests for create_tool_after_approve_hook function."""

    def test_approve_tool_call(self, agent_state, messages, parsed_response):
        """Test approving a tool call."""
        with patch("builtins.input", return_value="y"):
            hook = create_tool_after_approve_hook(tool_name="test_tool")
            hook_input = AfterModelHookInput(
                agent_state=agent_state,
                max_iterations=10,
                current_iteration=5,
                messages=messages,
                original_response="test response",
                parsed_response=parsed_response,
            )

            result = hook(hook_input)

            assert isinstance(result, AfterModelHookResult)
            assert result.has_modifications() is False

    def test_reject_tool_call(self, agent_state, messages, parsed_response):
        """Test rejecting a tool call."""
        with patch("builtins.input", return_value="n"):
            hook = create_tool_after_approve_hook(tool_name="test_tool")
            hook_input = AfterModelHookInput(
                agent_state=agent_state,
                max_iterations=10,
                current_iteration=5,
                messages=messages,
                original_response="test response",
                parsed_response=parsed_response,
            )

            result = hook(hook_input)

            assert isinstance(result, AfterModelHookResult)
            assert result.has_modifications() is True
            assert len(result.parsed_response.tool_calls) == 0

    def test_invalid_input_then_approve(self, agent_state, messages, parsed_response):
        """Test invalid input then approve."""
        with patch("builtins.input", side_effect=["invalid", "y"]):
            with patch("builtins.print") as mock_print:
                hook = create_tool_after_approve_hook(tool_name="test_tool")
                hook_input = AfterModelHookInput(
                    agent_state=agent_state,
                    max_iterations=10,
                    current_iteration=5,
                    messages=messages,
                    original_response="test response",
                    parsed_response=parsed_response,
                )

                result = hook(hook_input)

                assert isinstance(result, AfterModelHookResult)
                mock_print.assert_called()
                assert result.has_modifications() is False

    def test_no_tool_calls(self, agent_state, messages):
        """Test with no tool calls."""
        hook = create_tool_after_approve_hook(tool_name="test_tool")
        parsed_response_empty = ParsedResponse(
            original_response="test",
            tool_calls=[],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=parsed_response_empty,
        )

        result = hook(hook_input)

        assert isinstance(result, AfterModelHookResult)
        assert result.has_modifications() is False

    def test_no_parsed_response(self, agent_state, messages):
        """Test with no parsed response."""
        hook = create_tool_after_approve_hook(tool_name="test_tool")
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=None,
        )

        result = hook(hook_input)

        assert isinstance(result, AfterModelHookResult)
        assert result.has_modifications() is False

    def test_different_tool_name(self, agent_state, messages, parsed_response):
        """Test with different tool name."""
        hook = create_tool_after_approve_hook(tool_name="other_tool")
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=parsed_response,
        )

        result = hook(hook_input)

        assert isinstance(result, AfterModelHookResult)
        assert result.has_modifications() is False


class TestCreateToolLoggingHook:
    """Tests for create_tool_logging_hook function."""

    def test_tool_logging_hook_short_output(self, agent_state, caplog):
        """Test tool logging hook with short output."""
        caplog.set_level(logging.INFO)

        hook = create_tool_logging_hook(logger_name="test_logger")
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="short output",
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is False
        assert "AFTER TOOL HOOK TRIGGERED" in caplog.text
        assert "test_tool" in caplog.text
        assert "short output" in caplog.text

    def test_tool_logging_hook_long_output(self, agent_state, caplog):
        """Test tool logging hook with long output (truncated)."""
        caplog.set_level(logging.INFO)

        hook = create_tool_logging_hook(logger_name="test_logger")
        long_output = "x" * 600
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output=long_output,
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is False
        assert "truncated" in caplog.text


class TestMiddlewareManager:
    """Tests for the unified middleware manager."""

    def test_run_before_model_in_order(self, agent_state, messages):
        """Before-model hooks execute from first to last."""
        order: list[str] = []

        def make_hook(name: str):
            def hook(hook_input: BeforeModelHookInput) -> HookResult:
                order.append(name)
                new_messages = hook_input.messages + [{"role": "system", "content": name}]
                return HookResult.with_modifications(messages=new_messages)

            return hook

        manager = MiddlewareManager(
            [
                FunctionMiddleware(before_model_hook=make_hook("first")),
                FunctionMiddleware(before_model_hook=make_hook("second")),
            ],
        )

        hook_input = BeforeModelHookInput(
            agent_state=agent_state,
            max_iterations=5,
            current_iteration=1,
            messages=messages.copy(),
        )

        updated_messages = manager.run_before_model(hook_input)
        assert order == ["first", "second"]
        assert [msg["content"] for msg in updated_messages[-2:]] == ["first", "second"]

    def test_run_after_model_reverse_order_with_force_continue(self, agent_state, messages, parsed_response):
        """After-model hooks execute in reverse order and can set force_continue."""
        order: list[str] = []

        def feedback_hook(hook_input: AfterModelHookInput) -> HookResult:
            order.append("feedback")
            new_messages = hook_input.messages + [{"role": "user", "content": "feedback"}]
            return HookResult.with_modifications(messages=new_messages)

        def cleanup_hook(hook_input: AfterModelHookInput) -> HookResult:
            order.append("cleanup")
            hook_input.parsed_response.tool_calls = []
            return HookResult.with_modifications(parsed_response=hook_input.parsed_response, force_continue=True)

        manager = MiddlewareManager(
            [
                FunctionMiddleware(after_model_hook=feedback_hook),
                FunctionMiddleware(after_model_hook=cleanup_hook),
            ],
        )

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=5,
            current_iteration=1,
            messages=messages.copy(),
            original_response="resp",
            parsed_response=parsed_response,
        )

        parsed, updated_messages, force_continue = manager.run_after_model(hook_input)
        assert order == ["cleanup", "feedback"]
        assert parsed is parsed_response
        assert not parsed.tool_calls
        assert updated_messages[-1]["content"] == "feedback"
        assert force_continue is True

    def test_run_after_tool_reverse_order(self, agent_state):
        """After-tool hooks execute from last to first."""
        order: list[str] = []

        def make_tool_hook(name: str):
            def hook(hook_input: AfterToolHookInput) -> HookResult:
                order.append(name)
                return HookResult.with_modifications(tool_output=f"{hook_input.tool_output}-{name}")

            return hook

        manager = MiddlewareManager(
            [
                FunctionMiddleware(after_tool_hook=make_tool_hook("first")),
                FunctionMiddleware(after_tool_hook=make_tool_hook("second")),
            ],
        )

        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="demo",
            tool_call_id="call_1",
            tool_input={},
            tool_output="base",
        )

        result = manager.run_after_tool(hook_input, "base")
        assert order == ["second", "first"]
        assert result == "base-second-first"

    def test_wrap_model_call_nested(self):
        """wrap_model_call applies middleware in a nested fashion."""
        call_log: list[str] = []

        class RecordingMiddleware(Middleware):
            def __init__(self, name: str) -> None:
                self.name = name

            def wrap_model_call(self, call_next):  # type: ignore[override]
                def wrapped(params: ModelCallParams):
                    call_log.append(f"before_{self.name}")
                    result = call_next(params)
                    call_log.append(f"after_{self.name}")
                    return result

                return wrapped

        manager = MiddlewareManager(
            [RecordingMiddleware("outer"), RecordingMiddleware("inner")],
        )

        params = ModelCallParams(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            openai_client=None,
            llm_config=None,
        )

        def base_call(_: ModelCallParams) -> ModelResponse:
            call_log.append("base")
            return ModelResponse(content="ok")

        wrapped = manager.wrap_model_call(base_call)
        response = wrapped(params)
        assert response.content == "ok"
        assert call_log == ["before_outer", "before_inner", "base", "after_inner", "after_outer"]

    def test_wrap_tool_call_nested(self, agent_state):
        """wrap_tool_call applies middleware in a nested fashion for tools."""
        call_log: list[str] = []

        class RecordingMiddleware(Middleware):
            def __init__(self, name: str) -> None:
                self.name = name

            def wrap_tool_call(self, call_next):  # type: ignore[override]
                def wrapped(params: ToolCallParams):
                    call_log.append(f"before_{self.name}")
                    result = call_next(params)
                    call_log.append(f"after_{self.name}")
                    return result

                return wrapped

        manager = MiddlewareManager(
            [RecordingMiddleware("outer"), RecordingMiddleware("inner")],
        )

        params = ToolCallParams(
            agent_state=agent_state,
            tool_name="demo",
            parameters={},
            tool_call_id="call_1",
            execution_params={},
        )

        def base_call(_: ToolCallParams) -> dict[str, str]:
            call_log.append("base")
            return {"result": "ok"}

        wrapped = manager.wrap_tool_call(base_call)
        result = wrapped(params)
        assert result == {"result": "ok"}
        assert call_log == ["before_outer", "before_inner", "base", "after_inner", "after_outer"]

    def test_run_before_tool(self, agent_state):
        """before_tool hooks run first-to-last and can modify input."""
        order: list[str] = []

        def make_hook(name: str):
            def hook(hook_input: BeforeToolHookInput) -> HookResult:
                order.append(name)
                updated = dict(hook_input.tool_input)
                updated[name] = True
                return HookResult.with_modifications(tool_input=updated)

            return hook

        manager = MiddlewareManager(
            [
                FunctionMiddleware(before_tool_hook=make_hook("first")),
                FunctionMiddleware(before_tool_hook=make_hook("second")),
            ],
        )

        hook_input = BeforeToolHookInput(
            agent_state=agent_state,
            tool_name="demo",
            tool_call_id="call_1",
            tool_input={"initial": True},
        )

        updated = manager.run_before_tool(hook_input)
        assert order == ["first", "second"]
        assert updated == {"initial": True, "first": True, "second": True}

    def test_logging_middleware_wrap_model_call(self, agent_state, capsys):
        """LoggingMiddleware can wrap model calls and emit console output."""
        middleware = LoggingMiddleware(log_model_calls=True)

        params = ModelCallParams(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            force_stop_reason=None,
            agent_state=agent_state,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            openai_client=None,
            llm_config=None,
            retry_attempts=1,
        )

        def base_call(_: ModelCallParams) -> ModelResponse:
            return ModelResponse(content="hi")

        wrapped = middleware.wrap_model_call(base_call)
        result = wrapped(params)
        assert isinstance(result, ModelResponse)
        captured = capsys.readouterr().out
        assert "Custom LLM Generator called with 1 messages" in captured


class TestHookProtocols:
    """Tests for hook protocol compliance."""

    def test_before_model_hook_protocol(self, agent_state, messages):
        """Test that a function conforms to BeforeModelHook protocol."""

        def my_hook(hook_input: BeforeModelHookInput) -> BeforeModelHookResult:
            return BeforeModelHookResult.no_changes()

        # This should type-check and work
        hook_input = BeforeModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
        )
        result = my_hook(hook_input)
        assert isinstance(result, BeforeModelHookResult)

    def test_after_model_hook_protocol(self, agent_state, messages, parsed_response):
        """Test that a function conforms to AfterModelHook protocol."""

        def my_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
            return AfterModelHookResult.no_changes()

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test",
            parsed_response=parsed_response,
        )
        result = my_hook(hook_input)
        assert isinstance(result, AfterModelHookResult)

    def test_after_tool_hook_protocol(self, agent_state):
        """Test that a function conforms to AfterToolHook protocol."""

        def my_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
            return AfterToolHookResult.no_changes()

        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="result",
        )
        result = my_hook(hook_input)
        assert isinstance(result, AfterToolHookResult)
