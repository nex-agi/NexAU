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
    HookManager,
    ToolHookManager,
    create_filter_hook,
    create_logging_hook,
    create_remaining_reminder_hook,
    create_tool_after_approve_hook,
    create_tool_logging_hook,
    create_tool_output_filter_hook,
    create_tool_result_transformer_hook,
)
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
            assert result.has_modifications() is True
            assert len(result.parsed_response.tool_calls) == 1

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
                assert len(result.parsed_response.tool_calls) == 1

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
        # Should not filter because tool name doesn't match
        assert len(result.parsed_response.tool_calls) == 1


class TestCreateFilterHook:
    """Tests for create_filter_hook function."""

    def test_filter_allowed_tools(self, agent_state, messages, parsed_response, caplog):
        """Test filtering with allowed tools."""
        caplog.set_level(logging.WARNING)

        hook = create_filter_hook(allowed_tools={"allowed_tool"})
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
        assert "Filtered out 1 disallowed tool calls" in caplog.text

    def test_filter_allowed_agents(self, agent_state, messages, parsed_response, caplog):
        """Test filtering with allowed agents."""
        caplog.set_level(logging.WARNING)

        hook = create_filter_hook(allowed_agents={"allowed_agent"})
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
        assert len(result.parsed_response.sub_agent_calls) == 0
        assert len(result.parsed_response.batch_agent_calls) == 0
        assert "Filtered out 1 disallowed sub-agent calls" in caplog.text
        assert "Filtered out 1 disallowed batch agent calls" in caplog.text

    def test_no_filtering_needed(self, agent_state, messages, parsed_response):
        """Test when no filtering is needed."""
        hook = create_filter_hook(
            allowed_tools={"test_tool"},
            allowed_agents={"sub_agent", "batch_agent"},
        )
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

    def test_none_parsed_response(self, agent_state, messages):
        """Test with None parsed response."""
        hook = create_filter_hook(allowed_tools={"test_tool"})
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

    def test_none_allowed_sets(self, agent_state, messages, parsed_response):
        """Test with None allowed sets (allow all)."""
        hook = create_filter_hook(allowed_tools=None, allowed_agents=None)
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


class TestCreateToolOutputFilterHook:
    """Tests for create_tool_output_filter_hook function."""

    def test_filter_sensitive_keys(self, agent_state, caplog):
        """Test filtering sensitive keys from dict output."""
        caplog.set_level(logging.INFO)

        hook = create_tool_output_filter_hook(filter_keys={"password", "secret"})
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output={
                "username": "user",
                "password": "secret123",
                "data": "public",
                "secret": "hidden",
            },
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is True
        assert "username" in result.tool_output
        assert "data" in result.tool_output
        assert "password" not in result.tool_output
        assert "secret" not in result.tool_output
        assert "Filtered 2 sensitive keys" in caplog.text

    def test_no_sensitive_keys(self, agent_state):
        """Test with no sensitive keys to filter."""
        hook = create_tool_output_filter_hook(filter_keys={"password"})
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output={"username": "user", "data": "public"},
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is False

    def test_non_dict_output(self, agent_state):
        """Test with non-dict output."""
        hook = create_tool_output_filter_hook(filter_keys={"password"})
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="string output",
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is False


class TestCreateToolResultTransformerHook:
    """Tests for create_tool_result_transformer_hook function."""

    def test_successful_transformation(self, agent_state):
        """Test successful transformation."""

        def transform_func(agent_name, agent_id, tool_name, tool_input, tool_output):
            return tool_output.upper()

        hook = create_tool_result_transformer_hook(transform_func)
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="hello",
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is True
        assert result.tool_output == "HELLO"

    def test_no_transformation(self, agent_state):
        """Test when transformation returns same output."""

        def transform_func(agent_name, agent_id, tool_name, tool_input, tool_output):
            return tool_output

        hook = create_tool_result_transformer_hook(transform_func)
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="hello",
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is False

    def test_transformation_error(self, agent_state, caplog):
        """Test when transformation raises an error."""
        caplog.set_level(logging.WARNING)

        def transform_func(agent_name, agent_id, tool_name, tool_input, tool_output):
            raise ValueError("Transformation failed")

        hook = create_tool_result_transformer_hook(transform_func)
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="hello",
        )

        result = hook(hook_input)

        assert isinstance(result, AfterToolHookResult)
        assert result.has_modifications() is False
        assert "Tool transformation failed" in caplog.text


class TestToolHookManager:
    """Tests for ToolHookManager class."""

    def test_init_empty(self):
        """Test initialization with no hooks."""
        manager = ToolHookManager()
        assert len(manager) == 0
        assert bool(manager) is False

    def test_init_with_hooks(self):
        """Test initialization with hooks."""
        hook1 = create_tool_logging_hook()
        hook2 = create_tool_output_filter_hook({"password"})
        manager = ToolHookManager(hooks=[hook1, hook2])
        assert len(manager) == 2
        assert bool(manager) is True

    def test_add_hook(self):
        """Test adding a hook."""
        manager = ToolHookManager()
        hook = create_tool_logging_hook()
        manager.add_hook(hook)
        assert len(manager) == 1
        assert hook in manager.hooks

    def test_remove_hook(self):
        """Test removing a hook."""
        hook = create_tool_logging_hook()
        manager = ToolHookManager(hooks=[hook])
        assert len(manager) == 1
        manager.remove_hook(hook)
        assert len(manager) == 0

    def test_remove_nonexistent_hook(self):
        """Test removing a hook that doesn't exist."""
        hook1 = create_tool_logging_hook()
        hook2 = create_tool_output_filter_hook({"password"})
        manager = ToolHookManager(hooks=[hook1])
        manager.remove_hook(hook2)  # Should not raise error
        assert len(manager) == 1

    def test_execute_hooks_single(self, agent_state, caplog):
        """Test executing a single hook."""
        caplog.set_level(logging.INFO)

        manager = ToolHookManager()
        manager.add_hook(create_tool_logging_hook())

        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="original",
        )

        result = manager.execute_hooks(hook_input)

        assert result == "original"
        assert "Executing tool hook 1/1" in caplog.text

    def test_execute_hooks_multiple(self, agent_state, caplog):
        """Test executing multiple hooks in sequence."""
        caplog.set_level(logging.INFO)

        def transform_upper(agent_name, agent_id, tool_name, tool_input, tool_output):
            return tool_output.upper()

        def transform_suffix(agent_name, agent_id, tool_name, tool_input, tool_output):
            return tool_output + "_SUFFIX"

        manager = ToolHookManager()
        manager.add_hook(create_tool_result_transformer_hook(transform_upper))
        manager.add_hook(create_tool_result_transformer_hook(transform_suffix))

        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="hello",
        )

        result = manager.execute_hooks(hook_input)

        assert result == "HELLO_SUFFIX"
        assert "Executing tool hook 1/2" in caplog.text
        assert "Executing tool hook 2/2" in caplog.text

    def test_execute_hooks_with_error(self, agent_state, caplog):
        """Test executing hooks when one fails."""
        caplog.set_level(logging.WARNING)

        def failing_transform(agent_name, agent_id, tool_name, tool_input, tool_output):
            raise ValueError("Hook failed")

        def working_transform(agent_name, agent_id, tool_name, tool_input, tool_output):
            return tool_output.upper()

        manager = ToolHookManager()
        manager.add_hook(create_tool_result_transformer_hook(failing_transform))
        manager.add_hook(create_tool_result_transformer_hook(working_transform))

        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output="hello",
        )

        result = manager.execute_hooks(hook_input)

        # Should still apply second hook despite first failing
        assert result == "HELLO"
        assert "Tool transformation failed" in caplog.text


class TestHookManager:
    """Tests for HookManager class."""

    def test_init_empty(self):
        """Test initialization with no hooks."""
        manager = HookManager()
        assert len(manager) == 0
        assert bool(manager) is False

    def test_init_with_hooks(self):
        """Test initialization with hooks."""
        hook1 = create_logging_hook()
        hook2 = create_remaining_reminder_hook()
        manager = HookManager(hooks=[hook1, hook2])
        assert len(manager) == 2
        assert bool(manager) is True

    def test_add_hook(self):
        """Test adding a hook."""
        manager = HookManager()
        hook = create_logging_hook()
        manager.add_hook(hook)
        assert len(manager) == 1
        assert hook in manager.hooks

    def test_remove_hook(self):
        """Test removing a hook."""
        hook = create_logging_hook()
        manager = HookManager(hooks=[hook])
        assert len(manager) == 1
        manager.remove_hook(hook)
        assert len(manager) == 0

    def test_remove_nonexistent_hook(self):
        """Test removing a hook that doesn't exist."""
        hook1 = create_logging_hook()
        hook2 = create_remaining_reminder_hook()
        manager = HookManager(hooks=[hook1])
        manager.remove_hook(hook2)  # Should not raise error
        assert len(manager) == 1

    def test_execute_hooks_after_model_single(self, agent_state, messages, parsed_response, caplog):
        """Test executing a single after model hook."""
        caplog.set_level(logging.INFO)

        manager = HookManager()
        manager.add_hook(create_logging_hook())

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=parsed_response,
        )

        result_parsed, result_messages = manager.execute_hooks(hook_input)

        assert result_parsed == parsed_response
        assert result_messages == messages
        assert "Executing hook 1/1" in caplog.text

    def test_execute_hooks_after_model_multiple(self, agent_state, messages, parsed_response, caplog):
        """Test executing multiple after model hooks."""
        caplog.set_level(logging.INFO)

        manager = HookManager()
        manager.add_hook(create_logging_hook())
        manager.add_hook(create_remaining_reminder_hook())

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages.copy(),
            original_response="test response",
            parsed_response=parsed_response,
        )

        result_parsed, result_messages = manager.execute_hooks(hook_input)

        assert result_parsed == parsed_response
        assert len(result_messages) == len(messages) + 1
        assert "Executing hook 1/2" in caplog.text
        assert "Executing hook 2/2" in caplog.text

    def test_execute_hooks_before_model(self, agent_state, messages, caplog):
        """Test executing before model hooks."""
        caplog.set_level(logging.INFO)

        def before_hook(hook_input: BeforeModelHookInput) -> BeforeModelHookResult:
            modified_messages = hook_input.messages + [{"role": "user", "content": "Extra message"}]
            return BeforeModelHookResult.with_modifications(messages=modified_messages)

        manager = HookManager()
        manager.add_hook(before_hook)

        hook_input = BeforeModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages.copy(),
        )

        result_messages = manager.execute_hooks(hook_input)

        assert len(result_messages) == len(messages) + 1
        assert result_messages[-1]["content"] == "Extra message"

    def test_execute_hooks_with_error(self, agent_state, messages, parsed_response, caplog):
        """Test executing hooks when one fails."""
        caplog.set_level(logging.WARNING)

        def failing_hook(hook_input):
            raise ValueError("Hook failed")

        manager = HookManager()
        manager.add_hook(failing_hook)
        manager.add_hook(create_logging_hook())

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=parsed_response,
        )

        result_parsed, result_messages = manager.execute_hooks(hook_input)

        # Should still complete despite first hook failing
        assert result_parsed == parsed_response
        assert result_messages == messages
        assert "Hook 1 failed" in caplog.text

    def test_execute_hooks_modify_parsed_response(self, agent_state, messages, parsed_response, caplog):
        """Test hook that modifies parsed response."""
        caplog.set_level(logging.INFO)

        def modify_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
            # Clear all tool calls
            hook_input.parsed_response.tool_calls = []
            return AfterModelHookResult.with_modifications(parsed_response=hook_input.parsed_response)

        manager = HookManager()
        manager.add_hook(modify_hook)

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages,
            original_response="test response",
            parsed_response=parsed_response,
        )

        result_parsed, result_messages = manager.execute_hooks(hook_input)

        assert len(result_parsed.tool_calls) == 0
        assert "modified the parsed response" in caplog.text

    def test_execute_hooks_modify_messages(self, agent_state, messages, parsed_response, caplog):
        """Test hook that modifies messages."""
        caplog.set_level(logging.INFO)

        def modify_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
            new_messages = hook_input.messages + [{"role": "user", "content": "Additional context"}]
            return AfterModelHookResult.with_modifications(messages=new_messages)

        manager = HookManager()
        manager.add_hook(modify_hook)

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=5,
            messages=messages.copy(),
            original_response="test response",
            parsed_response=parsed_response,
        )

        result_parsed, result_messages = manager.execute_hooks(hook_input)

        assert len(result_messages) == len(messages) + 1
        assert "modified the message history" in caplog.text


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
