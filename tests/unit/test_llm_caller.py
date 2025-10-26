"""
Unit tests for LLM caller component.

Tests cover:
- LLMCaller initialization
- Basic LLM API calls
- Retry logic with exponential backoff
- Custom LLM generator functionality
- Force stop reason handling
- XML tag restoration
- Debug logging
- Stop sequence handling
- Error scenarios
"""

import logging
from unittest.mock import Mock, call, patch

import pytest

from northau.archs.main_sub.execution.llm_caller import LLMCaller, bypass_llm_generator
from northau.archs.main_sub.execution.stop_reason import AgentStopReason


class TestLLMCallerInitialization:
    """Test cases for LLMCaller initialization."""

    def test_initialization_with_defaults(self, mock_openai_client, mock_llm_config):
        """Test LLMCaller initialization with default parameters."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        assert caller.openai_client == mock_openai_client
        assert caller.llm_config == mock_llm_config
        assert caller.retry_attempts == 5
        assert caller.custom_llm_generator is None

    def test_initialization_with_custom_retry_attempts(self, mock_openai_client, mock_llm_config):
        """Test LLMCaller initialization with custom retry attempts."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=10,
        )

        assert caller.retry_attempts == 10

    def test_initialization_with_custom_llm_generator(self, mock_openai_client, mock_llm_config):
        """Test LLMCaller initialization with custom LLM generator."""

        def custom_generator(client, params, force_stop, state):
            return "custom response"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            custom_llm_generator=custom_generator,
        )

        assert caller.custom_llm_generator == custom_generator


class TestLLMCallerBasicCalls:
    """Test cases for basic LLM API calls."""

    def test_call_llm_success(self, mock_openai_client, mock_llm_config, agent_state):
        """Test successful LLM API call."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Hello! How can I help you?"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_call_llm_without_client_raises_error(self, mock_llm_config):
        """Test that calling LLM without client raises RuntimeError."""
        caller = LLMCaller(
            openai_client=None,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="OpenAI client is not available"):
            caller.call_llm(messages, max_tokens=100)

    def test_call_llm_includes_xml_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that LLM calls include XML stop sequences."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that stop sequences were added
        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]

        assert "</tool_use>" in stop_sequences
        assert "</use_parallel_tool_calls>" in stop_sequences
        assert "</use_batch_agent>" in stop_sequences

    def test_call_llm_merges_existing_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that existing stop sequences are preserved and merged."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with existing stop sequences
        mock_llm_config.to_openai_params = Mock(
            return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": ["custom_stop_1", "custom_stop_2"]}
        )

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]

        # Both custom and XML stop sequences should be present
        assert "custom_stop_1" in stop_sequences
        assert "custom_stop_2" in stop_sequences
        assert "</tool_use>" in stop_sequences

    def test_call_llm_handles_string_stop_sequence(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that string stop sequences are converted to list."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with string stop sequence
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": "single_stop"})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]

        assert "single_stop" in stop_sequences
        assert "</tool_use>" in stop_sequences


class TestLLMCallerXMLRestoration:
    """Test cases for XML tag restoration."""

    def test_call_llm_restores_closing_tags(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that closing XML tags are restored when missing."""
        # Response with missing closing tag
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<tool_use><tool_name>test"
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # The XMLUtils.restore_closing_tags should add </tool_use>
        assert "</tool_use>" in response

    def test_call_llm_splits_response_at_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that response is split at stop sequences."""
        # Response that contains stop sequence
        full_response = "Response content</tool_use>Extra content"
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = full_response
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Response should be split at the stop sequence
        # After split and restoration, it should have the closing tag but not extra content
        assert "Response content" in response
        assert "Extra content" not in response


class TestLLMCallerDebugLogging:
    """Test cases for debug logging."""

    def test_call_llm_debug_logging_enabled(self, mock_openai_client, mock_llm_config, agent_state, caplog):
        """Test that debug logging works when enabled."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Debug response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Enable debug mode
        mock_llm_config.debug = True

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "User message"}]

        with caplog.at_level(logging.INFO):
            caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that debug logs were created
        debug_logs = [rec.message for rec in caplog.records if "🐛 [DEBUG]" in rec.message]
        assert len(debug_logs) > 0

    def test_call_llm_debug_logging_disabled(self, mock_openai_client, mock_llm_config, agent_state, caplog):
        """Test that debug logging is disabled when debug mode is off."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Disable debug mode
        mock_llm_config.debug = False

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with caplog.at_level(logging.INFO):
            caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that debug logs were NOT created
        debug_logs = [rec.message for rec in caplog.records if "🐛 [DEBUG]" in rec.message]
        assert len(debug_logs) == 0


class TestLLMCallerRetryLogic:
    """Test cases for retry logic."""

    def test_call_llm_retry_on_failure(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that LLM calls retry on failure."""
        # First two calls fail, third succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(choices=[Mock(message=Mock(content="Success after retry"))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):  # Mock sleep to speed up test
            response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Success after retry"
        assert mock_openai_client.chat.completions.create.call_count == 3

    def test_call_llm_exhausts_retries(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that LLM calls raise error after exhausting retries."""
        # All calls fail
        mock_openai_client.chat.completions.create.side_effect = Exception("Persistent API Error")

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(Exception, match="Persistent API Error"):
                caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert mock_openai_client.chat.completions.create.call_count == 3

    def test_call_llm_exponential_backoff(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that retry uses exponential backoff."""
        # Fail a few times to trigger backoff
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Mock(choices=[Mock(message=Mock(content="Success"))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep") as mock_sleep:
            caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that sleep was called with exponential backoff
        sleep_calls = mock_sleep.call_args_list
        assert len(sleep_calls) == 2  # Two failures before success
        assert sleep_calls[0] == call(1)  # First backoff: 1 second
        assert sleep_calls[1] == call(2)  # Second backoff: 2 seconds

    def test_call_llm_empty_response_triggers_retry(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that empty response content triggers retry."""
        # First call returns empty content, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content=""))]),
            Mock(choices=[Mock(message=Mock(content="Valid response"))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):
            response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Valid response"
        assert mock_openai_client.chat.completions.create.call_count == 2


class TestLLMCallerForceStopReason:
    """Test cases for force stop reason handling."""

    def test_call_llm_with_force_stop_reason_success(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with SUCCESS stop reason proceeds normally."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Normal response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
        )

        assert response == "Normal response"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_call_llm_with_non_success_stop_reason_returns_none(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with non-SUCCESS stop reason returns None."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
            agent_state=agent_state,
        )

        assert response is None
        # OpenAI client should not be called
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_call_llm_with_error_stop_reason(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with ERROR_OCCURRED stop reason."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.ERROR_OCCURRED,
            agent_state=agent_state,
        )

        assert response is None
        mock_openai_client.chat.completions.create.assert_not_called()


class TestLLMCallerCustomGenerator:
    """Test cases for custom LLM generator functionality."""

    def test_call_llm_with_custom_generator(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with custom generator."""

        def custom_generator(client, params, force_stop, state):
            return "Custom generator response"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            custom_llm_generator=custom_generator,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
        )

        assert response == "Custom generator response"
        # OpenAI client should not be called when custom generator is used
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_call_llm_custom_generator_receives_correct_params(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that custom generator receives correct parameters."""
        received_params = {}

        def custom_generator(client, params, force_stop, state):
            received_params["client"] = client
            received_params["params"] = params
            received_params["force_stop"] = force_stop
            received_params["state"] = state
            return "Response"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            custom_llm_generator=custom_generator,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
        )

        assert received_params["client"] == mock_openai_client
        assert "messages" in received_params["params"]
        assert received_params["params"]["messages"] == messages
        assert received_params["params"]["max_tokens"] == 100
        assert received_params["force_stop"] == AgentStopReason.SUCCESS
        assert received_params["state"] == agent_state

    def test_call_llm_custom_generator_error_retry(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that custom generator errors trigger retry."""
        call_count = [0]

        def custom_generator(client, params, force_stop, state):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception(f"Custom generator error {call_count[0]}")
            return "Success after retry"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=5,
            custom_llm_generator=custom_generator,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):
            response = caller.call_llm(
                messages,
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        assert response == "Success after retry"
        assert call_count[0] == 3

    def test_call_llm_custom_generator_with_non_success_stop(self, mock_openai_client, mock_llm_config, agent_state):
        """Test custom generator with non-SUCCESS stop reason."""

        def custom_generator(client, params, force_stop, state):
            # Return a non-empty string when force_stop is not SUCCESS
            # to avoid AttributeError on split
            if force_stop != AgentStopReason.SUCCESS:
                return "Stopped"
            return "Response"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            custom_llm_generator=custom_generator,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
            agent_state=agent_state,
        )

        # Should return "Stopped" from the custom generator
        assert response == "Stopped"


class TestBypassLLMGenerator:
    """Test cases for bypass_llm_generator function."""

    def test_bypass_llm_generator_success(self, mock_openai_client, agent_state, capsys):
        """Test bypass LLM generator with successful API call."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "API response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.7,
        }

        result = bypass_llm_generator(
            mock_openai_client,
            kwargs,
            AgentStopReason.SUCCESS,
            agent_state,
        )

        assert result == "API response"
        captured = capsys.readouterr()
        assert "Custom LLM Generator called with 1 messages" in captured.out

    def test_bypass_llm_generator_with_error(self, mock_openai_client, agent_state, capsys):
        """Test bypass LLM generator when API call fails."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Test"}],
        }

        with pytest.raises(Exception, match="API Error"):
            bypass_llm_generator(
                mock_openai_client,
                kwargs,
                AgentStopReason.SUCCESS,
                agent_state,
            )

        captured = capsys.readouterr()
        assert "Bypass LLM generator error: API Error" in captured.out

    def test_bypass_llm_generator_with_multiple_messages(self, mock_openai_client, agent_state, capsys):
        """Test bypass LLM generator with multiple messages."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant reply"},
            ],
        }

        result = bypass_llm_generator(
            mock_openai_client,
            kwargs,
            AgentStopReason.SUCCESS,
            agent_state,
        )

        assert result == "Response"
        captured = capsys.readouterr()
        assert "Custom LLM Generator called with 3 messages" in captured.out


class TestLLMCallerEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def test_call_llm_with_empty_messages(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with empty messages list."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        response = caller.call_llm([], max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Response"
        # Check that empty messages were passed
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == []

    def test_call_llm_with_zero_max_tokens(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with zero max_tokens."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=0, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Response"
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["max_tokens"] == 0

    def test_call_llm_with_none_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call when stop sequences are None."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with None stop sequences
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": None})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Response"
        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]
        # Should only have XML stop sequences
        assert "</tool_use>" in stop_sequences

    def test_call_llm_with_complex_response_content(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with complex response containing special characters."""
        complex_response = """
<tool_use>
<tool_name>test_tool</tool_name>
<parameter>
<query>Search for "complex & special < > characters"</query>
</parameter>
</tool_use>
"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = complex_response
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Response should preserve special characters
        assert "complex & special < > characters" in response
        assert "<tool_use>" in response

    def test_call_llm_with_very_large_retry_attempts(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM caller with very large retry attempts."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=100,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Response"
        assert caller.retry_attempts == 100

    def test_call_llm_response_content_none_triggers_exception(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that None response content triggers retry."""
        # First call returns None, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content=None))]),
            Mock(choices=[Mock(message=Mock(content="Valid response"))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):
            response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response == "Valid response"
        assert mock_openai_client.chat.completions.create.call_count == 2


class TestLLMCallerIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_all_features(self, mock_openai_client, mock_llm_config, agent_state):
        """Test complete workflow with all features enabled."""
        # Setup mock with XML response containing stop sequence
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<tool_use><tool_name>test</tool_name>"
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Enable debug mode
        mock_llm_config.debug = True
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": ["custom_stop"]})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Execute tool"}]

        response = caller.call_llm(
            messages,
            max_tokens=200,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
        )

        # Verify response has closing tag restored
        assert "</tool_use>" in response
        assert "<tool_use>" in response

        # Verify API was called with correct parameters
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == messages
        assert call_args[1]["max_tokens"] == 200
        assert "custom_stop" in call_args[1]["stop"]
        assert "</tool_use>" in call_args[1]["stop"]

    def test_retry_with_custom_generator_and_backoff(self, mock_openai_client, mock_llm_config, agent_state):
        """Test retry logic with custom generator and exponential backoff."""
        call_count = [0]

        def flaky_generator(client, params, force_stop, state):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception(f"Temporary error {call_count[0]}")
            return "<tool_use><tool_name>success"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=5,
            custom_llm_generator=flaky_generator,
        )

        messages = [{"role": "user", "content": "Test"}]

        with patch("time.sleep") as mock_sleep:
            response = caller.call_llm(
                messages,
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        # Should succeed after retries
        assert "success" in response
        assert "</tool_use>" in response  # Tag restored
        assert call_count[0] == 3

        # Verify exponential backoff
        assert len(mock_sleep.call_args_list) == 2
        assert mock_sleep.call_args_list[0] == call(1)
        assert mock_sleep.call_args_list[1] == call(2)
