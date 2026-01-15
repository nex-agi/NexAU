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

"""Comprehensive tests for context compaction middleware."""

from unittest.mock import Mock, patch

import pytest

from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.hooks import AfterModelHookInput
from nexau.archs.main_sub.execution.middleware.context_compaction import (
    ContextCompactionMiddleware,
    SlidingWindowCompaction,
    TokenThresholdTrigger,
    ToolResultCompaction,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock


@pytest.fixture
def agent_state(mock_executor):
    """Create a mock agent state for testing."""
    context = AgentContext()
    global_storage = GlobalStorage()
    return AgentState(
        agent_name="test_agent",
        agent_id="test_id_123",
        context=context,
        global_storage=global_storage,
        executor=mock_executor,
    )


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    legacy = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is Python?"},
        {"role": "assistant", "content": "Python is a programming language.", "tool_calls": []},
        {"role": "user", "content": "Tell me more about it."},
        {"role": "assistant", "content": "Python is versatile and easy to learn.", "tool_calls": []},
        {"role": "user", "content": "What are its main features?"},
        {"role": "assistant", "content": "Python has dynamic typing and automatic memory management.", "tool_calls": []},
    ]
    return messages_from_legacy_openai_chat(legacy)


@pytest.fixture
def mock_token_counter():
    """Mock token counter for testing."""
    counter = Mock()
    counter.count_tokens.side_effect = lambda msgs: len(msgs) * 100  # Simple mock: 100 tokens per message
    return counter


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a comprehensive summary of the conversation."
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_model_response():
    """Create a mock model response with usage information."""
    return ModelResponse(
        content="Test response",
        role="assistant",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    )


@pytest.fixture
def temp_compact_prompt(tmp_path):
    """Create a temporary compact prompt file for testing."""
    prompt_file = tmp_path / "compact_prompt.md"
    prompt_file.write_text("Please summarize the conversation above concisely.")
    return str(prompt_file)


class TestTokenThresholdTrigger:
    """Tests for TokenThresholdTrigger strategy."""

    def test_initialization_default(self):
        """Test initialization with default threshold."""
        trigger = TokenThresholdTrigger()
        assert trigger.threshold == 0.75

    def test_initialization_custom(self):
        """Test initialization with custom threshold."""
        trigger = TokenThresholdTrigger(threshold=0.80)
        assert trigger.threshold == 0.80

    def test_should_compact_below_threshold(self, sample_messages):
        """Test should_compact returns False when below threshold."""
        trigger = TokenThresholdTrigger(threshold=0.75)
        current_tokens = 5000
        max_tokens = 10000  # 50% usage

        should_compact, reason = trigger.should_compact(sample_messages, current_tokens, max_tokens)

        assert should_compact is False
        assert reason == ""

    def test_should_compact_at_threshold(self, sample_messages):
        """Test should_compact returns True when at threshold."""
        trigger = TokenThresholdTrigger(threshold=0.75)
        current_tokens = 7500
        max_tokens = 10000  # 75% usage

        should_compact, reason = trigger.should_compact(sample_messages, current_tokens, max_tokens)

        assert should_compact is True
        assert "75.0%" in reason
        assert "7500/10000" in reason

    def test_should_compact_above_threshold(self, sample_messages):
        """Test should_compact returns True when above threshold."""
        trigger = TokenThresholdTrigger(threshold=0.75)
        current_tokens = 8500
        max_tokens = 10000  # 85% usage

        should_compact, reason = trigger.should_compact(sample_messages, current_tokens, max_tokens)

        assert should_compact is True
        assert "85.0%" in reason


class TestSlidingWindowCompaction:
    """Tests for SlidingWindowCompaction strategy."""

    def test_initialization_requires_llm_config(self):
        """Test that initialization requires LLM configuration."""
        with pytest.raises(ValueError, match="LLM configuration is required"):
            SlidingWindowCompaction(
                keep_iterations=3,
                summary_model=None,
                summary_base_url=None,
                summary_api_key=None,
            )

    def test_initialization_with_llm_config(self, mock_openai_client, temp_compact_prompt):
        """Test initialization with valid LLM configuration."""
        with patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI") as mock_openai:
            mock_openai.return_value = mock_openai_client
            compaction = SlidingWindowCompaction(
                keep_iterations=3,
                summary_model="gpt-4o-mini",
                summary_base_url="https://api.openai.com/v1",
                summary_api_key="test-key",
                compact_prompt_path=temp_compact_prompt,
            )
            assert compaction.summary_model == "gpt-4o-mini"
            assert compaction.keep_iterations == 3

    def test_keep_iterations_validation(self, temp_compact_prompt):
        """Test that keep_iterations must be >= 1."""
        with pytest.raises(ValueError, match="keep_iterations must be >= 1"):
            SlidingWindowCompaction(
                keep_iterations=0,
                summary_model="gpt-4o-mini",
                summary_base_url="https://api.openai.com/v1",
                summary_api_key="test-key",
                compact_prompt_path=temp_compact_prompt,
            )

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_compact_preserves_system_message(self, mock_openai_class, mock_openai_client, sample_messages, temp_compact_prompt):
        """Test that compaction preserves system message."""
        mock_openai_class.return_value = mock_openai_client
        compaction = SlidingWindowCompaction(
            keep_iterations=2,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        # Create enough messages to trigger compaction
        messages = sample_messages * 3  # 21 messages, 3 iterations
        result = compaction.compact(messages)

        assert result[0].role == Role.SYSTEM
        assert result[0].get_text_content() == "You are a helpful assistant."


class TestToolResultCompaction:
    """Tests for ToolResultCompaction strategy."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        compaction = ToolResultCompaction(keep_system=True)
        assert compaction.keep_system is True

    def test_compact_preserves_system_message(self):
        """Test that compaction preserves system message."""
        compaction = ToolResultCompaction()
        messages = messages_from_legacy_openai_chat(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "tool", "tool_call_id": "call_1", "content": "Tool result"},
            ],
        )
        result = compaction.compact(messages)
        assert result[0].role == Role.SYSTEM

    def test_compact_tool_results(self):
        """Test that old tool results are compacted."""
        compaction = ToolResultCompaction()
        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "tool", "tool_call_id": "call_old", "content": "Old tool result"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},  # Last assistant
                {"role": "tool", "tool_call_id": "call_new", "content": "Recent tool result"},
            ],
        )
        result = compaction.compact(messages)

        # Old tool result should be compacted
        old_tool_blocks = [b for b in result[2].content if isinstance(b, ToolResultBlock)]
        assert old_tool_blocks and old_tool_blocks[0].content == "Tool call result has been compacted"
        # Recent tool result should be preserved
        new_tool_blocks = [b for b in result[5].content if isinstance(b, ToolResultBlock)]
        assert new_tool_blocks and new_tool_blocks[0].content == "Recent tool result"


class TestContextCompactionMiddleware:
    """Tests for ContextCompactionMiddleware."""

    def test_initialization_default(self, mock_token_counter):
        """Test initialization with default parameters."""
        middleware = ContextCompactionMiddleware(
            token_counter=mock_token_counter,
            compaction_strategy="tool_result_compaction",  # Specify strategy explicitly
        )

        assert middleware.max_context_tokens == 128000
        assert middleware.auto_compact is True
        assert isinstance(middleware.trigger_strategy, TokenThresholdTrigger)
        assert isinstance(middleware.compaction_strategy, ToolResultCompaction)

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_initialization_custom_threshold(self, mock_openai_class, mock_openai_client, mock_token_counter, temp_compact_prompt):
        """Test initialization with custom threshold."""
        mock_openai_class.return_value = mock_openai_client
        middleware = ContextCompactionMiddleware(
            threshold=0.80,
            token_counter=mock_token_counter,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        assert middleware.trigger_strategy.threshold == 0.80

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_after_model_no_compaction_needed(
        self,
        mock_openai_class,
        mock_openai_client,
        agent_state,
        sample_messages,
        mock_token_counter,
        mock_model_response,
        temp_compact_prompt,
    ):
        """Test after_model when compaction is not needed."""
        mock_openai_class.return_value = mock_openai_client
        # Setup: low token usage
        mock_token_counter.count_tokens.return_value = 5000  # 50% of 10000

        middleware = ContextCompactionMiddleware(
            max_context_tokens=10000,
            token_counter=mock_token_counter,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=sample_messages,
            original_response="Test response",
            model_response=mock_model_response,
            parsed_response=ParsedResponse(
                original_response="Test",
                tool_calls=[],
                sub_agent_calls=[],
                batch_agent_calls=[],
            ),
        )

        result = middleware.after_model(hook_input)

        assert result.has_modifications() is False
        assert middleware._compact_count == 0

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_after_model_auto_compact_disabled(
        self,
        mock_openai_class,
        mock_openai_client,
        agent_state,
        sample_messages,
        mock_token_counter,
        mock_model_response,
        temp_compact_prompt,
    ):
        """Test after_model with auto_compact disabled."""
        mock_openai_class.return_value = mock_openai_client
        mock_token_counter.count_tokens.return_value = 9000  # High usage

        middleware = ContextCompactionMiddleware(
            auto_compact=False,
            token_counter=mock_token_counter,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=sample_messages,
            original_response="Test response",
            model_response=mock_model_response,
            parsed_response=ParsedResponse(
                original_response="Test",
                tool_calls=[],
                sub_agent_calls=[],
                batch_agent_calls=[],
            ),
        )

        result = middleware.after_model(hook_input)

        assert result.has_modifications() is False
        assert middleware._compact_count == 0

    def test_tool_result_compaction_strategy(self, mock_token_counter):
        """Test initialization with tool_result_compaction strategy."""
        middleware = ContextCompactionMiddleware(
            compaction_strategy="tool_result_compaction",
            token_counter=mock_token_counter,
        )

        assert isinstance(middleware.compaction_strategy, ToolResultCompaction)

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_string_enum_strategy(self, mock_openai_class, mock_openai_client, mock_token_counter, temp_compact_prompt):
        """Test initialization with string enum strategy."""
        mock_openai_class.return_value = mock_openai_client
        middleware = ContextCompactionMiddleware(
            compaction_strategy="sliding_window",
            keep_iterations=5,
            token_counter=mock_token_counter,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        assert isinstance(middleware.compaction_strategy, SlidingWindowCompaction)
        assert middleware.compaction_strategy.keep_iterations == 5


class TestSlidingWindowCompactionAdvanced:
    """Advanced tests for SlidingWindowCompaction."""

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_compact_with_llm_summary(self, mock_openai_class, mock_openai_client, temp_compact_prompt):
        """Test compaction with LLM-generated summary.

        With the iteration definition:
        [USER or FRAMEWORK](optional) -> [ASSISTANT] -> [TOOL results](optional)

        Messages: [system, user1, assistant1, user2, assistant2, user3, assistant3]
        - Iteration 1: [user1, assistant1]
        - Iteration 2: [user2, assistant2]
        - Iteration 3: [user3, assistant3]

        With keep_iterations=2, we keep iterations 2-3 and compress iteration 1.
        """
        mock_openai_class.return_value = mock_openai_client

        compaction = SlidingWindowCompaction(
            keep_iterations=2,  # Keep 2 iterations to ensure USER message is in kept iterations
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )
        compaction._llm_caller.call_llm = Mock(  # type: ignore[method-assign]
            return_value=ModelResponse(content="This is a comprehensive summary of the conversation.", role="assistant"),
        )

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},
                {"role": "user", "content": "Question 3"},
                {"role": "assistant", "content": "Answer 3"},
            ],
        )

        result = compaction.compact(messages)

        # Should have system + kept iterations with summary injected into first USER message
        assert len(result) >= 2
        assert result[0].role == Role.SYSTEM
        # Find the first USER message in result and check for summary
        user_msgs = [m for m in result if m.role == Role.USER]
        assert len(user_msgs) > 0
        assert "This session is being continued" in user_msgs[0].get_text_content()
        assert user_msgs[0].metadata.get("isSummary") is True

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_compact_llm_failure_returns_original(self, mock_openai_class, mock_openai_client, temp_compact_prompt):
        """Test that LLM failure returns original messages."""
        mock_openai_class.return_value = mock_openai_client

        compaction = SlidingWindowCompaction(
            keep_iterations=1,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )
        compaction._llm_caller.call_llm = Mock(side_effect=Exception("LLM Error"))  # type: ignore[method-assign]

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},
            ],
        )

        result = compaction.compact(messages)

        # Should return original messages on error
        assert result == messages

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_group_into_iterations_complex(self, mock_openai_class, mock_openai_client, temp_compact_prompt):
        """Test iteration grouping with tools.

        An iteration is bounded by ASSISTANT messages:
        [USER or FRAMEWORK](optional) -> [ASSISTANT] -> [TOOL results](optional)

        Messages: [user1, assistant1, tool1, tool2, user2, assistant2]
        - Iteration 1: [user1, assistant1, tool1, tool2]
        - Iteration 2: [user2, assistant2]
        """
        mock_openai_class.return_value = mock_openai_client

        compaction = SlidingWindowCompaction(
            keep_iterations=1,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "tool", "tool_call_id": "call_1", "content": "T1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "T2"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
        )

        iterations = compaction._group_into_iterations(messages)

        # Should group by ASSISTANT boundaries with USER moved to new iteration:
        # Iteration 1: [user1, assistant1, tool1, tool2]
        # Iteration 2: [user2, assistant2]
        assert len(iterations) == 2
        assert len(iterations[0]) == 4  # user1, assistant1, tool1, tool2
        assert len(iterations[1]) == 2  # user2, assistant2

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_group_into_iterations_with_framework(self, mock_openai_class, mock_openai_client, temp_compact_prompt):
        """Test iteration grouping with FRAMEWORK messages.

        An iteration is bounded by ASSISTANT messages:
        [USER or FRAMEWORK](optional) -> [ASSISTANT] -> [TOOL results](optional)

        FRAMEWORK messages are injected by middleware after TOOL results and before
        the next ASSISTANT response. So a realistic sequence is:
        [user1, assistant1, tool1, framework1, assistant2, user2, assistant3]

        Expected iterations:
        - Iteration 1: [user1, assistant1, tool1]
        - Iteration 2: [framework1, assistant2]
        - Iteration 3: [user2, assistant3]
        """
        mock_openai_class.return_value = mock_openai_client

        compaction = SlidingWindowCompaction(
            keep_iterations=1,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        messages = [
            Message(role=Role.USER, content=[TextBlock(text="Q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="A1")]),
            Message(role=Role.TOOL, content=[ToolResultBlock(tool_use_id="call_1", content="T1")]),
            Message(role=Role.FRAMEWORK, content=[TextBlock(text="System reminder")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="A2")]),
            Message(role=Role.USER, content=[TextBlock(text="Q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="A3")]),
        ]

        iterations = compaction._group_into_iterations(messages)

        # Iteration 1: [user1, assistant1, tool1]
        # Iteration 2: [framework1, assistant2]
        # Iteration 3: [user2, assistant3]
        assert len(iterations) == 3
        assert len(iterations[0]) == 3  # user1, assistant1, tool1
        assert len(iterations[1]) == 2  # framework1, assistant2
        assert len(iterations[2]) == 2  # user2, assistant3
        assert iterations[0][0].role == Role.USER
        assert iterations[0][1].role == Role.ASSISTANT
        assert iterations[0][2].role == Role.TOOL
        assert iterations[1][0].role == Role.FRAMEWORK
        assert iterations[1][1].role == Role.ASSISTANT
        assert iterations[2][0].role == Role.USER
        assert iterations[2][1].role == Role.ASSISTANT

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_compact_without_system_message(self, mock_openai_class, mock_openai_client, temp_compact_prompt):
        """Test compaction without system message."""
        mock_openai_class.return_value = mock_openai_client

        compaction = SlidingWindowCompaction(
            keep_system=False,
            keep_iterations=1,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
        )

        result = compaction.compact(messages)

        # Should not include system message in result
        assert all(msg.role != Role.SYSTEM for msg in result)

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_compact_with_structured_content(self, mock_openai_class, mock_openai_client, temp_compact_prompt):
        """Test compaction with structured content in messages."""
        mock_openai_class.return_value = mock_openai_client

        compaction = SlidingWindowCompaction(
            keep_iterations=1,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Q1"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "A1"},
                        {"type": "tool_use", "id": "call_1", "name": "test_tool", "input": {}},
                    ],
                },
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
        )

        result = compaction.compact(messages)

        # Should handle structured content
        assert len(result) >= 2


class TestToolResultCompactionAdvanced:
    """Advanced tests for ToolResultCompaction."""

    def test_no_assistant_message(self):
        """Test compaction when no assistant message exists."""
        compaction = ToolResultCompaction()

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Question"},
                {"role": "tool", "tool_call_id": "call_1", "content": "Tool result"},
            ],
        )

        result = compaction.compact(messages)

        # Should return original messages
        assert result == messages

    def test_multiple_tool_results_after_last_assistant(self):
        """Test that multiple tool results after last assistant are preserved."""
        compaction = ToolResultCompaction()

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "tool", "tool_call_id": "call_old", "content": "Old result"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "tool", "tool_call_id": "call_1", "content": "Recent result 1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "Recent result 2"},
            ],
        )

        result = compaction.compact(messages)

        # Old tool result should be compacted
        blocks = [b for b in result[2].content if isinstance(b, ToolResultBlock)]
        assert blocks and blocks[0].content == "Tool call result has been compacted"
        # Recent tool results should be preserved
        blocks_1 = [b for b in result[5].content if isinstance(b, ToolResultBlock)]
        blocks_2 = [b for b in result[6].content if isinstance(b, ToolResultBlock)]
        assert blocks_1 and blocks_1[0].content == "Recent result 1"
        assert blocks_2 and blocks_2[0].content == "Recent result 2"

    def test_compact_without_system(self):
        """Test compaction without system message preservation."""
        compaction = ToolResultCompaction(keep_system=False)

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
            ],
        )

        result = compaction.compact(messages)

        # System message should not be at start if keep_system=False
        # (it will still be in result as a regular message)
        assert len(result) == 4

    def test_only_user_and_assistant_messages(self):
        """Test compaction with no tool messages."""
        compaction = ToolResultCompaction()

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        )

        result = compaction.compact(messages)

        # Should preserve all messages
        assert result == messages


class TestContextCompactionMiddlewareAdvanced:
    """Advanced tests for ContextCompactionMiddleware."""

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_keep_iterations_validation_in_middleware(self, mock_openai_class, mock_openai_client, mock_token_counter, temp_compact_prompt):
        """Test that keep_iterations validation works in middleware."""
        mock_openai_class.return_value = mock_openai_client

        with pytest.raises(ValueError, match="keep_iterations must be >= 1"):
            ContextCompactionMiddleware(
                compaction_strategy="sliding_window",
                keep_iterations=0,
                token_counter=mock_token_counter,
                summary_model="gpt-4o-mini",
                summary_base_url="https://api.openai.com/v1",
                summary_api_key="test-key",
                compact_prompt_path=temp_compact_prompt,
            )

    def test_invalid_strategy_name(self, mock_token_counter):
        """Test that invalid strategy name raises error."""
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError, match="Input should be 'sliding_window' or 'tool_result_compaction'"):
            ContextCompactionMiddleware(
                compaction_strategy="invalid_strategy",
                token_counter=mock_token_counter,
            )

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_after_model_with_tool_calls_openai_format(
        self, mock_openai_class, mock_openai_client, agent_state, mock_token_counter, temp_compact_prompt
    ):
        """Test after_model with tool_calls in OpenAI format."""
        mock_openai_class.return_value = mock_openai_client
        # Only one call to count_tokens: for the compacted messages
        mock_token_counter.count_tokens.return_value = 4000

        # Create a custom model response with high token usage to trigger compaction
        high_usage_model_response = ModelResponse(
            content="Test response",
            role="assistant",
            usage={
                "prompt_tokens": 7500,
                "completion_tokens": 500,
                "total_tokens": 8000,
            },
        )

        middleware = ContextCompactionMiddleware(
            max_context_tokens=10000,
            threshold=0.75,
            token_counter=mock_token_counter,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        messages = [
            Message(role=Role.USER, content=[TextBlock(text="Question 1")]),
            Message(
                role=Role.ASSISTANT,
                content=[
                    TextBlock(text="Answer 1"),
                    ToolUseBlock(id="call_1", name="test_tool", input={}),
                ],
            ),
            Message(role=Role.USER, content=[TextBlock(text="Question 2")]),
            Message(
                role=Role.ASSISTANT,
                content=[
                    TextBlock(text="Answer 2"),
                    ToolUseBlock(id="call_2", name="test_tool", input={}),
                ],
            ),
        ]

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=messages,
            original_response="Test",
            model_response=high_usage_model_response,
            parsed_response=ParsedResponse(
                original_response="Test",
                tool_calls=[],
                sub_agent_calls=[],
                batch_agent_calls=[],
            ),
        )

        result = middleware.after_model(hook_input)

        # Should trigger compaction
        assert result.has_modifications() is True

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_after_model_with_tool_use_content_format(
        self, mock_openai_class, mock_openai_client, agent_state, mock_token_counter, temp_compact_prompt
    ):
        """Test after_model with tool_use in content list format."""
        mock_openai_class.return_value = mock_openai_client
        # Only one call to count_tokens: for the compacted messages
        mock_token_counter.count_tokens.return_value = 4000

        # Create a custom model response with high token usage to trigger compaction
        high_usage_model_response = ModelResponse(
            content="Test response",
            role="assistant",
            usage={
                "prompt_tokens": 7500,
                "completion_tokens": 500,
                "total_tokens": 8000,
            },
        )

        middleware = ContextCompactionMiddleware(
            max_context_tokens=10000,
            threshold=0.75,
            token_counter=mock_token_counter,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        messages = [
            Message(role=Role.USER, content=[TextBlock(text="Question 1")]),
            Message(
                role=Role.ASSISTANT,
                content=[
                    TextBlock(text="Answer 1"),
                    ToolUseBlock(id="call_1", name="test_tool", input={}),
                ],
            ),
            Message(role=Role.USER, content=[TextBlock(text="Question 2")]),
            Message(
                role=Role.ASSISTANT,
                content=[
                    TextBlock(text="Answer 2"),
                    ToolUseBlock(id="call_2", name="test_tool", input={}),
                ],
            ),
        ]

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=messages,
            original_response="Test",
            model_response=high_usage_model_response,
            parsed_response=ParsedResponse(
                original_response="Test",
                tool_calls=[],
                sub_agent_calls=[],
                batch_agent_calls=[],
            ),
        )

        result = middleware.after_model(hook_input)

        # Should trigger compaction
        assert result.has_modifications() is True

    @patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI")
    def test_after_model_no_tool_calls_skips_compaction(
        self, mock_openai_class, mock_openai_client, agent_state, mock_token_counter, temp_compact_prompt
    ):
        """Test that compaction is skipped when last assistant has no tool calls."""
        mock_openai_class.return_value = mock_openai_client
        mock_token_counter.count_tokens.return_value = 9000  # High usage

        # Create a custom model response with high token usage
        high_usage_model_response = ModelResponse(
            content="Test response",
            role="assistant",
            usage={
                "prompt_tokens": 8500,
                "completion_tokens": 500,
                "total_tokens": 9000,
            },
        )

        middleware = ContextCompactionMiddleware(
            max_context_tokens=10000,
            threshold=0.75,
            token_counter=mock_token_counter,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            compact_prompt_path=temp_compact_prompt,
        )

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer without tool calls"},
            ],
        )

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=3,
            messages=messages,
            original_response="Test",
            model_response=high_usage_model_response,
            parsed_response=ParsedResponse(
                original_response="Test",
                tool_calls=[],
                sub_agent_calls=[],
                batch_agent_calls=[],
            ),
        )

        result = middleware.after_model(hook_input)

        # Should skip compaction
        assert result.has_modifications() is True
