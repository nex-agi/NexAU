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

"""Coverage improvement tests for SlidingWindowCompaction.

Targets uncovered paths in:
- nexau/archs/main_sub/execution/middleware/context_compaction/compact_stratigies/sliding_window.py
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window import (
    SlidingWindowCompaction,
    _load_compact_prompt,
    with_handoff_prefix,
)
from nexau.core.messages import Message, Role, TextBlock, ToolUseBlock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compact_prompt_file(tmp_path: Path) -> str:
    """Create a temporary compact prompt file."""
    prompt_file = tmp_path / "compact_prompt.txt"
    prompt_file.write_text("Please summarize the conversation.")
    return str(prompt_file)


@pytest.fixture
def mock_token_counter():
    """Token counter returning predictable counts."""
    counter = Mock()
    counter.count_tokens.side_effect = lambda msgs: len(msgs) * 100
    return counter


@pytest.fixture
def mock_llm_config():
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_chat_completion",
    )


@pytest.fixture
def compaction(compact_prompt_file, mock_token_counter, mock_llm_config):
    """A SlidingWindowCompaction wired with mocks."""
    sw = SlidingWindowCompaction(
        keep_iterations=2,
        compact_prompt_path=compact_prompt_file,
        token_counter=mock_token_counter,
        max_context_tokens=50000,
        summary_model="gpt-4o-mini",
        summary_base_url="https://api.openai.com/v1",
        summary_api_key="test-key",
        summary_api_type="openai_chat_completion",
    )
    return sw


# ---------------------------------------------------------------------------
# with_handoff_prefix
# ---------------------------------------------------------------------------


class TestWithHandoffPrefix:
    def test_nonempty_summary(self):
        result = with_handoff_prefix("Hello world")
        assert "Hello world" in result
        assert "Another language model" in result

    def test_empty_summary(self):
        result = with_handoff_prefix("")
        assert "Another language model" in result

    def test_whitespace_only_summary(self):
        result = with_handoff_prefix("   ")
        assert "Another language model" in result


# ---------------------------------------------------------------------------
# _load_compact_prompt
# ---------------------------------------------------------------------------


class TestLoadCompactPrompt:
    def test_load_success(self, compact_prompt_file):
        content = _load_compact_prompt(compact_prompt_file)
        assert "summarize" in content.lower()

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_compact_prompt("/nonexistent/path/prompt.txt")


# ---------------------------------------------------------------------------
# SlidingWindowCompaction.__init__ validation
# ---------------------------------------------------------------------------


class TestSlidingWindowInit:
    def test_keep_iterations_and_user_rounds_conflict(self, compact_prompt_file, mock_token_counter):
        with pytest.raises(ValueError, match="Cannot set both"):
            SlidingWindowCompaction(
                keep_iterations=5,
                keep_user_rounds=3,
                compact_prompt_path=compact_prompt_file,
                token_counter=mock_token_counter,
            )

    def test_keep_iterations_below_one(self, compact_prompt_file, mock_token_counter):
        with pytest.raises(ValueError, match="keep_iterations must be >= 1"):
            SlidingWindowCompaction(
                keep_iterations=0,
                compact_prompt_path=compact_prompt_file,
                token_counter=mock_token_counter,
            )

    def test_keep_user_rounds_negative(self, compact_prompt_file, mock_token_counter):
        with pytest.raises(ValueError, match="keep_user_rounds must be >= 0"):
            SlidingWindowCompaction(
                keep_user_rounds=-1,
                compact_prompt_path=compact_prompt_file,
                token_counter=mock_token_counter,
            )

    def test_missing_compact_prompt_path(self, mock_token_counter):
        with pytest.raises(ValueError, match="compact_prompt_path is required"):
            SlidingWindowCompaction(
                compact_prompt_path=None,
                token_counter=mock_token_counter,
            )

    def test_keep_user_rounds_mode(self, compact_prompt_file, mock_token_counter):
        sw = SlidingWindowCompaction(
            keep_user_rounds=2,
            compact_prompt_path=compact_prompt_file,
            token_counter=mock_token_counter,
        )
        assert sw.keep_user_rounds == 2


# ---------------------------------------------------------------------------
# _summary_input_limit
# ---------------------------------------------------------------------------


class TestSummaryInputLimit:
    def test_raises_when_max_context_tokens_not_set(self, compact_prompt_file, mock_token_counter):
        sw = SlidingWindowCompaction(
            compact_prompt_path=compact_prompt_file,
            token_counter=mock_token_counter,
            max_context_tokens=None,
        )
        with pytest.raises(RuntimeError, match="max_context_tokens not resolved"):
            _ = sw._summary_input_limit

    def test_returns_limit_minus_reserved(self, compact_prompt_file, mock_token_counter):
        sw = SlidingWindowCompaction(
            compact_prompt_path=compact_prompt_file,
            token_counter=mock_token_counter,
            max_context_tokens=50000,
        )
        expected = 50000 - SlidingWindowCompaction._SUMMARY_RESERVED_TOKENS
        assert sw._summary_input_limit == expected


# ---------------------------------------------------------------------------
# configure_llm_runtime
# ---------------------------------------------------------------------------


class TestConfigureLlmRuntime:
    def test_inherits_max_context_tokens(self, compaction, mock_llm_config):
        compaction.max_context_tokens = None
        compaction.configure_llm_runtime(mock_llm_config, Mock(), max_context_tokens=100000)
        assert compaction.max_context_tokens == 100000

    def test_preserves_existing_max_context_tokens(self, compaction, mock_llm_config):
        compaction.max_context_tokens = 50000
        compaction.configure_llm_runtime(mock_llm_config, Mock(), max_context_tokens=200000)
        assert compaction.max_context_tokens == 50000


# ---------------------------------------------------------------------------
# _build_client
# ---------------------------------------------------------------------------


class TestBuildClient:
    def test_gemini_rest_returns_none(self, compaction):
        llm_config = LLMConfig(
            model="gemini-pro",
            base_url="https://api.example.com",
            api_key="test-key",
            api_type="gemini_rest",
        )
        assert compaction._build_client(llm_config) is None

    def test_anthropic_client(self, compaction):
        llm_config = LLMConfig(
            model="claude-3",
            base_url="https://api.anthropic.com",
            api_key="test-key",
            api_type="anthropic_chat_completion",
        )
        with patch(
            "nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.Anthropic"
        ) as mock_anthropic:
            mock_anthropic.return_value = Mock()
            client = compaction._build_client(llm_config)
            assert client is not None
            mock_anthropic.assert_called_once()

    def test_openai_client(self, compaction):
        llm_config = LLMConfig(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_type="openai_chat_completion",
        )
        with patch("nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window.OpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            client = compaction._build_client(llm_config)
            assert client is not None
            mock_openai.assert_called_once()

    def test_invalid_api_type(self, compaction):
        llm_config = LLMConfig(
            model="m",
            base_url="https://api.example.com",
            api_key="test-key",
            api_type="invalid_type",
        )
        with pytest.raises(ValueError, match="Invalid API type"):
            compaction._build_client(llm_config)


# ---------------------------------------------------------------------------
# _ensure_llm_caller
# ---------------------------------------------------------------------------


class TestEnsureLlmCaller:
    def test_returns_caller_when_set(self, compaction):
        compaction._llm_caller = Mock()
        caller = compaction._ensure_llm_caller()
        assert caller is compaction._llm_caller

    def test_raises_when_refresh_fails(self, compact_prompt_file, mock_token_counter):
        sw = SlidingWindowCompaction(
            compact_prompt_path=compact_prompt_file,
            token_counter=mock_token_counter,
        )
        # No summary config → _resolve_summary_llm_config will fail
        with pytest.raises((RuntimeError, ValueError)):
            sw._ensure_llm_caller()


# ---------------------------------------------------------------------------
# _group_into_iterations / _group_into_user_rounds
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_into_iterations_basic(self, compaction):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
        ]
        groups = compaction._group_into_iterations(msgs)
        assert len(groups) == 2

    def test_group_into_iterations_with_tool(self, compaction):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id="t1", name="foo", input={})],
            ),
            Message(role=Role.TOOL, content=[TextBlock(text="result")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
        ]
        groups = compaction._group_into_iterations(msgs)
        assert len(groups) == 2

    def test_group_into_user_rounds_basic(self, compaction):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
        ]
        rounds = compaction._group_into_user_rounds(msgs)
        assert len(rounds) == 2

    def test_group_into_user_rounds_with_tool_calls(self, compaction):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id="t1", name="foo", input={})],
            ),
            Message(role=Role.TOOL, content=[TextBlock(text="result")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="final answer")]),
        ]
        rounds = compaction._group_into_user_rounds(msgs)
        # Only 1 round because only the last assistant has no tool calls
        assert len(rounds) == 1


# ---------------------------------------------------------------------------
# compact() — skip when few iterations
# ---------------------------------------------------------------------------


class TestCompactSkip:
    def test_compact_skips_when_too_few_iterations(self, compaction):
        msgs = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
        ]
        result = compaction.compact(msgs)
        assert len(result) == len(msgs)


# ---------------------------------------------------------------------------
# compact() — with summarization
# ---------------------------------------------------------------------------


class TestCompactWithSummary:
    def test_compact_triggers_summarization(self, compaction):
        msgs = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
            Message(role=Role.USER, content=[TextBlock(text="q3")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a3")]),
            Message(role=Role.USER, content=[TextBlock(text="q4")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a4")]),
        ]
        with patch.object(compaction, "_generate_summary_safe", return_value="Summary of conversation") as mock_summary:
            result = compaction.compact(msgs)
            mock_summary.assert_called_once()

        # Result should contain system + summary + kept messages
        assert len(result) < len(msgs)

    def test_compact_with_user_rounds_mode(self, compact_prompt_file, mock_token_counter):
        sw = SlidingWindowCompaction(
            keep_user_rounds=1,
            compact_prompt_path=compact_prompt_file,
            token_counter=mock_token_counter,
            max_context_tokens=50000,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            summary_api_type="openai_chat_completion",
        )
        msgs = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
        ]
        with patch.object(sw, "_generate_summary_safe", return_value="Summary"):
            result = sw.compact(msgs)
        assert len(result) < len(msgs)


# ---------------------------------------------------------------------------
# compact_async()
# ---------------------------------------------------------------------------


class TestCompactAsync:
    @pytest.mark.anyio
    async def test_compact_async_skips_when_too_few_iterations(self, compaction):
        msgs = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
        ]
        result = await compaction.compact_async(msgs)
        assert len(result) == len(msgs)

    @pytest.mark.anyio
    async def test_compact_async_triggers_summarization(self, compaction):
        msgs = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
            Message(role=Role.USER, content=[TextBlock(text="q3")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a3")]),
            Message(role=Role.USER, content=[TextBlock(text="q4")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a4")]),
        ]
        with patch.object(compaction, "_generate_summary_safe_async", new_callable=AsyncMock, return_value="Async summary"):
            result = await compaction.compact_async(msgs)
        assert len(result) < len(msgs)


# ---------------------------------------------------------------------------
# _generate_summary_safe / _generate_summary_safe_async
# ---------------------------------------------------------------------------


class TestGenerateSummarySafe:
    def test_direct_summary_success(self, compaction):
        with patch.object(compaction, "_generate_summary", return_value="Summary OK"):
            result = compaction._generate_summary_safe([Message(role=Role.USER, content=[TextBlock(text="Hello")])])
        assert result == "Summary OK"
        assert compaction._consecutive_fallback_count == 0

    def test_direct_summary_failure_triggers_fallback(self, compaction):
        with patch.object(compaction, "_generate_summary", side_effect=RuntimeError("LLM Error")):
            result = compaction._generate_summary_safe([Message(role=Role.USER, content=[TextBlock(text="Hello")])])
        assert compaction._consecutive_fallback_count == 1
        # Result is hard truncation fallback text
        assert isinstance(result, str)

    def test_max_consecutive_fallbacks_raises(self, compaction):
        compaction._consecutive_fallback_count = SlidingWindowCompaction._MAX_CONSECUTIVE_FALLBACKS
        with pytest.raises(RuntimeError, match="persistently unavailable"):
            compaction._generate_summary_safe([Message(role=Role.USER, content=[TextBlock(text="Hello")])])

    def test_chunked_summary_on_large_input(self, compaction):
        # Make token count exceed limit
        compaction.token_counter.count_tokens.side_effect = lambda msgs: len(msgs) * 100000
        with patch.object(compaction, "_chunked_summary", return_value="Chunked summary"):
            result = compaction._generate_summary_safe([Message(role=Role.USER, content=[TextBlock(text="Hello")])])
        assert result == "Chunked summary"

    def test_chunked_summary_failure_triggers_fallback(self, compaction):
        compaction.token_counter.count_tokens.side_effect = lambda msgs: len(msgs) * 100000
        with patch.object(compaction, "_chunked_summary", side_effect=RuntimeError("All chunks failed")):
            _ = compaction._generate_summary_safe([Message(role=Role.USER, content=[TextBlock(text="Hello")])])
        assert compaction._consecutive_fallback_count == 1

    @pytest.mark.anyio
    async def test_async_direct_summary_success(self, compaction):
        with patch.object(compaction, "_generate_summary_async", new_callable=AsyncMock, return_value="Async Summary"):
            result = await compaction._generate_summary_safe_async([Message(role=Role.USER, content=[TextBlock(text="Hello")])])
        assert result == "Async Summary"
        assert compaction._consecutive_fallback_count == 0

    @pytest.mark.anyio
    async def test_async_direct_summary_failure_triggers_fallback(self, compaction):
        with patch.object(compaction, "_generate_summary_async", new_callable=AsyncMock, side_effect=RuntimeError("Async LLM Error")):
            _ = await compaction._generate_summary_safe_async([Message(role=Role.USER, content=[TextBlock(text="Hello")])])
        assert compaction._consecutive_fallback_count == 1

    @pytest.mark.anyio
    async def test_async_max_consecutive_fallbacks_raises(self, compaction):
        compaction._consecutive_fallback_count = SlidingWindowCompaction._MAX_CONSECUTIVE_FALLBACKS
        with pytest.raises(RuntimeError, match="persistently unavailable"):
            await compaction._generate_summary_safe_async([Message(role=Role.USER, content=[TextBlock(text="Hello")])])

    @pytest.mark.anyio
    async def test_async_chunked_summary_on_large_input(self, compaction):
        compaction.token_counter.count_tokens.side_effect = lambda msgs: len(msgs) * 100000
        with patch.object(compaction, "_chunked_summary_async", new_callable=AsyncMock, return_value="Async chunked"):
            result = await compaction._generate_summary_safe_async([Message(role=Role.USER, content=[TextBlock(text="Hello")])])
        assert result == "Async chunked"


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_splits_respecting_iteration_boundaries(self, compaction):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
        ]
        # Each iteration ~ 200 tokens, limit 250 → 2 chunks
        chunks = compaction._split_into_chunks(msgs, chunk_token_limit=250)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# _hard_truncation_fallback
# ---------------------------------------------------------------------------


class TestHardTruncationFallback:
    def test_keeps_system_and_recent_messages(self, compaction):
        msgs = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system prompt")]),
            Message(role=Role.USER, content=[TextBlock(text="old question")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="old answer")]),
            Message(role=Role.USER, content=[TextBlock(text="new question")]),
        ]
        result = compaction._hard_truncation_fallback(msgs)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _inject_summary
# ---------------------------------------------------------------------------


class TestInjectSummary:
    def test_merge_summary_into_first_user_message(self, compaction):
        result: list[Message] = []
        groups = [
            [
                Message(role=Role.USER, content=[TextBlock(text="original user msg")]),
                Message(role=Role.ASSISTANT, content=[TextBlock(text="response")]),
            ]
        ]
        compaction._inject_summary(result, groups, "my summary")
        assert len(result) == 2
        first_text = result[0].get_text_content()
        assert "my summary" in first_text
        assert "original user msg" in first_text

    def test_standalone_summary_when_first_is_not_user(self, compaction):
        result: list[Message] = []
        groups = [
            [
                Message(role=Role.ASSISTANT, content=[TextBlock(text="response")]),
                Message(role=Role.TOOL, content=[TextBlock(text="result")]),
            ]
        ]
        compaction._inject_summary(result, groups, "my summary")
        # Should have standalone summary + 2 kept messages
        assert len(result) == 3
        assert result[0].role == Role.USER
        assert "my summary" in result[0].get_text_content()


# ---------------------------------------------------------------------------
# _record_fallback
# ---------------------------------------------------------------------------


class TestRecordFallback:
    def test_increments_counter(self, compaction):
        assert compaction._consecutive_fallback_count == 0
        compaction._record_fallback([Message(role=Role.USER, content=[TextBlock(text="test")])])
        assert compaction._consecutive_fallback_count == 1


# ---------------------------------------------------------------------------
# _chunked_summary
# ---------------------------------------------------------------------------


class TestChunkedSummary:
    def test_single_chunk_returned_directly(self, compaction):
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hello")])]
        with patch.object(compaction, "_generate_summary", return_value="summary1"):
            result = compaction._chunked_summary(msgs, chunk_token_limit=200)
        assert result == "summary1"

    def test_all_chunks_fail_raises(self, compaction):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="q1")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a1")]),
            Message(role=Role.USER, content=[TextBlock(text="q2")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a2")]),
        ]
        with patch.object(compaction, "_generate_summary", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError, match="All chunk summaries failed"):
                compaction._chunked_summary(msgs, chunk_token_limit=150)
