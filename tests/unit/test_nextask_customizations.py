"""Tests for NexTask customizations.

Covers:
- ReasoningBlock redacted_data field
- Anthropic thinking passthrough (model_response.py)
- Anthropic adapter ReasoningBlock conversion + tool result batching
- Legacy adapter reasoning signature/redacted_data roundtrip
- Context compaction middleware (token fallback, pre-compaction check, full trace)
- LLM caller encrypted_content summary fix
- Web tool 64KB content truncation
- Agent stop(_from_del) and cleanup_manager logging protection
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import Mock, patch

import pytest

from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.hooks import AfterModelHookInput
from nexau.archs.main_sub.execution.middleware.context_compaction.middleware import (
    ContextCompactionMiddleware,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.adapters.anthropic_messages import AnthropicMessagesAdapter
from nexau.core.adapters.legacy import (
    messages_from_legacy_openai_chat,
    messages_to_legacy_openai_chat,
)
from nexau.core.messages import (
    Message,
    ReasoningBlock,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_executor():
    executor = Mock()
    executor.add_tool = Mock()
    executor.cleanup = Mock()
    return executor


@pytest.fixture
def agent_state(mock_executor):
    context = AgentContext()
    global_storage = GlobalStorage()
    return AgentState(
        agent_name="test_agent",
        agent_id="test_id",
        run_id="run_1",
        root_run_id="run_1",
        context=context,
        global_storage=global_storage,
        executor=mock_executor,
    )


# ===========================================================================
# 1. ReasoningBlock redacted_data field (messages.py)
# ===========================================================================


class TestReasoningBlockRedactedData:
    """Tests for the redacted_data field on ReasoningBlock."""

    def test_reasoning_block_default_redacted_data_is_none(self):
        block = ReasoningBlock(text="thinking...")
        assert block.redacted_data is None
        assert block.signature is None

    def test_reasoning_block_with_redacted_data(self):
        block = ReasoningBlock(text="", signature="sig123", redacted_data="encrypted_blob")
        assert block.text == ""
        assert block.signature == "sig123"
        assert block.redacted_data == "encrypted_blob"

    def test_reasoning_block_serialization_roundtrip(self):
        block = ReasoningBlock(text="hello", signature="s", redacted_data="rd")
        d = block.model_dump()
        assert d["redacted_data"] == "rd"
        assert d["signature"] == "s"
        restored = ReasoningBlock.model_validate(d)
        assert restored.redacted_data == "rd"
        assert restored.signature == "s"


# ===========================================================================
# 2. Anthropic thinking passthrough (model_response.py)
# ===========================================================================


class TestAnthropicThinkingPassthrough:
    """Tests for from_anthropic_message thinking/redacted_thinking extraction."""

    def test_extract_thinking_block(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think...", "signature": "sig_abc"},
                {"type": "text", "text": "Here is my answer."},
            ],
        }
        resp = ModelResponse.from_anthropic_message(message)
        assert resp.reasoning_content == "Let me think..."
        assert resp.reasoning_signature == "sig_abc"
        assert resp.reasoning_redacted_data is None
        assert resp.content == "Here is my answer."

    def test_extract_redacted_thinking_block(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "visible thought", "signature": "sig1"},
                {"type": "redacted_thinking", "data": "encrypted_data_here"},
                {"type": "text", "text": "Answer"},
            ],
        }
        resp = ModelResponse.from_anthropic_message(message)
        assert resp.reasoning_content == "visible thought"
        assert resp.reasoning_signature == "sig1"
        assert resp.reasoning_redacted_data == "encrypted_data_here"

    def test_no_thinking_blocks(self):
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Just text"}],
        }
        resp = ModelResponse.from_anthropic_message(message)
        assert resp.reasoning_content is None
        assert resp.reasoning_signature is None
        assert resp.reasoning_redacted_data is None

    def test_to_message_dict_includes_reasoning_fields(self):
        resp = ModelResponse(
            content="answer",
            reasoning_content="think",
            reasoning_signature="sig",
            reasoning_redacted_data="rd",
        )
        d = resp.to_message_dict()
        assert d["reasoning_content"] == "think"
        assert d["reasoning_signature"] == "sig"
        assert d["reasoning_redacted_data"] == "rd"

    def test_to_ump_message_creates_reasoning_block(self):
        resp = ModelResponse(
            content="answer",
            reasoning_content="thought",
            reasoning_signature="s1",
            reasoning_redacted_data="r1",
        )
        msg = resp.to_ump_message()
        reasoning_blocks = [b for b in msg.content if isinstance(b, ReasoningBlock)]
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0].text == "thought"
        assert reasoning_blocks[0].signature == "s1"
        assert reasoning_blocks[0].redacted_data == "r1"


class TestNormalizeUsageAnthropicCache:
    """Tests for _normalize_usage handling of Anthropic cache tokens."""

    def test_anthropic_cache_tokens_included(self):
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 300,
        }
        resp = ModelResponse.from_anthropic_message(message, usage=usage)
        assert resp.usage is not None
        # input_tokens should be 100 + 200 + 300 = 600
        assert resp.usage["input_tokens"] == 600
        assert resp.usage["input_tokens_uncached"] == 100
        assert resp.usage["completion_tokens"] == 50

    def test_usage_without_cache(self):
        message = {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        resp = ModelResponse.from_anthropic_message(message, usage=usage)
        assert resp.usage is not None
        assert resp.usage["total_tokens"] == 150


# ===========================================================================
# 3. Anthropic adapter (anthropic_messages.py)
# ===========================================================================


class TestAnthropicAdapterReasoningBlock:
    """Tests for ReasoningBlock conversion in AnthropicMessagesAdapter."""

    def test_reasoning_block_to_thinking(self):
        adapter = AnthropicMessagesAdapter()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ReasoningBlock(text="my thinking", signature="sig1"),
                    TextBlock(text="my answer"),
                ],
            ),
        ]
        system_blocks, convo = adapter.to_vendor_format(messages)
        assert len(convo) == 1
        blocks = convo[0]["content"]
        thinking_blocks = [b for b in blocks if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "my thinking"
        assert thinking_blocks[0]["signature"] == "sig1"

    def test_redacted_thinking_block(self):
        adapter = AnthropicMessagesAdapter()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ReasoningBlock(text="", redacted_data="encrypted_blob"),
                    TextBlock(text="answer"),
                ],
            ),
        ]
        _, convo = adapter.to_vendor_format(messages)
        blocks = convo[0]["content"]
        redacted = [b for b in blocks if b.get("type") == "redacted_thinking"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "encrypted_blob"

    def test_tool_result_batching(self):
        """Consecutive TOOL messages should be merged into a single user message."""
        adapter = AnthropicMessagesAdapter()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ToolUseBlock(id="call_1", tool_use_id="t1", name="foo", input={"a": 1}),
                    ToolUseBlock(id="call_2", tool_use_id="t2", name="bar", input={"b": 2}),
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="t1", content="result1")],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="t2", content="result2")],
            ),
            Message(role=Role.USER, content=[TextBlock(text="next question")]),
        ]
        _, convo = adapter.to_vendor_format(messages)
        # Should be: assistant, user(merged tool results), user(question)
        assert len(convo) == 3
        # The merged tool results message
        merged = convo[1]
        assert merged["role"] == "user"
        assert len(merged["content"]) == 2
        assert merged["content"][0]["type"] == "tool_result"
        assert merged["content"][1]["type"] == "tool_result"


# ===========================================================================
# 4. Legacy adapter roundtrip (legacy.py)
# ===========================================================================


class TestLegacyReasoningSignatureRoundtrip:
    """Tests for reasoning_signature/redacted_data roundtrip through legacy format."""

    def test_from_legacy_with_reasoning_signature(self):
        legacy = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "thinking...",
                "reasoning_signature": "sig123",
                "reasoning_redacted_data": "rd_data",
            }
        ]
        msgs = messages_from_legacy_openai_chat(legacy)
        assert len(msgs) == 1
        reasoning_blocks = [b for b in msgs[0].content if isinstance(b, ReasoningBlock)]
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0].text == "thinking..."
        assert reasoning_blocks[0].signature == "sig123"
        assert reasoning_blocks[0].redacted_data == "rd_data"

    def test_reasoning_block_inserted_at_beginning(self):
        """Anthropic requires thinking blocks before text/tool blocks."""
        legacy = [
            {
                "role": "assistant",
                "content": "answer text",
                "reasoning_content": "thought",
                "reasoning_signature": "s",
            }
        ]
        msgs = messages_from_legacy_openai_chat(legacy)
        # ReasoningBlock should be first
        assert isinstance(msgs[0].content[0], ReasoningBlock)
        assert isinstance(msgs[0].content[1], TextBlock)

    def test_to_legacy_preserves_signature(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ReasoningBlock(text="think", signature="sig", redacted_data="rd"),
                TextBlock(text="answer"),
            ],
        )
        legacy = messages_to_legacy_openai_chat([msg])
        assert legacy[0]["reasoning_content"] == "think"
        assert legacy[0]["reasoning_signature"] == "sig"
        assert legacy[0]["reasoning_redacted_data"] == "rd"

    def test_full_roundtrip(self):
        original = [
            {
                "role": "assistant",
                "content": "result",
                "reasoning_content": "deep thought",
                "reasoning_signature": "abc",
                "reasoning_redacted_data": "xyz",
            }
        ]
        msgs = messages_from_legacy_openai_chat(original)
        roundtripped = messages_to_legacy_openai_chat(msgs)
        assert roundtripped[0]["reasoning_content"] == "deep thought"
        assert roundtripped[0]["reasoning_signature"] == "abc"
        assert roundtripped[0]["reasoning_redacted_data"] == "xyz"

    def test_structured_content_reasoning_with_signature(self):
        """Test parsing reasoning from structured content list format."""
        legacy = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "think", "signature": "s1", "redacted_data": "r1"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
        msgs = messages_from_legacy_openai_chat(legacy)
        reasoning = [b for b in msgs[0].content if isinstance(b, ReasoningBlock)]
        assert len(reasoning) == 1
        assert reasoning[0].signature == "s1"
        assert reasoning[0].redacted_data == "r1"


# ===========================================================================
# 5. Context compaction middleware
# ===========================================================================


class TestContextCompactionTokenFallback:
    """Tests for token usage fallback calculation in middleware."""

    def _make_middleware(self, **kwargs):
        defaults = {
            "max_context_tokens": 100000,
            "auto_compact": True,
            "compaction_strategy": "tool_result_compaction",
        }
        defaults.update(kwargs)
        return ContextCompactionMiddleware(**defaults)

    def test_get_current_tokens_from_total(self):
        mw = self._make_middleware()
        mock_response = Mock()
        mock_response.usage = {"total_tokens": 5000}
        hook_input = Mock(spec=AfterModelHookInput)
        hook_input.model_response = mock_response
        result = mw._get_current_tokens(hook_input)
        assert result == 5000

    def test_get_current_tokens_fallback_from_components(self):
        """When total_tokens is missing, should calculate from input + output."""
        mw = self._make_middleware()
        mock_response = Mock()
        mock_response.usage = {"input_tokens": 3000, "output_tokens": 1000}
        hook_input = Mock(spec=AfterModelHookInput)
        hook_input.model_response = mock_response
        result = mw._get_current_tokens(hook_input)
        assert result == 4000

    def test_get_current_tokens_with_anthropic_cache(self):
        """Should include cache tokens when calculating from components."""
        mw = self._make_middleware()
        mock_response = Mock()
        mock_response.usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 300,
        }
        hook_input = Mock(spec=AfterModelHookInput)
        hook_input.model_response = mock_response
        result = mw._get_current_tokens(hook_input)
        # 100 + 200 + 300 + 50 = 650
        assert result == 650

    def test_get_current_tokens_none_when_no_usage(self):
        mw = self._make_middleware()
        mock_response = Mock()
        mock_response.usage = None
        hook_input = Mock(spec=AfterModelHookInput)
        hook_input.model_response = mock_response
        result = mw._get_current_tokens(hook_input)
        assert result is None


class TestContextCompactionPreCompactionCheck:
    """Tests for the pre-compaction check (skip if no tool calls)."""

    def _make_middleware(self, **kwargs):
        defaults = {
            "max_context_tokens": 1000,
            "auto_compact": True,
            "compaction_strategy": "tool_result_compaction",
        }
        defaults.update(kwargs)
        return ContextCompactionMiddleware(**defaults)

    def test_skip_compaction_when_last_assistant_has_no_tool_calls(self, agent_state):
        """Should skip compaction if last assistant message has no tool calls."""
        mw = self._make_middleware()
        messages = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
            Message(role=Role.USER, content=[TextBlock(text="hello")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="just text, no tools")]),
        ]
        mock_response = Mock()
        mock_response.usage = {"total_tokens": 950}  # 95% usage, should trigger
        hook_input = AfterModelHookInput(
            messages=messages,
            model_response=mock_response,
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=1,
            original_response="just text, no tools",
        )
        result = mw.after_model(hook_input)
        assert result.messages is None  # no_changes

    def test_proceed_compaction_when_last_assistant_has_tool_calls(self, agent_state):
        """Should proceed with compaction check if last assistant has tool calls."""
        mw = self._make_middleware(max_context_tokens=1000)
        messages = [
            Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
            Message(role=Role.USER, content=[TextBlock(text="hello")]),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id="call_1", tool_use_id="t1", name="bash", input={"cmd": "ls"})],
            ),
        ]
        mock_response = Mock()
        mock_response.usage = {"total_tokens": 500}  # 50%, below threshold
        hook_input = AfterModelHookInput(
            messages=messages,
            model_response=mock_response,
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=1,
            original_response="",
        )
        result = mw.after_model(hook_input)
        # Should not compact (below threshold) but should have gone through the check
        assert result.messages is None


class TestContextCompactionFullTrace:
    """Tests for full trace recording in middleware."""

    def _make_middleware(self, **kwargs):
        defaults = {
            "max_context_tokens": 100000,
            "auto_compact": False,
            "compaction_strategy": "tool_result_compaction",
        }
        defaults.update(kwargs)
        return ContextCompactionMiddleware(**defaults)

    def _make_hook_input(self, messages, agent_state):
        mock_response = Mock()
        mock_response.usage = {"total_tokens": 100}
        return AfterModelHookInput(
            messages=messages,
            model_response=mock_response,
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=1,
            original_response="",
        )

    def test_full_trace_records_messages(self, agent_state):
        mw = self._make_middleware()
        messages = [
            Message(role=Role.USER, content=[TextBlock(text="hello")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="hi")]),
        ]
        hook_input = self._make_hook_input(messages, agent_state)
        mw.after_model(hook_input)

        full = agent_state.get_context_value("__nexau_full_trace_messages__", [])
        assert isinstance(full, list)
        assert len(full) == 2

    def test_full_trace_deduplicates(self, agent_state):
        mw = self._make_middleware()
        msg1 = Message(role=Role.USER, content=[TextBlock(text="hello")])
        msg2 = Message(role=Role.ASSISTANT, content=[TextBlock(text="hi")])
        messages = [msg1, msg2]
        hook_input = self._make_hook_input(messages, agent_state)
        # Call twice with same messages
        mw.after_model(hook_input)
        mw.after_model(hook_input)

        full = agent_state.get_context_value("__nexau_full_trace_messages__", [])
        assert len(full) == 2  # Not 4

    def test_full_trace_filters_compaction_artifacts(self, agent_state):
        mw = self._make_middleware()
        normal_msg = Message(role=Role.USER, content=[TextBlock(text="hello")])
        compacted_msg = Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="t1", content="Tool call result has been compacted")],
        )
        hook_input = self._make_hook_input([normal_msg, compacted_msg], agent_state)
        mw.after_model(hook_input)

        full = agent_state.get_context_value("__nexau_full_trace_messages__", [])
        # Compacted message should be filtered out
        assert len(full) == 1


# ===========================================================================
# 6. LLM caller encrypted_content summary fix
# ===========================================================================


class TestResponsesApiIncludeEncryptedContent:
    """Tests for include=reasoning.encrypted_content in Responses API requests."""

    @staticmethod
    def _make_mock_response() -> dict[str, Any]:
        """Create a mock Responses API response as a plain dict (avoids Mock serialization issues)."""
        return {
            "id": "resp_1",
            "model": "o4-mini",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    def test_include_added_to_empty_payload(self):
        """Should auto-add include with reasoning.encrypted_content."""
        from nexau.archs.main_sub.execution.llm_caller import (
            call_llm_with_openai_responses,
        )

        captured: dict[str, Any] = {}

        mock_client = Mock()

        def capture_create(**kw: Any) -> dict[str, Any]:
            captured.update(kw)
            return self._make_mock_response()

        mock_client.responses.create = capture_create

        mock_config = Mock()
        mock_config.stream = False

        call_llm_with_openai_responses(
            mock_client,
            {"model": "o4-mini", "input": "hello"},
            llm_config=mock_config,
        )

        assert "include" in captured
        assert "reasoning.encrypted_content" in captured["include"]

    def test_include_preserves_existing_entries(self):
        """Should not duplicate if already present."""
        from nexau.archs.main_sub.execution.llm_caller import (
            call_llm_with_openai_responses,
        )

        captured: dict[str, Any] = {}

        mock_client = Mock()

        def capture_create(**kw: Any) -> dict[str, Any]:
            captured.update(kw)
            return self._make_mock_response()

        mock_client.responses.create = capture_create

        mock_config = Mock()
        mock_config.stream = False

        call_llm_with_openai_responses(
            mock_client,
            {
                "model": "o4-mini",
                "input": "hello",
                "include": ["reasoning.encrypted_content"],
            },
            llm_config=mock_config,
        )

        assert captured["include"].count("reasoning.encrypted_content") == 1


class TestSanitizeResponseItemsEncryptedContent:
    """Tests for _sanitize_response_items_for_input encrypted_content handling."""

    def test_encrypted_content_preserves_empty_summary(self):
        from nexau.archs.main_sub.execution.llm_caller import _sanitize_response_items_for_input

        items = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "base64blob==",
                "summary": [],
            }
        ]
        result = _sanitize_response_items_for_input(items)
        reasoning = [i for i in result if i.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["summary"] == []  # Preserved as empty list
        assert "id" not in reasoning[0]  # id should be removed

    def test_non_encrypted_reasoning_gets_summary_filled(self):
        from nexau.archs.main_sub.execution.llm_caller import _sanitize_response_items_for_input

        items = [
            {
                "type": "reasoning",
                "id": "rs_2",
                "summary": [],
            }
        ]
        result = _sanitize_response_items_for_input(items)
        reasoning = [i for i in result if i.get("type") == "reasoning"]
        assert len(reasoning) == 1
        # Should have been filled by _ensure_reasoning_summary
        assert len(reasoning[0]["summary"]) > 0

    def test_encrypted_with_nonempty_summary_preserved(self):
        from nexau.archs.main_sub.execution.llm_caller import _sanitize_response_items_for_input

        items = [
            {
                "type": "reasoning",
                "id": "rs_3",
                "encrypted_content": "blob",
                "summary": [{"type": "summary_text", "text": "existing"}],
            }
        ]
        result = _sanitize_response_items_for_input(items)
        reasoning = [i for i in result if i.get("type") == "reasoning"]
        assert reasoning[0]["summary"] == [{"type": "summary_text", "text": "existing"}]


# ===========================================================================
# 7. Web tool 64KB truncation
# ===========================================================================


class TestWebReadTruncation:
    """Tests for web_read 64KB content truncation."""

    def test_max_web_content_length_constant(self):
        from nexau.archs.tool.builtin.web_tool import MAX_WEB_CONTENT_LENGTH

        assert MAX_WEB_CONTENT_LENGTH == 64 * 1024

    def test_html_parser_truncation(self):
        """Test that HTML parser results are truncated at 64KB."""
        from nexau.archs.tool.builtin.web_tool import MAX_WEB_CONTENT_LENGTH, web_read

        large_content = "A" * (MAX_WEB_CONTENT_LENGTH + 1000)

        with patch("nexau.archs.tool.builtin.web_tool._html_parser") as mock_parser:
            mock_parser.base_url = "http://test"
            mock_parser.api_key = "key"
            mock_parser.secret = "secret"
            mock_parser.parse.return_value = (True, large_content)

            result = web_read("http://example.com")

        assert result["status"] == "success"
        assert result["content_truncated"] is True
        assert len(result["content"].encode("utf-8")) <= MAX_WEB_CONTENT_LENGTH + 10  # +... suffix

    def test_html_parser_no_truncation_for_small_content(self):
        from nexau.archs.tool.builtin.web_tool import web_read

        small_content = "Hello World"

        with patch("nexau.archs.tool.builtin.web_tool._html_parser") as mock_parser:
            mock_parser.base_url = "http://test"
            mock_parser.api_key = "key"
            mock_parser.secret = "secret"
            mock_parser.parse.return_value = (True, small_content)

            result = web_read("http://example.com")

        assert result["status"] == "success"
        assert result["content_truncated"] is False
        assert result["content"] == small_content


# ===========================================================================
# 8. Agent stop(_from_del) and cleanup_manager
# ===========================================================================


class TestAgentStopFromDel:
    """Tests for agent.stop(_from_del=True) skipping logging."""

    def test_stop_from_del_does_not_log(self, caplog):
        """When _from_del=True, stop() should not produce log messages."""
        mock_agent = Mock()
        mock_agent.config = Mock()
        mock_agent.config.name = "test"
        mock_agent.executor = Mock()
        mock_agent.executor.cleanup = Mock()

        # Simulate the stop logic directly
        from nexau.archs.main_sub.agent import Agent

        # Use the actual method with a patched agent
        with patch.object(Agent, "__init__", lambda self, **kwargs: None):
            agent = Agent.__new__(Agent)
            agent.config = mock_agent.config
            agent.executor = mock_agent.executor

            with caplog.at_level(logging.INFO):
                agent.stop(_from_del=True)

            # Should not have any cleanup log messages
            cleanup_logs = [r for r in caplog.records if "Cleaning up" in r.message]
            assert len(cleanup_logs) == 0
            mock_agent.executor.cleanup.assert_called_once()

    def test_stop_normal_does_log(self, caplog):
        """When _from_del=False (default), stop() should produce log messages."""
        from nexau.archs.main_sub.agent import Agent

        with patch.object(Agent, "__init__", lambda self, **kwargs: None):
            agent = Agent.__new__(Agent)
            agent.config = Mock()
            agent.config.name = "test_agent"
            agent.executor = Mock()
            agent.executor.cleanup = Mock()

            with caplog.at_level(logging.INFO):
                agent.stop()

            cleanup_logs = [r for r in caplog.records if "Cleaning up" in r.message]
            assert len(cleanup_logs) == 1


class TestCleanupManagerLoggingProtection:
    """Tests for cleanup_manager logging crash protection."""

    def test_cleanup_handles_logging_failure(self):
        """Cleanup should not crash even if logging raises ValueError."""
        from nexau.archs.main_sub.utils.cleanup_manager import CleanupManager

        manager = CleanupManager()

        # Create a mock agent that raises on stop
        mock_agent = Mock()
        mock_agent.stop.side_effect = RuntimeError("agent stop failed")
        mock_agent.name = "failing_agent"
        manager.register_agent(mock_agent)

        # Patching logger to raise ValueError (simulating interpreter shutdown)
        with patch("nexau.archs.main_sub.utils.cleanup_manager.logger") as mock_logger:
            mock_logger.info.side_effect = ValueError("I/O operation on closed file")
            mock_logger.error.side_effect = ValueError("I/O operation on closed file")
            # Should not raise
            manager._cleanup_all_agents()


# ===========================================================================
# 9. Is-compaction-artifact detection
# ===========================================================================


class TestIsCompactionArtifact:
    """Tests for ContextCompactionMiddleware._is_compaction_artifact."""

    def test_normal_message_not_artifact(self):
        msg = Message(role=Role.USER, content=[TextBlock(text="hello")])
        assert ContextCompactionMiddleware._is_compaction_artifact(msg) is False

    def test_compacted_tool_result_is_artifact(self):
        msg = Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="t1", content="Tool call result has been compacted")],
        )
        assert ContextCompactionMiddleware._is_compaction_artifact(msg) is True

    def test_normal_tool_result_not_artifact(self):
        msg = Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="t1", content="actual result")],
        )
        assert ContextCompactionMiddleware._is_compaction_artifact(msg) is False
