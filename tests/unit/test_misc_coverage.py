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

"""Additional coverage tests for smaller files.

Covers uncovered paths in:
- history_list.py: _schedule_async cross-thread, _on_task_done, has_pending_messages
- token_trace_session.py: _build_headers, _build_url, _post_json, detokenize_async, timeout property
- multiplexer.py: _put cross-thread, create_event_handler, emit, close, stream
- engine.py: LoopSafeDatabaseEngine bridge paths (via cross-thread test)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nexau.core.messages import Message, Role, TextBlock

# ---------------------------------------------------------------------------
# HistoryList — _schedule_async, _on_task_done, has_pending_messages
# ---------------------------------------------------------------------------


class TestHistoryListScheduleAsync:
    def test_has_pending_messages(self):
        from nexau.archs.main_sub.history_list import HistoryList

        hl = HistoryList()
        assert hl.has_pending_messages is False

    @pytest.mark.anyio
    async def test_schedule_async_same_loop(self):
        from nexau.archs.main_sub.history_list import HistoryList

        hl = HistoryList()
        hl._owner_loop = asyncio.get_running_loop()
        executed = asyncio.Event()

        async def coro():
            executed.set()

        hl._schedule_async(coro())
        await asyncio.sleep(0.1)
        assert executed.is_set()

    def test_schedule_async_no_loop(self):
        from nexau.archs.main_sub.history_list import HistoryList

        hl = HistoryList()
        hl._owner_loop = None
        executed = {"done": False}

        async def coro():
            executed["done"] = True

        hl._schedule_async(coro())
        assert executed["done"] is True

    def test_on_task_done_with_exception(self):
        from nexau.archs.main_sub.history_list import HistoryList

        hl = HistoryList()
        mock_task = MagicMock()
        mock_task.exception.return_value = RuntimeError("persist failed")
        # Should not raise, just log
        hl._on_task_done(mock_task)

    def test_on_task_done_cancelled(self):
        from nexau.archs.main_sub.history_list import HistoryList

        hl = HistoryList()
        mock_task = MagicMock()
        mock_task.exception.side_effect = asyncio.CancelledError()
        # Should not raise
        hl._on_task_done(mock_task)


# ---------------------------------------------------------------------------
# TokenTraceSession — _build_headers, _build_url, timeout, _post_json
# ---------------------------------------------------------------------------


class TestTokenTraceSessionHelpers:
    def test_timeout_default(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="test-key",
            api_type="generate_with_token",
        )
        session = TokenTraceSession(llm_config=config)
        assert session.timeout == 60.0

    def test_timeout_from_config(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="test-key",
            api_type="generate_with_token",
            timeout=30,
        )
        session = TokenTraceSession(llm_config=config)
        assert session.timeout == 30.0

    def test_build_headers_with_api_key(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="my-key",
            api_type="generate_with_token",
        )
        session = TokenTraceSession(llm_config=config)
        headers = session._build_headers()
        assert headers["Authorization"] == "Bearer my-key"
        assert headers["Content-Type"] == "application/json"

    def test_build_url_default(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="key",
            api_type="generate_with_token",
        )
        session = TokenTraceSession(llm_config=config)
        url = session._build_url("detokenize_path", "/detokenize")
        assert url == "http://localhost:8000/detokenize"

    def test_build_url_custom_path(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="key",
            api_type="generate_with_token",
        )
        config.set_param("detokenize_path", "/custom/detokenize")
        session = TokenTraceSession(llm_config=config)
        url = session._build_url("detokenize_path", "/detokenize")
        assert url == "http://localhost:8000/custom/detokenize"

    def test_build_url_full_url(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="key",
            api_type="generate_with_token",
        )
        config.set_param("detokenize_path", "https://other-server.com/detokenize")
        session = TokenTraceSession(llm_config=config)
        url = session._build_url("detokenize_path", "/detokenize")
        assert url == "https://other-server.com/detokenize"

    def test_build_url_no_base_url_raises(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="key",
            api_type="generate_with_token",
        )
        session = TokenTraceSession(llm_config=config)
        # Override base_url after construction
        session.llm_config.base_url = ""
        with pytest.raises(ValueError, match="base_url is required"):
            session._build_url("detokenize_path", "/detokenize")

    def test_model_property_raises_when_empty(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="key",
            api_type="generate_with_token",
        )
        session = TokenTraceSession(llm_config=config)
        # Override model after construction
        session.llm_config.model = ""
        with pytest.raises(ValueError, match="model is required"):
            _ = session.model

    def test_post_json(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="key",
            api_type="generate_with_token",
        )
        session = TokenTraceSession(llm_config=config)
        mock_response = Mock()
        mock_response.json.return_value = {"text": "hello"}
        mock_response.raise_for_status = Mock()
        with patch("nexau.archs.main_sub.token_trace_session.requests.post", return_value=mock_response):
            result = session._post_json("http://localhost:8000/detokenize", {"token_ids": [1, 2]})
        assert result["text"] == "hello"

    @pytest.mark.anyio
    async def test_detokenize_async_empty(self):
        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.token_trace_session import TokenTraceSession

        config = LLMConfig(
            model="test-model",
            base_url="http://localhost:8000",
            api_key="key",
            api_type="generate_with_token",
        )
        session = TokenTraceSession(llm_config=config)
        result = await session.detokenize_async([])
        assert result == ""


# ---------------------------------------------------------------------------
# TeamSSEMultiplexer — _put, create_event_handler, emit, close, stream
# ---------------------------------------------------------------------------


class TestMultiplexerCoverage:
    @pytest.mark.anyio
    async def test_create_event_handler_and_stream(self):
        from nexau.archs.llm.llm_aggregators.events import TextMessageStartEvent
        from nexau.archs.main_sub.team.sse.multiplexer import TeamSSEMultiplexer

        mux = TeamSSEMultiplexer(team_id="team1")
        handler = mux.create_event_handler("agent1", "coder")
        event = TextMessageStartEvent(message_id="msg1", role="assistant", runId="run1")
        handler(event)

        # Close to end stream
        mux.close()

        envelopes = []
        async for envelope in mux.stream():
            envelopes.append(envelope)
        assert len(envelopes) == 1
        assert envelopes[0].agent_id == "agent1"

    @pytest.mark.anyio
    async def test_emit_with_on_envelope(self):
        from nexau.archs.llm.llm_aggregators.events import TextMessageStartEvent
        from nexau.archs.main_sub.team.sse.multiplexer import TeamSSEMultiplexer

        captured = []
        mux = TeamSSEMultiplexer(team_id="team1", on_envelope=captured.append)
        event = TextMessageStartEvent(message_id="msg1", role="assistant", runId="run1")
        mux.emit("agent2", event, role_name="reviewer")
        mux.close()

        assert len(captured) == 1
        assert captured[0].agent_id == "agent2"

    def test_put_no_loop(self):
        from nexau.archs.main_sub.team.sse.multiplexer import TeamSSEMultiplexer

        mux = TeamSSEMultiplexer.__new__(TeamSSEMultiplexer)
        mux._team_id = "team1"
        mux._queue = asyncio.Queue()
        mux._on_envelope = None
        mux._loop = None
        # Should use direct put_nowait
        mux._put(None)
        assert mux._queue.qsize() == 1

    def test_put_closed_loop(self):
        from nexau.archs.main_sub.team.sse.multiplexer import TeamSSEMultiplexer

        loop = asyncio.new_event_loop()
        loop.close()
        mux = TeamSSEMultiplexer.__new__(TeamSSEMultiplexer)
        mux._team_id = "team1"
        mux._queue = asyncio.Queue()
        mux._on_envelope = None
        mux._loop = loop
        # Should fall back to direct put_nowait
        mux._put(None)
        assert mux._queue.qsize() == 1


# ---------------------------------------------------------------------------
# SlidingWindowCompaction — additional async paths
# ---------------------------------------------------------------------------


class TestSlidingWindowAsyncPaths:
    @pytest.mark.anyio
    async def test_async_max_consecutive_fallbacks_raises(self, tmp_path):
        from nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window import (
            SlidingWindowCompaction,
        )

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Summarize.")
        counter = Mock()
        counter.count_tokens.return_value = 100

        sw = SlidingWindowCompaction(
            compact_prompt_path=str(prompt_file),
            token_counter=counter,
            max_context_tokens=50000,
        )
        sw._consecutive_fallback_count = sw._MAX_CONSECUTIVE_FALLBACKS
        with pytest.raises(RuntimeError, match="persistently unavailable"):
            await sw._generate_summary_safe_async([Message(role=Role.USER, content=[TextBlock(text="Hello")])])

    @pytest.mark.anyio
    async def test_async_chunked_summary_all_fail_raises(self, tmp_path):
        from nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window import (
            SlidingWindowCompaction,
        )

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Summarize.")
        counter = Mock()
        counter.count_tokens.return_value = 100

        sw = SlidingWindowCompaction(
            compact_prompt_path=str(prompt_file),
            token_counter=counter,
            max_context_tokens=50000,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            summary_api_type="openai_chat_completion",
        )
        with patch.object(sw, "_generate_summary_async", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError, match="All async chunk summaries failed"):
                await sw._chunked_summary_async(
                    [Message(role=Role.USER, content=[TextBlock(text="Hello")])],
                    chunk_token_limit=50,
                )

    @pytest.mark.anyio
    async def test_async_chunked_summary_single_chunk(self, tmp_path):
        from nexau.archs.main_sub.execution.middleware.context_compaction.compact_stratigies.sliding_window import (
            SlidingWindowCompaction,
        )

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Summarize.")
        counter = Mock()
        counter.count_tokens.return_value = 100

        sw = SlidingWindowCompaction(
            compact_prompt_path=str(prompt_file),
            token_counter=counter,
            max_context_tokens=50000,
            summary_model="gpt-4o-mini",
            summary_base_url="https://api.openai.com/v1",
            summary_api_key="test-key",
            summary_api_type="openai_chat_completion",
        )
        with patch.object(sw, "_generate_summary_async", return_value="chunk summary"):
            result = await sw._chunked_summary_async(
                [Message(role=Role.USER, content=[TextBlock(text="Hello")])],
                chunk_token_limit=1000000,  # Large enough for single chunk
            )
        assert result == "chunk summary"


# ---------------------------------------------------------------------------
# LoopSafeDatabaseEngine — _get_bridge_loop edge cases
# ---------------------------------------------------------------------------


class TestLoopSafeDatabaseEngineEdgeCases:
    def test_get_bridge_loop_no_running_loop(self):
        from nexau.archs.session.orm.engine import LoopSafeDatabaseEngine

        inner = AsyncMock()
        wrapper = LoopSafeDatabaseEngine(inner)
        # Set a mock loop that is running, but no current running loop
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        wrapper._owner_loop = mock_loop
        # get_running_loop will raise RuntimeError since we're not in an async context
        result = wrapper._get_bridge_loop()
        # Should return None because get_running_loop raises RuntimeError
        assert result is None
