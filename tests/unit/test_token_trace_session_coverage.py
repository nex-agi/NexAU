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

"""Coverage improvement tests for token_trace_session.py.

Targets uncovered paths: _coerce_text, _serialize_tool_calls, _copy_mapping,
TokenTraceSession properties, _build_headers, _build_url, detokenize,
detokenize_async, record_round, export_trace.
"""

from collections import OrderedDict
from unittest.mock import AsyncMock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.token_trace_session import (
    TokenTraceSession,
    _coerce_text,
    _copy_mapping,
    _serialize_tool_calls,
)
from nexau.core.messages import Message


@pytest.fixture
def llm_config():
    return LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )


class TestCoerceText:
    def test_none_returns_none(self):
        assert _coerce_text(None) is None

    def test_string_returned_as_is(self):
        assert _coerce_text("hello") == "hello"

    def test_non_string_converted(self):
        assert _coerce_text(42) == "42"


class TestSerializeToolCalls:
    def test_valid_tool_calls(self):
        calls = [{"name": "foo", "args": {"x": 1}}]
        result = _serialize_tool_calls(calls)
        assert result == calls

    def test_non_serializable_returns_empty(self):
        # object() is not JSON serializable
        result = _serialize_tool_calls([{"func": object()}])
        assert result == []

    def test_empty_list(self):
        assert _serialize_tool_calls([]) == []


class TestCopyMapping:
    def test_dict_input(self):
        result = _copy_mapping({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_ordered_dict_input(self):
        result = _copy_mapping(OrderedDict([("x", 10)]))
        assert result == {"x": 10}

    def test_non_mapping_returns_empty(self):
        assert _copy_mapping("not a mapping") == {}
        assert _copy_mapping(42) == {}
        assert _copy_mapping(None) == {}


class TestTokenTraceSessionProperties:
    def test_model_property(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        assert session.model == "token-model"

    def test_model_property_raises_when_missing(self):
        config = LLMConfig(
            base_url="http://test",
            api_key="test",
            api_type="generate_with_token",
        )
        # Override model to empty after construction to bypass LLMConfig validation
        object.__setattr__(config, "model", "")
        session = TokenTraceSession(llm_config=config)
        with pytest.raises(ValueError, match="model is required"):
            _ = session.model

    def test_timeout_property_default(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        assert session.timeout == 60.0

    def test_timeout_property_custom(self):
        config = LLMConfig(
            model="m",
            base_url="http://test",
            api_key="test",
            api_type="generate_with_token",
            timeout=30,
        )
        session = TokenTraceSession(llm_config=config)
        assert session.timeout == 30.0


class TestBuildHeaders:
    def test_includes_auth_header(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        headers = session._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"

    def test_no_auth_without_api_key(self):
        config = LLMConfig(
            model="m",
            base_url="http://test",
            api_key="placeholder",
            api_type="generate_with_token",
        )
        # Override api_key to empty after construction to test header logic
        object.__setattr__(config, "api_key", "")
        session = TokenTraceSession(llm_config=config)
        headers = session._build_headers()
        assert "Authorization" not in headers


class TestBuildUrl:
    def test_default_path(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        url = session._build_url("detokenize_path", "/detokenize")
        assert url == "http://token-gateway/detokenize"

    def test_custom_path(self):
        config = LLMConfig(
            model="m",
            base_url="http://test",
            api_key="test",
            api_type="generate_with_token",
        )
        config.extra_params = {"detokenize_path": "/v2/detokenize"}
        session = TokenTraceSession(llm_config=config)
        url = session._build_url("detokenize_path", "/detokenize")
        assert url == "http://test/v2/detokenize"

    def test_raises_without_base_url(self):
        config = LLMConfig(
            model="m",
            base_url="http://placeholder",
            api_key="test",
            api_type="generate_with_token",
        )
        # Override base_url to empty to test _build_url validation
        object.__setattr__(config, "base_url", "")
        session = TokenTraceSession(llm_config=config)
        with pytest.raises(ValueError, match="base_url is required"):
            session._build_url("detokenize_path", "/detokenize")


class TestRecordRound:
    def test_record_round_with_usage(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        session.record_round(
            request_tokens=[1, 2],
            response_tokens=[3, 4],
            response_text="Hello",
            tool_calls=[{"name": "foo"}],
            usage={"prompt_tokens": 2, "completion_tokens": 2},
        )
        assert len(session.round_traces) == 1
        assert session.round_traces[0]["response_text"] == "Hello"
        assert len(session.token_provider_usage) == 1

    def test_record_round_without_usage(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        session.record_round(
            request_tokens=[1],
            response_tokens=[2],
            response_text=None,
            tool_calls=[],
            usage=None,
        )
        assert len(session.round_traces) == 1
        assert session.round_traces[0]["response_text"] == ""
        assert len(session.token_provider_usage) == 0

    def test_record_round_with_non_serializable_usage(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        session.record_round(
            request_tokens=[1],
            response_tokens=[2],
            response_text="ok",
            tool_calls=[],
            usage={"obj": object()},
        )
        # Non-serializable usage should result in empty dict
        assert session.token_provider_usage == [{}]


class TestExportTrace:
    def test_export_trace_structure(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        session.token_ids = [1, 2, 3]
        session.response_mask = [0, 1, 0]
        trace = session.export_trace()
        assert trace["final_token_list"] == [1, 2, 3]
        assert trace["response_mask"] == [0, 1, 0]
        assert isinstance(trace["round_traces"], list)
        assert isinstance(trace["token_provider_usage"], list)


class TestDetokenize:
    def test_empty_tokens_returns_empty(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        assert session.detokenize([]) == ""

    def test_detokenize_calls_post_json(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        with patch.object(session, "_post_json", return_value={"text": "decoded text"}):
            result = session.detokenize([1, 2, 3])
        assert result == "decoded text"

    def test_detokenize_raises_on_missing_text(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        with patch.object(session, "_post_json", return_value={}):
            with pytest.raises(ValueError, match="missing text"):
                session.detokenize([1, 2, 3])


class TestDetokenizeAsync:
    @pytest.mark.anyio
    async def test_empty_tokens_returns_empty(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        result = await session.detokenize_async([])
        assert result == ""

    @pytest.mark.anyio
    async def test_detokenize_async_calls_post(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        with patch.object(session, "_post_json_async", new_callable=AsyncMock, return_value={"text": "async decoded"}):
            result = await session.detokenize_async([1, 2])
        assert result == "async decoded"

    @pytest.mark.anyio
    async def test_detokenize_async_raises_on_missing_text(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config)
        with patch.object(session, "_post_json_async", new_callable=AsyncMock, return_value={}):
            with pytest.raises(ValueError, match="missing text"):
                await session.detokenize_async([1, 2])


class TestInitializeFromMessages:
    def test_skips_if_already_synced(self, llm_config):
        session = TokenTraceSession(llm_config=llm_config, synced_message_count=5)
        with patch.object(session, "tokenize_messages") as mock_tokenize:
            session.initialize_from_messages([Message.user("Hello")])
        mock_tokenize.assert_not_called()
