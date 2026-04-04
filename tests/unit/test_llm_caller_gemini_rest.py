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

"""Coverage tests for Gemini REST functions in llm_caller.py (lines 2528+).

Targets:
- GeminiRestStreamAggregator (.consume, .finalize)
- _iter_gemini_sse_chunks (sync, SSE + JSON array fallback)
- _iter_gemini_sse_chunks_async (async, SSE + JSON array fallback)
- _gemini_sanitize_parameters
- convert_tools_to_gemini
- call_llm_with_gemini_rest (sync: non-stream, stream, tracing, errors)
- call_llm_with_gemini_rest_async (async: non-stream, stream, tracing, errors)
"""

import json
import threading
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import requests

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.hooks import MiddlewareManager, ModelCallParams
from nexau.archs.main_sub.execution.llm_caller import (
    GeminiRestStreamAggregator,
    _gemini_sanitize_parameters,
    _iter_gemini_sse_chunks,
    _iter_gemini_sse_chunks_async,
    call_llm_with_gemini_rest,
    call_llm_with_gemini_rest_async,
    convert_tools_to_gemini,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.messages import Message, Role, TextBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_chunk(text: str, *, thought: bool = False, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a minimal Gemini REST SSE chunk dict."""
    part: dict[str, Any] = {"text": text}
    if thought:
        part["thought"] = True
    chunk: dict[str, Any] = {
        "candidates": [{"content": {"parts": [part], "role": "model"}}],
    }
    if usage is not None:
        chunk["usageMetadata"] = usage
    return chunk


def _tool_call_chunk(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"functionCall": {"name": name, "args": args}}],
                    "role": "model",
                }
            }
        ],
    }


def _gemini_config(**overrides: Any) -> LLMConfig:
    defaults: dict[str, Any] = {
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com",
        "api_key": "test-key",
        "api_type": "gemini_rest",
        "temperature": 0.5,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _make_gemini_params(
    msg_text: str = "Hi",
    shutdown_event: threading.Event | None = None,
) -> ModelCallParams:
    """Build a real ModelCallParams for Gemini REST tests."""
    return ModelCallParams(
        messages=[Message(role=Role.USER, content=[TextBlock(text=msg_text)])],
        max_tokens=100,
        force_stop_reason=None,
        agent_state=None,
        tool_call_mode="structured",
        tools=None,
        api_params={},
        shutdown_event=shutdown_event,
    )


class _AsyncStreamCM:
    """Mock async context manager for httpx.AsyncClient.stream().

    Wraps a list of SSE line strings into a mock response with aiter_lines().
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> "_AsyncStreamCM":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


# ===========================================================================
# GeminiRestStreamAggregator
# ===========================================================================


class TestGeminiRestStreamAggregator:
    """Tests for GeminiRestStreamAggregator.consume / .finalize."""

    def test_single_text_chunk(self):
        agg = GeminiRestStreamAggregator()
        agg.consume(_text_chunk("Hello"))
        result = agg.finalize()
        assert result["candidates"][0]["content"]["parts"][-1]["text"] == "Hello"

    def test_multiple_text_chunks_concatenated(self):
        agg = GeminiRestStreamAggregator()
        agg.consume(_text_chunk("Hello "))
        agg.consume(_text_chunk("world"))
        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        text_parts = [p for p in parts if "text" in p and not p.get("thought")]
        assert text_parts[-1]["text"] == "Hello world"

    def test_thinking_content_aggregated(self):
        agg = GeminiRestStreamAggregator()
        agg.consume(_text_chunk("think-1", thought=True))
        agg.consume(_text_chunk("think-2", thought=True))
        agg.consume(_text_chunk("answer"))
        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        thought_parts = [p for p in parts if p.get("thought")]
        assert thought_parts[0]["text"] == "think-1think-2"

    def test_thought_signature_captured(self):
        agg = GeminiRestStreamAggregator()
        agg.consume(_text_chunk("answer"))
        sig_chunk: dict[str, Any] = {"candidates": [{"content": {"parts": [{"thoughtSignature": "sig123"}], "role": "model"}}]}
        agg.consume(sig_chunk)
        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        sig_parts = [p for p in parts if "thoughtSignature" in p]
        assert sig_parts[0]["thoughtSignature"] == "sig123"

    def test_tool_calls_aggregated(self):
        agg = GeminiRestStreamAggregator()
        agg.consume(_tool_call_chunk("search", {"q": "test"}))
        agg.consume(_tool_call_chunk("fetch", {"url": "http://x"}))
        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        fc_parts = [p for p in parts if "functionCall" in p]
        assert len(fc_parts) == 2

    def test_usage_metadata_from_last_chunk(self):
        usage = {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}
        agg = GeminiRestStreamAggregator()
        agg.consume(_text_chunk("a"))
        agg.consume(_text_chunk("b", usage=usage))
        result = agg.finalize()
        assert result["usageMetadata"] == usage

    def test_model_version_captured(self):
        agg = GeminiRestStreamAggregator()
        chunk = _text_chunk("hi")
        chunk["modelVersion"] = "gemini-2.0-flash-001"
        agg.consume(chunk)
        assert agg.model_name == "gemini-2.0-flash-001"

    def test_finalize_raises_without_chunks(self):
        agg = GeminiRestStreamAggregator()
        with pytest.raises(RuntimeError, match="No stream chunks"):
            agg.finalize()

    def test_consume_skips_non_sequence_candidates(self):
        """candidates field is not a list — should silently skip."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": "not-a-list"})
        # Also feed a real chunk so finalize doesn't raise
        agg.consume(_text_chunk("ok"))
        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        assert any(p.get("text") == "ok" for p in parts)

    def test_consume_skips_candidate_without_content_mapping(self):
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": "not-a-dict"}]})
        agg.consume(_text_chunk("ok"))
        agg.finalize()  # should not raise

    def test_consume_no_parts(self):
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"role": "model"}}]})
        agg.consume(_text_chunk("ok"))
        agg.finalize()

    def test_consume_parts_not_a_sequence(self):
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": "string-parts", "role": "model"}}]})
        agg.consume(_text_chunk("ok"))
        agg.finalize()

    def test_usage_metadata_absent_gives_no_key(self):
        agg = GeminiRestStreamAggregator()
        agg.consume(_text_chunk("ok"))
        result = agg.finalize()
        assert "usageMetadata" not in result


# ===========================================================================
# _iter_gemini_sse_chunks (sync)
# ===========================================================================


class TestIterGeminiSseChunks:
    """Tests for the sync SSE chunk iterator."""

    def test_sse_data_lines(self):
        """Standard SSE data: lines produce parsed JSON chunks."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.iter_lines.return_value = [
            b'data: {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}',
            b"",
            b'data: {"candidates": [{"content": {"parts": [{"text": " world"}]}}]}',
        ]
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 2
        assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "hello"

    def test_sse_empty_data_skipped(self):
        mock_resp = Mock(spec=requests.Response)
        mock_resp.iter_lines.return_value = [b"data: ", b"data:", b'data: {"ok": true}']
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1

    def test_sse_invalid_json_logged(self, caplog):
        mock_resp = Mock(spec=requests.Response)
        mock_resp.iter_lines.return_value = [b"data: {bad-json}"]
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert chunks == []
        assert "Failed to parse Gemini SSE chunk" in caplog.text

    def test_json_array_fallback(self):
        """No data: lines → falls back to parsing as JSON array."""
        body = json.dumps(
            [
                {"candidates": [{"content": {"parts": [{"text": "a"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "b"}]}}]},
            ]
        )
        mock_resp = Mock(spec=requests.Response)
        mock_resp.iter_lines.return_value = [line.encode() for line in body.split("\n")]
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 2

    def test_json_dict_fallback(self):
        """Single JSON object (not array) as fallback."""
        body = json.dumps({"candidates": [{"content": {"parts": [{"text": "single"}]}}]})
        mock_resp = Mock(spec=requests.Response)
        mock_resp.iter_lines.return_value = [line.encode() for line in body.split("\n")]
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1

    def test_json_fallback_invalid_logged(self, caplog):
        mock_resp = Mock(spec=requests.Response)
        mock_resp.iter_lines.return_value = [b"not json at all"]
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert chunks == []
        assert "Failed to parse Gemini streaming response as JSON" in caplog.text

    def test_string_lines_handled(self):
        """iter_lines may return str instead of bytes."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.iter_lines.return_value = ['data: {"ok": true}']
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1


# ===========================================================================
# _iter_gemini_sse_chunks_async
# ===========================================================================


class TestIterGeminiSseChunksAsync:
    """Tests for the async SSE chunk iterator."""

    @staticmethod
    async def _collect(aiter: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item async for item in aiter]

    @staticmethod
    def _mock_httpx_response(lines: list[str]) -> Mock:
        """Create a mock httpx.Response with aiter_lines returning given lines."""
        mock_resp = Mock(spec=httpx.Response)

        async def aiter_lines() -> AsyncIterator[str]:
            for line in lines:
                yield line

        mock_resp.aiter_lines = aiter_lines
        return mock_resp

    @pytest.mark.anyio
    async def test_sse_data_lines(self):
        resp = self._mock_httpx_response(
            [
                'data: {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}',
                "",
                'data: {"candidates": [{"content": {"parts": [{"text": " there"}]}}]}',
            ]
        )
        chunks = await self._collect(_iter_gemini_sse_chunks_async(resp))
        assert len(chunks) == 2

    @pytest.mark.anyio
    async def test_sse_empty_data_skipped(self):
        resp = self._mock_httpx_response(["data: ", "data:", 'data: {"ok": true}'])
        chunks = await self._collect(_iter_gemini_sse_chunks_async(resp))
        assert len(chunks) == 1

    @pytest.mark.anyio
    async def test_sse_invalid_json_logged(self, caplog):
        resp = self._mock_httpx_response(["data: {bad}"])
        chunks = await self._collect(_iter_gemini_sse_chunks_async(resp))
        assert chunks == []
        assert "Failed to parse Gemini SSE chunk (async)" in caplog.text

    @pytest.mark.anyio
    async def test_json_array_fallback(self):
        body = json.dumps(
            [
                {"candidates": [{"content": {"parts": [{"text": "x"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "y"}]}}]},
            ]
        )
        resp = self._mock_httpx_response(body.split("\n"))
        chunks = await self._collect(_iter_gemini_sse_chunks_async(resp))
        assert len(chunks) == 2

    @pytest.mark.anyio
    async def test_json_dict_fallback(self):
        body = json.dumps({"candidates": [{"content": {"parts": [{"text": "z"}]}}]})
        resp = self._mock_httpx_response(body.split("\n"))
        chunks = await self._collect(_iter_gemini_sse_chunks_async(resp))
        assert len(chunks) == 1

    @pytest.mark.anyio
    async def test_json_fallback_invalid_logged(self, caplog):
        resp = self._mock_httpx_response(["not-json"])
        chunks = await self._collect(_iter_gemini_sse_chunks_async(resp))
        assert chunks == []
        assert "Failed to parse Gemini streaming response as JSON (async)" in caplog.text

    @pytest.mark.anyio
    async def test_empty_lines_skipped(self):
        resp = self._mock_httpx_response(["", "", 'data: {"ok":1}', ""])
        chunks = await self._collect(_iter_gemini_sse_chunks_async(resp))
        assert len(chunks) == 1


# ===========================================================================
# _gemini_sanitize_parameters
# ===========================================================================


class TestGeminiSanitizeParameters:
    def test_strips_disallowed_keys(self):
        params: dict[str, object] = {
            "type": "object",
            "$schema": "http://...",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "description": "The name"},
            },
            "required": ["name"],
        }
        result = _gemini_sanitize_parameters(params)
        assert "$schema" not in result
        assert "additionalProperties" not in result
        assert result["type"] == "object"
        assert "name" in result["properties"]

    def test_recursively_sanitizes_properties(self):
        params: dict[str, object] = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "$schema": "remove-me",
                    "properties": {"x": {"type": "string"}},
                },
            },
        }
        result = _gemini_sanitize_parameters(params)
        child = result["properties"]["child"]
        assert "$schema" not in child

    def test_sanitizes_items(self):
        params: dict[str, object] = {
            "type": "array",
            "items": {"type": "string", "$schema": "remove"},
        }
        result = _gemini_sanitize_parameters(params)
        assert "$schema" not in result["items"]
        assert result["items"]["type"] == "string"

    def test_keeps_allowed_keys(self):
        params: dict[str, object] = {
            "type": "string",
            "description": "A field",
            "enum": ["a", "b"],
            "format": "date",
            "nullable": True,
        }
        result = _gemini_sanitize_parameters(params)
        assert result == params

    def test_ignores_non_dict_property_values(self):
        """Property values that are not dicts should be filtered out."""
        params: dict[str, object] = {
            "type": "object",
            "properties": {
                "good": {"type": "string"},
                "bad": "not-a-dict",
            },
        }
        result = _gemini_sanitize_parameters(params)
        assert "good" in result["properties"]
        assert "bad" not in result["properties"]


# ===========================================================================
# convert_tools_to_gemini
# ===========================================================================


class TestConvertToolsToGemini:
    def test_basic_conversion(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "kind": "tool",
            }
        ]
        result = convert_tools_to_gemini(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert "parameters" in result[0]
        # Schema should be sanitized
        assert "$schema" not in result[0]["parameters"]

    def test_multiple_tools(self):
        tools = [
            {"name": f"tool_{i}", "description": f"Tool {i}", "input_schema": {"type": "object", "properties": {}}, "kind": "tool"}
            for i in range(3)
        ]
        result = convert_tools_to_gemini(tools)
        assert len(result) == 3

    def test_skips_non_function_type_on_normalize_error(self):
        """Tools with type != 'function' that fail normalization are skipped."""
        tools = [
            {"type": "not_function", "name": "skip_me"},  # will fail normalize, type != function → skip
            {"name": "good", "description": "ok", "input_schema": {"type": "object", "properties": {}}, "kind": "tool"},
        ]
        result = convert_tools_to_gemini(tools)
        assert len(result) == 1
        assert result[0]["name"] == "good"


# ===========================================================================
# call_llm_with_gemini_rest (sync)
# ===========================================================================


class TestCallLlmWithGeminiRestSync:
    """Tests for the sync Gemini REST call function."""

    def _make_gemini_response_json(self, text: str = "ok") -> dict[str, Any]:
        return {
            "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
        }

    def test_non_streaming_basic(self):
        config = _gemini_config()
        resp_json = self._make_gemini_response_json("Hello")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = resp_json

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

        assert isinstance(result, ModelResponse)
        assert result.content == "Hello"
        mock_post.assert_called_once()
        # Verify URL contains generateContent (not stream)
        call_url = mock_post.call_args[0][0]
        assert "generateContent" in call_url
        assert "stream" not in call_url.split("?")[0]

    def test_non_streaming_with_tracing(self):
        config = _gemini_config()
        resp_json = self._make_gemini_response_json("traced")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = resp_json

        mock_tracer = Mock()
        mock_trace_ctx = MagicMock()

        with (
            patch("requests.post", return_value=mock_response),
            patch("nexau.archs.main_sub.execution.llm_caller.get_current_span", return_value=Mock()),
            patch("nexau.archs.main_sub.execution.llm_caller.TraceContext", return_value=mock_trace_ctx),
        ):
            mock_trace_ctx.__enter__ = Mock(return_value=mock_trace_ctx)
            mock_trace_ctx.__exit__ = Mock(return_value=False)
            result = call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                tracer=mock_tracer,
                model_call_params=_make_gemini_params(),
            )

        assert result.content == "traced"

    def test_non_streaming_request_exception(self):
        config = _gemini_config()
        mock_response = Mock()
        mock_response.text = "Bad Request"

        exc = requests.exceptions.HTTPError(response=mock_response)

        with (
            patch("requests.post", side_effect=exc),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

    def test_non_streaming_generic_exception(self):
        config = _gemini_config()

        with (
            patch("requests.post", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

    def test_streaming_basic(self):
        config = _gemini_config()
        sse_lines = [
            b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}}]}',
            (
                b'data: {"candidates": [{"content": {"parts": [{"text": " world"}],'
                b' "role": "model"}}], "usageMetadata": {"totalTokenCount": 10}}'
            ),
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = iter(sse_lines)

        with patch("requests.post", return_value=mock_response):
            result = call_llm_with_gemini_rest(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

        assert isinstance(result, ModelResponse)
        assert result.content is not None
        assert "Hello" in result.content
        assert "world" in result.content

    def test_streaming_with_shutdown_event(self):
        """Shutdown event already set → breaks before consuming, finalize raises."""
        config = _gemini_config()
        sse_lines = [
            b'data: {"candidates": [{"content": {"parts": [{"text": "partial"}], "role": "model"}}]}',
            b'data: {"candidates": [{"content": {"parts": [{"text": " more"}], "role": "model"}}]}',
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = iter(sse_lines)

        shutdown_ev = threading.Event()
        shutdown_ev.set()

        model_params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            shutdown_event=shutdown_ev,
        )

        with (
            patch("requests.post", return_value=mock_response),
            pytest.raises(RuntimeError, match="No stream chunks"),
        ):
            call_llm_with_gemini_rest(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=model_params,
            )

    def test_streaming_request_exception(self):
        config = _gemini_config()
        mock_response = Mock()
        mock_response.text = "Server Error"
        exc = requests.exceptions.HTTPError(response=mock_response)

        with (
            patch("requests.post", side_effect=exc),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            call_llm_with_gemini_rest(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

    def test_streaming_generic_exception(self):
        config = _gemini_config()

        with (
            patch("requests.post", side_effect=RuntimeError("stream-boom")),
            pytest.raises(RuntimeError, match="stream-boom"),
        ):
            call_llm_with_gemini_rest(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

    def test_image_block_supported(self):
        """RFC-0014: ImageBlock is now supported for Gemini REST via GeminiMessagesAdapter."""
        from nexau.core.messages import ImageBlock

        config = _gemini_config()
        model_params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[ImageBlock(base64="abc123", mime_type="image/png")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json("image ok")

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = call_llm_with_gemini_rest(
                {"messages": []},
                llm_config=config,
                model_call_params=model_params,
            )

        assert isinstance(result, ModelResponse)
        assert result.content == "image ok"
        # Verify the image was sent as inline_data in the request body
        request_body = mock_post.call_args[1]["json"]
        parts = request_body["contents"][0]["parts"]
        inline_parts = [p for p in parts if "inlineData" in p or "inline_data" in p]
        assert len(inline_parts) == 1

    def test_url_custom_base_url(self):
        """Custom base_url (non-googleapis) should use /models/ without v1beta."""
        config = _gemini_config(base_url="https://custom-proxy.example.com")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json()

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )
        call_url = mock_post.call_args[0][0]
        assert "custom-proxy.example.com/models/" in call_url
        assert "v1beta" not in call_url

    def test_url_empty_base_url_defaults(self):
        config = _gemini_config()
        # Force base_url to None so the function defaults to googleapis.com
        config.base_url = None

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json()

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )
        call_url = mock_post.call_args[0][0]
        assert "generativelanguage.googleapis.com" in call_url

    def test_generation_config_params(self):
        config = _gemini_config(temperature=0.9, max_tokens=500, top_p=0.95)
        config.extra_params["top_k"] = "40"

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json()

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )
        body = mock_post.call_args[1]["json"]
        gen_config = body["generationConfig"]
        assert gen_config["temperature"] == 0.9
        assert gen_config["maxOutputTokens"] == 500
        assert gen_config["topP"] == 0.95
        assert gen_config["topK"] == 40

    def test_top_k_invalid_ignored(self):
        config = _gemini_config()
        config.extra_params["top_k"] = "not-a-number"

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json()

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )
        body = mock_post.call_args[1]["json"]
        assert "topK" not in body["generationConfig"]

    def test_thinking_config_added(self):
        config = _gemini_config()
        config.extra_params["thinkingConfig"] = {"thinkingBudget": 1024}

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json()

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )
        body = mock_post.call_args[1]["json"]
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}

    def test_tools_converted_and_attached(self):
        config = _gemini_config()
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                "kind": "tool",
            }
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json()

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}], "tools": tools},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )
        body = mock_post.call_args[1]["json"]
        assert "tools" in body
        assert "functionDeclarations" in body["tools"][0]

    def test_with_model_call_params(self):
        """When model_call_params is provided, uses GeminiMessagesAdapter."""
        config = _gemini_config()
        model_params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi from UMP")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json("adapter-ok")

        with patch("requests.post", return_value=mock_response):
            result = call_llm_with_gemini_rest(
                {"messages": []},
                llm_config=config,
                model_call_params=model_params,
            )
        assert result.content == "adapter-ok"

    def test_no_llm_config_raises(self):
        with pytest.raises(ValueError, match="llm_config is required"):
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=None,
                model_call_params=_make_gemini_params(),
            )

    def test_stream_from_llm_config_attribute(self):
        """stream flag can come from llm_config.stream instead of kwargs."""
        config = _gemini_config()
        config.stream = True

        sse_lines = [
            (
                b'data: {"candidates": [{"content": {"parts": [{"text": "streamed"}],'
                b' "role": "model"}}], "usageMetadata": {"totalTokenCount": 5}}'
            ),
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = iter(sse_lines)

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )
        call_url = mock_post.call_args[0][0]
        assert "streamGenerateContent" in call_url
        assert "alt=sse" in call_url

    def test_system_instruction_included(self):
        config = _gemini_config()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json()

        sys_params = ModelCallParams(
            messages=[
                Message(role=Role.SYSTEM, content=[TextBlock(text="Be helpful")]),
                Message(role=Role.USER, content=[TextBlock(text="Hi")]),
            ],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="structured",
            tools=None,
            api_params={},
        )

        with patch("requests.post", return_value=mock_response) as mock_post:
            call_llm_with_gemini_rest(
                {},
                llm_config=config,
                model_call_params=sys_params,
            )
        body = mock_post.call_args[1]["json"]
        assert "systemInstruction" in body


# ===========================================================================
# call_llm_with_gemini_rest_async
# ===========================================================================


class TestCallLlmWithGeminiRestAsync:
    """Tests for the async Gemini REST call function."""

    def _make_gemini_response_json(self, text: str = "ok") -> dict[str, Any]:
        return {
            "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
        }

    @pytest.mark.anyio
    async def test_non_streaming_basic(self):
        config = _gemini_config()
        resp_json = self._make_gemini_response_json("async-ok")

        # httpx.Response.json() and .raise_for_status() are sync methods
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = resp_json

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await call_llm_with_gemini_rest_async(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

        assert isinstance(result, ModelResponse)
        assert result.content == "async-ok"

    @pytest.mark.anyio
    async def test_non_streaming_with_tracing(self):
        config = _gemini_config()
        resp_json = self._make_gemini_response_json("traced-async")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = resp_json

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_tracer = Mock()
        mock_trace_ctx = MagicMock()
        mock_trace_ctx.__enter__ = Mock(return_value=mock_trace_ctx)
        mock_trace_ctx.__exit__ = Mock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("nexau.archs.main_sub.execution.llm_caller.get_current_span", return_value=Mock()),
            patch("nexau.archs.main_sub.execution.llm_caller.TraceContext", return_value=mock_trace_ctx),
        ):
            result = await call_llm_with_gemini_rest_async(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                tracer=mock_tracer,
                model_call_params=_make_gemini_params(),
            )

        assert result.content == "traced-async"
        mock_trace_ctx.set_outputs.assert_called_once()

    @pytest.mark.anyio
    async def test_non_streaming_http_error(self):
        config = _gemini_config()

        mock_request = Mock(spec=httpx.Request)
        mock_err_response = Mock(spec=httpx.Response)
        mock_err_response.status_code = 400
        mock_err_response.text = "Bad request"

        exc = httpx.HTTPStatusError("Bad Request", request=mock_request, response=mock_err_response)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=exc)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await call_llm_with_gemini_rest_async(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

    @pytest.mark.anyio
    async def test_non_streaming_generic_error(self):
        config = _gemini_config()

        mock_client = AsyncMock()
        mock_client.post.side_effect = RuntimeError("async-boom")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(RuntimeError, match="async-boom"),
        ):
            await call_llm_with_gemini_rest_async(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

    @pytest.mark.anyio
    async def test_streaming_basic(self):
        config = _gemini_config()

        sse_lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "async "}], "role": "model"}}]}',
            (
                'data: {"candidates": [{"content": {"parts": [{"text": "stream"}],'
                ' "role": "model"}}], "usageMetadata": {"totalTokenCount": 8}}'
            ),
        ]

        stream_cm = _AsyncStreamCM(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_cm
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await call_llm_with_gemini_rest_async(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

        assert isinstance(result, ModelResponse)
        assert result.content is not None
        assert "async " in result.content
        assert "stream" in result.content

    @pytest.mark.anyio
    async def test_streaming_with_shutdown(self):
        """Shutdown event already set → breaks before consuming, finalize raises."""
        config = _gemini_config()

        sse_lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "partial"}], "role": "model"}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": " more"}], "role": "model"}}]}',
        ]

        stream_cm = _AsyncStreamCM(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_cm
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        shutdown_ev = threading.Event()
        shutdown_ev.set()

        model_params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            shutdown_event=shutdown_ev,
        )

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(RuntimeError, match="No stream chunks"),
        ):
            await call_llm_with_gemini_rest_async(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=model_params,
            )

    @pytest.mark.anyio
    async def test_streaming_with_tracing(self):
        config = _gemini_config()

        sse_lines = [
            (
                'data: {"candidates": [{"content": {"parts": [{"text": "traced-stream"}],'
                ' "role": "model"}}], "usageMetadata": {"totalTokenCount": 5}}'
            ),
        ]

        stream_cm = _AsyncStreamCM(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_cm
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_tracer = Mock()
        mock_trace_ctx = MagicMock()
        mock_trace_ctx.__enter__ = Mock(return_value=mock_trace_ctx)
        mock_trace_ctx.__exit__ = Mock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("nexau.archs.main_sub.execution.llm_caller.get_current_span", return_value=Mock()),
            patch("nexau.archs.main_sub.execution.llm_caller.TraceContext", return_value=mock_trace_ctx),
        ):
            result = await call_llm_with_gemini_rest_async(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                tracer=mock_tracer,
                model_call_params=_make_gemini_params(),
            )

        assert result.content == "traced-stream"
        mock_trace_ctx.set_outputs.assert_called_once()
        mock_trace_ctx.set_attributes.assert_called_once()

    @pytest.mark.anyio
    async def test_image_block_supported(self):
        """RFC-0014: ImageBlock is now supported for Gemini REST via GeminiMessagesAdapter."""
        from nexau.core.messages import ImageBlock

        config = _gemini_config()
        model_params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[ImageBlock(base64="abc123", mime_type="image/png")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = self._make_gemini_response_json("image ok")

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await call_llm_with_gemini_rest_async(
                {"messages": []},
                llm_config=config,
                model_call_params=model_params,
            )

        assert isinstance(result, ModelResponse)
        assert result.content == "image ok"

    @pytest.mark.anyio
    async def test_no_api_key_raises(self):
        config = _gemini_config()
        config.api_key = None  # Force no api_key after construction
        with pytest.raises(ValueError, match="API key is required"):
            await call_llm_with_gemini_rest_async(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

    @pytest.mark.anyio
    async def test_url_custom_base_url(self):
        config = _gemini_config(base_url="https://my-proxy.com")
        resp_json = self._make_gemini_response_json()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = resp_json

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client) as _:
            result = await call_llm_with_gemini_rest_async(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

        assert result.content == "ok"

    @pytest.mark.anyio
    async def test_with_model_call_params_uses_adapter(self):
        config = _gemini_config()
        model_params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="UMP msg")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )

        resp_json = self._make_gemini_response_json("adapter-async-ok")
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = resp_json

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await call_llm_with_gemini_rest_async(
                {"messages": []},
                llm_config=config,
                model_call_params=model_params,
            )

        assert result.content == "adapter-async-ok"

    @pytest.mark.anyio
    async def test_generation_config_all_params(self):
        config = _gemini_config(temperature=0.3, max_tokens=200, top_p=0.8)
        config.extra_params["top_k"] = "20"
        config.extra_params["thinkingConfig"] = {"thinkingBudget": 512}

        resp_json = self._make_gemini_response_json()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = resp_json

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await call_llm_with_gemini_rest_async(
                {"messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                model_call_params=_make_gemini_params(),
            )

        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        gen = body["generationConfig"]
        assert gen["temperature"] == 0.3
        assert gen["maxOutputTokens"] == 200
        assert gen["topP"] == 0.8
        assert gen["topK"] == 20
        assert gen["thinkingConfig"] == {"thinkingBudget": 512}

    @pytest.mark.anyio
    async def test_streaming_with_middleware(self):
        """_process_stream_chunk returning None should skip the chunk."""
        config = _gemini_config()

        sse_lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "visible"}], "role": "model"}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": " filtered"}], "role": "model"}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": " kept"}], "role": "model"}}], "usageMetadata": {"totalTokenCount": 5}}',
        ]

        stream_cm = _AsyncStreamCM(sse_lines)

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_cm
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Middleware that filters the second chunk
        call_count = 0

        def mock_stream_chunk(chunk: Any, params: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return None  # filter out
            return chunk

        mm = Mock(spec=MiddlewareManager)
        mm.stream_chunk = mock_stream_chunk

        model_params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await call_llm_with_gemini_rest_async(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                llm_config=config,
                middleware_manager=mm,
                model_call_params=model_params,
            )

        assert isinstance(result, ModelResponse)
        # "filtered" chunk was dropped by middleware
        assert result.content is not None
        assert "visible" in result.content
        assert "kept" in result.content
