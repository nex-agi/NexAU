# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Live end-to-end coverage for Set A streaming aggregators.

RFC-0023 §阶段 ③ — Set A aggregators became the single canonical stream
parsers and Set B retired. The deleted parity tests gave us
**equivalence** coverage (Set A ≡ Set B); this file gives us
**correctness** coverage against real provider endpoints.

For each of the 4 supported APIs (OpenAI Chat / OpenAI Responses /
Anthropic / Gemini REST) we issue one streaming call via the
``call_llm_with_*`` entry point ``llm_caller`` exposes, then assert
the resulting ``ModelResponse``:

- has non-empty ``.content`` (the aggregator built blocks correctly
  and the ``ModelResponse.from_X`` adapter extracted them — this
  is the exact axis the deleted Set B ``finalize() → from_X`` parity
  used to cover, and that let the Chat ``ChatCompletion`` shape bug
  slip past local tests),
- reports ``.usage.total_tokens > 0`` (usage propagated through the
  stream → build → ``UsageUpdateEvent`` chain).

Each test pulls credentials from a dedicated env var and skips
cleanly when the env isn't provisioned, so the file is safe to keep
alongside the broader live test suite without forcing every dev to
hold every provider's key.

These are gated behind ``@pytest.mark.llm`` (auto-skip unless
``NEXAU_RUN_LIVE_LLM_TESTS=1``) like the rest of our live suite.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import openai
import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.hooks import ModelCallParams
from nexau.archs.main_sub.execution.llm_caller import (
    call_llm_with_anthropic_chat_completion,
    call_llm_with_gemini_rest,
    call_llm_with_openai_chat_completion,
    call_llm_with_openai_responses,
)
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.core.messages import Message, Role, TextBlock

pytestmark = [pytest.mark.integration, pytest.mark.llm]


# ── Provider config ─────────────────────────────────────────────────
#
# Northgate (``northgate.xiaobei.top``) proxies multiple upstream
# providers behind a single key (``NORTHGATE_API_KEY``); Gemini uses
# Google's public REST endpoint with ``GEMINI_API_KEY``. Models picked
# to be the cheapest stable offering on each route.

_NORTHGATE_BASE = "https://northgate.xiaobei.top"
_NORTHGATE_OPENAI_BASE = f"{_NORTHGATE_BASE}/v1"

# Gemini route uses the same 14.103 gateway the matrix targets — the
# public ``generativelanguage.googleapis.com`` free tier caps daily
# requests at 20 for ``gemini-2.5-flash`` and 0 for everything else,
# which made our prior CI runs flake-skip on quota exhaustion. The
# gateway proxies a paid tier with no per-test quotas.
#
# ``call_llm_with_gemini_rest`` builds custom-gateway URLs as
# ``{base_url}/models/...`` (no automatic ``/v1beta`` insertion —
# only the public Google host gets that prefix added), so the base
# must include ``/v1beta`` itself.
_GEMINI_GATEWAY_BASE = "http://14.103.60.158:3001/v1beta"
# Default Gemini model for happy-path / tools / async / multi-chunk
# tests — flash-lite-preview doesn't auto-think (``thoughtsTokenCount=0``
# on a "say PONG" prompt), so a 32-token budget actually leaves room
# for the answer. The Gemini-3 ``pro``/``flash`` variants both think
# 28-44 tokens before answering anything, which would consume a
# small max_tokens before generating any content.
_GEMINI_GATEWAY_MODEL = "gemini-3.1-flash-lite-preview"
# Pro variant always thinks — used by the reasoning test below.
_GEMINI_GATEWAY_THINKING_MODEL = "gemini-3.1-pro-preview"


def _northgate_key() -> str:
    key = os.getenv("NORTHGATE_API_KEY")
    if not key:
        pytest.skip("NORTHGATE_API_KEY not set")
    return key


def _gemini_key() -> str:
    """Resolve the Gemini gateway key.

    Prefer the gateway-specific key (``TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY``,
    same name the existing live token matrix uses). Fall back to a
    public-tier ``GEMINI_API_KEY`` if someone wires that for ad-hoc local
    runs against ``generativelanguage.googleapis.com``.
    """
    key = os.getenv("TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        pytest.skip("TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY (or GEMINI_API_KEY) not set")
    return key


def _retry_on_transient_gateway_error(thunk, *, attempts: int = 3, label: str = ""):
    """Retry transient gateway 5xx (502 Bad Gateway, 503 Unavailable).

    The 14.103 Gemini route occasionally bounces a 502 mid-stream
    (gateway-side blip, not our code). One retry is almost always
    enough; cap at 3 to prevent runaway delays.
    """
    import time  # noqa: PLC0415

    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return thunk()
        except Exception as exc:
            text = str(exc)
            if "502" in text or "503" in text or "Bad Gateway" in text or "Service Unavailable" in text:
                last_exc = exc
                if i < attempts - 1:
                    time.sleep(2)
                continue
            raise
    raise last_exc if last_exc else RuntimeError(f"{label}: gateway error retries exhausted")


_USER_PROMPT = "Reply with the single word: PONG"
_TOOL_PROMPT = "What is the current weather in Tokyo? You MUST call the get_weather tool to find out — do not answer from memory."

# Tool definitions per provider wire format. Same logical contract
# (``get_weather(city: str)``) so each test can assert on the same
# ``tool_calls[0].name`` / arguments shape regardless of provider.

_OPENAI_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_OPENAI_RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

_ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

# ``call_llm_with_gemini_rest`` runs nexau's structured tool definitions
# through ``convert_tools_to_gemini`` — same neutral shape as the rest
# of the codebase (name + description + input_schema).
_GEMINI_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def _assert_minimal_streaming_response(model_response, *, label: str) -> None:
    """Shared correctness checks for a streaming ``ModelResponse``.

    Whitespace-tolerant content check (Gemini sometimes wraps the
    answer in a sentence; we just need *some* assistant text). Usage
    check guards against the regression class where the aggregator
    ``build()`` returned the right object but the ``ModelResponse``
    adapter dropped fields (e.g. the OpenAI Chat ``ChatCompletion``
    shape bug fixed during PR-C.2 review).
    """
    assert model_response is not None, f"{label}: ModelResponse is None"
    assert model_response.content, f"{label}: empty content (raw={model_response!r})"
    assert isinstance(model_response.content, str), f"{label}: content is {type(model_response.content)}"
    assert model_response.usage is not None, f"{label}: usage is None"
    assert model_response.usage.total_tokens > 0, f"{label}: usage.total_tokens={model_response.usage.total_tokens} (expected >0)"


def test_openai_chat_streaming_e2e_northgate_deepseek():
    """Set A ``OpenAIChatCompletionAggregator`` → ``ModelResponse.from_openai_message``.

    ``deepseek-v4-flash`` is the cheapest unrestricted OpenAI-Chat-shaped
    model on northgate (``gpt-5.x`` are Codex-locked).
    """
    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
    response = call_llm_with_openai_chat_completion(
        client,
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": _USER_PROMPT}],
            "max_tokens": 32,
            "stream": True,
        },
    )
    _assert_minimal_streaming_response(response, label="openai_chat/deepseek")


def test_openai_responses_streaming_e2e_northgate_gpt52():
    """Set A ``OpenAIResponsesAggregator`` → ``ModelResponse.from_openai_response``."""
    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
    response = call_llm_with_openai_responses(
        client,
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": _USER_PROMPT}],
            "max_output_tokens": 32,
            "stream": True,
        },
    )
    _assert_minimal_streaming_response(response, label="openai_responses/gpt-5.2")


def _ump_model_call_params(*, llm_config: LLMConfig, max_tokens: int = 32) -> ModelCallParams:
    user_msg = Message(role=Role.USER, content=[TextBlock(text=_USER_PROMPT)])
    return ModelCallParams(
        messages=[user_msg],
        max_tokens=max_tokens,
        force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
        agent_state=None,
        tool_call_mode="auto",
        tools=None,
        api_params={},
        llm_config=llm_config,
    )


def test_anthropic_streaming_e2e_northgate_sonnet45():
    """Set A ``AnthropicEventAggregator`` → ``ModelResponse.from_anthropic_message``."""
    import anthropic  # local import — anthropic SDK is optional in some installs  # noqa: PLC0415

    key = _northgate_key()
    client = anthropic.Anthropic(api_key=key, base_url=_NORTHGATE_BASE)
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        base_url=_NORTHGATE_BASE,
        api_key=key,
        api_type="anthropic_chat_completion",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )
    response = call_llm_with_anthropic_chat_completion(
        client,
        {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 32,
            "stream": True,
        },
        model_call_params=_ump_model_call_params(llm_config=llm_config),
        llm_config=llm_config,
    )
    _assert_minimal_streaming_response(response, label="anthropic/sonnet-4-5")


def test_gemini_rest_streaming_e2e_gateway_31pro():
    """Set A ``GeminiRestEventAggregator`` → ``ModelResponse.from_gemini_rest``.

    Hits Google's public ``generativelanguage.googleapis.com`` REST
    endpoint directly; ``call_llm_with_gemini_rest`` handles URL
    construction and SSE parsing. Unlike the OpenAI/Anthropic entry
    points (which take a ``client`` and a loose dict), the Gemini
    helper builds requests itself from UMP messages on
    ``ModelCallParams`` plus the ``LLMConfig``.
    """
    key = _gemini_key()
    llm_config = LLMConfig(
        model=_GEMINI_GATEWAY_MODEL,
        base_url=_GEMINI_GATEWAY_BASE,
        api_key=key,
        api_type="gemini_rest",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )
    response = _retry_on_transient_gateway_error(
        lambda: call_llm_with_gemini_rest(
            kwargs={"stream": True},
            model_call_params=_ump_model_call_params(llm_config=llm_config),
            llm_config=llm_config,
        ),
        label="gemini_rest/3.1-flash-lite",
    )
    _assert_minimal_streaming_response(response, label="gemini_rest/3.1-flash-lite")


# ── Tool calling ────────────────────────────────────────────────────
#
# Tool calling is the most common reason streaming aggregators get
# exercised in production agents. Each provider has its own wire
# shape for tool_use blocks; the aggregator ``build()`` → adapter
# chain is responsible for unifying them into ``ModelResponse.tool_calls``.
# A regression here silently breaks every agent loop.


def _assert_tool_call_get_weather(response, *, label: str) -> None:
    """Assert the model emitted a ``get_weather(city=...)`` tool call."""
    assert response is not None, f"{label}: ModelResponse is None"
    assert response.tool_calls, f"{label}: tool_calls empty (raw={response!r})"
    call = response.tool_calls[0]
    assert call.name == "get_weather", f"{label}: expected get_weather, got {call.name}"
    assert isinstance(call.arguments, dict), f"{label}: arguments not dict ({type(call.arguments)})"
    # The model should have filled in a city — Tokyo or some variant.
    # Don't pin the exact value (model paraphrases); just assert the
    # required field surfaced through the aggregator's tool-input parse.
    assert "city" in call.arguments or "location" in call.arguments, f"{label}: tool arguments missing city/location: {call.arguments!r}"
    assert response.usage is not None and response.usage.total_tokens > 0, f"{label}: usage missing or zero (got {response.usage!r})"


def test_openai_chat_streaming_tools_e2e_northgate_deepseek():
    """Set A OpenAI Chat aggregator: tool_calls deltas → ModelResponse.tool_calls."""
    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
    response = call_llm_with_openai_chat_completion(
        client,
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": _TOOL_PROMPT}],
            "tools": _OPENAI_CHAT_TOOLS,
            # ``tool_choice="required"`` rejected by the deepseek route on
            # this gateway ("does not support this tool_choice"); leave it
            # at the default and rely on the prompt's MUST directive.
            "max_tokens": 128,
            "stream": True,
        },
    )
    _assert_tool_call_get_weather(response, label="openai_chat/deepseek/tools")


def test_openai_responses_streaming_tools_e2e_northgate_gpt52():
    """Set A OpenAI Responses aggregator: function_call output items → tool_calls."""
    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
    response = call_llm_with_openai_responses(
        client,
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": _TOOL_PROMPT}],
            "tools": _OPENAI_RESPONSES_TOOLS,
            "tool_choice": "required",
            "max_output_tokens": 256,
            "stream": True,
        },
    )
    _assert_tool_call_get_weather(response, label="openai_responses/gpt-5.2/tools")


def test_anthropic_streaming_tools_e2e_northgate_sonnet45():
    """Set A Anthropic aggregator: tool_use blocks + input_json deltas → tool_calls."""
    import anthropic  # noqa: PLC0415

    key = _northgate_key()
    client = anthropic.Anthropic(api_key=key, base_url=_NORTHGATE_BASE)
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        base_url=_NORTHGATE_BASE,
        api_key=key,
        api_type="anthropic_chat_completion",
        temperature=0.0,
        max_tokens=256,
        max_retries=1,
        stream=True,
    )
    user_msg = Message(role=Role.USER, content=[TextBlock(text=_TOOL_PROMPT)])
    model_call_params = ModelCallParams(
        messages=[user_msg],
        max_tokens=256,
        force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
        agent_state=None,
        tool_call_mode="auto",
        tools=None,
        api_params={},
        llm_config=llm_config,
    )
    response = call_llm_with_anthropic_chat_completion(
        client,
        {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 256,
            "tools": _ANTHROPIC_TOOLS,
            "tool_choice": {"type": "any"},
            "stream": True,
        },
        model_call_params=model_call_params,
        llm_config=llm_config,
    )
    _assert_tool_call_get_weather(response, label="anthropic/sonnet-4-5/tools")


def test_gemini_rest_streaming_tools_e2e_gateway_31pro():
    """Set A Gemini aggregator: functionCall parts → tool_calls."""
    key = _gemini_key()
    llm_config = LLMConfig(
        model=_GEMINI_GATEWAY_MODEL,
        base_url=_GEMINI_GATEWAY_BASE,
        api_key=key,
        api_type="gemini_rest",
        temperature=0.0,
        max_tokens=256,
        max_retries=1,
        stream=True,
    )
    user_msg = Message(role=Role.USER, content=[TextBlock(text=_TOOL_PROMPT)])
    model_call_params = ModelCallParams(
        messages=[user_msg],
        max_tokens=256,
        force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
        agent_state=None,
        tool_call_mode="auto",
        tools=None,
        api_params={},
        llm_config=llm_config,
    )
    response = _retry_on_transient_gateway_error(
        lambda: call_llm_with_gemini_rest(
            kwargs={"tools": _GEMINI_TOOLS, "stream": True},
            model_call_params=model_call_params,
            llm_config=llm_config,
        ),
        label="gemini_rest/3.1-flash-lite/tools",
    )
    _assert_tool_call_get_weather(response, label="gemini_rest/3.1-flash-lite/tools")


# ── Reasoning / thinking ─────────────────────────────────────────────
#
# Each provider exposes "reasoning before answer" with its own wire
# shape — Anthropic ``thinking`` blocks + signature, Gemini's hidden
# ``thoughtSignature`` + ``thoughtsTokenCount`` usage, OpenAI Responses
# reasoning summary items, OpenAI Chat ``reasoning_content`` (DeepSeek
# style). Each shape feeds a different code path through the Set A
# aggregator and a different branch in ``ModelResponse.from_X``;
# regressions here show up as quiet token-accounting drift rather than
# obvious crashes, so a positive token-count assertion is the cheapest
# guardrail.

_REASONING_PROMPT = "Compute 47 * 53 step by step, then state the final number."


def test_anthropic_streaming_thinking_e2e_northgate_sonnet45():
    """Anthropic extended thinking → reasoning_content + reasoning_tokens."""
    import anthropic  # noqa: PLC0415

    key = _northgate_key()
    client = anthropic.Anthropic(api_key=key, base_url=_NORTHGATE_BASE)
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        base_url=_NORTHGATE_BASE,
        api_key=key,
        api_type="anthropic_chat_completion",
        # Extended thinking requires temperature=1.0 per Anthropic docs.
        temperature=1.0,
        max_tokens=2048,
        max_retries=1,
        stream=True,
    )
    user_msg = Message(role=Role.USER, content=[TextBlock(text=_REASONING_PROMPT)])
    model_call_params = ModelCallParams(
        messages=[user_msg],
        max_tokens=2048,
        force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
        agent_state=None,
        tool_call_mode="auto",
        tools=None,
        api_params={},
        llm_config=llm_config,
    )
    response = call_llm_with_anthropic_chat_completion(
        client,
        {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 2048,
            "temperature": 1.0,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "stream": True,
        },
        model_call_params=model_call_params,
        llm_config=llm_config,
    )
    label = "anthropic/sonnet-4-5/thinking"
    assert response is not None, f"{label}: ModelResponse is None"
    assert response.reasoning_content, f"{label}: reasoning_content empty (raw_message={response.raw_message!r})"
    assert response.usage.reasoning_tokens > 0 or response.reasoning_signature is not None, (
        f"{label}: usage.reasoning_tokens={response.usage.reasoning_tokens}, sig={response.reasoning_signature!r}"
    )


def test_gemini_rest_streaming_reasoning_e2e_gateway_31pro():
    """Gemini 3.1-pro-preview always reasons → ``thoughtsTokenCount`` → ``reasoning_tokens``."""
    key = _gemini_key()
    llm_config = LLMConfig(
        # Pro variant always thinks; budget enough tokens for reasoning + answer.
        model=_GEMINI_GATEWAY_THINKING_MODEL,
        base_url=_GEMINI_GATEWAY_BASE,
        api_key=key,
        api_type="gemini_rest",
        temperature=0.0,
        max_tokens=2048,
        max_retries=1,
        stream=True,
    )
    user_msg = Message(role=Role.USER, content=[TextBlock(text=_REASONING_PROMPT)])
    model_call_params = ModelCallParams(
        messages=[user_msg],
        max_tokens=2048,
        force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
        agent_state=None,
        tool_call_mode="auto",
        tools=None,
        api_params={},
        llm_config=llm_config,
    )
    response = _retry_on_transient_gateway_error(
        lambda: call_llm_with_gemini_rest(
            kwargs={"stream": True},
            model_call_params=model_call_params,
            llm_config=llm_config,
        ),
        label="gemini_rest/3.1-pro/reasoning",
    )
    label = "gemini_rest/3.1-pro/reasoning"
    assert response is not None, f"{label}: ModelResponse is None"
    assert response.content, f"{label}: empty content"
    # Gemini 2.5 always produces thinking tokens even on simple queries
    # — captured into ``thoughtsTokenCount`` and surfaced as
    # ``reasoning_tokens``. A regression in ``from_gemini_rest`` usage
    # extraction would zero this out silently.
    assert response.usage.reasoning_tokens > 0, (
        f"{label}: reasoning_tokens={response.usage.reasoning_tokens}, full usage={response.usage!r}"
    )


def test_openai_chat_streaming_reasoning_content_e2e_northgate_v4pro():
    """OpenAI Chat ``reasoning_content`` (DeepSeek style) → ``ModelResponse.reasoning_content``.

    ``deepseek-v4-pro`` exposes the chain of thought as a separate
    ``reasoning_content`` field on the assistant message — distinct
    from ``content``. ``ModelResponse.from_openai_message`` extracts
    it via ``getattr(message, 'reasoning_content', None)``. A
    regression in either the aggregator preserving the field through
    chunk concatenation or the adapter reading it would silently
    drop reasoning text.
    """
    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
    response = call_llm_with_openai_chat_completion(
        client,
        {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": _REASONING_PROMPT}],
            "max_tokens": 1024,
            "stream": True,
        },
    )
    label = "openai_chat/deepseek-v4-pro/reasoning_content"
    assert response is not None and response.content
    assert response.reasoning_content, f"{label}: reasoning_content empty"
    assert len(response.reasoning_content) > 30, f"{label}: reasoning_content too short ({len(response.reasoning_content)} chars)"


def test_openai_responses_streaming_reasoning_e2e_northgate_gpt52():
    """OpenAI Responses with reasoning effort: aggregator handles
    reasoning items in the stream without dropping content.

    The northgate gpt-5.x route proxies the Responses API but the
    upstream backing it doesn't actually emit reasoning tokens
    (``reasoning_tokens=0`` even with ``effort: medium``), so we
    can't assert on ``usage.reasoning_tokens`` here. What we *can*
    assert is that requesting reasoning doesn't break the aggregator
    chain — ``response.content`` still surfaces normally and the
    final ``ModelResponse`` is well-formed. The Anthropic + Gemini
    reasoning tests above carry the strict reasoning-tokens guarantee.
    """
    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
    response = call_llm_with_openai_responses(
        client,
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": _REASONING_PROMPT}],
            "reasoning": {"effort": "low"},
            "max_output_tokens": 1024,
            "stream": True,
        },
    )
    label = "openai_responses/gpt-5.2/reasoning"
    assert response is not None, f"{label}: ModelResponse is None"
    assert response.content, f"{label}: empty content"
    assert response.usage.total_tokens > 0, f"{label}: total_tokens=0"


# ── Async streaming variants ────────────────────────────────────────
#
# ``call_llm_with_*_async`` mirrors the sync entry points but uses
# ``AsyncOpenAI`` / ``AsyncAnthropic`` / ``httpx.AsyncClient``. They
# were refactored in lockstep with sync (PR-C.2) and need their own
# coverage — same code paths run, but ``async with``/``async for``
# is a separate branch in each ``call_llm_with_X_async`` body.

from nexau.archs.main_sub.execution.llm_caller import (  # noqa: E402
    call_llm_with_anthropic_chat_completion_async,
    call_llm_with_gemini_rest_async,
    call_llm_with_openai_chat_completion_async,
    call_llm_with_openai_responses_async,
)


def test_openai_chat_streaming_async_e2e_northgate_deepseek():
    """Async variant exercises ``async with``/``async for`` branch.

    The 4 ``call_llm_with_*_async`` functions are independent code
    paths (they use ``AsyncOpenAI`` / ``AsyncAnthropic`` /
    ``httpx.AsyncClient`` rather than the sync clients) — a regression
    in just the async variant would slip past the sync-only tests.
    Wrap in ``asyncio.run`` so we don't need a pytest async plugin.
    """
    key = _northgate_key()

    async def _run() -> Any:
        client = openai.AsyncOpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
        return await call_llm_with_openai_chat_completion_async(
            client,
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": _USER_PROMPT}],
                "max_tokens": 32,
                "stream": True,
            },
        )

    response = asyncio.run(_run())
    _assert_minimal_streaming_response(response, label="openai_chat_async/deepseek")


def test_openai_responses_streaming_async_e2e_northgate_gpt52():
    key = _northgate_key()

    async def _run() -> Any:
        client = openai.AsyncOpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
        return await call_llm_with_openai_responses_async(
            client,
            {
                "model": "gpt-5.2",
                "messages": [{"role": "user", "content": _USER_PROMPT}],
                "max_output_tokens": 32,
                "stream": True,
            },
        )

    response = asyncio.run(_run())
    _assert_minimal_streaming_response(response, label="openai_responses_async/gpt-5.2")


def test_anthropic_streaming_async_e2e_northgate_sonnet45():
    import anthropic  # noqa: PLC0415

    key = _northgate_key()
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        base_url=_NORTHGATE_BASE,
        api_key=key,
        api_type="anthropic_chat_completion",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )

    async def _run() -> Any:
        client = anthropic.AsyncAnthropic(api_key=key, base_url=_NORTHGATE_BASE)
        return await call_llm_with_anthropic_chat_completion_async(
            client,
            {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 32,
                "stream": True,
            },
            model_call_params=_ump_model_call_params(llm_config=llm_config),
            llm_config=llm_config,
        )

    response = asyncio.run(_run())
    _assert_minimal_streaming_response(response, label="anthropic_async/sonnet-4-5")


def test_gemini_rest_streaming_async_e2e_gateway_31pro():
    key = _gemini_key()
    llm_config = LLMConfig(
        model=_GEMINI_GATEWAY_MODEL,
        base_url=_GEMINI_GATEWAY_BASE,
        api_key=key,
        api_type="gemini_rest",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )

    async def _run() -> Any:
        return await call_llm_with_gemini_rest_async(
            kwargs={"stream": True},
            model_call_params=_ump_model_call_params(llm_config=llm_config),
            llm_config=llm_config,
        )

    response = _retry_on_transient_gateway_error(lambda: asyncio.run(_run()), label="gemini_rest_async/3.1-flash-lite")
    _assert_minimal_streaming_response(response, label="gemini_rest_async/3.1-flash-lite")


# ── Multi-chunk content (long output) ────────────────────────────────
#
# Short prompts often complete in 1-2 chunks. Long output forces the
# aggregator to handle many delta concatenations + index tracking
# across dozens of frames — a different code path than the
# happy-path tests above (which mostly exercise single-chunk
# completions).

_LONG_PROMPT = "List 10 distinct interesting facts about the planet Mars. Number them 1 through 10, one per line."


def test_openai_chat_streaming_multichunk_e2e_northgate_deepseek():
    """Multi-chunk delta accumulation through the Set A Chat aggregator."""
    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
    response = call_llm_with_openai_chat_completion(
        client,
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": _LONG_PROMPT}],
            "max_tokens": 1024,
            "stream": True,
        },
    )
    label = "openai_chat/deepseek/multichunk"
    assert response is not None and response.content
    # Long enough to guarantee the stream produced many chunks. A
    # regression in delta concatenation (off-by-one, lost chunk,
    # double-append) would surface as content shorter than the
    # request shape implies.
    assert len(response.content) > 200, f"{label}: content too short ({len(response.content)} chars)"
    assert response.usage.completion_tokens > 30, f"{label}: completion_tokens={response.usage.completion_tokens}"


def test_anthropic_streaming_multichunk_e2e_northgate_sonnet45():
    """Multi-chunk delta accumulation through the Set A Anthropic aggregator."""
    import anthropic  # noqa: PLC0415

    key = _northgate_key()
    client = anthropic.Anthropic(api_key=key, base_url=_NORTHGATE_BASE)
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        base_url=_NORTHGATE_BASE,
        api_key=key,
        api_type="anthropic_chat_completion",
        temperature=0.0,
        max_tokens=1024,
        max_retries=1,
        stream=True,
    )
    user_msg = Message(role=Role.USER, content=[TextBlock(text=_LONG_PROMPT)])
    model_call_params = ModelCallParams(
        messages=[user_msg],
        max_tokens=1024,
        force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
        agent_state=None,
        tool_call_mode="auto",
        tools=None,
        api_params={},
        llm_config=llm_config,
    )
    response = call_llm_with_anthropic_chat_completion(
        client,
        {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "stream": True,
        },
        model_call_params=model_call_params,
        llm_config=llm_config,
    )
    label = "anthropic/sonnet-4-5/multichunk"
    assert response is not None and response.content
    assert len(response.content) > 200, f"{label}: content too short ({len(response.content)} chars)"
    assert response.usage.completion_tokens > 30, f"{label}: completion_tokens={response.usage.completion_tokens}"


# ── shutdown_event break path ────────────────────────────────────────
#
# The streaming loop in each ``call_llm_with_*`` polls
# ``model_call_params.shutdown_event`` and breaks out without raising
# when set. The aggregator should still produce a well-formed (possibly
# partial) ``ModelResponse`` from whatever blocks were already sealed
# — silently swallowing the cancellation OR returning a truncated /
# malformed response would confuse the agent loop.


def test_openai_chat_streaming_shutdown_event_break_yields_partial():
    """Setting shutdown_event after some chunks land should produce a
    clean partial ``ModelResponse`` — not raise, not return ``None``,
    not silently swallow the cancellation.

    We let the stream run for ~600 ms (enough to seal at least the
    first delta on a long prompt), then trip ``shutdown_event``. The
    loop in ``call_llm_with_openai_chat_completion`` should break,
    ``aggregator.build()`` runs against partial state, and the
    adapter returns a truncated but well-formed ``ModelResponse``.
    A regression where the break path silently corrupts state
    (missing usage, ``None`` content where there should be text)
    would surface here.
    """
    import threading  # noqa: PLC0415

    key = _northgate_key()
    client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)

    shutdown = threading.Event()
    # Trip shutdown after ~3 s so the upstream has time to set up
    # TLS + ship several deltas. The deepseek-v4-flash route through
    # northgate has variable latency on the first chunk (sometimes
    # >1s), so a short timer races the first chunk and triggers the
    # ``RuntimeError("Chat completion stream did not receive any
    # valid chunks")`` path instead of the partial-response path.
    timer = threading.Timer(3.0, shutdown.set)
    timer.daemon = True
    timer.start()

    llm_config = LLMConfig(
        model="deepseek-v4-flash",
        base_url=_NORTHGATE_OPENAI_BASE,
        api_key=key,
        api_type="openai_chat_completion",
        temperature=0.0,
        max_tokens=2048,
        max_retries=1,
        stream=True,
    )
    user_msg = Message(role=Role.USER, content=[TextBlock(text=_LONG_PROMPT)])
    model_call_params = ModelCallParams(
        messages=[user_msg],
        max_tokens=2048,
        force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
        agent_state=None,
        tool_call_mode="auto",
        tools=None,
        api_params={},
        llm_config=llm_config,
        shutdown_event=shutdown,
    )

    try:
        response = call_llm_with_openai_chat_completion(
            client,
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": _LONG_PROMPT}],
                "max_tokens": 2048,
                "stream": True,
            },
            model_call_params=model_call_params,
            llm_config=llm_config,
        )
    except RuntimeError as exc:
        # Documented contract: if shutdown trips before *any* chunk
        # lands, the OpenAI Chat aggregator's ``build()`` raises
        # ``RuntimeError("...did not receive any valid chunks")`` /
        # ``RuntimeError("Choice 0 was never aggregated...")`` rather
        # than silently fabricating an empty Choice. Either branch is
        # acceptable — both prove the shutdown path doesn't corrupt
        # state. We just need to ensure the exception is the expected
        # aggregator-state error, not something else.
        assert "did not receive" in str(exc) or "never aggregated" in str(exc), f"unexpected RuntimeError after shutdown: {exc}"
        return
    finally:
        timer.cancel()

    # Shutdown landed *after* some chunks (or stream completed first):
    # either way, the aggregator built a well-formed ModelResponse.
    assert response is not None
    assert isinstance(response.content, str)
    assert response.usage is not None


# ── Tracing branch coverage ──────────────────────────────────────────
#
# Each ``call_llm_with_*`` has a ``should_trace and tracer is not None``
# branch that wraps the stream loop in a ``TraceContext``. The Set A
# build() result is serialized into ``trace_ctx.set_outputs(
# _to_serializable_dict(message))`` — a code path the no-tracer tests
# above can't reach. Hit it once per provider with a recording tracer
# + a parent span so the tracing instrumentation actually exercises
# the post-PR-C.2 build() output shape.

from nexau.archs.tracer.context import reset_current_span, set_current_span  # noqa: E402
from nexau.archs.tracer.core import BaseTracer, Span, SpanType  # noqa: E402


class _RecordingTracer(BaseTracer):
    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.end_calls: list[dict[str, Any]] = []

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        self.start_calls.append({"name": name, "span_type": span_type})
        return Span(
            id="span",
            name=name,
            type=span_type,
            parent_id=getattr(parent_span, "id", None),
            inputs=inputs or {},
            attributes=attributes or {},
        )

    def end_span(self, span: Span, outputs: Any = None, error: Exception | None = None, attributes: dict[str, Any] | None = None) -> None:
        self.end_calls.append({"span": span, "outputs": outputs, "error": error, "attributes": attributes})


def _with_traced_span():
    """Context manager: install a parent Span so ``should_trace`` fires."""
    parent = Span(id="parent", name="parent", type=SpanType.AGENT)
    return parent


def test_openai_chat_streaming_traced_e2e_northgate_deepseek():
    """OpenAI Chat traced branch: ``trace_ctx.set_outputs(_to_serializable_dict(completion))``."""
    key = _northgate_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)
    try:
        client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
        response = call_llm_with_openai_chat_completion(
            client,
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": _USER_PROMPT}],
                "max_tokens": 32,
                "stream": True,
            },
            tracer=tracer,
        )
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="openai_chat/deepseek/traced")
    assert any("OpenAI chat.completions.create (stream)" == c["name"] for c in tracer.start_calls)
    end = next(c for c in tracer.end_calls if c["span"].name == "OpenAI chat.completions.create (stream)")
    # build() output serialized — the post-PR-C.2 shape (full ChatCompletion, not flat message dict).
    assert end["outputs"]["choices"][0]["message"]["content"]


def test_openai_responses_streaming_traced_e2e_northgate_gpt52():
    """OpenAI Responses traced branch."""
    key = _northgate_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)
    try:
        client = openai.OpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
        response = call_llm_with_openai_responses(
            client,
            {
                "model": "gpt-5.2",
                "messages": [{"role": "user", "content": _USER_PROMPT}],
                "max_output_tokens": 32,
                "stream": True,
            },
            tracer=tracer,
        )
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="openai_responses/gpt-5.2/traced")
    assert any("OpenAI responses.stream" == c["name"] for c in tracer.start_calls)


def test_anthropic_streaming_traced_e2e_northgate_sonnet45():
    """Anthropic traced branch."""
    import anthropic  # noqa: PLC0415

    key = _northgate_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        base_url=_NORTHGATE_BASE,
        api_key=key,
        api_type="anthropic_chat_completion",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )
    try:
        client = anthropic.Anthropic(api_key=key, base_url=_NORTHGATE_BASE)
        response = call_llm_with_anthropic_chat_completion(
            client,
            {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 32,
                "stream": True,
            },
            model_call_params=_ump_model_call_params(llm_config=llm_config),
            llm_config=llm_config,
            tracer=tracer,
        )
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="anthropic/sonnet-4-5/traced")
    assert any("Anthropic messages.stream" == c["name"] for c in tracer.start_calls)


def test_gemini_rest_streaming_traced_e2e_gateway():
    """Gemini REST traced branch."""
    key = _gemini_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)
    llm_config = LLMConfig(
        model=_GEMINI_GATEWAY_MODEL,
        base_url=_GEMINI_GATEWAY_BASE,
        api_key=key,
        api_type="gemini_rest",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )
    try:
        response = _retry_on_transient_gateway_error(
            lambda: call_llm_with_gemini_rest(
                kwargs={"stream": True},
                model_call_params=_ump_model_call_params(llm_config=llm_config),
                llm_config=llm_config,
                tracer=tracer,
            ),
            label="gemini_rest/3.1-flash-lite/traced",
        )
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="gemini_rest/3.1-flash-lite/traced")
    assert any("Gemini REST streamGenerateContent" == c["name"] for c in tracer.start_calls)


# ── Async traced branch coverage ─────────────────────────────────────
#
# Mirrors the sync traced tests above for ``call_llm_with_*_async``.
# The async paths have their own ``should_trace and tracer is not None``
# branches with ``trace_ctx.set_outputs(_to_serializable_dict(...))``
# calls that PR-C.2 changed; without explicit coverage the patch
# coverage on these ~3 lines per provider went 0%.


def test_openai_chat_streaming_async_traced_e2e_northgate_deepseek():
    key = _northgate_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)

    async def _run() -> Any:
        client = openai.AsyncOpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
        return await call_llm_with_openai_chat_completion_async(
            client,
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": _USER_PROMPT}],
                "max_tokens": 32,
                "stream": True,
            },
            tracer=tracer,
        )

    try:
        response = asyncio.run(_run())
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="openai_chat_async/deepseek/traced")
    assert any("OpenAI chat.completions.create (async stream)" == c["name"] for c in tracer.start_calls)


def test_openai_responses_streaming_async_traced_e2e_northgate_gpt52():
    key = _northgate_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)

    async def _run() -> Any:
        client = openai.AsyncOpenAI(api_key=key, base_url=_NORTHGATE_OPENAI_BASE)
        return await call_llm_with_openai_responses_async(
            client,
            {
                "model": "gpt-5.2",
                "messages": [{"role": "user", "content": _USER_PROMPT}],
                "max_output_tokens": 32,
                "stream": True,
            },
            tracer=tracer,
        )

    try:
        response = asyncio.run(_run())
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="openai_responses_async/gpt-5.2/traced")
    assert any("OpenAI responses.stream (async)" == c["name"] for c in tracer.start_calls)


def test_anthropic_streaming_async_traced_e2e_northgate_sonnet45():
    import anthropic  # noqa: PLC0415

    key = _northgate_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        base_url=_NORTHGATE_BASE,
        api_key=key,
        api_type="anthropic_chat_completion",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )

    async def _run() -> Any:
        client = anthropic.AsyncAnthropic(api_key=key, base_url=_NORTHGATE_BASE)
        return await call_llm_with_anthropic_chat_completion_async(
            client,
            {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 32,
                "stream": True,
            },
            model_call_params=_ump_model_call_params(llm_config=llm_config),
            llm_config=llm_config,
            tracer=tracer,
        )

    try:
        response = asyncio.run(_run())
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="anthropic_async/sonnet-4-5/traced")
    assert any("Anthropic messages.stream (async)" == c["name"] for c in tracer.start_calls)


def test_gemini_rest_streaming_async_traced_e2e_gateway():
    key = _gemini_key()
    tracer = _RecordingTracer()
    parent = _with_traced_span()
    token = set_current_span(parent)
    llm_config = LLMConfig(
        model=_GEMINI_GATEWAY_MODEL,
        base_url=_GEMINI_GATEWAY_BASE,
        api_key=key,
        api_type="gemini_rest",
        temperature=0.0,
        max_tokens=32,
        max_retries=1,
        stream=True,
    )

    async def _run() -> Any:
        return await call_llm_with_gemini_rest_async(
            kwargs={"stream": True},
            model_call_params=_ump_model_call_params(llm_config=llm_config),
            llm_config=llm_config,
            tracer=tracer,
        )

    try:
        response = _retry_on_transient_gateway_error(
            lambda: asyncio.run(_run()),
            label="gemini_rest_async/3.1-flash-lite/traced",
        )
    finally:
        reset_current_span(token)

    _assert_minimal_streaming_response(response, label="gemini_rest_async/3.1-flash-lite/traced")
    assert any("Gemini REST streamGenerateContent (async)" == c["name"] for c in tracer.start_calls)
