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

"""Unit tests for token counting utilities."""

import json
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from pytest import MonkeyPatch

from nexau.archs.main_sub.utils import token_counter
from nexau.archs.main_sub.utils.token_counter import TokenCounter
from nexau.core.messages import ImageBlock, Message, Role, TextBlock


def test_fallback_counter_counts_content_and_tools():
    """Ensure fallback strategy counts message text and tool call overhead."""
    counter = TokenCounter(strategy="fallback")
    messages = [
        {
            "role": "user",
            "content": "hello world",
            "reasoning_content": "abcde",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": {"query": "hi"},
                    },
                },
            ],
        },
    ]

    result = counter.count_tokens(messages)

    # role=4 ->1 token, content=11 ->2, reasoning=5 ->1, overhead=4
    # tool call: base 3 + name(6)->1 + args('{"query": "hi"}'=15)->3
    assert result == 15


def test_fallback_counter_enforces_minimum_token():
    """Fallback counter should never return zero tokens."""
    counter = TokenCounter(strategy="fallback")

    assert counter.count_tokens([]) == 1


def test_tiktoken_strategy_uses_stub_encoding(monkeypatch: MonkeyPatch):
    """Verify tiktoken strategy counts tokens via provided encoding."""

    class DummyEncoding:
        def encode(
            self,
            text: str,
            allowed_special: Iterable[str] | None = None,
        ) -> list[int]:
            return [0] * len(text)

    def dummy_encoding_for_model(model: str):
        return DummyEncoding()

    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(
        token_counter,
        "tiktoken",
        SimpleNamespace(encoding_for_model=dummy_encoding_for_model),
    )

    counter = TokenCounter(strategy="tiktoken", model="dummy-model")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "hi",
            "reasoning_content": "why",
            "tool_calls": [
                {
                    "function": {
                        "name": "do",
                        "arguments": {"x": 1},
                    },
                },
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "do",
                "description": "test",
            },
        },
    ]

    result = counter.count_tokens(messages, tools=tools)

    tool_args_str = json.dumps({"x": 1})
    tool_def_str = json.dumps(tools[0])
    expected = (
        len("hi")  # message content
        + len("why")  # reasoning content
        + 3  # tool call overhead
        + len("do")  # tool name
        + len(tool_args_str)  # tool arguments
        + len("ok")  # second message content
        + len(tool_def_str)  # tool definition
    )

    assert result == expected


def test_tiktoken_strategy_supports_multimodal_message_content(monkeypatch: MonkeyPatch):
    """tiktoken strategy should not crash when legacy content is a list (e.g. image + text)."""

    class DummyEncoding:
        def encode(
            self,
            text: str,
            allowed_special: Iterable[str] | None = None,
        ) -> list[int]:
            return [0] * len(text)

    def dummy_encoding_for_model(model: str):
        return DummyEncoding()

    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(
        token_counter,
        "tiktoken",
        SimpleNamespace(encoding_for_model=dummy_encoding_for_model),
    )

    counter = TokenCounter(strategy="tiktoken", model="dummy-model")
    messages = [
        Message(
            role=Role.USER,
            content=[
                TextBlock(text="hello"),
                ImageBlock(base64="AAAA", mime_type="image/png"),
            ],
        ),
    ]

    # Adapter emits [{"type":"text","text":"hello"},{"type":"image_url",...}]; token counter should coerce to "hello<image>".
    assert counter.count_tokens(messages) == len("hello<image>")


def test_tiktoken_strategy_falls_back_on_encoder_error(monkeypatch: MonkeyPatch):
    """When tiktoken encoder fails to initialize, fallback counter is used."""

    def broken_encoding_for_model(model: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(
        token_counter,
        "tiktoken",
        SimpleNamespace(encoding_for_model=broken_encoding_for_model),
    )

    counter = TokenCounter(strategy="tiktoken", model="broken")
    messages = [{"role": "user", "content": "abcd"}]

    # Fallback uses len(role)//4 + len(content)//4 + overhead 4 -> 1 + 1 + 4
    assert counter.count_tokens(messages) == 6


def test_fallback_counter_skips_invalid_tool_function():
    """Invalid tool function payload still counts only base overhead."""
    counter = TokenCounter(strategy="fallback")
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": "bad"}]},
    ]

    # role len=9 ->2 tokens, overhead 4, tool call base 3 => total 9
    assert counter.count_tokens(messages) == 9


def test_fallback_counter_counts_tool_definitions():
    """Fallback should include tools parameter in token count."""
    counter = TokenCounter(strategy="fallback")
    tools = [{"name": "tool", "description": "abcd"}]
    expected_tools_tokens = len(json.dumps(tools[0])) // 4

    assert counter.count_tokens([], tools=tools) == max(expected_tools_tokens, 1)
