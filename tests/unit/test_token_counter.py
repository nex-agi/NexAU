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

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pytest import MonkeyPatch

from nexau.archs.main_sub.utils import token_counter
from nexau.archs.main_sub.utils.token_counter import TokenCounter
from nexau.core.messages import ImageBlock, Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock


def _approximate(text: str) -> int:
    if not text:
        return 0
    return max((len(text) + 3) // 4, 1)


def test_count_tokens_rejects_legacy_dict_messages() -> None:
    counter = TokenCounter(strategy="fallback")
    legacy_messages: Any = [{"role": "user", "content": "hello"}]

    with pytest.raises(TypeError, match=r"only accepts Sequence\[Message\]"):
        counter.count_tokens(legacy_messages)


def test_fallback_counter_counts_ump_blocks_and_tools() -> None:
    counter = TokenCounter(strategy="fallback")

    messages = [
        Message(
            role=Role.USER,
            content=[
                TextBlock(text="hello"),
                ReasoningBlock(text="think"),
                ImageBlock(base64="AAAA", mime_type="image/png"),
            ],
        ),
        Message(
            role=Role.ASSISTANT,
            content=[
                ToolUseBlock(
                    id="call_1",
                    name="search",
                    input={"q": "weather"},
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content="sunny",
                )
            ],
        ),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"type": "object"},
            },
        }
    ]

    tool_use_args = json.dumps({"q": "weather"}, ensure_ascii=False, sort_keys=True)
    tool_schema = json.dumps(tools[0], ensure_ascii=False, sort_keys=True)

    expected = 0
    expected += 3 + _approximate("user")
    expected += 1 + _approximate("hello")
    expected += 1 + _approximate("think")
    expected += 1 + 85

    expected += 3 + _approximate("assistant")
    expected += 8 + _approximate("call_1") + _approximate("search") + _approximate(tool_use_args)

    expected += 3 + _approximate("tool")
    expected += 6 + _approximate("call_1") + _approximate("sunny")

    expected += 4 + _approximate(tool_schema)

    assert counter.count_tokens(messages, tools=tools) == expected


def test_fallback_counter_enforces_minimum_token() -> None:
    counter = TokenCounter(strategy="fallback")

    assert counter.count_tokens([]) == 1


def test_tiktoken_uses_model_fallback_order(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    class DummyEncoding:
        def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
            return [0] * len(text)

    def encoding_for_model(model: str) -> DummyEncoding:
        calls.append(f"encoding_for_model:{model}")
        raise KeyError("unknown model")

    def get_encoding(name: str) -> DummyEncoding:
        calls.append(f"get_encoding:{name}")
        if name == "o200k_base":
            return DummyEncoding()
        raise KeyError(name)

    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(
        token_counter,
        "tiktoken",
        SimpleNamespace(encoding_for_model=encoding_for_model, get_encoding=get_encoding),
    )

    counter = TokenCounter(strategy="tiktoken", model="custom-model")
    result = counter.count_tokens([Message.user("hello")])

    assert result > 0
    assert calls == [
        "encoding_for_model:custom-model",
        "get_encoding:o200k_base",
    ]


def test_tiktoken_falls_back_to_character_estimator_when_no_encoder(monkeypatch: MonkeyPatch) -> None:
    def encoding_for_model(model: str) -> Any:
        raise KeyError(model)

    def get_encoding(name: str) -> Any:
        raise KeyError(name)

    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(
        token_counter,
        "tiktoken",
        SimpleNamespace(encoding_for_model=encoding_for_model, get_encoding=get_encoding),
    )

    counter = TokenCounter(strategy="tiktoken", model="unknown-model")

    assert counter.count_tokens([Message.user("abcd")]) == 6


def test_tiktoken_regex_backtracking_uses_chunk_fallback(monkeypatch: MonkeyPatch) -> None:
    class RegexFailEncoding:
        def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
            raise ValueError("Regex error while tokenizing")

    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(
        token_counter,
        "tiktoken",
        SimpleNamespace(
            encoding_for_model=lambda _model: RegexFailEncoding(),
            get_encoding=lambda _name: RegexFailEncoding(),
        ),
    )

    counter = TokenCounter(strategy="tiktoken", model="regex-model")
    long_text = "x" * 9000

    expected = 3 + _approximate("user") + 1 + _approximate(long_text)
    assert counter.count_tokens([Message.user(long_text)]) == expected
