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

import base64
import json
import math
import struct
from collections.abc import Callable
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
    # ImageBlock(base64="AAAA") decodes to 3 bytes — no parseable header — so
    # it charges the conservative fallback, not a real pixel-area estimate.
    expected += 1 + token_counter._FALLBACK_IMAGE_TOKENS

    expected += 3 + _approximate("assistant")
    expected += 8 + _approximate("call_1") + _approximate("search") + _approximate(tool_use_args)

    expected += 3 + _approximate("tool")
    expected += 6 + _approximate("call_1") + _approximate("sunny")

    expected += 4 + _approximate(tool_schema)

    assert counter.count_tokens(messages, tools=tools) == expected


def test_fallback_counter_enforces_minimum_token() -> None:
    counter = TokenCounter(strategy="fallback")

    assert counter.count_tokens([]) == 1


# ── Image token estimation (incident fix; Rust counterpart nexau-rs#94) ──────


def _png_header(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


def _gif_header(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height)


def _bmp_header(width: int, height: int) -> bytes:
    # "BM" + 16 filler bytes, then signed 32-bit LE width/height at offset 18.
    return b"BM" + b"\x00" * 16 + struct.pack("<ii", width, height)


def _jpeg_header(width: int, height: int) -> bytes:
    # SOI, then a SOF0 segment carrying [precision:1][height:2 BE][width:2 BE].
    sof0 = b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", height, width) + b"\x03\x01\x22\x00"
    return b"\xff\xd8" + sof0


def _image_block(raw: bytes, mime_type: str) -> ImageBlock:
    return ImageBlock(base64=base64.b64encode(raw).decode("ascii"), mime_type=mime_type)


def _expected_patch_tokens(width: int, height: int) -> int:
    # Official Anthropic patch formula — one token per 28x28-pixel patch.
    # Pins the shared IMAGE_TOKEN_PATCH_SIZE so a drift fails loudly.
    return math.ceil(width / 28) * math.ceil(height / 28)


@pytest.mark.parametrize(
    ("header_factory", "mime_type"),
    [
        (_png_header, "image/png"),
        (_gif_header, "image/gif"),
        (_bmp_header, "image/bmp"),
        (_jpeg_header, "image/jpeg"),
    ],
)
def test_estimate_image_tokens_uses_real_pixel_dimensions(header_factory: Callable[[int, int], bytes], mime_type: str) -> None:
    # 1000x1000 → ceil(1000/28)^2 = 36^2 = 1296 official patch tokens.
    block = _image_block(header_factory(1000, 1000), mime_type)
    assert token_counter._estimate_image_tokens(block) == 1296
    assert token_counter._estimate_image_tokens(block) == _expected_patch_tokens(1000, 1000)


def test_estimate_image_tokens_scales_with_both_axes_for_tall_image() -> None:
    # A tall-narrow image (long screenshot) is charged over both axes' patches.
    block = _image_block(_jpeg_header(500, 6000), "image/jpeg")
    assert token_counter._estimate_image_tokens(block) == _expected_patch_tokens(500, 6000)


def test_estimate_image_tokens_url_only_uses_conservative_fallback() -> None:
    # No local bytes to probe → conservative fallback, not the old flat 85.
    block = ImageBlock(url="https://example.com/pic.png")
    assert token_counter._estimate_image_tokens(block) == token_counter._FALLBACK_IMAGE_TOKENS


def test_estimate_image_tokens_undecodable_bytes_use_fallback() -> None:
    block = _image_block(b"not an image at all", "image/png")
    assert token_counter._estimate_image_tokens(block) == token_counter._FALLBACK_IMAGE_TOKENS


def test_estimate_image_tokens_truncated_jpeg_before_sof_uses_fallback() -> None:
    # SOI present but SOF never reached within the bytes → fallback, no crash.
    block = _image_block(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00", "image/jpeg")
    assert token_counter._estimate_image_tokens(block) == token_counter._FALLBACK_IMAGE_TOKENS


def test_count_tokens_charges_image_by_patch_formula_end_to_end() -> None:
    counter = TokenCounter(strategy="fallback")
    block = _image_block(_png_header(1200, 800), "image/png")
    messages = [Message(role=Role.USER, content=[block])]

    expected = 3 + _approximate("user") + 1 + _expected_patch_tokens(1200, 800)

    assert counter.count_tokens(messages) == expected


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
