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

"""Token counting utilities for agents."""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, cast

from nexau.core.messages import ImageBlock, Message, ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock

logger = logging.getLogger(__name__)

TokenCounterFn = Callable[[Sequence[Message], list[dict[str, Any]] | None], int]

_IMAGE_TOKEN_ESTIMATE: Final[int] = 85
_MESSAGE_OVERHEAD_TOKENS: Final[int] = 3
_BLOCK_OVERHEAD_TOKENS: Final[int] = 1
_TOOL_USE_OVERHEAD_TOKENS: Final[int] = 8
_TOOL_RESULT_OVERHEAD_TOKENS: Final[int] = 6
_TOOL_DEFINITION_OVERHEAD_TOKENS: Final[int] = 4
_TIKTOKEN_SAFE_CHUNK_SIZE: Final[int] = 8192

_tiktoken: Any
try:
    import tiktoken as _tiktoken
except ImportError:
    _tiktoken = None

tiktoken: Any | None = _tiktoken
TIKTOKEN_AVAILABLE: Final[bool] = _tiktoken is not None


class TokenCounter:
    """Handles token counting for UMP messages."""

    def __init__(self, strategy: str = "fallback", model: str = "gpt-4o"):
        """Initialize token counter with specified strategy.

        Args:
            strategy: "tiktoken" or "fallback"
            model: Model name for tiktoken encoding resolution
        """
        self.strategy = strategy
        self.model = model
        self._counter = self._create_counter()

    def set_counter(self, counter: TokenCounterFn) -> None:
        """Override token counting function for custom integration points."""
        self._counter = counter

    def _create_counter(self) -> TokenCounterFn:
        """Create the appropriate token counter based on strategy."""
        if self.strategy == "tiktoken":
            if not TIKTOKEN_AVAILABLE:
                logger.warning("tiktoken not available, using fallback counter")
                return self._create_fallback_counter()

            encoding = self._resolve_tiktoken_encoding()
            if encoding is not None:
                return self._create_tiktoken_counter(encoding)

            logger.warning("Failed to resolve tiktoken encoding, using fallback counter")
            return self._create_fallback_counter()

        if self.strategy != "fallback":
            logger.warning("Unknown token counter strategy '%s', using fallback", self.strategy)
        return self._create_fallback_counter()

    def _resolve_tiktoken_encoding(self) -> Any | None:
        """Resolve tiktoken encoding with model-aware fallback order."""
        if tiktoken is None:
            return None

        attempts: list[tuple[str, Callable[[], Any]]] = []

        encoding_for_model = getattr(tiktoken, "encoding_for_model", None)
        if callable(encoding_for_model):
            attempts.append((f"encoding_for_model({self.model})", lambda: encoding_for_model(self.model)))

        get_encoding = getattr(tiktoken, "get_encoding", None)
        if callable(get_encoding):
            attempts.append(("get_encoding(o200k_base)", lambda: get_encoding("o200k_base")))
            attempts.append(("get_encoding(cl100k_base)", lambda: get_encoding("cl100k_base")))

        errors: list[str] = []
        for label, resolver in attempts:
            try:
                encoding = resolver()
                if label != f"encoding_for_model({self.model})":
                    logger.info("Using tiktoken fallback encoder via %s", label)
                return encoding
            except Exception as exc:  # pragma: no cover - defensive against tiktoken internals
                errors.append(f"{label}: {exc}")

        if errors:
            logger.warning("tiktoken encoder resolution failed for model '%s': %s", self.model, "; ".join(errors))
        return None

    def _create_tiktoken_counter(self, encoding: Any) -> TokenCounterFn:
        """Create tiktoken-based counter."""

        def encode_text(text: str) -> int:
            return self._encode_with_tiktoken(text, encoding)

        def tiktoken_message_counter(messages: Sequence[Message], tools: list[dict[str, Any]] | None = None) -> int:
            return self._count_tokens_with_text_encoder(messages, tools, encode_text)

        return tiktoken_message_counter

    def _create_fallback_counter(self) -> TokenCounterFn:
        """Create fallback counter using character approximation."""

        def fallback_message_counter(messages: Sequence[Message], tools: list[dict[str, Any]] | None = None) -> int:
            total = self._count_tokens_with_text_encoder(messages, tools, self._approximate_text_tokens)
            return max(total, 1)

        return fallback_message_counter

    @staticmethod
    def _approximate_text_tokens(text: str) -> int:
        if not text:
            return 0
        # Character approximation: roughly one token per four characters.
        return max((len(text) + 3) // 4, 1)

    @staticmethod
    def _is_regex_tokenization_error(exc: Exception) -> bool:
        return "Regex error while tokenizing" in str(exc)

    def _encode_with_tiktoken(self, text: str, encoding: Any) -> int:
        if not text:
            return 0

        try:
            return len(encoding.encode(text, allowed_special=set()))
        except ValueError as exc:
            if not self._is_regex_tokenization_error(exc):
                raise

            logger.warning(
                "tiktoken regex backtracking hit while counting %d chars; falling back to chunked token counting",
                len(text),
            )
            return self._encode_with_chunked_tiktoken(text, encoding)

    def _encode_with_chunked_tiktoken(self, text: str, encoding: Any) -> int:
        total_tokens = 0

        for start in range(0, len(text), _TIKTOKEN_SAFE_CHUNK_SIZE):
            chunk = text[start : start + _TIKTOKEN_SAFE_CHUNK_SIZE]
            try:
                total_tokens += len(encoding.encode(chunk, allowed_special=set()))
            except ValueError as exc:
                if not self._is_regex_tokenization_error(exc):
                    raise
                total_tokens += self._approximate_text_tokens(chunk)

        return total_tokens

    def _count_tokens_with_text_encoder(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None,
        text_encoder: Callable[[str], int],
    ) -> int:
        total_tokens = 0

        for index, raw_message in enumerate(cast(Sequence[Any], messages)):
            if not isinstance(raw_message, Message):
                self._raise_invalid_message_type(index, raw_message)
            message = raw_message

            total_tokens += _MESSAGE_OVERHEAD_TOKENS
            total_tokens += text_encoder(message.role.value)

            for block in message.content:
                total_tokens += self._count_block_tokens(block, text_encoder)

        if tools:
            for tool in tools:
                total_tokens += _TOOL_DEFINITION_OVERHEAD_TOKENS
                total_tokens += text_encoder(self._serialize_payload(tool))

        return total_tokens

    @staticmethod
    def _raise_invalid_message_type(index: int, value: Any) -> None:
        if isinstance(value, Mapping):
            raise TypeError(
                "TokenCounter.count_tokens now only accepts Sequence[Message]. "
                "Legacy dict messages are no longer supported. "
                "Please migrate callers to pass nexau.core.messages.Message objects "
                f"(got dict-like value at index {index})."
            )

        raise TypeError(f"TokenCounter.count_tokens now only accepts Sequence[Message]. Got {type(value).__name__} at index {index}.")

    def _count_block_tokens(self, block: Any, text_encoder: Callable[[str], int]) -> int:
        if isinstance(block, TextBlock):
            return _BLOCK_OVERHEAD_TOKENS + text_encoder(block.text)

        if isinstance(block, ReasoningBlock):
            total = _BLOCK_OVERHEAD_TOKENS + text_encoder(block.text)
            if block.signature:
                total += text_encoder(block.signature)
            if block.redacted_data:
                total += text_encoder(block.redacted_data)
            return total

        if isinstance(block, ImageBlock):
            return _BLOCK_OVERHEAD_TOKENS + _IMAGE_TOKEN_ESTIMATE

        if isinstance(block, ToolUseBlock):
            total = _TOOL_USE_OVERHEAD_TOKENS
            total += text_encoder(block.id)
            total += text_encoder(block.name)

            raw_arguments = block.raw_input if block.raw_input is not None else self._serialize_payload(block.input)
            total += text_encoder(raw_arguments)
            return total

        if isinstance(block, ToolResultBlock):
            total = _TOOL_RESULT_OVERHEAD_TOKENS + text_encoder(block.tool_use_id)
            total += self._count_tool_result_content_tokens(block.content, text_encoder)
            if block.is_error:
                total += _BLOCK_OVERHEAD_TOKENS
            return total

        # Best-effort fallback for future/unknown block types.
        model_dump_json = getattr(block, "model_dump_json", None)
        if callable(model_dump_json):
            try:
                dumped = model_dump_json(exclude_none=True)
                return _BLOCK_OVERHEAD_TOKENS + text_encoder(str(dumped))
            except Exception:  # pragma: no cover - defensive
                pass

        return _BLOCK_OVERHEAD_TOKENS + text_encoder(str(block))

    def _count_tool_result_content_tokens(
        self,
        content: str | list[TextBlock | ImageBlock],
        text_encoder: Callable[[str], int],
    ) -> int:
        if isinstance(content, str):
            return text_encoder(content)

        total = 0
        for block in content:
            if isinstance(block, TextBlock):
                total += _BLOCK_OVERHEAD_TOKENS + text_encoder(block.text)
            else:
                total += _BLOCK_OVERHEAD_TOKENS + _IMAGE_TOKEN_ESTIMATE
        return total

    @staticmethod
    def _serialize_payload(payload: Any) -> str:
        payload_value: Any = payload

        if not isinstance(payload_value, Mapping):
            model_dump = getattr(payload_value, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump(mode="python", exclude_none=True)
                    if isinstance(dumped, Mapping):
                        payload_value = cast(Mapping[str, Any], dumped)
                except Exception:  # pragma: no cover - defensive
                    pass

        if isinstance(payload_value, Mapping):
            try:
                return json.dumps(payload_value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(cast(object, payload_value))

        return str(payload_value)

    def count_tokens(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count total tokens in a list of UMP messages."""
        return self._counter(messages, tools)
