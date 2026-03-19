from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import ClassVar, Protocol, cast


def _int(value: object) -> int:
    """Safely coerce a value to int, returning 0 on failure.

    Non-finite floats (NaN, Infinity, -Infinity) return 0, matching opencode's
    ``safe()`` semantics that guard against provider anomalies.
    """
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        return int(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    return 0


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Immutable token usage snapshot aligned with opencode accounting semantics.

    Frontend display formulas:
    - total input including cache = input_tokens + cache_read_tokens + cache_creation_tokens
    - total output including reasoning = completion_tokens + reasoning_tokens

    Field meanings:
    - input_tokens: non-cached input
    - cache_read_tokens: cached input read/hit tokens
    - cache_creation_tokens: cache write tokens
    - completion_tokens: raw provider output token count. For OpenAI this **includes**
      reasoning tokens; for Anthropic it does **not** include reasoning.
    - reasoning_tokens: reasoning output tokens (extracted separately)

    Notes:
    - total_tokens is provider/session total semantics, not a frontend "input total" or
      "output total" field.
    - input_tokens_uncached is a compatibility alias of input_tokens.
    """

    input_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    input_tokens_uncached: int = 0

    _LEGACY_ALIASES: ClassVar[dict[str, str]] = {
        "cached_tokens": "cache_read_tokens",
    }

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            input_tokens_uncached=self.input_tokens_uncached + other.input_tokens_uncached,
        )

    def context_used_tokens(self) -> int:
        """Return prompt-context usage aligned with opencode session updates."""
        return self.input_tokens + self.cache_read_tokens

    def session_total_tokens(self) -> int:
        """Return session/run totals including reasoning and cache dimensions."""
        return self.input_tokens + self.completion_tokens + self.reasoning_tokens + self.cache_read_tokens + self.cache_creation_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def __getitem__(self, key: str) -> int:
        resolved_key = self._LEGACY_ALIASES.get(key, key)
        usage_dict = self.to_dict()
        if resolved_key in usage_dict:
            return usage_dict[resolved_key]
        raise KeyError(key)

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default


_CACHE_WRITE_FIELDS: tuple[str, ...] = (
    "cache_creation_input_tokens",
    "cache_write_input_tokens",
    "cache_write_5m_input_tokens",
    "cache_write_1h_input_tokens",
)
_CACHE_READ_FIELDS: tuple[str, ...] = ("cache_read_input_tokens",)
_PROMPT_DETAIL_FIELDS: tuple[str, ...] = ("prompt_tokens_details", "input_tokens_details")
_COMPLETION_DETAIL_FIELDS: tuple[str, ...] = ("completion_tokens_details", "output_tokens_details")


def _has_key(mapping: Mapping[str, object], key: str) -> bool:
    return key in mapping and mapping.get(key) is not None


def _details_dict(raw_usage: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    for key in keys:
        value = raw_usage.get(key)
        if isinstance(value, Mapping):
            return dict(cast(Mapping[str, object], value))
    return {}


def _sum_int_fields(mapping: Mapping[str, object], keys: tuple[str, ...]) -> int:
    return sum(_int(mapping.get(key)) for key in keys if _has_key(mapping, key))


def _first_int_field(mapping: Mapping[str, object], keys: tuple[str, ...]) -> int:
    for key in keys:
        if _has_key(mapping, key):
            return _int(mapping.get(key))
    return 0


def _cached_input_tokens(raw_usage: Mapping[str, object], details: Mapping[str, object]) -> int:
    if _has_key(raw_usage, "cache_read_input_tokens"):
        return _int(raw_usage.get("cache_read_input_tokens"))
    if _has_key(raw_usage, "cached_tokens"):
        return _int(raw_usage.get("cached_tokens"))
    if _has_key(details, "cached_tokens"):
        return _int(details.get("cached_tokens"))
    return 0


def _adjust_input_tokens(raw_input_tokens: int, *, cache_read_tokens: int, cache_write_tokens: int, excludes_cached_tokens: bool) -> int:
    if excludes_cached_tokens:
        return raw_input_tokens
    return max(raw_input_tokens - cache_read_tokens - cache_write_tokens, 0)


def _resolve_total_tokens(
    raw_usage: Mapping[str, object],
    *,
    input_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> int:
    if _has_key(raw_usage, "total_tokens"):
        return _int(raw_usage.get("total_tokens"))
    if _has_key(raw_usage, "totalTokenCount"):
        return _int(raw_usage.get("totalTokenCount"))
    return input_tokens + completion_tokens + cache_read_tokens + cache_write_tokens


def _build_usage(
    *,
    input_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    total_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        input_tokens_uncached=input_tokens,
    )


class UsageConverter(Protocol):
    """Convert provider-specific usage payloads into TokenUsage."""

    def convert(self, raw_usage: dict[str, object]) -> TokenUsage: ...


class OpenAIChatUsageConverter:
    def convert(self, raw_usage: dict[str, object]) -> TokenUsage:
        completion_details = _details_dict(raw_usage, _COMPLETION_DETAIL_FIELDS)
        prompt_details = _details_dict(raw_usage, _PROMPT_DETAIL_FIELDS)
        raw_input_tokens = _int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0)))
        completion_tokens = _int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0)))
        cache_read_tokens = _cached_input_tokens(raw_usage, prompt_details)
        cache_write_tokens = _first_int_field(raw_usage, _CACHE_WRITE_FIELDS)
        input_tokens = _adjust_input_tokens(
            raw_input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            excludes_cached_tokens=False,
        )
        reasoning_tokens = _int(completion_details.get("reasoning_tokens", 0))
        total_tokens = _resolve_total_tokens(
            raw_usage,
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        return _build_usage(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cache_creation_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
        )


class OpenAIResponsesUsageConverter:
    def convert(self, raw_usage: dict[str, object]) -> TokenUsage:
        output_details = _details_dict(raw_usage, ("output_tokens_details",))
        input_details = _details_dict(raw_usage, ("input_tokens_details",))
        raw_input_tokens = _int(raw_usage.get("input_tokens", 0))
        completion_tokens = _int(raw_usage.get("output_tokens", 0))
        cache_read_tokens = _cached_input_tokens(raw_usage, input_details)
        cache_write_tokens = _first_int_field(raw_usage, _CACHE_WRITE_FIELDS)
        input_tokens = _adjust_input_tokens(
            raw_input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            excludes_cached_tokens=False,
        )
        reasoning_tokens = _int(output_details.get("reasoning_tokens", 0))
        total_tokens = _resolve_total_tokens(
            raw_usage,
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        return _build_usage(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cache_creation_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
        )


class AnthropicUsageConverter:
    def convert(self, raw_usage: dict[str, object]) -> TokenUsage:
        input_tokens = _int(raw_usage.get("input_tokens", 0))
        cache_creation_tokens = _first_int_field(raw_usage, _CACHE_WRITE_FIELDS)
        cache_read_tokens = _sum_int_fields(raw_usage, _CACHE_READ_FIELDS)
        completion_tokens = _int(raw_usage.get("output_tokens", 0))
        total_tokens = _resolve_total_tokens(
            raw_usage,
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_creation_tokens,
        )
        return _build_usage(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=0,
            total_tokens=total_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        )


class GeminiUsageConverter:
    def convert(self, raw_usage: dict[str, object]) -> TokenUsage:
        input_tokens = _int(raw_usage.get("promptTokenCount", 0))
        completion_tokens = _int(raw_usage.get("candidatesTokenCount", 0))
        reasoning_tokens = _int(raw_usage.get("thoughtsTokenCount", 0))
        cache_read_tokens = _int(raw_usage.get("cachedContentTokenCount", 0))
        total_tokens = _resolve_total_tokens(
            raw_usage,
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=0,
        )
        return _build_usage(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
        )


class UsageConverterRegistry:
    """Registry for built-in and custom usage converters."""

    _converters: ClassVar[dict[str, UsageConverter]] = {
        "openai_chat_completion": OpenAIChatUsageConverter(),
        "openai_responses": OpenAIResponsesUsageConverter(),
        "anthropic_chat_completion": AnthropicUsageConverter(),
        "gemini_rest": GeminiUsageConverter(),
    }

    @classmethod
    def register(cls, api_type: str, converter: UsageConverter) -> None:
        cls._converters[api_type] = converter

    @classmethod
    def get(cls, api_type: str) -> UsageConverter | None:
        return cls._converters.get(api_type)


def fallback_normalize_usage(raw_usage: dict[str, object]) -> TokenUsage:
    """Best-effort normalization for unregistered api_type payloads.

    Default to opencode's non-Anthropic/Bedrock assumption: unknown providers report
    input totals including cached tokens, so cache read/write is subtracted unless a
    registered converter overrides the rule.
    """

    prompt_details = _details_dict(raw_usage, _PROMPT_DETAIL_FIELDS)
    completion_details = _details_dict(raw_usage, _COMPLETION_DETAIL_FIELDS)
    raw_input_tokens = _int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)))
    cache_creation_tokens = _first_int_field(raw_usage, _CACHE_WRITE_FIELDS)
    cache_read_tokens = _cached_input_tokens(raw_usage, prompt_details)
    input_tokens = _adjust_input_tokens(
        raw_input_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_creation_tokens,
        excludes_cached_tokens=False,
    )
    reasoning_tokens = _int(completion_details.get("reasoning_tokens", raw_usage.get("reasoning_tokens", 0)))
    completion_tokens = _int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0)))
    total_tokens = _resolve_total_tokens(
        raw_usage,
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_creation_tokens,
    )
    return _build_usage(
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )
