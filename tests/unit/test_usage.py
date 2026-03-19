from __future__ import annotations

from pathlib import Path

import yaml

from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.usage import (
    AnthropicUsageConverter,
    GeminiUsageConverter,
    OpenAIChatUsageConverter,
    OpenAIResponsesUsageConverter,
    TokenUsage,
    UsageConverterRegistry,
    _int,
    fallback_normalize_usage,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "token_usage_regression.yaml"


def test_token_usage_add_to_dict_and_accounting_helpers():
    left = TokenUsage(input_tokens=10, completion_tokens=2, total_tokens=12, cache_read_tokens=3)
    right = TokenUsage(
        input_tokens=5,
        completion_tokens=4,
        reasoning_tokens=6,
        total_tokens=9,
        cache_creation_tokens=7,
    )

    combined = left + right

    assert combined.to_dict() == {
        "input_tokens": 15,
        "completion_tokens": 6,
        "reasoning_tokens": 6,
        "total_tokens": 21,
        "cache_creation_tokens": 7,
        "cache_read_tokens": 3,
        "input_tokens_uncached": 0,
    }
    assert combined.context_used_tokens() == 18
    assert combined.session_total_tokens() == 37


def test_builtin_usage_converters():
    openai_usage = OpenAIChatUsageConverter().convert(
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 15,
        }
    )
    openai_responses_usage = OpenAIResponsesUsageConverter().convert(
        {
            "input_tokens": 12,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 4},
        }
    )
    anthropic_usage = AnthropicUsageConverter().convert(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
        }
    )
    gemini_usage = GeminiUsageConverter().convert(
        {
            "promptTokenCount": 8,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 2,
            "totalTokenCount": 14,
        }
    )

    assert openai_usage == TokenUsage(
        input_tokens=6,
        completion_tokens=5,
        reasoning_tokens=3,
        total_tokens=15,
        cache_read_tokens=4,
        input_tokens_uncached=6,
    )
    assert openai_responses_usage == TokenUsage(
        input_tokens=10,
        completion_tokens=5,
        reasoning_tokens=4,
        total_tokens=17,
        cache_read_tokens=2,
        input_tokens_uncached=10,
    )
    assert anthropic_usage == TokenUsage(
        input_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
        total_tokens=20,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        input_tokens_uncached=10,
    )
    assert gemini_usage == TokenUsage(
        input_tokens=8,
        completion_tokens=4,
        reasoning_tokens=2,
        total_tokens=14,
        input_tokens_uncached=8,
        cache_read_tokens=0,
    )


def test_anthropic_cache_write_uses_first_non_null_source():
    usage = AnthropicUsageConverter().convert(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 7,
            "cache_write_5m_input_tokens": 6,
            "cache_write_1h_input_tokens": 4,
        }
    )

    assert usage == TokenUsage(
        input_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
        total_tokens=22,
        cache_creation_tokens=7,
        cache_read_tokens=0,
        input_tokens_uncached=10,
    )


def test_fallback_normalize_usage_openai_style_cached_input():
    usage = fallback_normalize_usage(
        {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 10},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
    )

    assert usage == TokenUsage(
        input_tokens=70,
        completion_tokens=20,
        reasoning_tokens=3,
        total_tokens=100,
        cache_read_tokens=10,
        input_tokens_uncached=70,
    )


def test_fallback_normalize_usage_defaults_to_subtracting_cache_tokens():
    usage = fallback_normalize_usage(
        {
            "input_tokens": 40,
            "output_tokens": 15,
            "cache_read_input_tokens": 8,
            "cache_write_5m_input_tokens": 6,
            "cache_write_1h_input_tokens": 4,
            "reasoning_tokens": 2,
        }
    )

    assert usage == TokenUsage(
        input_tokens=26,
        completion_tokens=15,
        reasoning_tokens=2,
        total_tokens=55,
        cache_creation_tokens=6,
        cache_read_tokens=8,
        input_tokens_uncached=26,
    )


def test_custom_usage_converter_registration():
    class CustomConverter:
        def convert(self, raw_usage: dict[str, object]) -> TokenUsage:
            raw_in = raw_usage.get("in", 0)
            raw_out = raw_usage.get("out", 0)
            input_tokens = int(raw_in) if isinstance(raw_in, (int, float, str)) else 0
            completion_tokens = int(raw_out) if isinstance(raw_out, (int, float, str)) else 0
            return TokenUsage(
                input_tokens=input_tokens,
                completion_tokens=completion_tokens,
                total_tokens=input_tokens + completion_tokens,
            )

    UsageConverterRegistry.register("custom_gateway", CustomConverter())
    converter = UsageConverterRegistry.get("custom_gateway")
    assert converter is not None
    usage = converter.convert({"in": 7, "out": 5})
    response = ModelResponse(content="ok", usage=usage)

    assert response.usage.total_tokens == 12


def test_token_usage_regression_yaml_fixture_cases():
    fixtures = yaml.safe_load(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(fixtures, list)

    for case in fixtures:
        api_type = case["api_type"]
        raw_usage = case["raw_usage"]
        expected = case["expected"]

        if api_type == "fallback":
            usage = fallback_normalize_usage(raw_usage)
        else:
            converter = UsageConverterRegistry.get(api_type)
            assert converter is not None, api_type
            usage = converter.convert(raw_usage)

        assert usage.to_dict() == expected, case["name"]


def test_int_nan_and_infinity_return_zero():
    """Non-finite floats must return 0, matching opencode safe() semantics."""
    assert _int(float("nan")) == 0
    assert _int(float("inf")) == 0
    assert _int(float("-inf")) == 0
    # Normal floats still convert
    assert _int(3.9) == 3
    assert _int(-1.5) == -1


def test_gemini_cache_token_extraction():
    """GeminiUsageConverter should extract cachedContentTokenCount."""
    usage = GeminiUsageConverter().convert(
        {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 5,
            "cachedContentTokenCount": 30,
            "totalTokenCount": 150,
        }
    )
    assert usage == TokenUsage(
        input_tokens=100,
        completion_tokens=20,
        reasoning_tokens=5,
        total_tokens=150,
        cache_creation_tokens=0,
        cache_read_tokens=30,
        input_tokens_uncached=100,
    )


def test_model_response_to_ump_message_persists_usage_metadata():
    response = ModelResponse.from_openai_message(
        {"role": "assistant", "content": "hello"},
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 15,
        },
    )

    message = response.to_ump_message()

    assert message.metadata["usage"] == {
        "input_tokens": 6,
        "completion_tokens": 5,
        "reasoning_tokens": 2,
        "total_tokens": 15,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 4,
        "input_tokens_uncached": 6,
    }
