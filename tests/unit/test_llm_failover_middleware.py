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

"""Tests for LLM failover middleware."""

from __future__ import annotations

from unittest.mock import MagicMock

import anthropic
import openai
import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.hooks import ModelCallParams
from nexau.archs.main_sub.execution.middleware.llm_failover import (
    CircuitBreakerConfig,
    LLMFailoverMiddleware,
    _CircuitBreaker,
    _extract_status_code,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_params(llm_config: LLMConfig | None = None) -> ModelCallParams:
    """Create minimal ModelCallParams for testing."""
    config = llm_config or LLMConfig(
        model="test-model",
        base_url="https://primary.example.com/v1",
        api_key="sk-primary",
        api_type="openai_chat_completion",
    )
    return ModelCallParams(
        messages=[],
        max_tokens=100,
        force_stop_reason=None,
        agent_state=None,
        tool_call_mode="xml",
        tools=None,
        api_params=config.to_openai_params(),
        openai_client=MagicMock(),
        llm_config=config,
    )


def _make_middleware(**kwargs: object) -> LLMFailoverMiddleware:
    """Create middleware with sensible defaults."""
    defaults: dict[str, object] = {
        "fallback_providers": [
            {
                "name": "backup",
                "llm_config": {
                    "base_url": "https://backup.example.com/v1",
                    "api_key": "sk-backup",
                },
            },
        ],
        "trigger": {
            "status_codes": [500, 502, 503],
            "exception_types": ["ConnectionError"],
        },
    }
    defaults.update(kwargs)
    return LLMFailoverMiddleware(**defaults)  # type: ignore[arg-type]


def _ok_response() -> ModelResponse:
    return ModelResponse(content="ok")


# ---------------------------------------------------------------------------
# _extract_status_code
# ---------------------------------------------------------------------------


class TestExtractStatusCode:
    def test_openai_api_error(self) -> None:
        exc = openai.InternalServerError(
            message="internal",
            response=MagicMock(status_code=500, headers={}),
            body=None,
        )
        assert _extract_status_code(exc) == 500

    def test_anthropic_api_error(self) -> None:
        exc = anthropic.InternalServerError(
            message="internal",
            response=MagicMock(status_code=500, headers={}),
            body=None,
        )
        assert _extract_status_code(exc) == 500

    def test_generic_exception_returns_none(self) -> None:
        assert _extract_status_code(ValueError("boom")) is None


# ---------------------------------------------------------------------------
# _CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = _CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
        assert not cb.should_skip_primary()

    def test_opens_after_threshold(self) -> None:
        cb = _CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=60))
        cb.record_failure()
        assert not cb.should_skip_primary()
        cb.record_failure()
        assert cb.should_skip_primary()

    def test_success_resets(self) -> None:
        cb = _CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.should_skip_primary()
        cb.record_success()
        assert not cb.should_skip_primary()

    def test_half_open_after_timeout(self) -> None:
        cb = _CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0))
        cb.record_failure()
        # recovery_timeout=0 → 立即进入 HALF_OPEN
        assert not cb.should_skip_primary()


# ---------------------------------------------------------------------------
# LLMFailoverMiddleware — construction
# ---------------------------------------------------------------------------


class TestMiddlewareConstruction:
    def test_default_trigger(self) -> None:
        mw = LLMFailoverMiddleware(fallback_providers=[{"name": "fb", "llm_config": {}}])
        assert mw._trigger.status_codes == [500, 502, 503]
        assert mw._trigger.exception_types == []

    def test_custom_trigger(self) -> None:
        mw = _make_middleware(
            trigger={"status_codes": [429], "exception_types": ["RateLimitError"]},
        )
        assert mw._trigger.status_codes == [429]
        assert mw._trigger.exception_types == ["RateLimitError"]

    def test_no_circuit_breaker_by_default(self) -> None:
        mw = _make_middleware()
        assert mw._circuit_breaker is None

    def test_circuit_breaker_created(self) -> None:
        mw = _make_middleware(circuit_breaker={"failure_threshold": 5, "recovery_timeout_seconds": 30})
        assert mw._circuit_breaker is not None
        assert mw._circuit_breaker._threshold == 5


# ---------------------------------------------------------------------------
# LLMFailoverMiddleware — wrap_model_call
# ---------------------------------------------------------------------------


class TestWrapModelCall:
    def test_primary_success_no_failover(self) -> None:
        """Primary succeeds → no fallback attempted."""
        mw = _make_middleware()
        params = _make_params()
        call_count = 0

        def call_next(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            return _ok_response()

        result = mw.wrap_model_call(params, call_next)
        assert result is not None
        assert result.content == "ok"
        assert call_count == 1

    def test_primary_fails_fallback_succeeds(self) -> None:
        """Primary fails with matching error → fallback succeeds."""
        mw = _make_middleware()
        params = _make_params()
        call_count = 0

        def call_next(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 模拟主 provider 500 错误
                raise openai.InternalServerError(
                    message="internal",
                    response=MagicMock(status_code=500, headers={}),
                    body=None,
                )
            return _ok_response()

        result = mw.wrap_model_call(params, call_next)
        assert result is not None
        assert call_count == 2

    def test_non_matching_error_not_caught(self) -> None:
        """Non-matching error is re-raised immediately."""
        mw = _make_middleware()
        params = _make_params()

        def call_next(p: ModelCallParams) -> ModelResponse:
            raise ValueError("unrelated error")

        with pytest.raises(ValueError, match="unrelated error"):
            mw.wrap_model_call(params, call_next)

    def test_all_providers_fail(self) -> None:
        """All providers fail → last exception raised."""
        mw = _make_middleware()
        params = _make_params()

        def call_next(p: ModelCallParams) -> ModelResponse:
            raise openai.InternalServerError(
                message="all down",
                response=MagicMock(status_code=500, headers={}),
                body=None,
            )

        with pytest.raises(openai.InternalServerError):
            mw.wrap_model_call(params, call_next)

    def test_exception_type_matching(self) -> None:
        """Exception type name matching triggers failover."""
        mw = _make_middleware(
            trigger={"status_codes": [], "exception_types": ["ConnectionError"]},
        )
        params = _make_params()
        call_count = 0

        def call_next(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection refused")
            return _ok_response()

        result = mw.wrap_model_call(params, call_next)
        assert result is not None
        assert call_count == 2

    def test_fallback_params_inherit_primary_config(self) -> None:
        """Fallback params inherit model from primary when not overridden."""
        mw = _make_middleware(
            fallback_providers=[
                {
                    "name": "backup",
                    "llm_config": {
                        "base_url": "https://backup.example.com/v1",
                        "api_key": "sk-backup",
                    },
                },
            ],
        )
        primary_config = LLMConfig(
            model="gpt-4",
            base_url="https://primary.example.com/v1",
            api_key="sk-primary",
        )
        params = _make_params(primary_config)
        captured_params: list[ModelCallParams] = []
        call_count = 0

        def call_next(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            captured_params.append(p)
            if call_count == 1:
                raise openai.InternalServerError(
                    message="fail",
                    response=MagicMock(status_code=500, headers={}),
                    body=None,
                )
            return _ok_response()

        mw.wrap_model_call(params, call_next)

        # 验证 fallback params 继承了 primary 的 model
        fallback_config = captured_params[1].llm_config
        assert isinstance(fallback_config, LLMConfig)
        assert fallback_config.model == "gpt-4"
        assert fallback_config.base_url == "https://backup.example.com/v1"
        assert fallback_config.api_key == "sk-backup"

    def test_fallback_does_not_mutate_original_params(self) -> None:
        """Failover must not modify the original ModelCallParams."""
        mw = _make_middleware()
        params = _make_params()
        original_client = params.openai_client
        original_config = params.llm_config
        call_count = 0

        def call_next(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.InternalServerError(
                    message="fail",
                    response=MagicMock(status_code=500, headers={}),
                    body=None,
                )
            return _ok_response()

        mw.wrap_model_call(params, call_next)

        # 原始 params 不应被修改
        assert params.openai_client is original_client
        assert params.llm_config is original_config

    def test_multiple_fallback_providers(self) -> None:
        """Multiple fallbacks tried in order."""
        mw = _make_middleware(
            fallback_providers=[
                {"name": "fb1", "llm_config": {"base_url": "https://fb1.com/v1", "api_key": "sk-1"}},
                {"name": "fb2", "llm_config": {"base_url": "https://fb2.com/v1", "api_key": "sk-2"}},
            ],
        )
        params = _make_params()
        call_count = 0

        def call_next(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise openai.InternalServerError(
                    message="fail",
                    response=MagicMock(status_code=500, headers={}),
                    body=None,
                )
            return _ok_response()

        result = mw.wrap_model_call(params, call_next)
        assert result is not None
        assert call_count == 3  # primary + fb1 + fb2

    def test_circuit_breaker_skips_primary(self) -> None:
        """When circuit breaker is open, primary is skipped."""
        mw = _make_middleware(
            circuit_breaker={"failure_threshold": 1, "recovery_timeout_seconds": 9999},
        )
        params = _make_params()
        call_count = 0
        captured_configs: list[LLMConfig | None] = []

        def call_next(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            captured_configs.append(p.llm_config if isinstance(p.llm_config, LLMConfig) else None)
            if call_count == 1:
                # 第一次调用：primary 失败
                raise openai.InternalServerError(
                    message="fail",
                    response=MagicMock(status_code=500, headers={}),
                    body=None,
                )
            return _ok_response()

        # 第一次调用：primary 失败 → fallback 成功
        mw.wrap_model_call(params, call_next)
        assert call_count == 2

        # 第二次调用：熔断器打开，应该直接走 fallback
        call_count = 0
        captured_configs.clear()

        def call_next_2(p: ModelCallParams) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            captured_configs.append(p.llm_config if isinstance(p.llm_config, LLMConfig) else None)
            return _ok_response()

        mw.wrap_model_call(params, call_next_2)
        assert call_count == 1  # 只调用了一次（直接走 fallback）
        assert captured_configs[0] is not None
        assert captured_configs[0].base_url == "https://backup.example.com/v1"
