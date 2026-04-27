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

"""LLM configuration class for handling model-related arguments."""

import os
from collections.abc import Iterable
from typing import Any


class LLMConfig:
    """Configuration class for LLM-related parameters."""

    # 默认流式 idle 超时 5 分钟（与 Codex stream_idle_timeout 一致）
    DEFAULT_STREAM_IDLE_TIMEOUT_MS: int = 300_000
    # 默认连接超时 15 秒（与 Codex websocket_connect_timeout 一致）
    DEFAULT_CONNECT_TIMEOUT_MS: int = 15_000

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        debug: bool = False,
        stream: bool = False,
        additional_drop_params: Iterable[str] | None = None,
        api_type: str = "openai_chat_completion",
        cache_control_ttl: str | None = None,
        tokenizer_path: str | None = None,
        stream_idle_timeout_ms: int | None = None,
        connect_timeout_ms: int | None = None,
        tool_streaming: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize LLM configuration.

        Args:
            model: Model name/identifier
            base_url: API base URL (e.g., https://api.openai.com/v1)
            api_key: API key for authentication
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            debug: Enable debug logging of LLM messages
            stream: Enable streaming responses when supported by backend
            api_type: API type
            tokenizer_path: HuggingFace tokenizer 路径或模型名称，如 "meta-llama/Llama-3.1-8B-Instruct"。
                仅在 api_type="generate_with_token" 时生效，通过 AutoTokenizer.from_pretrained 加载。
            stream_idle_timeout_ms: Per-chunk idle timeout in ms for streaming responses.
                If no chunk arrives within this duration, the stream is aborted.
                None → DEFAULT_STREAM_IDLE_TIMEOUT_MS (300_000ms = 5 min).
            connect_timeout_ms: Connection-phase timeout in ms.
                Applied to the initial HTTP/WS handshake before any data flows.
                None → DEFAULT_CONNECT_TIMEOUT_MS (15_000ms = 15 s).
            tool_streaming: Enable Anthropic eager_input_streaming for tool calls.
                Only valid for api_type='anthropic_chat_completion'. Default True.
            **kwargs: Additional model-specific parameters
        """
        self.model = model or self._get_model_from_env()
        self.base_url = base_url or self._get_base_url_from_env()
        self.api_key = api_key or self._get_api_key_from_env()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self.stream = stream
        params_iterable = additional_drop_params or []
        self.additional_drop_params = tuple(params_iterable)
        self.api_type = api_type
        self.cache_control_ttl = cache_control_ttl
        self.tokenizer_path: str | None = tokenizer_path
        self.stream_idle_timeout_ms: int | None = stream_idle_timeout_ms
        self.connect_timeout_ms: int | None = connect_timeout_ms
        self.tool_streaming = tool_streaming

        # tool_streaming 仅适用于 Anthropic
        if tool_streaming is not True and api_type != "anthropic_chat_completion":
            raise ValueError("tool_streaming is only supported for api_type='anthropic_chat_completion'")

        # Store additional parameters
        self.extra_params = kwargs

    def get_stream_idle_timeout(self) -> float:
        """Resolved stream idle timeout in seconds (default 300s = 5 min)."""
        ms = self.stream_idle_timeout_ms if self.stream_idle_timeout_ms is not None else self.DEFAULT_STREAM_IDLE_TIMEOUT_MS
        return ms / 1000.0

    def get_connect_timeout(self) -> float:
        """Resolved connection-phase timeout in seconds (default 15s)."""
        ms = self.connect_timeout_ms if self.connect_timeout_ms is not None else self.DEFAULT_CONNECT_TIMEOUT_MS
        return ms / 1000.0

    def _get_model_from_env(self) -> str:
        """Get model from environment variables."""
        # Try common environment variable names for model (highest priority first)
        env_vars = [
            "MODEL",
            "OPENAI_MODEL",
            "LLM_MODEL",
        ]

        for var in env_vars:
            model = os.getenv(var)
            if model:
                return model

        raise ValueError("Model not found in environment variables")

    def _get_base_url_from_env(self) -> str | None:
        """Get base URL from environment variables."""
        # Try common environment variable names for base URL (highest priority first)
        env_vars = [
            "OPENAI_BASE_URL",
            "BASE_URL",
            "LLM_BASE_URL",
        ]

        for var in env_vars:
            url = os.getenv(var)
            if url:
                return url

        raise ValueError("Base URL not found in environment variables")

    def _get_api_key_from_env(self) -> str | None:
        """Get API key from environment variables."""
        # Try common environment variable names
        env_vars = [
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
            "ANTHROPIC_API_KEY",
        ]

        for var in env_vars:
            key = os.getenv(var)
            if key:
                return key

        raise ValueError("API key not found in environment variables")

    def to_openai_params(self) -> dict[str, Any]:
        """Convert to OpenAI client parameters."""
        params: dict[str, Any] = {}

        # Model parameters
        if self.model:
            params["model"] = self.model
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.stream:
            params["stream"] = True

        # For Gemini REST API specific parameters
        if self.api_type == "gemini_rest":
            if "thinking_budget" in self.extra_params:
                params["thinking_budget"] = self.extra_params["thinking_budget"]
            if "include_thoughts" in self.extra_params:
                params["include_thoughts"] = self.extra_params["include_thoughts"]

        # Add extra parameters
        params.update(self.extra_params)

        return self.apply_param_drops(params)

    def to_client_kwargs(self) -> dict[str, Any]:
        """Convert to OpenAI/Anthropic client initialization kwargs.

        stream_idle_timeout_ms → httpx read timeout，实现每帧超时而非总超时。
        connect_timeout_ms → httpx connect timeout。
        两者通过 httpx.Timeout 组合传入 SDK。
        """
        import httpx as _httpx

        kwargs: dict[str, Any] = {}

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.max_retries:
            kwargs["max_retries"] = self.max_retries

        # 构建 httpx.Timeout：read 对应 stream_idle_timeout，connect 对应 connect_timeout
        connect = self.get_connect_timeout()
        read = self.get_stream_idle_timeout()
        total = self.timeout if self.timeout else None
        kwargs["timeout"] = _httpx.Timeout(
            timeout=total,
            connect=connect,
            read=read,
        )

        return kwargs

    def apply_param_drops(self, params: dict[str, Any]) -> dict[str, Any]:
        """Remove any params requested via additional_drop_params."""
        if not self.additional_drop_params:
            return params

        for key in self.additional_drop_params:
            params.pop(key, None)

        return params

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra_params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """Set a parameter value."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra_params[key] = value

    def update(self, **kwargs: Any) -> None:
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            self.set_param(key, value)

    def copy(self) -> "LLMConfig":
        """Create a copy of this configuration."""
        return LLMConfig(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            timeout=self.timeout,
            max_retries=self.max_retries,
            debug=self.debug,
            stream=self.stream,
            additional_drop_params=self.additional_drop_params,
            api_type=self.api_type,
            cache_control_ttl=self.cache_control_ttl,
            tokenizer_path=self.tokenizer_path,
            stream_idle_timeout_ms=self.stream_idle_timeout_ms,
            connect_timeout_ms=self.connect_timeout_ms,
            tool_streaming=self.tool_streaming,
            **self.extra_params,
        )

    def __repr__(self) -> str:
        return f"LLMConfig(model='{self.model}', base_url='{self.base_url}', temperature={self.temperature})"

    def __str__(self) -> str:
        return self.__repr__()
