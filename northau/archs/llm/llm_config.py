"""LLM configuration class for handling model-related arguments."""

import os
from collections.abc import Iterable
from typing import Any


class LLMConfig:
    """Configuration class for LLM-related parameters."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        debug: bool = False,
        additional_drop_params: Iterable[str] | None = None,
        api_type: str = "openai_chat_completion",
        **kwargs,
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
            api_type: API type
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
        self.additional_drop_params = tuple(param for param in (additional_drop_params or []) if isinstance(param, str))
        self.api_type = api_type

        # Store additional parameters
        self.extra_params = kwargs

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
        params = {}

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

        # Add extra parameters
        params.update(self.extra_params)

        return self.apply_param_drops(params)

    def to_client_kwargs(self) -> dict[str, Any]:
        """Convert to OpenAI client initialization kwargs."""
        kwargs = {}

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.max_retries:
            kwargs["max_retries"] = self.max_retries

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

    def update(self, **kwargs) -> None:
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
            additional_drop_params=self.additional_drop_params,
            **self.extra_params,
        )

    def __repr__(self) -> str:
        return f"LLMConfig(model='{self.model}', base_url='{self.base_url}', temperature={self.temperature})"

    def __str__(self) -> str:
        return self.__repr__()
