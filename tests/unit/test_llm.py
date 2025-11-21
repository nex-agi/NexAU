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

"""
Unit tests for LLM configuration components.
"""

import os
from unittest.mock import patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig


class TestLLMConfig:
    """Test cases for LLM configuration."""

    def test_llm_config_initialization(self):
        """Test LLM config initialization with all parameters."""
        config = LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            timeout=30.0,
            max_retries=3,
            debug=True,
        )

        assert config.model == "gpt-4o-mini"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.1
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.debug is True
        assert config.api_type == "openai_chat_completion"

    def test_llm_config_defaults(self):
        """Test LLM config initialization with default values."""
        config = LLMConfig()

        assert config.temperature is None
        assert config.max_retries == 3
        assert config.debug is False
        assert config.stream is False
        assert config.extra_params == {}
        assert config.api_type == "openai_chat_completion"

    def test_llm_config_from_env(self):
        """Test LLM config initialization from environment variables."""
        with patch.dict(
            os.environ,
            {
                "LLM_MODEL": "gpt-4",
                "LLM_BASE_URL": "https://custom.api.com/v1",
                "LLM_API_KEY": "env-key",
            },
            clear=True,
        ):
            config = LLMConfig()

            assert config.model == "gpt-4"
            assert config.base_url == "https://custom.api.com/v1"
            assert config.api_key == "env-key"

    def test_llm_config_missing_env_vars(self):
        """Test LLM config with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Model not found in environment variables"):
                LLMConfig()

    def test_to_openai_params(self):
        """Test conversion to OpenAI parameters."""
        config = LLMConfig(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=2000,
            top_p=0.8,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            stream=True,
        )

        params = config.to_openai_params()

        assert params["model"] == "gpt-4o-mini"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 2000
        assert params["top_p"] == 0.8
        assert params["frequency_penalty"] == 0.2
        assert params["presence_penalty"] == 0.3
        assert params["stream"] is True

    def test_additional_drop_params(self):
        """Test that additional_drop_params removes entries from params."""
        config = LLMConfig(
            model="gpt-4o-mini",
            temperature=0.5,
            stop=["custom"],
            additional_drop_params=["temperature", "stop"],
        )

        params = config.to_openai_params()

        assert "temperature" not in params
        assert "stop" not in params

    def test_to_client_kwargs(self):
        """Test conversion to client kwargs."""
        config = LLMConfig(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            timeout=45.0,
            max_retries=5,
        )

        kwargs = config.to_client_kwargs()

        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "https://api.example.com/v1"
        assert kwargs["timeout"] == 45.0
        assert kwargs["max_retries"] == 5

    def test_get_set_param(self):
        """Test getting and setting parameters."""
        config = LLMConfig()

        # Test getting existing parameter
        assert config.get_param("temperature") is None

        # Test getting non-existing parameter with default
        assert config.get_param("nonexistent", "default") == "default"

        # Test setting new parameter
        config.set_param("custom_param", "custom_value")
        assert config.get_param("custom_param") == "custom_value"

        # Test setting existing parameter
        config.set_param("temperature", 0.5)
        assert config.get_param("temperature") == 0.5

    def test_update_config(self):
        """Test updating configuration with new parameters."""
        config = LLMConfig(temperature=0.7, max_tokens=1000)

        config.update(temperature=0.3, max_tokens=2000, new_param="value")

        assert config.temperature == 0.3
        assert config.max_tokens == 2000
        assert config.get_param("new_param") == "value"

    def test_copy_config(self):
        """Test copying configuration."""
        original = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            custom_param="value",
            additional_drop_params=["stop"],
        )

        copied = original.copy()

        assert copied.model == original.model
        assert copied.temperature == original.temperature
        assert copied.get_param("custom_param") == "value"
        assert copied.additional_drop_params == original.additional_drop_params
        assert copied.api_type == original.api_type
        assert copied is not original

    def test_repr_and_str(self):
        """Test string representations."""
        config = LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            temperature=0.7,
        )

        repr_str = repr(config)
        str_repr = str(config)

        assert "gpt-4o-mini" in repr_str
        assert "https://api.openai.com/v1" in repr_str
        assert repr_str == str_repr

    def test_extra_params_storage(self):
        """Test storage of extra parameters."""
        config = LLMConfig(
            model="gpt-4",
            extra_param1="value1",
            extra_param2="value2",
        )

        assert config.get_param("extra_param1") == "value1"
        assert config.get_param("extra_param2") == "value2"
        assert config.extra_params["extra_param1"] == "value1"

    def test_environment_variable_priority(self):
        """Test environment variable priority order."""
        with patch.dict(
            os.environ,
            {
                "LLM_MODEL": "low-priority",
                "OPENAI_MODEL": "medium-priority",
                "MODEL": "high-priority",
            },
        ):
            config = LLMConfig()
            assert config.model == "high-priority"

        with patch.dict(
            os.environ,
            {
                "LLM_BASE_URL": "low-priority",
                "OPENAI_BASE_URL": "high-priority",
            },
        ):
            config = LLMConfig()
            assert config.base_url == "high-priority"

    def test_api_key_priority(self):
        """Test API key priority from different sources."""
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "llm-key",
                "OPENAI_API_KEY": "openai-key",
                "API_KEY": "generic-key",
                "ANTHROPIC_API_KEY": "anthropic-key",
            },
        ):
            config = LLMConfig()
            assert config.api_key == "llm-key"  # Highest priority

    def test_invalid_temperature(self):
        """Test validation of temperature parameter."""
        config = LLMConfig(temperature=1.5)  # Should be valid (0.0-2.0)

        # Test boundary values
        config.temperature = 0.0
        assert config.temperature == 0.0

        config.temperature = 2.0
        assert config.temperature == 2.0

        # Test invalid values (should not raise, just store)
        config.temperature = 3.0  # Outside valid range but still stored
        assert config.temperature == 3.0
