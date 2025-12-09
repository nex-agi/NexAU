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
Pytest configuration and fixtures for nexau framework tests.

This module provides shared fixtures, configuration, and utilities
for all tests in the nexau test suite.
"""

import asyncio
import os
import shutil
import tempfile
import types
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest
import yaml

from nexau.archs.main_sub.execution.executor import Executor

# Provide a lightweight anthropic stub for environments without the package


def _load_nexau_dependencies():
    from nexau.archs.llm.llm_config import LLMConfig as _LLMConfig
    from nexau.archs.main_sub.agent import create_agent as _create_agent
    from nexau.archs.main_sub.agent_context import AgentContext as _AgentContext
    from nexau.archs.main_sub.agent_context import GlobalStorage as _GlobalStorage
    from nexau.archs.main_sub.agent_state import AgentState as _AgentState
    from nexau.archs.main_sub.config import AgentConfig as _AgentConfig
    from nexau.archs.main_sub.config import ExecutionConfig as _ExecutionConfig
    from nexau.archs.tool.tool import Tool as _Tool

    return (
        _LLMConfig,
        _create_agent,
        _AgentContext,
        _GlobalStorage,
        _AgentState,
        _AgentConfig,
        _ExecutionConfig,
        _Tool,
    )


(
    LLMConfig,
    create_agent,
    AgentContext,
    GlobalStorage,
    AgentState,
    AgentConfig,
    ExecutionConfig,
    Tool,
) = _load_nexau_dependencies()


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "test_data_dir": Path(__file__).parent / "test_data",
        "temp_dir": Path(__file__).parent / "temp",
        "mock_llm_responses": True,
        "enable_external_apis": False,
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(test_config):
    """Set up test environment for all tests."""
    # Create test directories
    test_config["test_data_dir"].mkdir(exist_ok=True)
    test_config["temp_dir"].mkdir(exist_ok=True)

    # Set environment variables for testing
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")  # Use smaller model for tests
    os.environ.setdefault("LLM_BASE_URL", "https://api.openai.com/v1")
    os.environ.setdefault("LLM_API_KEY", "test-key-not-used")

    yield

    # Cleanup after all tests
    if test_config["temp_dir"].exists():
        shutil.rmtree(test_config["temp_dir"])


# LLM Configuration Fixtures
@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration for testing."""
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        temperature=0.1,
        max_tokens=1000,
        api_type="openai_chat_completion",
    )


@pytest.fixture
def responses_llm_config():
    """LLM config configured for the Responses API."""
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        temperature=0.1,
        max_tokens=1000,
        api_type="openai_responses",
    )


@pytest.fixture
def real_llm_config():
    """Real LLM configuration using environment variables."""
    try:
        return LLMConfig()  # Will use environment variables
    except ValueError:
        pytest.skip("Real LLM credentials not available")


# Agent Context and State Fixtures
@pytest.fixture
def global_storage():
    """Global storage instance for testing."""
    return GlobalStorage()


@pytest.fixture
def agent_context():
    """Agent context for testing."""
    return AgentContext({"test": "value"})


@pytest.fixture
def mock_executor():
    """Lightweight executor mock for agent state interactions."""
    executor = Mock()
    executor.add_tool = Mock()
    return cast(Executor, executor)


@pytest.fixture
def agent_state(mock_llm_config, global_storage, agent_context, mock_executor):
    """Agent state for testing."""
    return AgentState(
        agent_name="test_agent",
        agent_id="test_agent_123",
        context=agent_context,
        global_storage=global_storage,
        executor=mock_executor,
    )


# Agent Configuration Fixtures
@pytest.fixture
def agent_config(mock_llm_config):
    """Basic agent configuration for testing."""
    return AgentConfig(
        name="test_agent",
        agent_id="test_agent_123",
        system_prompt="You are a helpful assistant.",
        system_prompt_type="string",
        tools=[],
        sub_agents=[],
        llm_config=mock_llm_config,
        stop_tools=set(),
    )


@pytest.fixture
def execution_config():
    """Execution configuration for testing."""
    return ExecutionConfig(
        max_iterations=10,
        max_context_tokens=8000,
        max_running_subagents=3,
        retry_attempts=2,
        timeout=60,
    )


# Tool Fixtures
@pytest.fixture
def sample_tool():
    """Sample tool for testing."""

    def sample_function(x: int, y: str = "default") -> dict:
        return {"result": f"{x}_{y}"}

    return Tool(
        name="sample_tool",
        description="A sample tool for testing",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "string", "default": "default"}},
            "required": ["x"],
        },
        implementation=sample_function,
    )


@pytest.fixture
def mock_tools(sample_tool):
    """List of mock tools for testing."""
    return [sample_tool]


# File and Directory Fixtures
@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    from nexau.archs.tool.builtin.file_tools.file_edit_tool import mark_file_as_read

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test content")
        temp_path = f.name

    # Mark the file as read so it can be edited/written by tests
    mark_file_as_read(temp_path)

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create a sample YAML configuration file."""
    config = {
        "name": "test_agent",
        "system_prompt": "You are a helpful assistant.",
        "llm_config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
        },
        "tools": [],
        "max_iterations": 10,
    }

    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


# Mock External Services
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()

    # Mock chat completions
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mocked LLM response"
    mock_client.chat.completions.create.return_value = mock_response

    # Mock Responses API
    message_item = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Mocked LLM response"}],
    }

    responses_payload = types.SimpleNamespace(
        output=[message_item],
        output_text="Mocked LLM response",
    )
    mock_client.responses.create.return_value = responses_payload

    return mock_client


# Agent Fixtures
@pytest.fixture
def mock_agent(mock_llm_config, execution_config, global_storage):
    """Create a mock agent for testing."""
    with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
        mock_openai.OpenAI.return_value = Mock()
        agent = create_agent(
            name="test_agent",
            llm_config=mock_llm_config,
            max_iterations=execution_config.max_iterations,
            max_context_tokens=execution_config.max_context_tokens,
            global_storage=global_storage,
        )
        yield agent


# Async Fixtures
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Environment Variables for Testing
@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for consistent testing."""
    with patch.dict(
        os.environ,
        {
            "TESTING": "true",
            "LLM_MODEL": "gpt-4o-mini",
            "LLM_BASE_URL": "https://api.openai.com/v1",
            "LLM_API_KEY": "test-key-not-used",
            "SERPER_API_KEY": "test-serper-key",
            "FEISHU_APP_ID": "test-feishu-app-id",
            "FEISHU_APP_SECRET": "test-feishu-secret",
        },
    ):
        yield


# Test Data Fixtures
@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing well, thank you for asking."},
    ]


@pytest.fixture
def sample_tool_call():
    """Sample tool call data for testing."""
    return {
        "tool_name": "sample_tool",
        "parameters": {"x": 42, "y": "test"},
    }


# Error Simulation Fixtures
@pytest.fixture
def mock_llm_error():
    """Mock LLM error for testing error handling."""
    return Exception("Mock LLM API error")


@pytest.fixture
def mock_tool_error():
    """Mock tool error for testing error handling."""
    return Exception("Mock tool execution error")


# Configuration Loading Fixtures
@pytest.fixture
def valid_agent_config_dict():
    """Valid agent configuration dictionary."""
    return {
        "name": "test_agent",
        "system_prompt": "You are a helpful assistant.",
        "llm_config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 1000,
        },
        "tools": [],
        "max_iterations": 10,
    }


@pytest.fixture
def invalid_agent_config_dict():
    """Invalid agent configuration dictionary."""
    return {
        "name": "",  # Invalid: empty name
        "llm_config": {
            "model": "",  # Invalid: empty model
        },
    }


# Test Utilities
@pytest.fixture
def assert_mock_calls():
    """Utility fixture for asserting mock calls."""

    def _assert_mock_calls(mock_obj, expected_calls):
        mock_obj.assert_has_calls(expected_calls)
        assert mock_obj.call_count == len(expected_calls)

    return _assert_mock_calls


# Performance and Load Testing Fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {
        "min_response_time": 0.1,
        "max_response_time": 5.0,
        "max_memory_mb": 100,
    }


# Skip Conditions
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "external: marks tests that require external services")
    config.addinivalue_line("markers", "llm: marks tests that require LLM services")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests that take longer
        if "performance" in item.name or "load" in item.name:
            item.add_marker(pytest.mark.slow)

        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add e2e marker to e2e tests
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add external marker to tests that might need external services
        if any(keyword in item.name for keyword in ["web", "search", "api", "external"]):
            item.add_marker(pytest.mark.external)

        # Add llm marker to tests that use LLM services
        if any(keyword in item.name for keyword in ["llm", "openai", "chat"]):
            item.add_marker(pytest.mark.llm)
