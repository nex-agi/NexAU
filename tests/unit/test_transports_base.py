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

"""Unit tests for transport base class."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.transports.base import TransportBase


@dataclass
class _TestConfig:
    """Test transport configuration."""

    host: str = "localhost"
    port: int = 8080


class ConcreteTransport(TransportBase[_TestConfig]):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False


class TestTransportBase:
    """Test cases for TransportBase class."""

    @pytest.fixture
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        """Create transport instance."""
        return ConcreteTransport(
            engine=engine,
            config=_TestConfig(),
            default_agent_config=agent_config,
        )

    def test_initialization(self, transport):
        """Test transport initialization."""
        assert transport._config.host == "localhost"
        assert transport._config.port == 8080
        assert transport._session_manager is not None

    def test_start_stop(self, transport):
        """Test start and stop methods."""
        assert not transport._started

        transport.start()
        assert transport._started

        transport.stop()
        assert not transport._started

    def test_context_manager(self, transport):
        """Test context manager protocol."""
        with transport as t:
            assert t._started
            assert t is transport

        assert not transport._started

    def test_recursively_apply_middlewares(self, agent_config):
        """Test _recursively_apply_middlewares method."""
        middleware = Mock()

        result = TransportBase._recursively_apply_middlewares(
            agent_config,
            middleware,
            enable_stream=True,
        )

        assert middleware in result.middlewares
        assert result.llm_config.stream is True

    def test_recursively_apply_middlewares_with_sub_agents(self):
        """Test _recursively_apply_middlewares applies to sub-agents."""
        sub_config = AgentConfig(
            name="sub_agent",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )
        parent_config = AgentConfig(
            name="parent_agent",
            llm_config=LLMConfig(model="gpt-4o-mini"),
            sub_agents={"sub": sub_config},
        )
        middleware = Mock()

        result = TransportBase._recursively_apply_middlewares(
            parent_config,
            middleware,
            enable_stream=True,
        )

        assert middleware in result.middlewares
        assert middleware in result.sub_agents["sub"].middlewares
        assert result.sub_agents["sub"].llm_config.stream is True

    def test_handle_request(self, transport):
        """Test handle_request method."""
        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.run_async = AsyncMock(return_value="Test response")
            mock_agent_cls.return_value = mock_agent

            response = asyncio.run(
                transport.handle_request(
                    message="Hello",
                    user_id="user_123",
                )
            )

            assert response == "Test response"
            mock_agent.run_async.assert_called_once()

    def test_handle_request_with_session_id(self, transport):
        """Test handle_request with existing session_id."""
        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.run_async = AsyncMock(return_value="Test response")
            mock_agent_cls.return_value = mock_agent

            response = asyncio.run(
                transport.handle_request(
                    message="Hello",
                    user_id="user_123",
                    session_id="existing_session",
                )
            )

            assert response == "Test response"

    def test_handle_request_with_context(self, transport):
        """Test handle_request with context."""
        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.run_async = AsyncMock(return_value="Test response")
            mock_agent_cls.return_value = mock_agent

            response = asyncio.run(
                transport.handle_request(
                    message="Hello",
                    user_id="user_123",
                    context={"key": "value"},
                )
            )

            assert response == "Test response"
            call_kwargs = mock_agent.run_async.call_args[1]
            assert call_kwargs["context"] == {"key": "value"}

    def test_handle_streaming_request(self, transport):
        """Test handle_streaming_request method."""
        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.run_async = AsyncMock(return_value="Test response")
            mock_agent_cls.return_value = mock_agent

            async def run_test():
                collected = []
                async for event in transport.handle_streaming_request(
                    message="Hello",
                    user_id="user_123",
                ):
                    collected.append(event)
                return collected

            asyncio.run(run_test())

            # Agent should have been called
            mock_agent.run_async.assert_called_once()

    def test_async_loop_nesting_problem_direct_agent_in_async(self, engine, agent_config):
        """Creating Agent() directly in async context succeeds with nest_asyncio."""
        session_manager = SessionManager(engine=engine)

        async def create_agent_in_async_context():
            return Agent(
                config=agent_config,
                session_manager=session_manager,
                user_id="user_1",
                session_id="session_1",
            )

        agent = asyncio.run(create_agent_in_async_context())
        assert agent is not None
        assert agent.agent_id
        assert agent.global_storage is not None
        assert agent._session_id == "session_1"

    def test_async_loop_nesting_fix_via_to_thread(self, engine, agent_config):
        """Verify the fix: creating Agent via asyncio.to_thread from async context completes.

        Transport creates agent in a worker thread via to_thread(create_agent); in that
        thread there is no running loop, so syncify() creates a new loop and runs
        session init there, no deadlock.
        """
        session_manager = SessionManager(engine=engine)

        def create_agent_sync():
            return Agent(
                config=agent_config,
                session_manager=session_manager,
                user_id="user_1",
                session_id="session_1",
            )

        async def create_via_to_thread():
            return await asyncio.to_thread(create_agent_sync)

        agent = asyncio.run(asyncio.wait_for(create_via_to_thread(), timeout=10.0))
        assert agent is not None
        assert agent.agent_id
        assert agent.global_storage is not None
        assert agent._session_id == "session_1"
