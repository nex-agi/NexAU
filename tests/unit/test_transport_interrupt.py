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

"""Unit tests for RFC-0001 Phase 4: Transport layer stop integration.

Tests cover:
- Agent registry in TransportBase (track/untrack running agents)
- handle_stop_request (find and stop running agents)
- StopRequest/StopResponse models
- SSE server /stop endpoint
- Stdio transport agent.stop method
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.execution.stop_result import StopResult
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.transports.base import TransportBase
from nexau.archs.transports.http.models import StopRequest, StopResponse


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


class TestAgentRegistry:
    """Test agent registry in TransportBase."""

    @pytest.fixture
    def engine(self):
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        return ConcreteTransport(
            engine=engine,
            config=_TestConfig(),
            default_agent_config=agent_config,
        )

    def test_running_agents_initialized_empty(self, transport):
        """_running_agents should be empty on init."""
        assert transport._running_agents == {}

    def test_running_agents_lock_exists(self, transport):
        """_running_agents_lock should be an asyncio.Lock."""
        assert isinstance(transport._running_agents_lock, asyncio.Lock)

    def test_handle_request_tracks_agent(self, transport):
        """handle_request should register agent in _running_agents during execution."""
        captured_agents = {}

        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.agent_id = "agent_123"

            async def fake_run_async(**kwargs):
                # 在执行期间检查 agent 是否已注册
                captured_agents.update(transport._running_agents)
                return "response"

            mock_agent.run_async = fake_run_async
            mock_agent_cls.return_value = mock_agent

            asyncio.run(
                transport.handle_request(
                    message="Hello",
                    user_id="user_1",
                    session_id="session_1",
                )
            )

        # Agent was registered during execution
        assert len(captured_agents) == 1
        # After completion, agent should be removed
        assert len(transport._running_agents) == 0

    def test_handle_request_unregisters_on_error(self, transport):
        """handle_request should unregister agent even on error."""
        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.agent_id = "agent_123"
            mock_agent.run_async = AsyncMock(side_effect=RuntimeError("test error"))
            mock_agent_cls.return_value = mock_agent

            with pytest.raises(RuntimeError):
                asyncio.run(
                    transport.handle_request(
                        message="Hello",
                        user_id="user_1",
                        session_id="session_1",
                    )
                )

        # Agent should be removed even after error
        assert len(transport._running_agents) == 0


class TestHandleStopRequest:
    """Test handle_stop_request in TransportBase."""

    @pytest.fixture
    def engine(self):
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        return ConcreteTransport(
            engine=engine,
            config=_TestConfig(),
            default_agent_config=agent_config,
        )

    @pytest.mark.anyio
    async def test_stop_no_running_agent(self, transport):
        """Should raise ValueError when no running agent found."""
        with pytest.raises(ValueError, match="No running agent found"):
            await transport.handle_stop_request(
                user_id="user_1",
                session_id="session_1",
            )

    @pytest.mark.anyio
    async def test_stop_with_agent_id_not_found(self, transport):
        """Should raise ValueError when specific agent_id not found."""
        with pytest.raises(ValueError, match="No running agent found"):
            await transport.handle_stop_request(
                user_id="user_1",
                session_id="session_1",
                agent_id="nonexistent",
            )

    @pytest.mark.anyio
    async def test_stop_finds_agent_by_session(self, transport):
        """Should find and stop agent by user_id + session_id."""
        mock_agent = Mock()
        mock_result = StopResult(
            messages=[],
            stop_reason=AgentStopReason.USER_INTERRUPTED,
        )
        mock_agent.stop = AsyncMock(return_value=mock_result)

        # 手动注册一个 agent
        transport._running_agents[("user_1", "session_1", "agent_abc")] = mock_agent

        result = await transport.handle_stop_request(
            user_id="user_1",
            session_id="session_1",
        )

        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED
        mock_agent.stop.assert_called_once_with(force=False, timeout=30.0)

    @pytest.mark.anyio
    async def test_stop_finds_agent_by_id(self, transport):
        """Should find agent by exact (user_id, session_id, agent_id) key."""
        mock_agent = Mock()
        mock_result = StopResult(
            messages=[],
            stop_reason=AgentStopReason.USER_INTERRUPTED,
        )
        mock_agent.stop = AsyncMock(return_value=mock_result)

        transport._running_agents[("user_1", "session_1", "agent_xyz")] = mock_agent

        result = await transport.handle_stop_request(
            user_id="user_1",
            session_id="session_1",
            agent_id="agent_xyz",
        )

        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED

    @pytest.mark.anyio
    async def test_stop_passes_timeout(self, transport):
        """Should pass custom timeout to agent.stop()."""
        mock_agent = Mock()
        mock_result = StopResult(
            messages=[],
            stop_reason=AgentStopReason.USER_INTERRUPTED,
        )
        mock_agent.stop = AsyncMock(return_value=mock_result)

        transport._running_agents[("user_1", "session_1", "agent_1")] = mock_agent

        await transport.handle_stop_request(
            user_id="user_1",
            session_id="session_1",
            timeout=10.0,
        )

        mock_agent.stop.assert_called_once_with(force=False, timeout=10.0)

    @pytest.mark.anyio
    async def test_stop_passes_force(self, transport):
        """Should pass force=True to agent.stop()."""
        mock_agent = Mock()
        mock_result = StopResult(
            messages=[],
            stop_reason=AgentStopReason.USER_INTERRUPTED,
        )
        mock_agent.stop = AsyncMock(return_value=mock_result)

        transport._running_agents[("user_1", "session_1", "agent_1")] = mock_agent

        await transport.handle_stop_request(
            user_id="user_1",
            session_id="session_1",
            force=True,
        )

        mock_agent.stop.assert_called_once_with(force=True, timeout=30.0)


class TestStopModels:
    """Test StopRequest and StopResponse models."""

    def test_stop_request_defaults(self):
        """StopRequest should have sensible defaults."""
        req = StopRequest(session_id="sess_1")
        assert req.user_id == "default-user"
        assert req.session_id == "sess_1"
        assert req.agent_id is None
        assert req.force is False
        assert req.timeout == 30.0

    def test_stop_request_full(self):
        """StopRequest should accept all fields."""
        req = StopRequest(
            user_id="user_1",
            session_id="sess_1",
            agent_id="agent_1",
            force=True,
            timeout=10.0,
        )
        assert req.user_id == "user_1"
        assert req.agent_id == "agent_1"
        assert req.force is True
        assert req.timeout == 10.0

    def test_stop_response_success(self):
        """StopResponse should represent success."""
        resp = StopResponse(
            status="success",
            stop_reason="USER_INTERRUPTED",
            message_count=12,
        )
        assert resp.status == "success"
        assert resp.stop_reason == "USER_INTERRUPTED"
        assert resp.message_count == 12
        assert resp.error is None

    def test_stop_response_error(self):
        """StopResponse should represent error."""
        resp = StopResponse(
            status="error",
            error="No running agent found",
        )
        assert resp.status == "error"
        assert resp.error == "No running agent found"
        assert resp.stop_reason is None
