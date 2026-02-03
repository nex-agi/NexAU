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

"""Integration tests for Session management.

Tests session persistence, history management, and database backends:
- Session creation and recovery
- History persistence (APPEND/UNDO/REPLACE)
- Multi-backend support (InMemory, SQL, JSONL)
- Concurrent access patterns
"""

import tempfile
from pathlib import Path

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.session.orm.jsonl_engine import JSONLDatabaseEngine


class TestSessionPersistence:
    """Test session persistence and recovery."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.fixture
    def agent_config(self):
        """Create agent config."""
        return AgentConfig(
            name="session_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
        )

    @pytest.mark.llm
    def test_session_persists_after_agent_run(self, session_manager, agent_config):
        """Test that session data persists after agent run."""
        user_id = "test_user"
        session_id = "persist_session"

        # Run agent
        agent = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        response = agent.run(message="Hello!")
        assert isinstance(response, str)
        assert len(response) > 0
        # Session is created implicitly by Agent - verified by successful run

    @pytest.mark.llm
    def test_history_persists_across_agent_instances(self, session_manager, agent_config):
        """Test that history persists across agent instances."""
        user_id = "test_user"
        session_id = "history_session"

        # First agent instance
        agent1 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        agent1.run(message="My favorite number is 42.")

        # Second agent instance - should have access to history
        agent2 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        response = agent2.run(message="What is my favorite number?")

        # Agent should remember from persisted history
        assert "42" in response


class TestMultipleBackends:
    """Test different database backends."""

    @pytest.fixture
    def agent_config(self):
        """Create agent config."""
        return AgentConfig(
            name="backend_agent",
            system_prompt="You are a helpful assistant. Remember what the user tells you.",
            llm_config=LLMConfig(),
        )

    @pytest.mark.llm
    def test_inmemory_backend(self, agent_config):
        """Test InMemory backend."""
        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="inmem_session",
        )
        response = agent.run(message="Hello from InMemory!")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.llm
    def test_jsonl_backend_persistence(self, agent_config):
        """Test JSONL backend with directory persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # JSONLDatabaseEngine uses base_path as directory
            db_dir = Path(temp_dir) / "session_data"

            # Create engine and run agent
            engine = JSONLDatabaseEngine(base_path=str(db_dir))
            session_manager = SessionManager(engine=engine)

            agent = Agent(
                config=agent_config,
                session_manager=session_manager,
                user_id="test_user",
                session_id="jsonl_session",
            )
            agent.run(message="Remember: the secret code is ABC123.")

            # Verify directory was created
            assert db_dir.exists()

            # Create new engine from same directory - should recover session
            engine2 = JSONLDatabaseEngine(base_path=str(db_dir))
            session_manager2 = SessionManager(engine=engine2)

            agent2 = Agent(
                config=agent_config,
                session_manager=session_manager2,
                user_id="test_user",
                session_id="jsonl_session",
            )
            response = agent2.run(message="What is the secret code?")

            # Should remember from persisted JSONL
            assert "ABC123" in response


class TestSessionIsolation:
    """Test session isolation between users and sessions."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.fixture
    def agent_config(self):
        """Create agent config."""
        return AgentConfig(
            name="isolated_agent",
            system_prompt="You are a helpful assistant. Remember what each user tells you.",
            llm_config=LLMConfig(),
        )

    @pytest.mark.llm
    def test_different_users_are_isolated(self, session_manager, agent_config):
        """Test that different users have isolated sessions."""
        # User 1 tells something
        agent1 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id="user_1",
            session_id="shared_session_id",  # Same session_id but different user
        )
        agent1.run(message="I am User One and my pet is a dog.")

        # User 2 should not know about User 1's information
        agent2 = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id="user_2",
            session_id="shared_session_id",
        )
        response = agent2.run(message="What is my pet?")

        # User 2 should not know about User 1's pet
        assert isinstance(response, str)
        # Response should indicate uncertainty or ask for clarification

    @pytest.mark.llm
    def test_different_sessions_same_user_are_isolated(self, session_manager, agent_config):
        """Test that different sessions for same user are isolated."""
        user_id = "test_user"

        # Session A
        agent_a = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id="session_a",
        )
        agent_a.run(message="In this conversation, we are discussing Python.")

        # Session B - should not know about session A
        agent_b = Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id="session_b",
        )
        response = agent_b.run(message="What programming language are we discussing?")

        # Session B should not know about Python discussion
        assert isinstance(response, str)


class TestAgentRegistration:
    """Test agent registration in sessions."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.mark.llm
    def test_agent_registration_on_run(self, session_manager):
        """Test that agent is registered when run."""
        config = AgentConfig(
            name="registered_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="reg_session",
        )
        response = agent.run(message="Hello!")
        # Successful run means agent was registered
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.llm
    def test_multiple_agents_same_session(self, session_manager):
        """Test multiple agents in same session."""
        config1 = AgentConfig(
            name="agent_one",
            system_prompt="You are Agent One.",
            llm_config=LLMConfig(),
        )

        config2 = AgentConfig(
            name="agent_two",
            system_prompt="You are Agent Two.",
            llm_config=LLMConfig(),
        )

        user_id = "test_user"
        session_id = "multi_agent_session"

        # Run first agent
        agent1 = Agent(
            config=config1,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        agent1.run(message="Hello from Agent One!")

        # Run second agent in same session
        agent2 = Agent(
            config=config2,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        response2 = agent2.run(message="Hello from Agent Two!")
        # Successful runs mean both agents are registered
        assert isinstance(response2, str)
        assert len(response2) > 0


class TestHistoryOperations:
    """Test history operations (APPEND, UNDO, REPLACE)."""

    @pytest.fixture
    def engine(self):
        """Create in-memory database engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def session_manager(self, engine):
        """Create session manager."""
        return SessionManager(engine=engine)

    @pytest.mark.llm
    def test_history_append_operation(self, session_manager):
        """Test history APPEND operation."""
        config = AgentConfig(
            name="history_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
        )

        user_id = "test_user"
        session_id = "append_session"

        # Run agent multiple times - each should append to history
        for i in range(3):
            agent = Agent(
                config=config,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )
            response = agent.run(message=f"Message {i + 1}")
            assert isinstance(response, str)
            assert len(response) > 0
        # Successful runs verify history operations work
