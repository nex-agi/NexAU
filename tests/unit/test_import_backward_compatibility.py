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

"""Tests for import paths.

Tests that import paths work correctly after the refactoring to the new
models directory structure.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st


class TestImportPaths:
    """Tests for import paths."""

    def test_models_import_path(self) -> None:
        """Test that models import path works."""
        from nexau.archs.session.models import (
            AgentModel,
            AgentRunActionModel,
            RunActionType,
            SessionModel,
        )

        # Verify all classes are importable and are classes
        assert isinstance(SessionModel, type)
        assert isinstance(AgentModel, type)
        assert isinstance(AgentRunActionModel, type)
        assert isinstance(RunActionType, type)

    def test_session_init_import_path(self) -> None:
        """Test that session __init__ import path works."""
        from nexau.archs.session import (
            AgentModel,
            AgentRunActionModel,
            DatabaseEngine,
            InMemoryDatabaseEngine,
            RemoteDatabaseEngine,
            RunActionType,
            SessionManager,
            SessionModel,
            SQLDatabaseEngine,
        )
        from nexau.archs.session.orm import (
            AndFilter,
            ComparisonFilter,
            NotFilter,
            OrFilter,
        )

        # Verify all classes are importable
        assert isinstance(SessionModel, type)
        assert isinstance(AgentModel, type)
        assert isinstance(AgentRunActionModel, type)
        assert isinstance(RunActionType, type)
        assert isinstance(DatabaseEngine, type)
        assert isinstance(InMemoryDatabaseEngine, type)
        assert isinstance(SQLDatabaseEngine, type)
        assert isinstance(RemoteDatabaseEngine, type)
        assert isinstance(SessionManager, type)
        # New Filter DSL types
        assert isinstance(ComparisonFilter, type)
        assert isinstance(AndFilter, type)
        assert isinstance(OrFilter, type)
        assert isinstance(NotFilter, type)

    def test_orm_init_import_path(self) -> None:
        """Test that orm __init__ import path works."""
        from nexau.archs.session.orm import (
            AndFilter,
            ComparisonFilter,
            DatabaseEngine,
            InMemoryDatabaseEngine,
            NotFilter,
            OrFilter,
            RemoteDatabaseEngine,
            SQLDatabaseEngine,
            get_table_name,
        )

        # Verify all classes are importable
        assert isinstance(DatabaseEngine, type)
        assert isinstance(InMemoryDatabaseEngine, type)
        assert isinstance(SQLDatabaseEngine, type)
        assert isinstance(RemoteDatabaseEngine, type)
        assert callable(get_table_name)
        # New Filter DSL types
        assert isinstance(ComparisonFilter, type)
        assert isinstance(AndFilter, type)
        assert isinstance(OrFilter, type)
        assert isinstance(NotFilter, type)

    @given(
        user_id=st.text(min_size=1, max_size=10),
        session_id=st.text(min_size=1, max_size=10),
    )
    @settings(max_examples=100)
    def test_session_model_instantiation(self, user_id: str, session_id: str) -> None:
        """Test SessionModel instantiation from both import paths."""
        from nexau.archs.session import SessionModel as NewSessionModel
        from nexau.archs.session.models import (
            SessionModel as ModelsSessionModel,
        )

        # Both should be the same class
        assert NewSessionModel is ModelsSessionModel

        # Create instances from both paths
        instance1 = NewSessionModel(user_id=user_id, session_id=session_id)
        instance2 = ModelsSessionModel(user_id=user_id, session_id=session_id)

        # Verify they have the same data
        assert instance1.user_id == instance2.user_id
        assert instance1.session_id == instance2.session_id

    @given(
        user_id=st.text(min_size=1, max_size=10),
        session_id=st.text(min_size=1, max_size=10),
        agent_id=st.text(min_size=1, max_size=10),
    )
    @settings(max_examples=100)
    def test_agent_model_instantiation(self, user_id: str, session_id: str, agent_id: str) -> None:
        """Test AgentModel instantiation from both import paths."""
        from nexau.archs.session import AgentModel as NewAgentModel
        from nexau.archs.session.models import AgentModel as ModelsAgentModel

        # Both should be the same class
        assert NewAgentModel is ModelsAgentModel

        # Create instances from both paths
        instance1 = NewAgentModel(user_id=user_id, session_id=session_id, agent_id=agent_id)
        instance2 = ModelsAgentModel(user_id=user_id, session_id=session_id, agent_id=agent_id)

        # Verify they have the same data
        assert instance1.user_id == instance2.user_id
        assert instance1.session_id == instance2.session_id
        assert instance1.agent_id == instance2.agent_id

    @given(
        user_id=st.text(min_size=1, max_size=10),
        session_id=st.text(min_size=1, max_size=10),
        agent_id=st.text(min_size=1, max_size=10),
        run_id=st.text(min_size=1, max_size=10),
    )
    @settings(max_examples=100)
    def test_run_action_model_instantiation(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
    ) -> None:
        """Test AgentRunActionModel instantiation from both import paths."""
        from nexau.archs.session import AgentRunActionModel
        from nexau.archs.session.models import (
            AgentRunActionModel as ModelsRunActionModel,
        )

        # Both should be the same class
        assert AgentRunActionModel is ModelsRunActionModel

        # Create instances from both paths
        instance1 = AgentRunActionModel(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=run_id,
            action_type="append",
        )
        instance2 = ModelsRunActionModel(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=run_id,
            action_type="append",
        )

        # Verify they have the same data
        assert instance1.user_id == instance2.user_id
        assert instance1.session_id == instance2.session_id
        assert instance1.agent_id == instance2.agent_id
        assert instance1.run_id == instance2.run_id
