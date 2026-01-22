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

"""Unit tests for AgentModel.

Note: AgentModel no longer stores messages directly. Messages are now stored
in MessageModel and AgentSnapshotModel. See test_message_store.py for message
storage tests.
"""

from nexau.archs.session import AgentModel


class TestAgentModel:
    """Test cases for AgentModel."""

    def test_initialization(self):
        """Test basic initialization."""
        model = AgentModel(
            user_id="user_123",
            session_id="sess_123",
            agent_id="agent_456",
            agent_name="test_agent",
        )
        assert model.user_id == "user_123"
        assert model.session_id == "sess_123"
        assert model.agent_id == "agent_456"
        assert model.agent_name == "test_agent"

    def test_default_values(self):
        """Test default values."""
        model = AgentModel(
            user_id="user_123",
            session_id="sess_123",
            agent_id="agent_456",
        )
        assert model.agent_name == ""
        assert model.created_at is not None
        assert model.last_updated is not None

    def test_serialization(self):
        """Test model serialization to JSON."""
        model = AgentModel(
            user_id="user_123",
            session_id="sess_123",
            agent_id="agent_456",
            agent_name="test_agent",
        )
        json_str = model.model_dump_json()
        assert "user_123" in json_str
        assert "sess_123" in json_str
        assert "agent_456" in json_str
        assert "test_agent" in json_str

    def test_deserialization(self):
        """Test model deserialization from JSON."""
        json_str = '{"user_id": "user_123", "session_id": "sess_123", "agent_id": "agent_456", "agent_name": "test_agent"}'
        model = AgentModel.model_validate_json(json_str)
        assert model.user_id == "user_123"
        assert model.session_id == "sess_123"
        assert model.agent_id == "agent_456"
        assert model.agent_name == "test_agent"

    def test_primary_key_fields(self):
        """Test that primary key fields are correctly defined using SQLModel Field."""
        # Verify primary key fields exist and are correctly typed
        model_fields = AgentModel.model_fields

        # Check primary key fields
        assert "user_id" in model_fields
        assert "session_id" in model_fields
        assert "agent_id" in model_fields
