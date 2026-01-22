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

"""Property-based tests for StorageModel serialization round-trip.

Feature: session-storage-simplification
Property 1: Serialization Round-Trip

*For any* StorageModel instance (SessionModel, AgentModel, or AgentRunActionModel),
serializing with `model_dump_json()` then deserializing with `model_validate_json()`
should produce an equivalent object.

**Validates: Requirements 2.7, 3.4, 5.3, 5.4**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from nexau.archs.session.models import (
    AgentModel,
    AgentRunActionModel,
    RunActionType,
    SessionModel,
)
from nexau.core.messages import Message, Role, TextBlock

# === Strategies for generating test data ===

# Strategy for non-empty strings (for IDs)
non_empty_string = st.text(min_size=1, max_size=50).filter(lambda x: x.strip())

# Strategy for optional strings
optional_string = st.one_of(st.none(), st.text(max_size=50))

# Strategy for string lists (removed - agent_ids no longer exists)

# Strategy for simple JSON-serializable values
json_value = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000000, max_value=1000000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=100),
)

# Strategy for simple JSON-serializable dicts
json_dict = st.dictionaries(
    keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    values=json_value,
    max_size=5,
)


# === SessionModel Strategy ===

session_model_strategy = st.builds(
    SessionModel,
    user_id=non_empty_string,
    session_id=non_empty_string,
    context=json_dict,
    storage=json_dict,
    root_agent_id=optional_string,
)


# === AgentModel Strategy ===

agent_model_strategy = st.builds(
    AgentModel,
    user_id=non_empty_string,
    session_id=non_empty_string,
    agent_id=non_empty_string,
    agent_name=st.text(max_size=50),
)


# === AgentRunActionModel Strategy ===

# Strategy for Message
message_strategy = st.builds(
    Message,
    role=st.sampled_from(list(Role)),
    content=st.lists(
        st.builds(TextBlock, text=st.text(max_size=100)),
        max_size=3,
    ),
    metadata=json_dict,
)

# Strategy for AgentRunActionModel with APPEND action
run_action_model_strategy = st.builds(
    AgentRunActionModel,
    user_id=non_empty_string,
    session_id=non_empty_string,
    agent_id=non_empty_string,
    run_id=non_empty_string,
    root_run_id=non_empty_string,
    parent_run_id=optional_string,
    action_type=st.sampled_from([RunActionType.APPEND, RunActionType.UNDO, RunActionType.REPLACE]),
    append_messages=st.one_of(st.none(), st.lists(message_strategy, min_size=1, max_size=3)),
    replace_messages=st.none(),
    undo_before_run_id=st.none(),
    agent_name=st.text(max_size=50),
)


class TestSerializationRoundTrip:
    """Property-based tests for StorageModel serialization round-trip.

    Property 1: Serialization Round-Trip

    *For any* StorageModel instance (SessionModel, AgentModel, or AgentRunActionModel),
    serializing with `model_dump_json()` then deserializing with `model_validate_json()`
    should produce an equivalent object.

    **Validates: Requirements 2.7, 3.4, 5.3, 5.4**
    """

    @settings(max_examples=100)
    @given(session=session_model_strategy)
    def test_session_model_serialization_round_trip(self, session: SessionModel):
        """Test SessionModel serialization round-trip.

        *For any* SessionModel instance, serializing with `model_dump_json()` then
        deserializing with `model_validate_json()` should produce an equivalent object.

        **Validates: Requirements 3.4**
        """
        # Serialize to JSON
        json_str = session.model_dump_json()

        # Deserialize back
        restored = SessionModel.model_validate_json(json_str)

        # Verify primary key fields
        assert restored.user_id == session.user_id
        assert restored.session_id == session.session_id

        # Verify data fields
        assert restored.context == session.context
        assert dict(restored.storage.items()) == dict(session.storage.items())
        assert restored.root_agent_id == session.root_agent_id

        # Verify timestamps are preserved (comparing ISO format strings)
        # After JSON round-trip, datetime may be string or datetime
        restored_created = restored.created_at if isinstance(restored.created_at, str) else restored.created_at.isoformat()
        restored_updated = restored.updated_at if isinstance(restored.updated_at, str) else restored.updated_at.isoformat()
        assert restored_created == session.created_at.isoformat()
        assert restored_updated == session.updated_at.isoformat()

    @settings(max_examples=100)
    @given(agent=agent_model_strategy)
    def test_agent_model_serialization_round_trip(self, agent: AgentModel):
        """Test AgentModel serialization round-trip.

        *For any* AgentModel instance, serializing with `model_dump_json()` then
        deserializing with `model_validate_json()` should produce an equivalent object.

        **Validates: Requirements 2.7**
        """
        # Serialize to JSON
        json_str = agent.model_dump_json()

        # Deserialize back
        restored = AgentModel.model_validate_json(json_str)

        # Verify primary key fields
        assert restored.user_id == agent.user_id
        assert restored.session_id == agent.session_id
        assert restored.agent_id == agent.agent_id

        # Verify metadata fields
        assert restored.agent_name == agent.agent_name

        # Verify timestamps are preserved
        # After JSON round-trip, datetime may be string or datetime
        restored_created = restored.created_at if isinstance(restored.created_at, str) else restored.created_at.isoformat()
        restored_updated = restored.last_updated if isinstance(restored.last_updated, str) else restored.last_updated.isoformat()
        assert restored_created == agent.created_at.isoformat()
        assert restored_updated == agent.last_updated.isoformat()

    @settings(max_examples=100)
    @given(record=run_action_model_strategy)
    def test_run_action_model_serialization_round_trip(self, record: AgentRunActionModel):
        """Test AgentRunActionModel serialization round-trip.

        *For any* AgentRunActionModel instance, serializing with `model_dump_json()`
        then deserializing with `model_validate(json.loads(...))` should produce an equivalent object.

        Note: We use model_validate(json.loads(...)) instead of model_validate_json() because
        Pydantic v2's model_validate_json() uses Rust-based parsing that bypasses Python validators.

        **Validates: Requirements 5.3, 5.4**
        """
        import json

        # Serialize to JSON
        json_str = record.model_dump_json()

        # Deserialize back using model_validate + json.loads (not model_validate_json)
        restored = AgentRunActionModel.model_validate(json.loads(json_str))

        # Verify primary key fields
        assert restored.user_id == record.user_id
        assert restored.session_id == record.session_id
        assert restored.agent_id == record.agent_id
        assert restored.run_id == record.run_id

        # Verify run tracking
        assert restored.root_run_id == record.root_run_id
        assert restored.parent_run_id == record.parent_run_id

        # Verify action type
        assert restored.action_type == record.action_type

        # Verify action fields
        if record.append_messages is not None:
            assert restored.append_messages is not None
            assert len(restored.append_messages) == len(record.append_messages)
            for restored_msg, original_msg in zip(restored.append_messages, record.append_messages):
                assert restored_msg.role == original_msg.role
                assert restored_msg.content == original_msg.content
                assert restored_msg.metadata == original_msg.metadata
        else:
            assert restored.append_messages is None

        assert restored.replace_messages == record.replace_messages
        assert restored.undo_before_run_id == record.undo_before_run_id

        # Verify metadata
        assert restored.agent_name == record.agent_name

        # Verify timestamp (nanosecond precision)
        assert restored.created_at_ns == record.created_at_ns
