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

"""Tests for HistoryList functionality."""

import asyncio

import pytest

from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.session import AgentRunActionKey, AgentRunActionModel, SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.core.messages import Message


@pytest.fixture
def engine():
    """Create an in-memory database engine."""
    return InMemoryDatabaseEngine()


@pytest.fixture
def session_manager(engine):
    """Create a SessionManager instance."""
    return SessionManager(engine=engine)


@pytest.fixture
def history_key():
    """Create a test history key."""
    return AgentRunActionKey(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
    )


def test_history_list_append(session_manager, history_key, engine):
    """Test that append automatically persists messages on flush."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Append a message
        msg = Message.user("hello")
        history.append(msg)

        # Flush to persist
        history.flush()

        # Give the async task time to complete
        await asyncio.sleep(0.01)

        # Verify it was persisted
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 1
        assert loaded[0].get_text_content() == "hello"

    asyncio.run(run())


def test_history_list_extend(session_manager, history_key, engine):
    """Test that extend automatically persists messages on flush."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Extend with messages
        msgs = [Message.user("msg1"), Message.assistant("msg2")]
        history.extend(msgs)

        # Flush to persist
        history.flush()

        # Give the async task time to complete
        await asyncio.sleep(0.01)

        # Verify they were persisted
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 2
        assert loaded[0].get_text_content() == "msg1"
        assert loaded[1].get_text_content() == "msg2"

    asyncio.run(run())


def test_history_list_setitem_single(session_manager, history_key, engine):
    """Test that single index assignment doesn't crash but doesn't persist (run-level API limitation)."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Create initial history
        msg1 = Message.user("original")
        await session_manager.agent_run_action.persist_append(
            key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[msg1],
        )

        history = HistoryList(
            [msg1],
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_002",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Update via index assignment (should not crash)
        new_msg = Message.user("updated")
        history[0] = new_msg

        # Give the async task time to complete
        await asyncio.sleep(0.01)

        # Verify local update worked
        assert history[0].get_text_content() == "updated"

        # But persistence doesn't support message-level updates in run-level API
        # So the stored version should still be "original"
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 1
        assert loaded[0].get_text_content() == "original"

    asyncio.run(run())


def test_history_list_replace_all_append(session_manager, history_key, engine):
    """Test that replace_all without update_baseline triggers smart append detection on flush."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Create initial history in storage
        msg1 = Message.user("msg1")
        await session_manager.agent_run_action.persist_append(
            key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[msg1],
        )

        history = HistoryList(
            [msg1],
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_002",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Replace with append pattern (old messages + new message)
        # Without update_baseline=True, flush will detect this as append
        msg2 = Message.assistant("msg2")
        history.replace_all([msg1, msg2])

        # flush after replace_all should detect msg2 as new and persist it
        history.flush()
        await asyncio.sleep(0.01)

        # Verify msg2 was persisted (smart append detection)
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 2
        assert loaded[0].get_text_content() == "msg1"
        assert loaded[1].get_text_content() == "msg2"

    asyncio.run(run())


def test_history_list_replace_all_true_replace(session_manager, history_key, engine):
    """Test that replace_all without update_baseline triggers smart replace detection on flush."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Create initial history in storage
        msg1 = Message.user("msg1")
        msg2 = Message.assistant("msg2")
        await session_manager.agent_run_action.persist_append(
            key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[msg1, msg2],
        )

        history = HistoryList(
            [msg1, msg2],
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_002",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Replace with different messages (true replacement)
        # Without update_baseline=True, flush will detect this as replacement
        msg3 = Message.user("msg3")
        history.replace_all([msg3])

        # flush after replace_all should detect this as true replacement and persist
        history.flush()
        await asyncio.sleep(0.01)

        # Verify replacement was persisted
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 1
        assert loaded[0].get_text_content() == "msg3"

    asyncio.run(run())


def test_history_list_mutable_message_edit_persists(session_manager, history_key, engine):
    async def run():
        await engine.setup_models([AgentRunActionModel])

        msg1 = Message.user("msg1")
        await session_manager.agent_run_action.persist_append(
            key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[msg1],
        )

        history = HistoryList(
            [msg1],
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_002",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        msg1.content[0].text = "edited"
        history.flush()
        await asyncio.sleep(0.01)

        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 1
        assert loaded[0].get_text_content() == "edited"

    asyncio.run(run())


def test_history_list_flush_under_existing_agent_lock(session_manager, history_key, engine):
    async def run():
        await session_manager.setup_models()

        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_lock_001",
            root_run_id="run_lock_001",
            agent_name="test_agent",
        )

        async with session_manager.agent_lock.acquire(
            session_id=history_key.session_id,
            agent_id=history_key.agent_id,
            user_id=history_key.user_id,
            run_id="run_lock_001",
        ):
            history.append(Message.user("hello"))
            history.flush()
            await asyncio.sleep(0.01)

        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 1
        assert loaded[0].get_text_content() == "hello"

    asyncio.run(run())


def test_history_list_without_persistence():
    """Test that HistoryList works without SessionManager (no persistence)."""
    history = HistoryList()

    # Should work like a normal list
    msg1 = Message.user("msg1")
    msg2 = Message.assistant("msg2")

    history.append(msg1)
    history.extend([msg2])

    assert len(history) == 2
    assert history[0].get_text_content() == "msg1"
    assert history[1].get_text_content() == "msg2"

    # Update should work
    msg3 = Message.user("msg3")
    history[0] = msg3
    assert history[0].get_text_content() == "msg3"


def test_history_list_setitem_slice_small(session_manager, history_key, engine):
    """Test that slice assignment with small changes doesn't crash but doesn't persist (run-level API limitation)."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Create initial history with 10 messages
        messages = [Message.user(f"msg{i}") for i in range(10)]
        await session_manager.agent_run_action.persist_append(
            key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=messages,
        )

        history = HistoryList(
            messages,
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_002",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Update 2 messages via slice (20% of history - should not crash)
        new_messages = [Message.user("updated0"), Message.user("updated1")]
        history[0:2] = new_messages

        # Give the async task time to complete
        await asyncio.sleep(0.01)

        # Verify local update worked
        assert history[0].get_text_content() == "updated0"
        assert history[1].get_text_content() == "updated1"

        # But persistence doesn't support message-level updates in run-level API
        # So the stored version should still be original
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 10
        assert loaded[0].get_text_content() == "msg0"
        assert loaded[1].get_text_content() == "msg1"
        assert loaded[2].get_text_content() == "msg2"

    asyncio.run(run())


def test_history_list_setitem_slice_large(session_manager, history_key, engine):
    """Test that slice assignment doesn't persist (run-level API limitation)."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Create initial history with 10 messages
        messages = [Message.user(f"msg{i}") for i in range(10)]
        await session_manager.agent_run_action.persist_append(
            key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=messages,
        )

        history = HistoryList(
            messages,
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_002",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Update 5 messages via slice (local only, no persistence)
        new_messages = [Message.user(f"updated{i}") for i in range(5)]
        history[0:5] = new_messages

        # Give the async task time to complete
        await asyncio.sleep(0.01)

        # Verify local update worked
        assert len(history) == 10
        assert history[0].get_text_content() == "updated0"
        assert history[4].get_text_content() == "updated4"
        assert history[5].get_text_content() == "msg5"

        # But persistence doesn't support slice assignment in run-level API
        # So the stored version should still be original
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 10
        assert loaded[0].get_text_content() == "msg0"
        assert loaded[4].get_text_content() == "msg4"
        assert loaded[5].get_text_content() == "msg5"

    asyncio.run(run())


def test_history_list_replace_all_updates_baseline_fingerprints(session_manager, history_key, engine):
    """Test that replace_all updates baseline fingerprints so subsequent appends only persist new messages.

    This is a regression test for the bug where each run's persist_append would include
    all previous messages because _baseline_fingerprints was not updated after replace_all.
    """

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Simulate first run: create empty HistoryList, then replace_all with loaded history
        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Simulate loading history from storage and setting it via replace_all
        # (This is what Agent.run() does: self._history.replace_all([system_msg] + stored_messages, update_baseline=True))
        msg1 = Message.user("msg1")
        msg2 = Message.assistant("msg2")
        history.replace_all([msg1, msg2], update_baseline=True)

        # Now append a new message (simulating user input in this run)
        msg3 = Message.user("msg3")
        history.append(msg3)

        # Flush to persist
        history.flush()
        await asyncio.sleep(0.01)

        # Verify only the new message (msg3) was persisted, not all messages
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 1, f"Expected 1 message (only new), got {len(loaded)}"
        assert loaded[0].get_text_content() == "msg3"

    asyncio.run(run())


def test_history_list_multi_run_no_duplicate_messages(session_manager, history_key, engine):
    """Test that multiple runs don't accumulate duplicate messages in storage.

    This simulates the real-world scenario where:
    1. Run 1: User sends message, agent responds
    2. Run 2: Load history, user sends another message, agent responds
    3. Run 3: Load history, user sends another message, agent responds

    Each run should only persist its own new messages, not re-persist previous messages.
    """

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # === Run 1 ===
        history1 = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # User message and assistant response
        history1.append(Message.user("hello"))
        history1.append(Message.assistant("hi there"))
        history1.flush()
        await asyncio.sleep(0.01)

        # Verify run 1 persisted 2 messages
        loaded1 = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded1) == 2

        # === Run 2 ===
        history2 = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_002",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Simulate loading history from storage (like Agent.run does)
        history2.replace_all(loaded1, update_baseline=True)

        # User sends another message
        history2.append(Message.user("how are you"))
        history2.append(Message.assistant("I am fine"))
        history2.flush()
        await asyncio.sleep(0.01)

        # Verify total is 4 messages (not 6 if duplicates were persisted)
        loaded2 = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded2) == 4, f"Expected 4 messages, got {len(loaded2)}"

        # === Run 3 ===
        history3 = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_003",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Simulate loading history from storage
        history3.replace_all(loaded2, update_baseline=True)

        # User sends another message
        history3.append(Message.user("goodbye"))
        history3.append(Message.assistant("bye"))
        history3.flush()
        await asyncio.sleep(0.01)

        # Verify total is 6 messages (not 12 if duplicates were persisted)
        loaded3 = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded3) == 6, f"Expected 6 messages, got {len(loaded3)}"

        # Verify message order is correct
        assert loaded3[0].get_text_content() == "hello"
        assert loaded3[1].get_text_content() == "hi there"
        assert loaded3[2].get_text_content() == "how are you"
        assert loaded3[3].get_text_content() == "I am fine"
        assert loaded3[4].get_text_content() == "goodbye"
        assert loaded3[5].get_text_content() == "bye"

    asyncio.run(run())


def test_history_list_setitem_slice_with_single_message():
    """Test __setitem__ with slice assignment using a single Message."""
    history = HistoryList()
    history.append(Message.user("msg1"))
    history.append(Message.user("msg2"))

    # Assign single message to slice
    new_msg = Message.user("new_msg")
    history[0:1] = new_msg

    assert len(history) == 2
    assert history[0].get_text_content() == "new_msg"


def test_history_list_setitem_index_with_list():
    """Test __setitem__ with index assignment using a list."""
    history = HistoryList()
    history.append(Message.user("msg1"))
    history.append(Message.user("msg2"))

    # Assign list to index (takes first element)
    new_msg = Message.user("new_msg")
    history[0] = [new_msg]

    assert len(history) == 2
    assert history[0].get_text_content() == "new_msg"


def test_history_list_flush_without_persistence():
    """Test flush when persistence is not enabled."""
    history = HistoryList()

    # Should not raise or do anything
    history.append(Message.user("msg1"))
    history.flush()

    # Just verify it didn't crash and list is intact
    assert len(history) == 1


def test_history_list_update_context(session_manager, history_key, engine):
    """Test update_context method updates run context."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Add a message
        history.append(Message.user("hello"))

        # Update context
        history.update_context(
            run_id="run_002",
            root_run_id="run_001",
            parent_run_id="parent_001",
        )

        # Verify context was updated
        assert history._run_id == "run_002"
        assert history._root_run_id == "run_001"
        assert history._parent_run_id == "parent_001"

        # Wait for flush to complete
        await asyncio.sleep(0.01)

    asyncio.run(run())


def test_history_list_persist_flush_async_early_returns(session_manager, history_key, engine):
    """Test _persist_flush_async early returns."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Test with empty replace_messages (should return early)
        await history._persist_flush_async(append_messages=None, replace_messages=[])

        # Test with empty append_messages (should return early)
        await history._persist_flush_async(append_messages=[], replace_messages=None)

        # Verify nothing was persisted
        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 0

    asyncio.run(run())


def test_history_list_persist_flush_async_no_session_manager():
    """Test _persist_flush_async when session_manager is None."""

    async def run():
        history = HistoryList()

        # Should return early without error
        await history._persist_flush_async(append_messages=[Message.user("test")], replace_messages=None)

    asyncio.run(run())


def test_history_list_schedule_async_no_running_loop():
    """Test _schedule_async when there's no running event loop."""
    from unittest.mock import Mock, patch

    history = HistoryList()

    # Mock coroutine
    mock_coro = Mock()

    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no running loop")):
        with patch("asyncio.run") as mock_run:
            history._schedule_async(mock_coro)
            mock_run.assert_called_once_with(mock_coro)


def test_history_list_persist_flush_async_exception_handling(session_manager, history_key, engine):
    """Test _persist_flush_async handles exceptions gracefully."""
    from unittest.mock import AsyncMock, patch

    async def run():
        await engine.setup_models([AgentRunActionModel])

        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="test_agent",
        )

        # Mock persist_append to raise an exception
        with patch.object(
            session_manager.agent_run_action,
            "persist_append",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Database error"),
        ):
            # Should not raise, just log the error
            await history._persist_flush_async(
                append_messages=[Message.user("test")],
                replace_messages=None,
            )

    asyncio.run(run())
