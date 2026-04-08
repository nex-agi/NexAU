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

"""Unit tests for /resume runtime commands in CLI agent runner."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nexau.cli.agent_runner import CliAgentRuntime, CliSessionStore


def _build_runtime_stub() -> CliAgentRuntime:
    runtime = CliAgentRuntime.__new__(CliAgentRuntime)
    runtime._session_metadata = Mock(return_value={"session_id": "sess-1"})
    runtime.resume = Mock()
    return runtime


class TestResumeCommands:
    def test_handle_resume_command_default(self):
        runtime = _build_runtime_stub()
        runtime.handle_resume_command("/resume")
        runtime.resume.assert_called_once_with()

    def test_handle_resume_command_invalid_usage(self):
        runtime = _build_runtime_stub()
        with pytest.raises(RuntimeError, match="Usage: /resume"):
            runtime.handle_resume_command("/resume 2")

    def test_resolve_resume_target_prefers_current(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        runtime._session_store = Mock()
        runtime._session_store.list_sessions.return_value = [
            {"session_id": "local_1", "current": False},
            {"session_id": "local_2", "current": True},
        ]

        assert runtime._resolve_resume_target_session_id() == "local_2"

    def test_resolve_resume_target_falls_back_to_latest(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        runtime._session_store = Mock()
        runtime._session_store.list_sessions.return_value = [
            {"session_id": "local_1", "current": False},
            {"session_id": "local_2", "current": False},
        ]

        assert runtime._resolve_resume_target_session_id() == "local_1"

    def test_resolve_resume_target_no_sessions_raises(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        runtime._session_store = Mock()
        runtime._session_store.list_sessions.return_value = []

        with pytest.raises(RuntimeError, match="No saved sessions found"):
            runtime._resolve_resume_target_session_id()

    def test_list_resume_sessions_returns_structured_items(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        runtime.is_busy = Mock(return_value=False)
        runtime._persist_state = Mock()
        runtime._session_store = Mock()
        runtime._session_store.list_sessions.return_value = [
            {
                "session_id": "local_1",
                "created_at": "2026-04-08T08:00:00",
                "updated_at": "2026-04-08T09:00:00",
                "first_user_input": "first hello",
                "preview": "hello world",
                "current": True,
            },
            {
                "session_id": "local_2",
                "updated_at": "2026-04-08T07:00:00",
                "cwd": "/repo/project",
                "preview": "",
                "current": False,
            },
        ]
        runtime._session_store.load_session.side_effect = (
            lambda session_id: {
                "history": [
                    {"role": "user", "content": [{"type": "text", "text": "earliest user input"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
                ]
            }
            if session_id == "local_2"
            else None
        )

        result = runtime.list_resume_sessions()

        assert result == [
            {
                "index": 1,
                "id": "local_1",
                "created_at": "2026-04-08T08:00:00",
                "updated_at": "2026-04-08T09:00:00",
                "conversation": "first hello",
                "current": True,
            },
            {
                "index": 2,
                "id": "local_2",
                "created_at": "2026-04-08T07:00:00",
                "updated_at": "2026-04-08T07:00:00",
                "conversation": "earliest user input",
                "current": False,
            },
        ]


class TestSessionStoreLoadCurrent:
    def test_load_current_uses_current_session_id_when_present(self):
        store = CliSessionStore.__new__(CliSessionStore)
        store._load_manifest = Mock(return_value={"current_session_id": "local_current", "sessions": []})
        store.load_session = Mock(return_value={"session_id": "local_current"})

        snapshot = store.load_current()

        assert snapshot == {"session_id": "local_current"}
        store.load_session.assert_called_once_with("local_current")

    def test_load_current_falls_back_to_latest_same_cwd(self):
        store = CliSessionStore.__new__(CliSessionStore)
        store._load_manifest = Mock(
            return_value={
                "current_session_id": "",
                "sessions": [
                    {"session_id": "other_cwd", "updated_at": "2026-04-08T10:00:00", "cwd": "/tmp/other"},
                    {"session_id": "same_cwd", "updated_at": "2026-04-08T11:00:00", "cwd": "/repo/current"},
                ],
            }
        )
        store._sort_entries = Mock(
            return_value=[
                {"session_id": "same_cwd", "updated_at": "2026-04-08T11:00:00", "cwd": "/repo/current"},
                {"session_id": "other_cwd", "updated_at": "2026-04-08T10:00:00", "cwd": "/tmp/other"},
            ]
        )
        store.load_session = Mock(return_value={"session_id": "same_cwd"})

        with patch("nexau.cli.agent_runner._project_root", "/repo/current"):
            snapshot = store.load_current()

        assert snapshot == {"session_id": "same_cwd"}
        store.load_session.assert_called_once_with("same_cwd")


class TestSessionStoreJsonlEvents:
    def test_snapshot_paths_use_per_session_directory_layout(self, tmp_path):
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        store = CliSessionStore(config_path=config_path, user_id="cli_user")

        paths = store.snapshot_paths("local_20260408_100000_000001")

        assert paths.file_path.name == "messages.jsonl"
        assert paths.readable_path.name == "session_checkpoint.json"
        assert paths.file_path.parent == paths.readable_path.parent

    def test_append_jsonl_snapshot_writes_message_events(self, tmp_path):
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        store = CliSessionStore(config_path=config_path, user_id="cli_user")

        jsonl_path = tmp_path / "sessions" / "rollout.jsonl"
        checkpoint_path = tmp_path / "sessions" / "session_checkpoint.json"
        agent = SimpleNamespace(
            agent_id="agent-1",
            config=SimpleNamespace(llm_config=SimpleNamespace(model="gpt-test")),
        )

        payload = {
            "session_id": "local_1",
            "updated_at": "2026-04-08T10:00:00",
            "history": [
                {
                    "id": "m-user-1",
                    "role": "user",
                    "created_at": "2026-04-08T10:00:00",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "id": "m-assistant-1",
                    "role": "assistant",
                    "created_at": "2026-04-08T10:00:01",
                    "content": [
                        {"type": "text", "text": "calling tool"},
                        {"type": "tool_use", "id": "call-1", "name": "search", "input": {"q": "nexau"}},
                    ],
                },
                {
                    "id": "m-tool-1",
                    "role": "tool",
                    "created_at": "2026-04-08T10:00:02",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "ok",
                            "is_error": False,
                        }
                    ],
                },
            ],
            "global_storage": {},
            "last_context": {},
        }

        store._append_jsonl_snapshot(
            jsonl_path,
            checkpoint_path=checkpoint_path,
            payload=payload,
            agent=agent,
        )

        events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        event_types = [event.get("type") for event in events]

        assert "session_meta" in event_types
        assert "user_message" in event_types
        assert "agent_message" in event_types
        assert "function_call" in event_types
        assert "function_result" in event_types
        assert "session_checkpoint" not in event_types
        assert event_types[-1] == "checkpoint_ref"

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["type"] == "session_checkpoint"
        assert checkpoint["payload"]["session_id"] == "local_1"

    def test_append_jsonl_snapshot_appends_only_new_message_events(self, tmp_path):
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        store = CliSessionStore(config_path=config_path, user_id="cli_user")

        jsonl_path = tmp_path / "sessions" / "rollout.jsonl"
        checkpoint_path = tmp_path / "sessions" / "session_checkpoint.json"
        agent = SimpleNamespace(
            agent_id="agent-1",
            config=SimpleNamespace(llm_config=SimpleNamespace(model="gpt-test")),
        )

        first_payload = {
            "session_id": "local_1",
            "updated_at": "2026-04-08T10:00:00",
            "history": [
                {
                    "id": "m-user-1",
                    "role": "user",
                    "created_at": "2026-04-08T10:00:00",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "global_storage": {},
            "last_context": {},
        }
        second_payload = {
            **first_payload,
            "updated_at": "2026-04-08T10:01:00",
            "history": [
                *first_payload["history"],
                {
                    "id": "m-assistant-2",
                    "role": "assistant",
                    "created_at": "2026-04-08T10:01:00",
                    "content": [{"type": "text", "text": "hi"}],
                },
            ],
        }

        store._append_jsonl_snapshot(
            jsonl_path,
            checkpoint_path=checkpoint_path,
            payload=first_payload,
            agent=agent,
        )
        store._append_jsonl_snapshot(
            jsonl_path,
            checkpoint_path=checkpoint_path,
            payload=second_payload,
            agent=agent,
        )

        events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        user_events = [event for event in events if event.get("type") == "user_message"]
        agent_events = [event for event in events if event.get("type") == "agent_message"]
        checkpoint_refs = [event for event in events if event.get("type") == "checkpoint_ref"]

        assert len(user_events) == 1
        assert len(agent_events) == 1
        assert len(checkpoint_refs) == 2

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["type"] == "session_checkpoint"
        assert len(checkpoint["payload"]["history"]) == 2
