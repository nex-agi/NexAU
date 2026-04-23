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
Agent runner for NexAU CLI with real-time progress tracking.
This script runs the NexAU agent and communicates via stdin/stdout.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.main_sub.config import (
    ConfigError,
)
from nexau.archs.main_sub.config.config import AgentConfigBuilder
from nexau.archs.main_sub.config.schema import normalize_agent_config_dict
from nexau.archs.main_sub.execution.hooks import (
    AfterModelHookInput,
    AfterModelHookResult,
    AfterToolHookInput,
    AfterToolHookResult,
)
from nexau.archs.main_sub.utils import load_yaml_with_vars
from nexau.archs.session.models.serialization_utils import sanitize_for_serialization
from nexau.cli.cli_subagent_adapter import attach_cli_to_agent
from nexau.core.messages import Message, Role

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from nexau.archs.main_sub.agent import Agent
    from nexau.archs.main_sub.execution.stop_result import StopResult

logger = logging.getLogger(__name__)


# Ensure the active project root is importable.
#
# When Python runs a script by filename, it adds the script directory to sys.path,
# not the current working directory. The workbench runs this runner from a
# project/worktree and expects project-local packages (e.g. `tools.*`) to be
# importable.
_project_root = os.environ.get("projectRoot") or os.environ.get("PROJECT_ROOT") or os.getcwd()
if _project_root and _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _ensure_config_dir_on_sys_path(config_path: Path) -> None:
    """Ensure local config modules (e.g. `tool_impl.*`) are importable."""
    config_dir = str(config_path.resolve().parent)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)


def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_stdout_lock = threading.Lock()
_CLI_TRANSIENT_STORAGE_KEYS = {"tracer", "skill_registry"}


def send_message(msg_type, content, metadata=None):
    """Send a JSON message to stdout."""
    message = {"type": msg_type, "content": content}
    if metadata:
        message["metadata"] = metadata
    with _stdout_lock:
        print(json.dumps(message), flush=True)


def _resolve_cli_snapshot_dir() -> Path:
    env_dir = os.environ.get("NEXAU_CLI_STATE_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    candidates.extend(
        [
            Path(_project_root) / ".nexau" / "cli-sessions",
            Path.home() / ".nexau" / "cli-sessions",
            Path("/tmp") / "nexau" / "cli-sessions",
        ]
    )

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue

    raise OSError("Unable to create a writable CLI session snapshot directory")


class CliSessionStore:
    """Multi-session snapshot store for CLI resume (JSON/JSONL only)."""

    @dataclass(frozen=True)
    class SnapshotPaths:
        file_path: Path
        readable_path: Path

    def __init__(self, *, config_path: Path, user_id: str):
        self.config_path = config_path.resolve()
        self.user_id = user_id
        self.base_dir = _resolve_cli_snapshot_dir()
        self.store_key = self._build_store_key()
        self.store_dir = self.base_dir / self.store_key
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.store_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.store_dir / "manifest.json"
        self.manifest_readable_path = self.manifest_path
        legacy_file_name = f"{self.store_key}.json"
        self.legacy_file_path = self.base_dir / legacy_file_name
        self.legacy_readable_path = self.legacy_file_path
        self._jsonl_history_cursor: dict[str, int] = {}

    @staticmethod
    def _slugify(value: str, *, default: str) -> str:
        safe_value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-_")
        return safe_value or default

    def _build_store_key(self) -> str:
        base_name = self.config_path.stem or "agent"
        safe_name = self._slugify(base_name, default="agent")
        digest = hashlib.sha1(f"{self.config_path}:{self.user_id}".encode()).hexdigest()[:10]
        return f"{safe_name}-{digest}"

    def snapshot_paths(self, session_id: str) -> SnapshotPaths:
        manifest = self._load_manifest()
        for entry in manifest.get("sessions", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("session_id") != session_id:
                continue
            existing_paths = self._manifest_entry_paths(entry)
            if existing_paths is not None:
                return existing_paths

        session_slug = self._slugify(session_id, default="session")
        if len(session_slug) > 64:
            digest = hashlib.sha1(session_id.encode()).hexdigest()[:12]
            session_slug = f"{session_slug[:48]}-{digest}"
        session_dir = self.sessions_dir / session_slug
        session_dir.mkdir(parents=True, exist_ok=True)
        file_path = session_dir / "messages.jsonl"
        checkpoint_path = session_dir / "session_checkpoint.json"
        return self.SnapshotPaths(file_path=file_path, readable_path=checkpoint_path)

    def _read_jsonl_snapshot(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None

        latest_checkpoint: dict[str, Any] | None = None
        session_meta_payload: dict[str, Any] | None = None
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        continue
                    event_type = event.get("type")
                    payload = event.get("payload")
                    if event_type == "session_meta" and isinstance(payload, dict):
                        session_meta_payload = payload
                        continue
                    if event_type == "session_checkpoint" and isinstance(payload, dict):
                        latest_checkpoint = payload
                        continue
                    # Compatibility path: a pure snapshot object stored as one jsonl line.
                    if "history" in event and "session_id" in event:
                        latest_checkpoint = event
        except Exception as exc:
            logger.warning("Failed to parse CLI session jsonl snapshot from %s: %s", path, exc)
            return None

        if latest_checkpoint is None:
            return None

        snapshot = dict(latest_checkpoint)
        if session_meta_payload:
            snapshot.setdefault("session_id", session_meta_payload.get("id", ""))
            snapshot.setdefault("agent_config_path", session_meta_payload.get("agent_config_path", str(self.config_path)))
            snapshot.setdefault("user_id", session_meta_payload.get("user_id", self.user_id))
        return snapshot

    def _read_snapshot_file(self, path: Path) -> dict[str, Any] | None:
        if path.suffix == ".jsonl":
            return self._read_jsonl_snapshot(path)
        return self._read_json_file(path)

    def _read_checkpoint_file(self, path: Path) -> dict[str, Any] | None:
        if not path.exists() or path.suffix == ".jsonl":
            return None
        data = self._read_json_file(path)
        if not isinstance(data, dict):
            return None

        if data.get("type") == "session_checkpoint" and isinstance(data.get("payload"), dict):
            return data["payload"]

        if "history" in data and "session_id" in data:
            return data
        return None

    def _extract_text_blocks(self, content: Any) -> str:
        if not isinstance(content, list):
            return ""
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        return "".join(texts).strip()

    def _build_message_events(
        self,
        message_payload: dict[str, Any],
        *,
        timestamp: str,
        message_index: int,
    ) -> list[dict[str, Any]]:
        role = message_payload.get("role")
        if not isinstance(role, str):
            return []

        content = message_payload.get("content")
        blocks = content if isinstance(content, list) else []
        text = self._extract_text_blocks(blocks)
        message_id = message_payload.get("id", "")
        created_at = message_payload.get("created_at", "")
        base_payload = {
            "message_id": message_id,
            "message_index": message_index,
            "role": role,
            "created_at": created_at,
        }

        events: list[dict[str, Any]] = []

        if role in {Role.USER.value, Role.FRAMEWORK.value}:
            events.append(
                {
                    "timestamp": timestamp,
                    "type": "user_message",
                    "payload": {
                        **base_payload,
                        "text": text,
                    },
                }
            )
            return events

        if role == Role.ASSISTANT.value:
            events.append(
                {
                    "timestamp": timestamp,
                    "type": "agent_message",
                    "payload": {
                        **base_payload,
                        "text": text,
                    },
                }
            )
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                events.append(
                    {
                        "timestamp": timestamp,
                        "type": "function_call",
                        "payload": {
                            **base_payload,
                            "call_id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                            "raw_arguments": block.get("raw_input"),
                        },
                    }
                )
            return events

        if role == Role.TOOL.value:
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                events.append(
                    {
                        "timestamp": timestamp,
                        "type": "function_result",
                        "payload": {
                            **base_payload,
                            "tool_use_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                            "is_error": bool(block.get("is_error", False)),
                        },
                    }
                )
            return events

        # Keep non-standard roles observable in log stream.
        if text:
            events.append(
                {
                    "timestamp": timestamp,
                    "type": "message",
                    "payload": {
                        **base_payload,
                        "text": text,
                    },
                }
            )
        return events

    def _latest_history_count_in_jsonl(self, path: Path, *, checkpoint_path: Path) -> int:
        cache_key = str(path.resolve())
        cached = self._jsonl_history_cursor.get(cache_key)
        if cached is not None:
            return cached

        checkpoint = self._read_checkpoint_file(checkpoint_path)
        if isinstance(checkpoint, dict):
            history = checkpoint.get("history")
            if isinstance(history, list):
                history_count = len(history)
                self._jsonl_history_cursor[cache_key] = history_count
                return history_count

        history_count = 0
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            continue
                        payload = event.get("payload")
                        if isinstance(payload, dict):
                            message_index = payload.get("message_index")
                            if isinstance(message_index, int):
                                history_count = max(history_count, message_index + 1)
            except Exception as exc:
                logger.warning("Failed to infer history cursor from %s: %s", path, exc)
                history_count = 0

        self._jsonl_history_cursor[cache_key] = history_count
        return history_count

    def _write_checkpoint_file(self, path: Path, *, payload: dict[str, Any], timestamp: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "timestamp": timestamp,
            "type": "session_checkpoint",
            "payload": payload,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _append_jsonl_snapshot(
        self,
        path: Path,
        *,
        checkpoint_path: Path,
        payload: dict[str, Any],
        agent: Agent,
    ) -> None:
        now = datetime.now().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_key = str(path.resolve())
        previous_history_count = self._latest_history_count_in_jsonl(path, checkpoint_path=checkpoint_path)
        raw_history = payload.get("history")
        history_payload: list[Any] = raw_history if isinstance(raw_history, list) else []

        start_index = previous_history_count
        if previous_history_count > len(history_payload):
            start_index = 0

        with path.open("a", encoding="utf-8") as f:
            if path.stat().st_size == 0:
                session_meta = {
                    "timestamp": now,
                    "type": "session_meta",
                    "payload": {
                        "id": payload.get("session_id", ""),
                        "timestamp": now,
                        "cwd": _project_root,
                        "source": "nexau-cli",
                        "agent_config_path": str(self.config_path),
                        "user_id": self.user_id,
                        "agent_id": agent.agent_id,
                        "model": getattr(agent.config.llm_config, "model", ""),
                    },
                }
                f.write(json.dumps(session_meta, ensure_ascii=False) + "\n")

            for message_index, history_item in enumerate(history_payload[start_index:], start=start_index):
                if not isinstance(history_item, dict):
                    continue
                for event in self._build_message_events(
                    history_item,
                    timestamp=now,
                    message_index=message_index,
                ):
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

            checkpoint_ref = {
                "timestamp": now,
                "type": "checkpoint_ref",
                "payload": {
                    "checkpoint_path": str(checkpoint_path),
                    "history_count": len(history_payload),
                },
            }
            f.write(json.dumps(checkpoint_ref, ensure_ascii=False) + "\n")
        self._write_checkpoint_file(checkpoint_path, payload=payload, timestamp=now)
        self._jsonl_history_cursor[cache_key] = len(history_payload)

    def _read_json_file(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load CLI session snapshot from %s: %s", path, exc)
            return None

        if not isinstance(data, dict):
            return None

        return data

    def _load_manifest(self) -> dict[str, Any]:
        data = self._read_json_file(self.manifest_path)
        if not data:
            legacy_snapshot = self._load_legacy_snapshot()
            if legacy_snapshot is None:
                return self._empty_manifest()
            legacy_entry = self._snapshot_entry(
                payload=legacy_snapshot,
                paths=self.SnapshotPaths(
                    file_path=self.legacy_file_path,
                    readable_path=self.legacy_readable_path,
                ),
            )
            return {
                **self._empty_manifest(),
                "updated_at": legacy_snapshot.get("updated_at", ""),
                "current_session_id": legacy_entry["session_id"],
                "sessions": [legacy_entry],
            }

        if data.get("agent_config_path") != str(self.config_path):
            return self._empty_manifest()
        if data.get("user_id") != self.user_id:
            return self._empty_manifest()

        sessions = data.get("sessions")
        if not isinstance(sessions, list):
            sessions = []

        return {
            "version": data.get("version", 2),
            "agent_config_path": str(self.config_path),
            "user_id": self.user_id,
            "updated_at": data.get("updated_at", ""),
            "current_session_id": data.get("current_session_id", "") or "",
            "sessions": [entry for entry in sessions if isinstance(entry, dict)],
        }

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "version": 2,
            "agent_config_path": str(self.config_path),
            "user_id": self.user_id,
            "updated_at": "",
            "current_session_id": "",
            "sessions": [],
        }

    def _load_legacy_snapshot(self) -> dict[str, Any] | None:
        data = self._read_json_file(self.legacy_file_path)
        if not data:
            return None
        if data.get("agent_config_path") != str(self.config_path):
            return None
        if data.get("user_id") != self.user_id:
            return None
        return data

    def _manifest_entry_paths(self, entry: dict[str, Any]) -> SnapshotPaths | None:
        json_path = entry.get("json_path")
        readable_path = entry.get("readable_path")
        if not isinstance(json_path, str):
            return None
        if not isinstance(readable_path, str) or not readable_path:
            readable_path = json_path
        return self.SnapshotPaths(
            file_path=(self.store_dir / json_path).resolve(),
            readable_path=(self.store_dir / readable_path).resolve(),
        )

    def _snapshot_entry(self, *, payload: dict[str, Any], paths: SnapshotPaths) -> dict[str, Any]:
        history = payload.get("history")
        history_payload = history if isinstance(history, list) else []
        preview = self._build_preview(history_payload)
        first_user_input = self._build_first_user_input(history_payload)
        created_at = payload.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = payload.get("updated_at", "")
        last_context = payload.get("last_context")
        cwd = ""
        if isinstance(last_context, dict):
            cwd_value = last_context.get("working_directory")
            if isinstance(cwd_value, str):
                cwd = cwd_value
        return {
            "session_id": payload.get("session_id", ""),
            "agent_id": payload.get("agent_id", ""),
            "created_at": created_at,
            "updated_at": payload.get("updated_at", ""),
            "cwd": cwd,
            "message_count": len(history_payload),
            "preview": preview,
            "first_user_input": first_user_input,
            "json_path": os.path.relpath(paths.file_path, self.store_dir),
            "readable_path": os.path.relpath(paths.readable_path, self.store_dir),
        }

    def _build_preview(self, history_payload: list[Any]) -> str:
        for item in reversed(history_payload):
            if not isinstance(item, dict):
                continue
            try:
                message = Message.model_validate(item)
            except Exception:
                continue
            preview = message.get_text_content().strip()
            if preview:
                preview = " ".join(preview.split())
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                return preview
        return ""

    def _build_first_user_input(self, history_payload: list[Any]) -> str:
        for item in history_payload:
            if not isinstance(item, dict):
                continue
            try:
                message = Message.model_validate(item)
            except Exception:
                continue
            if message.role not in {Role.USER, Role.FRAMEWORK}:
                continue
            content = message.get_text_content().strip()
            if content:
                content = " ".join(content.split())
                if len(content) > 120:
                    content = content[:117] + "..."
                return content
        return ""

    def _sort_entries(self, entries: list[dict[str, Any]], *, current_session_id: str = "") -> list[dict[str, Any]]:
        def sort_key(entry: dict[str, Any]) -> tuple[bool, str]:
            session_id = entry.get("session_id", "")
            updated_at = entry.get("updated_at", "") or ""
            return (session_id == current_session_id, updated_at)

        return sorted(entries, key=sort_key, reverse=True)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        sessions = manifest.get("sessions", [])
        current_session_id = manifest.get("current_session_id", "") or ""
        normalized_manifest = {
            "version": 2,
            "agent_config_path": str(self.config_path),
            "user_id": self.user_id,
            "updated_at": manifest.get("updated_at", ""),
            "current_session_id": current_session_id,
            "sessions": self._sort_entries(
                [entry for entry in sessions if isinstance(entry, dict)],
                current_session_id=current_session_id,
            ),
        }

        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(normalized_manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self._write_readable_manifest(normalized_manifest)

    def _write_readable_manifest(self, manifest: dict[str, Any]) -> None:
        _ = manifest
        return

    def load_current(self) -> dict[str, Any] | None:
        manifest = self._load_manifest()
        current_session_id = manifest.get("current_session_id")
        if not isinstance(current_session_id, str) or not current_session_id:
            target_cwd = _project_root
            ordered_entries = self._sort_entries(
                [entry for entry in manifest.get("sessions", []) if isinstance(entry, dict)],
                current_session_id="",
            )
            cwd_entries = [entry for entry in ordered_entries if isinstance(entry.get("cwd"), str) and entry.get("cwd") == target_cwd]
            candidate_entries = cwd_entries if cwd_entries else ordered_entries
            for entry in candidate_entries:
                session_id = entry.get("session_id")
                if isinstance(session_id, str) and session_id:
                    return self.load_session(session_id)
            return None
        return self.load_session(current_session_id)

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        manifest = self._load_manifest()
        for entry in manifest.get("sessions", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("session_id") != session_id:
                continue
            paths = self._manifest_entry_paths(entry)
            if paths is None:
                continue
            snapshot = self._read_checkpoint_file(paths.readable_path)
            if snapshot is None:
                snapshot = self._read_snapshot_file(paths.file_path)
            if snapshot is not None:
                return snapshot

        legacy_snapshot = self._load_legacy_snapshot()
        if legacy_snapshot is not None and legacy_snapshot.get("session_id") == session_id:
            return legacy_snapshot

        return None

    def list_sessions(self) -> list[dict[str, Any]]:
        manifest = self._load_manifest()
        current_session_id = manifest.get("current_session_id", "") or ""
        sessions = []
        for entry in manifest.get("sessions", []):
            if not isinstance(entry, dict):
                continue
            paths = self._manifest_entry_paths(entry)
            session_id = entry.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                continue
            sessions.append(
                {
                    **entry,
                    "created_at": entry.get("created_at", "") or entry.get("updated_at", "") or "",
                    "first_user_input": entry.get("first_user_input", "") or "",
                    "current": session_id == current_session_id,
                    "file_path": str(paths.file_path) if paths else "",
                    "readable_path": str(paths.readable_path) if paths else "",
                }
            )

        return self._sort_entries(sessions, current_session_id=current_session_id)

    def set_current_session(self, session_id: str) -> None:
        manifest = self._load_manifest()
        sessions = manifest.get("sessions", [])
        if not any(isinstance(entry, dict) and entry.get("session_id") == session_id for entry in sessions):
            legacy_snapshot = self._load_legacy_snapshot()
            if legacy_snapshot is None or legacy_snapshot.get("session_id") != session_id:
                raise KeyError(f"Unknown session id: {session_id}")
            sessions = [
                self._snapshot_entry(
                    payload=legacy_snapshot,
                    paths=self.SnapshotPaths(
                        file_path=self.legacy_file_path,
                        readable_path=self.legacy_readable_path,
                    ),
                )
            ]

        manifest["updated_at"] = datetime.now().isoformat()
        manifest["current_session_id"] = session_id
        manifest["sessions"] = sessions
        self._write_manifest(manifest)

    def save(
        self,
        *,
        agent: Agent,
        history: list[Message],
        last_context: dict[str, Any] | None = None,
    ) -> None:
        serializable_storage = {
            key: value
            for key, value in sanitize_for_serialization(agent.global_storage.to_dict()).items()
            if key not in _CLI_TRANSIENT_STORAGE_KEYS
        }
        non_system_history = [msg for msg in history if msg.role != Role.SYSTEM]
        session_id = getattr(agent, "_session_id", None)
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Agent session_id is required for CLI snapshot persistence")

        paths = self.snapshot_paths(session_id)
        payload = {
            "version": 2,
            "agent_config_path": str(self.config_path),
            "user_id": self.user_id,
            "agent_id": agent.agent_id,
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "history": [msg.model_dump(mode="json", exclude_none=True) for msg in non_system_history],
            "global_storage": serializable_storage,
            "last_context": sanitize_for_serialization(last_context or {}),
        }

        manifest = self._load_manifest()
        existing_entry = next(
            (entry for entry in manifest.get("sessions", []) if isinstance(entry, dict) and entry.get("session_id") == session_id),
            None,
        )
        existing_created_at = existing_entry.get("created_at") if isinstance(existing_entry, dict) else ""
        payload["created_at"] = (
            existing_created_at if isinstance(existing_created_at, str) and existing_created_at else payload["updated_at"]
        )

        if paths.file_path.suffix == ".jsonl":
            self._append_jsonl_snapshot(
                paths.file_path,
                checkpoint_path=paths.readable_path,
                payload=payload,
                agent=agent,
            )
        else:
            with paths.file_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")

        self._write_readable_snapshot(payload, paths=paths)

        updated_entry = self._snapshot_entry(payload=payload, paths=paths)
        existing_sessions = [
            entry for entry in manifest.get("sessions", []) if isinstance(entry, dict) and entry.get("session_id") != session_id
        ]
        manifest["updated_at"] = payload["updated_at"]
        manifest["current_session_id"] = session_id
        manifest["sessions"] = [updated_entry, *existing_sessions]
        self._write_manifest(manifest)

    def _write_readable_snapshot(self, payload: dict[str, Any], *, paths: SnapshotPaths) -> None:
        _ = (payload, paths)
        return


def create_cli_progress_hook():
    """Create a hook that reports tool calls to the CLI."""

    def progress_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        agent_state = hook_input.agent_state
        agent_name = getattr(agent_state, "agent_name", "agent")
        agent_id = getattr(agent_state, "agent_id", "")
        parent_state = getattr(agent_state, "parent_agent_state", None)
        is_sub_agent = parent_state is not None

        base_metadata = {
            "agent_name": agent_name,
            "agent_id": agent_id,
            "is_sub_agent": is_sub_agent,
            "parent_agent_name": getattr(parent_state, "agent_name", None) if parent_state else None,
            "parent_agent_id": getattr(parent_state, "agent_id", None) if parent_state else None,
            "iteration": hook_input.current_iteration,
        }

        def build_metadata(extra: dict | None = None) -> dict:
            metadata = {k: v for k, v in base_metadata.items() if v is not None}
            if extra:
                metadata.update(extra)
            return metadata

        if hook_input.parsed_response:
            # Extract and display agent's text response (non-tool thinking)
            # This shows what the agent is thinking before executing tools
            if hook_input.original_response:
                # Try to extract text that's not tool calls
                response_text = hook_input.original_response.strip()

                # Send the agent's thinking/reasoning text if it exists
                # This is the text the agent writes before making tool calls
                if response_text and len(response_text) > 10:  # Avoid very short strings
                    # Truncate very long responses for display
                    if len(response_text) > 500:
                        display_text = response_text[:500] + f"... [truncated {len(response_text) - 500} chars]"
                    else:
                        display_text = response_text + f" ({len(response_text)} chars)"

                    # Check if this looks like meaningful text (not just XML/JSON)
                    if not response_text.startswith(("<", "{", "[")):
                        message_type = "subagent_text" if is_sub_agent else "agent_text"
                        send_message(
                            message_type,
                            display_text,
                            metadata=build_metadata({"type": "agent_thinking"}),
                        )

            # Report tool calls with prettier formatting
            if hook_input.parsed_response.tool_calls:
                tool_count = len(hook_input.parsed_response.tool_calls)
                is_parallel = hook_input.parsed_response.is_parallel_tools

                # Format tool calls for display
                tool_details = []
                for i, call in enumerate(hook_input.parsed_response.tool_calls, 1):
                    tool_name = call.tool_name

                    # Format parameters nicely
                    params_preview = ""
                    if hasattr(call, "tool_input") and call.tool_input:
                        # Truncate long parameter values for preview
                        params = {}
                        for key, value in call.tool_input.items():
                            str_value = str(value)
                            if len(str_value) > 100:
                                params[key] = str_value[:100] + f"... [truncated {len(str_value) - 100} chars]"
                            else:
                                params[key] = str_value

                        # Format as readable string
                        params_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:3]])
                        if len(call.tool_input) > 3:
                            params_str += f", ... (+{len(call.tool_input) - 3} more)"
                        params_preview = f"({params_str})"

                    tool_details.append(f"{tool_name}{params_preview}")

                # Send summary message
                execution_type = "parallel" if is_parallel else "sequential"
                message_type = "subagent_step" if is_sub_agent else "step"
                send_message(
                    message_type,
                    f"Planning to execute {tool_count} tool(s) [{execution_type}]:",
                    metadata=build_metadata(
                        {
                            "type": "tool_plan_header",
                            "tool_count": tool_count,
                            "is_parallel": is_parallel,
                        }
                    ),
                )

                # Send each tool as a separate step for better readability
                for i, tool_detail in enumerate(tool_details, 1):
                    send_message(
                        message_type,
                        f"  {i}. {tool_detail}",
                        metadata=build_metadata({"type": "tool_detail"}),
                    )

            # Report sub-agent calls
            sub_agent_tool_calls = [tc for tc in hook_input.parsed_response.tool_calls if tc.tool_name == "Agent"]
            if sub_agent_tool_calls:
                agent_count = len(sub_agent_tool_calls)
                agent_names = [tc.parameters.get("sub_agent_name", "unknown") for tc in sub_agent_tool_calls]

                send_message(
                    message_type,
                    f"Calling {agent_count} sub-agent(s): {', '.join(agent_names)}",
                    metadata=build_metadata(
                        {
                            "type": "subagent_plan",
                            "agent_count": agent_count,
                            "agents": agent_names,
                        }
                    ),
                )

        return AfterModelHookResult.no_changes()

    return progress_hook


def create_cli_tool_hook():
    """Create a hook that reports tool execution to the CLI."""

    def tool_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        # Truncate long outputs for display with clear indicator
        output_preview = str(hook_input.tool_output)
        if len(output_preview) > 200:
            output_preview = output_preview[:200] + f"... [truncated {len(output_preview) - 200} chars]"
        else:
            output_preview = output_preview + f" ({len(output_preview)} chars)"

        agent_state = hook_input.agent_state
        parent_state = getattr(agent_state, "parent_agent_state", None)
        is_sub_agent = parent_state is not None

        metadata = {
            "type": "tool_executed",
            "tool_name": hook_input.tool_name,
            "output_preview": output_preview,
            "agent_name": getattr(agent_state, "agent_name", "agent"),
            "agent_id": getattr(agent_state, "agent_id", ""),
            "is_sub_agent": is_sub_agent,
            "parent_agent_name": getattr(parent_state, "agent_name", None) if parent_state else None,
            "parent_agent_id": getattr(parent_state, "agent_id", None) if parent_state else None,
        }

        # Remove None values for cleanliness
        metadata = {k: v for k, v in metadata.items() if v is not None}

        message_type = "subagent_step" if is_sub_agent else "step"

        send_message(
            message_type,
            f"Tool '{hook_input.tool_name}' completed",
            metadata=metadata,
        )

        return AfterToolHookResult.no_changes()

    return tool_hook


class CliAgentRuntime:
    """Run the CLI agent in a background worker thread so control messages stay responsive."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.user_id = os.environ.get("NEXAU_CLI_USER_ID", "cli_user")
        self._session_store = CliSessionStore(config_path=config_path, user_id=self.user_id)
        self._requested_session_id = os.environ.get("NEXAU_CLI_SESSION_ID") or None
        self._restored_snapshot = (
            self._session_store.load_session(self._requested_session_id)
            if self._requested_session_id
            else self._session_store.load_current()
        )
        self.initial_session_id = self._requested_session_id or (self._restored_snapshot or {}).get("session_id") or None
        self._agent: Agent | None = None
        self._run_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._interrupt_requested = False
        self._last_stop_result: StopResult | None = None
        self._llm_override: dict[str, str] | None = None

        self._cli_progress_hook = create_cli_progress_hook()
        self._cli_tool_hook = create_cli_tool_hook()

    def start(self) -> None:
        self._activate_agent(restored_snapshot=self._restored_snapshot, session_id=self.initial_session_id)
        send_message("status", "Agent loaded successfully")
        self._emit_session_event(restored=self._restored_snapshot is not None)
        self._persist_state()
        send_message("ready", "Agent is ready for input", metadata=self._session_metadata())

    def shutdown(self) -> None:
        if self.is_busy():
            try:
                self.interrupt(force=True, timeout=5.0)
            except Exception:
                logger.exception("Failed to interrupt active run during shutdown")

        if self._agent is not None:
            self._persist_state()
            self._agent.sync_cleanup()
            self._agent = None

    def is_busy(self) -> bool:
        with self._state_lock:
            return self._run_thread is not None and self._run_thread.is_alive()

    def start_message(self, user_message: str) -> None:
        with self._state_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                raise RuntimeError("Agent is already processing a request")
            if self._agent is None:
                raise RuntimeError("Agent is not initialized")

            self._interrupt_requested = False
            self._last_stop_result = None
            self._run_thread = threading.Thread(
                target=self._run_message_sync,
                args=(user_message,),
                name="nexau-cli-run",
                daemon=True,
            )
            self._run_thread.start()

    def interrupt(self, *, force: bool = True, timeout: float = 10.0) -> None:
        with self._state_lock:
            if self._agent is None:
                raise RuntimeError("Agent is not initialized")
            if self._run_thread is None or not self._run_thread.is_alive():
                raise RuntimeError("No active run to interrupt")
            run_thread = self._run_thread
            self._interrupt_requested = True

        self._last_stop_result = asyncio.run(self._agent.stop(force=force, timeout=timeout))
        run_thread.join(timeout + 5.0)
        if run_thread.is_alive():
            logger.warning("Timed out waiting for interrupted run thread to finish")

    def clear(self) -> None:
        self.new_session()

    def new_session(self) -> None:
        if self.is_busy():
            raise RuntimeError("Cannot start a fresh session while the agent is processing. Interrupt it first.")

        send_message("status", "Starting a fresh session...")
        self._activate_agent(restored_snapshot=None, session_id=None)
        self._restored_snapshot = None
        self._persist_state()
        self._emit_session_event(restored=False, reset_ui=True)
        response_text = "Started a fresh session. Previous sessions are still available via /resume."
        send_message("response", response_text)
        send_message("ready", "", metadata=self._session_metadata())

    def use_session(self, session_id: str) -> None:
        if self.is_busy():
            raise RuntimeError("Cannot switch sessions while the agent is processing. Interrupt it first.")

        snapshot = self._session_store.load_session(session_id)
        if snapshot is None:
            raise RuntimeError(f"Unknown session id: {session_id}")

        send_message("status", f"Switching to session {session_id}...")
        self._activate_agent(
            restored_snapshot=snapshot,
            session_id=snapshot.get("session_id") if isinstance(snapshot.get("session_id"), str) else session_id,
        )
        self._persist_state()
        assert self._agent is not None
        self._emit_session_event(
            restored=True,
            reset_ui=True,
            content=f"Session ready: {self._agent._session_id}",
        )
        send_message("response", f"Switched to session {self._agent._session_id}.")
        send_message("ready", "", metadata=self._session_metadata())

    def _resolve_resume_target_session_id(self) -> str:
        sessions = self._session_store.list_sessions()
        if not sessions:
            raise RuntimeError("No saved sessions found. Use /resume to start or select a session.")

        current_entry = next((entry for entry in sessions if entry.get("current")), None)
        if current_entry and isinstance(current_entry.get("session_id"), str):
            return current_entry["session_id"]

        newest = sessions[0].get("session_id")
        if isinstance(newest, str) and newest:
            return newest

        raise RuntimeError("No resumable session found.")

    def resume(self) -> None:
        session_id = self._resolve_resume_target_session_id()
        self.use_session(session_id)

    def format_session_list(self) -> str:
        if self.is_busy():
            raise RuntimeError("Cannot list sessions while the agent is processing. Interrupt it first.")

        self._persist_state()
        sessions = self._session_store.list_sessions()
        current_session_id = getattr(self._agent, "_session_id", None) or ""
        if not sessions:
            return f"No saved sessions yet.\nRaw index: {self._session_store.manifest_path}\nUse /resume to choose or start a session."

        lines = [
            f"Saved sessions for {self.config_path.name} / {self.user_id}:",
            f"Raw index: {self._session_store.manifest_path}",
            "",
        ]

        for index, entry in enumerate(sessions, start=1):
            session_id = entry.get("session_id", "")
            current_marker = " [current]" if session_id == current_session_id else ""
            updated_at = entry.get("updated_at", "") or "unknown"
            created_at = entry.get("created_at", "") or updated_at
            message_count = entry.get("message_count", 0)
            lines.append(f"{index}. {session_id}{current_marker} | created {created_at} | updated {updated_at} | messages {message_count}")
            preview = entry.get("preview", "")
            if isinstance(preview, str) and preview:
                lines.append(f"  {preview}")
        return "\n".join(lines)

    @staticmethod
    def _extract_first_user_input_from_history(history_payload: Any) -> str:
        if not isinstance(history_payload, list):
            return ""

        for item in history_payload:
            if not isinstance(item, dict):
                continue
            try:
                message = Message.model_validate(item)
            except Exception:
                continue
            if message.role not in {Role.USER, Role.FRAMEWORK}:
                continue
            content = message.get_text_content().strip()
            if content:
                content = " ".join(content.split())
                if len(content) > 120:
                    content = content[:117] + "..."
                return content
        return ""

    @staticmethod
    def _resume_conversation_preview(entry: dict[str, Any], *, fallback_first_user_input: str = "") -> str:
        if fallback_first_user_input.strip():
            return fallback_first_user_input.strip()

        first_user_input = entry.get("first_user_input", "")
        if isinstance(first_user_input, str) and first_user_input.strip():
            return first_user_input.strip()

        preview = entry.get("preview", "")
        if isinstance(preview, str) and preview.strip():
            return preview.strip()

        cwd = entry.get("cwd", "")
        if isinstance(cwd, str) and cwd.strip():
            return cwd.strip()

        session_id = entry.get("session_id", "")
        if isinstance(session_id, str):
            return session_id
        return ""

    def list_resume_sessions(self) -> list[dict[str, Any]]:
        if self.is_busy():
            raise RuntimeError("Cannot list sessions while the agent is processing. Interrupt it first.")

        self._persist_state()
        sessions = self._session_store.list_sessions()
        structured: list[dict[str, Any]] = []
        for index, entry in enumerate(sessions, start=1):
            session_id = entry.get("session_id", "")
            if not isinstance(session_id, str) or not session_id:
                continue

            first_user_input = entry.get("first_user_input", "")
            if not isinstance(first_user_input, str) or not first_user_input.strip():
                snapshot = self._session_store.load_session(session_id)
                if isinstance(snapshot, dict):
                    first_user_input = self._extract_first_user_input_from_history(snapshot.get("history"))
                else:
                    first_user_input = ""

            created_at = entry.get("created_at", "") or entry.get("updated_at", "") or ""
            updated_at = entry.get("updated_at", "") or ""
            structured.append(
                {
                    "index": index,
                    "id": session_id,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "conversation": self._resume_conversation_preview(
                        entry,
                        fallback_first_user_input=first_user_input,
                    ),
                    "current": bool(entry.get("current", False)),
                }
            )

        return structured

    def handle_session_command(self, command_text: str) -> None:
        parts = command_text.split()
        if len(parts) == 1 or parts[1] in {"help", "-h", "--help"}:
            send_message(
                "response",
                "Session commands:\n/session list\n/session new\n/session use <session_id>\n/resume",
            )
            send_message("ready", "", metadata=self._session_metadata())
            return

        action = parts[1]
        if action == "list" and len(parts) == 2:
            send_message("response", self.format_session_list())
            send_message("ready", "", metadata=self._session_metadata())
            return
        if action == "new" and len(parts) == 2:
            self.new_session()
            return
        if action == "use" and len(parts) == 3:
            self.use_session(parts[2])
            return

        raise RuntimeError("Usage: /session list | /session new | /session use <session_id>")

    def handle_resume_command(self, command_text: str) -> None:
        parts = command_text.split()
        if len(parts) == 1:
            self.resume()
            return

        raise RuntimeError("Usage: /resume")

    def _resolve_model_inventory(self) -> tuple[str, tuple[str, ...]]:
        if self._agent is None or self._agent.config.llm_config is None:
            return "", ()

        llm_config = self._agent.config.llm_config
        current_model = llm_config.model or ""
        available_models = tuple(llm_config.list_available_models() if hasattr(llm_config, "list_available_models") else ())
        if current_model and current_model not in available_models:
            available_models = (current_model, *available_models)
        return current_model, available_models

    def _discover_llm_profiles(self) -> dict[str, dict[str, str]]:
        """Discover profile-style env configs like NEX_MODEL/NEX_BASE_URL/NEX_API_KEY."""
        profiles: dict[str, dict[str, str]] = {}
        for env_name, model_value in os.environ.items():
            if not env_name.endswith("_MODEL"):
                continue
            prefix = env_name[: -len("_MODEL")]
            if prefix in {"LLM", "SUMMARY", "OPENAI", "MODEL"}:
                continue
            if not model_value:
                continue

            base_url = os.getenv(f"{prefix}_BASE_URL")
            api_key = os.getenv(f"{prefix}_API_KEY")
            if not base_url or not api_key:
                continue

            alias = prefix.lower()
            profiles[alias] = {
                "alias": alias,
                "model": model_value,
                "base_url": base_url,
                "api_key": api_key,
            }
        return profiles

    @staticmethod
    def _merge_models_with_profiles(
        available_models: tuple[str, ...],
        profiles: dict[str, dict[str, str]],
    ) -> tuple[str, ...]:
        merged: list[str] = [model for model in available_models if isinstance(model, str) and model]
        for profile in profiles.values():
            model_name = profile.get("model")
            if not isinstance(model_name, str) or not model_name:
                continue
            if model_name in merged:
                continue
            merged.append(model_name)
        return tuple(merged)

    def format_model_list(self) -> str:
        if self.is_busy():
            raise RuntimeError("Cannot list models while the agent is processing. Interrupt it first.")

        profiles = self._discover_llm_profiles()
        current_model, available_models = self._resolve_model_inventory()
        available_models = self._merge_models_with_profiles(available_models, profiles)
        if not available_models and not current_model:
            return "No model configured."

        if not available_models and current_model:
            available_models = (current_model,)

        lines = ["Available models:"]
        for idx, model_name in enumerate(available_models, start=1):
            marker = " [current]" if model_name == current_model else ""
            lines.append(f"{idx}. {model_name}{marker}")
        return "\n".join(lines)

    def list_model_options(self) -> list[dict[str, Any]]:
        if self.is_busy():
            raise RuntimeError("Cannot list models while the agent is processing. Interrupt it first.")

        current_model, available_models = self._resolve_model_inventory()
        profiles = self._discover_llm_profiles()
        available_models = self._merge_models_with_profiles(available_models, profiles)
        if not available_models and current_model:
            available_models = (current_model,)

        profile_by_model = {profile["model"]: profile for profile in profiles.values()}
        options: list[dict[str, Any]] = []
        for idx, model_name in enumerate(available_models, start=1):
            profile = profile_by_model.get(model_name)
            options.append(
                {
                    "index": idx,
                    "name": model_name,
                    "current": model_name == current_model,
                    "profile_alias": profile.get("alias") if profile else "",
                }
            )
        return options

    def switch_model(self, target: str) -> None:
        if self.is_busy():
            raise RuntimeError("Cannot switch model while the agent is processing. Interrupt it first.")
        if self._agent is None:
            raise RuntimeError("Agent is not initialized")
        if self._agent.config.llm_config is None:
            raise RuntimeError("Agent LLM config is not initialized")

        candidate = target.strip()
        if not candidate:
            raise RuntimeError("Usage: /model use <index|model_name>")

        current_model, available_models_raw = self._resolve_model_inventory()
        current_base_url = self._agent.config.llm_config.base_url or ""
        current_api_key = self._agent.config.llm_config.api_key or ""
        profiles = self._discover_llm_profiles()
        available_models = self._merge_models_with_profiles(available_models_raw, profiles)
        profile_by_model = {profile["model"]: profile for profile in profiles.values()}

        selected_profile: dict[str, str] | None = None
        if candidate.isdigit():
            if not available_models:
                raise RuntimeError("No available models configured.")
            index = int(candidate)
            if index < 1 or index > len(available_models):
                raise RuntimeError(f"Model index must be between 1 and {len(available_models)}")
            selected_model = available_models[index - 1]
            selected_profile = profile_by_model.get(selected_model)
        else:
            selected_profile = profiles.get(candidate.lower())
            selected_model = selected_profile["model"] if selected_profile else candidate
            if available_models and selected_model not in available_models and selected_model not in profile_by_model:
                raise RuntimeError(f"Unknown model: {selected_model}")
            if selected_profile is None:
                selected_profile = profile_by_model.get(selected_model)

        target_runtime = {
            "model": selected_model,
            "base_url": selected_profile["base_url"] if selected_profile else current_base_url,
            "api_key": selected_profile["api_key"] if selected_profile else current_api_key,
        }
        if not target_runtime["base_url"] or not target_runtime["api_key"]:
            raise RuntimeError("Target model is missing base_url or api_key")

        if (
            selected_model == current_model
            and target_runtime["base_url"] == current_base_url
            and target_runtime["api_key"] == current_api_key
        ):
            send_message("response", f"Model already in use: {selected_model}")
            send_message("ready", "", metadata=self._session_metadata())
            return

        current_session_id = getattr(self._agent, "_session_id", None)
        if not isinstance(current_session_id, str) or not current_session_id:
            raise RuntimeError("Current session is not initialized")

        self._persist_state()
        snapshot = self._session_store.load_session(current_session_id)
        if snapshot is None:
            raise RuntimeError("Failed to load current session snapshot for model switch")

        profile_suffix = f" (profile: {selected_profile['alias']})" if selected_profile else ""
        send_message("status", f"Switching model to {selected_model}{profile_suffix}...")
        self._llm_override = target_runtime
        self._activate_agent(restored_snapshot=snapshot, session_id=current_session_id)
        self._persist_state()
        send_message("response", f"Switched model to {selected_model}{profile_suffix}.")
        send_message("ready", "", metadata=self._session_metadata())

    def handle_model_command(self, command_text: str) -> None:
        parts = command_text.split()
        if len(parts) == 1:
            send_message("response", self.format_model_list())
            send_message("ready", "", metadata=self._session_metadata())
            return

        raise RuntimeError("Usage: /model")

    def _session_metadata(self) -> dict[str, Any]:
        session_id = getattr(self._agent, "_session_id", None) or ""
        agent_id = getattr(self._agent, "agent_id", None) or ""
        current_model, available_models = self._resolve_model_inventory()
        metadata = {
            "user_id": self.user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "model": current_model,
            "index_path": str(self._session_store.manifest_path),
            "index_raw_path": str(self._session_store.manifest_path),
        }
        if available_models:
            metadata["available_models"] = list(available_models)
        if session_id:
            snapshot_paths = self._session_store.snapshot_paths(session_id)
            metadata["storage_path"] = str(snapshot_paths.file_path)
            metadata["storage_raw_path"] = str(snapshot_paths.file_path)
        return metadata

    def _emit_session_event(
        self,
        *,
        restored: bool,
        reset_ui: bool = False,
        content: str | None = None,
    ) -> None:
        session_id = getattr(self._agent, "_session_id", None) or ""
        send_message(
            "session",
            content or f"Session ready: {session_id}",
            metadata={
                **self._session_metadata(),
                "restored": restored,
                "reset_ui": reset_ui,
            },
        )

    def _handle_subagent_event(self, event_type: str, payload: dict) -> None:
        event_mapping = {
            "start": ("subagent_start", "message"),
            "complete": ("subagent_complete", "result"),
            "error": ("subagent_error", "error"),
        }

        if event_type not in event_mapping:
            return

        message_type, content_field = event_mapping[event_type]
        content = payload.get(content_field, "")
        send_message(message_type, content, metadata=payload)

    def _build_agent_from_config(self, *, session_id: str | None) -> Agent:
        raw_config = load_yaml_with_vars(str(self.config_path))
        if not raw_config:
            raise ConfigError(
                f"Empty or invalid configuration file: {self.config_path}",
            )

        if not isinstance(raw_config, dict):
            raise ConfigError(
                f"Invalid configuration type in {self.config_path}; expected mapping",
            )

        normalized_config = normalize_agent_config_dict(raw_config)

        if self._llm_override:
            llm_config_raw = normalized_config.get("llm_config")
            if not isinstance(llm_config_raw, dict):
                raise ConfigError("llm_config is required and must be a mapping in agent configuration")
            llm_config_raw["model"] = self._llm_override["model"]
            llm_config_raw["base_url"] = self._llm_override["base_url"]
            llm_config_raw["api_key"] = self._llm_override["api_key"]

        existing_model_hooks = list(normalized_config.get("after_model_hooks", []))
        existing_tool_hooks = list(normalized_config.get("after_tool_hooks", []))

        normalized_config["after_model_hooks"] = [self._cli_progress_hook] + existing_model_hooks
        normalized_config["after_tool_hooks"] = [self._cli_tool_hook] + existing_tool_hooks

        builder = AgentConfigBuilder(normalized_config, self.config_path.parent)
        agent_config = (
            builder.build_core_properties()
            .build_llm_config()
            .build_mcp_servers()
            .build_hooks()
            .build_tracers()
            .build_tools()
            .build_sub_agents()
            .build_skills()
            .build_system_prompt_path()
            .build_sandbox()
            .get_agent_config()
        )
        restored_storage = self._build_restored_global_storage()
        restored_agent_id = None if self._restored_snapshot is None else self._restored_snapshot.get("agent_id")
        return Agent(
            config=agent_config,
            agent_id=restored_agent_id if isinstance(restored_agent_id, str) and restored_agent_id else None,
            global_storage=restored_storage,
            user_id=self.user_id,
            session_id=session_id,
        )

    def _activate_agent(self, *, restored_snapshot: dict[str, Any] | None, session_id: str | None) -> None:
        if self._agent is not None:
            self._persist_state()
            self._agent.sync_cleanup()

        self._restored_snapshot = restored_snapshot
        self._agent = self._build_agent_from_config(session_id=session_id)
        self._restore_agent_state()
        attach_cli_to_agent(
            self._agent,
            self._cli_progress_hook,
            self._cli_tool_hook,
            self._handle_subagent_event,
        )

    def _build_restored_global_storage(self) -> GlobalStorage | None:
        if self._restored_snapshot is None:
            return None

        stored = self._restored_snapshot.get("global_storage")
        if not isinstance(stored, dict):
            return None

        restored = GlobalStorage()
        restored.update(stored)
        return restored

    def _restore_agent_state(self) -> None:
        if self._agent is None or self._restored_snapshot is None:
            return

        history_payload = self._restored_snapshot.get("history")
        if isinstance(history_payload, list):
            restored_history = []
            for item in history_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    restored_history.append(Message.model_validate(item))
                except Exception as exc:
                    logger.warning("Skipping invalid persisted message in CLI session snapshot: %s", exc)
            if restored_history:
                self._agent.history = restored_history

        last_context = self._restored_snapshot.get("last_context")
        if isinstance(last_context, dict):
            self._agent._last_context = last_context

    def _persist_state(self) -> None:
        if self._agent is None:
            return

        try:
            self._session_store.save(
                agent=self._agent,
                history=list(self._agent.history),
                last_context=getattr(self._agent, "_last_context", {}) or {},
            )
        except Exception as exc:
            logger.warning("Failed to persist CLI session snapshot: %s", exc)

    def _rebuild_agent_from_current_session(self) -> None:
        if self._agent is None:
            return

        current_session_id = getattr(self._agent, "_session_id", None)
        if not isinstance(current_session_id, str) or not current_session_id:
            return

        snapshot = self._session_store.load_session(current_session_id)
        if snapshot is None:
            logger.warning("Failed to reload interrupted session snapshot for %s", current_session_id)
            return

        self._activate_agent(restored_snapshot=snapshot, session_id=current_session_id)

    def _run_message_sync(self, user_message: str) -> None:
        if self._agent is None:
            raise RuntimeError("Agent is not initialized")

        send_message("step", "Processing request...", metadata={"type": "start"})

        sandbox = self._agent.sandbox_manager.instance

        try:
            response = self._agent.run(
                message=user_message,
                context={
                    "date": get_date(),
                    "username": os.getenv("USER", "user"),
                    "working_directory": str(sandbox.work_dir if sandbox else os.getcwd()),
                    "env_content": {
                        "date": get_date(),
                        "username": os.getenv("USER", "user"),
                        "working_directory": str(sandbox.work_dir if sandbox else os.getcwd()),
                    },
                },
            )

            if not self._interrupt_requested:
                send_message("step", "Request completed", metadata={"type": "complete"})
                final_response = response[0] if isinstance(response, tuple) else response
                send_message("response", final_response)
        except Exception as e:
            if not self._interrupt_requested:
                send_message("error", f"{str(e)}\n{traceback.format_exc()}")
        finally:
            self._persist_state()
            interrupted_result = self._last_stop_result

            with self._state_lock:
                self._run_thread = None
                self._interrupt_requested = False
                self._last_stop_result = None

            if interrupted_result is not None:
                self._rebuild_agent_from_current_session()
                send_message(
                    "interrupted",
                    "Current run interrupted. Session preserved.",
                    metadata={
                        **self._session_metadata(),
                        "stop_reason": interrupted_result.stop_reason.name,
                        "message_count": len(interrupted_result.messages),
                    },
                )

            send_message("ready", "", metadata=self._session_metadata())


def main():
    if len(sys.argv) < 2:
        send_message("error", "No agent configuration file provided")
        sys.exit(1)

    yaml_path = sys.argv[1]
    runtime: CliAgentRuntime | None = None

    try:
        send_message("status", "Loading agent configuration...")

        config_path = Path(yaml_path)
        _ensure_config_dir_on_sys_path(config_path)
        runtime = CliAgentRuntime(config_path)
        runtime.start()

        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                message_type = data.get("type")

                if message_type == "exit":
                    send_message("status", "Shutting down...")
                    runtime.shutdown()
                    break

                if message_type == "interrupt":
                    send_message("status", "Interrupting current run...")
                    runtime.interrupt(force=True, timeout=10.0)
                    continue

                if message_type == "clear":
                    runtime.clear()
                    continue

                if message_type == "resume_list":
                    send_message("resume_list", "", metadata={"sessions": runtime.list_resume_sessions()})
                    continue

                if message_type == "resume_use":
                    session_id = data.get("session_id", "")
                    if not isinstance(session_id, str) or not session_id.strip():
                        raise RuntimeError("resume_use requires a non-empty session_id")
                    runtime.use_session(session_id.strip())
                    continue

                if message_type == "resume_new":
                    runtime.new_session()
                    continue

                if message_type == "model_list":
                    send_message("model_list", "", metadata={"models": runtime.list_model_options()})
                    continue

                if message_type == "model_use":
                    model_name = data.get("model_name", "")
                    if not isinstance(model_name, str) or not model_name.strip():
                        raise RuntimeError("model_use requires a non-empty model_name")
                    runtime.switch_model(model_name.strip())
                    continue

                if message_type == "message":
                    user_message = data.get("content", "")
                    if not user_message:
                        continue

                    stripped_message = user_message.strip()
                    if stripped_message == "/clear":
                        runtime.clear()
                        continue
                    if stripped_message == "/interrupt":
                        send_message("status", "Interrupting current run...")
                        runtime.interrupt(force=True, timeout=10.0)
                        continue
                    if stripped_message.startswith("/resume"):
                        runtime.handle_resume_command(stripped_message)
                        continue
                    if stripped_message.startswith("/session"):
                        runtime.handle_session_command(stripped_message)
                        continue
                    if stripped_message.startswith("/model"):
                        runtime.handle_model_command(stripped_message)
                        continue

                    runtime.start_message(user_message)
                    continue

                send_message("error", f"Unsupported message type: {message_type}")
            except ConfigError as e:
                send_message("error", str(e))
                send_message("ready", "", metadata=runtime._session_metadata())
            except json.JSONDecodeError:
                send_message("error", f"Invalid JSON received: {line}")
            except Exception as e:
                send_message("error", f"{str(e)}\n{traceback.format_exc()}")
                send_message("ready", "", metadata=runtime._session_metadata())

    except ConfigError as e:
        if runtime is not None:
            runtime.shutdown()
        send_message("error", str(e))
        sys.exit(1)
    except Exception as e:
        if runtime is not None:
            runtime.shutdown()
        send_message("error", f"Failed to load agent: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
