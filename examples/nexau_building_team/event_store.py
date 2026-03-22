"""In-memory event store for team SSE events.

RFC-0045: 团队 SSE 事件存储（含快照优化）

Stores serialized TeamStreamEnvelope dicts keyed by (user_id, session_id),
enabling history replay after frontend reconnection.

Maintains a running compacted state snapshot alongside raw events so the
frontend can load the pre-reduced snapshot on reconnect instead of
replaying thousands of granular delta events.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from snapshot_reducer import SnapshotState, apply_envelope, snapshot_deep_copy

logger = logging.getLogger(__name__)


class EventStore:
    """Thread-safe in-memory event store with snapshot support.

    RFC-0045: 带快照的线程安全事件存储

    Each append() call both stores the raw event and incrementally updates
    a compacted state snapshot.  get_snapshot() returns the pre-reduced
    state so the frontend can skip replaying thousands of delta events.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[tuple[str, str], list[dict[str, object]]] = {}
        # 增量快照：每次 append 时原地更新，避免重播
        self._snapshots: dict[tuple[str, str], SnapshotState] = {}

    def append(self, user_id: str, session_id: str, envelope_dict: dict[str, object]) -> None:
        """Persist one envelope dict and update the snapshot.

        RFC-0045: 追加事件并增量更新快照
        """
        with self._lock:
            key = (user_id, session_id)
            self._events.setdefault(key, []).append(envelope_dict)
            count = len(self._events[key])

            # 增量更新快照（in-place mutation）
            if key not in self._snapshots:
                self._snapshots[key] = {}
            apply_envelope(self._snapshots[key], envelope_dict)

            if count <= 3 or count % 50 == 0:
                logger.info("EventStore: appended event #%d for (%s, %s)", count, user_id, session_id)

    def get_history(self, user_id: str, session_id: str, after: int = 0) -> list[dict[str, object]]:
        """Return stored envelopes, optionally skipping the first `after` entries."""
        with self._lock:
            return list(self._events.get((user_id, session_id), [])[after:])

    def get_snapshot(self, user_id: str, session_id: str) -> dict[str, Any]:
        """Return the compacted state snapshot and total event count.

        RFC-0045: 返回预压缩的快照状态

        Response shape:
            {
                "snapshot": { "<agent_id>": AgentSnapshot, ... },
                "event_count": int,
            }

        The snapshot is a deep copy to avoid concurrent modification issues
        during JSON serialization.
        """
        with self._lock:
            key = (user_id, session_id)
            snapshot = self._snapshots.get(key, {})
            event_count = len(self._events.get(key, []))
            return {
                "snapshot": snapshot_deep_copy(snapshot),
                "event_count": event_count,
            }

    def list_sessions(self, user_id: str) -> list[str]:
        """Return all session IDs for a given user."""
        with self._lock:
            return [sid for (uid, sid) in self._events if uid == user_id]

    def has_session(self, user_id: str, session_id: str) -> bool:
        """Check if any events exist for this session."""
        with self._lock:
            return (user_id, session_id) in self._events

    def count(self, user_id: str, session_id: str) -> int:
        """Return the number of stored events for a session."""
        with self._lock:
            return len(self._events.get((user_id, session_id), []))
