"""In-memory event store for team SSE events.

Stores serialized TeamStreamEnvelope dicts keyed by (user_id, session_id),
enabling history replay after frontend reconnection.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class EventStore:
    """Thread-safe in-memory event store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[tuple[str, str], list[dict[str, object]]] = {}

    def append(self, user_id: str, session_id: str, envelope_dict: dict[str, object]) -> None:
        """Persist one envelope dict."""
        with self._lock:
            self._events.setdefault((user_id, session_id), []).append(envelope_dict)
            count = len(self._events[(user_id, session_id)])
            if count <= 3 or count % 50 == 0:
                logger.info("EventStore: appended event #%d for (%s, %s)", count, user_id, session_id)

    def get_history(self, user_id: str, session_id: str, after: int = 0) -> list[dict[str, object]]:
        """Return stored envelopes, optionally skipping the first `after` entries."""
        with self._lock:
            return list(self._events.get((user_id, session_id), [])[after:])

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
