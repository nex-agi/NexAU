"""Server-side snapshot reducer for team SSE events.

RFC-0045: 服务端状态快照压缩

Mirrors the frontend applyEnvelope() logic in Python so the backend can
maintain a running compacted state snapshot.  When the frontend loads
history, it receives the pre-reduced snapshot instead of replaying
thousands of granular delta events.

The snapshot shape matches the frontend AgentState:
  {
    "<agent_id>": {
      "agentId": str,
      "roleName": str,
      "blocks": [ ContentBlock, ... ],
      "isActive": bool,
    }
  }
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Type aliases for clarity
AgentSnapshot = dict[str, Any]
SnapshotState = dict[str, AgentSnapshot]


def apply_envelope(state: SnapshotState, envelope: dict[str, Any]) -> SnapshotState:
    """Apply a single TeamStreamEnvelope dict to the snapshot state.

    RFC-0045: 将单条事件增量合并到快照状态

    This is the Python equivalent of the frontend's applyEnvelope() function.
    The state is mutated in-place for performance (caller should not assume
    immutability).

    Args:
        state: Current snapshot state (agent_id → AgentSnapshot). Modified in-place.
        envelope: Serialized TeamStreamEnvelope dict with keys:
            agent_id, role_name, event (dict with type, delta, etc.)

    Returns:
        The same state dict (modified in-place).
    """
    agent_id: str = envelope.get("agent_id", "")
    role_name: str | None = envelope.get("role_name")
    event: dict[str, Any] = envelope.get("event", {})

    if not agent_id or not event:
        return state

    effective_role = role_name if (role_name and role_name != "user") else None

    # 1. 获取或创建 agent 快照
    if agent_id not in state:
        state[agent_id] = {
            "agentId": agent_id,
            "roleName": effective_role or agent_id,
            "blocks": [],
            "isActive": True,
        }

    agent = state[agent_id]
    if effective_role:
        agent["roleName"] = effective_role
    agent["isActive"] = True
    blocks: list[dict[str, Any]] = agent["blocks"]

    event_type: str = event.get("type", "")

    # 2. 按事件类型更新 blocks（与前端 applyEnvelope 完全对应）
    if event_type == "THINKING_TEXT_MESSAGE_CONTENT" and event.get("delta"):
        last = blocks[-1] if blocks else None
        if last and last.get("kind") == "thinking":
            last["content"] += event["delta"]
        else:
            blocks.append({"kind": "thinking", "content": event["delta"]})

    elif event_type == "THINKING_TEXT_MESSAGE_START":
        last = blocks[-1] if blocks else None
        if not last or last.get("kind") != "thinking":
            blocks.append({"kind": "thinking", "content": ""})

    elif event_type == "THINKING_TEXT_MESSAGE_END":
        pass  # no-op

    elif event_type == "TEXT_MESSAGE_CONTENT" and event.get("delta"):
        last = blocks[-1] if blocks else None
        if last and last.get("kind") == "text":
            last["content"] += event["delta"]
        else:
            blocks.append({"kind": "text", "content": event["delta"]})

    elif event_type == "TOOL_CALL_START":
        blocks.append({
            "kind": "tool_call",
            "id": event.get("tool_call_id", ""),
            "name": event.get("tool_call_name") or event.get("name", ""),
            "args": "",
            "argsDone": False,
            "result": None,
        })

    elif event_type == "TOOL_CALL_ARGS" and event.get("delta"):
        tool_call_id = event.get("tool_call_id", "")
        updated = False
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            if b.get("kind") == "tool_call" and (
                tool_call_id == "" or b.get("id") == tool_call_id
            ):
                b["args"] += event["delta"]
                updated = True
                break
        if not updated:
            blocks.append({
                "kind": "tool_call",
                "id": tool_call_id,
                "name": "",
                "args": event["delta"],
                "argsDone": False,
                "result": None,
            })

    elif event_type == "TOOL_CALL_END":
        tool_call_id = event.get("tool_call_id", "")
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            if b.get("kind") == "tool_call" and (
                tool_call_id == "" or b.get("id") == tool_call_id
            ):
                b["argsDone"] = True
                break

    elif event_type == "TOOL_CALL_RESULT":
        tool_call_id = event.get("tool_call_id", "")
        result_content = event.get("content", "")
        updated = False
        if tool_call_id:
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                if b.get("kind") == "tool_call" and b.get("id") == tool_call_id:
                    b["argsDone"] = True
                    b["result"] = result_content
                    updated = True
                    break
        if not updated:
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                if b.get("kind") == "tool_call" and b.get("result") is None:
                    b["argsDone"] = True
                    b["result"] = result_content
                    break

    elif event_type == "RUN_FINISHED":
        agent["isActive"] = False

    elif event_type == "RUN_ERROR" and event.get("message"):
        blocks.append({"kind": "error", "message": event["message"]})
        agent["isActive"] = False

    elif event_type == "USER_MESSAGE" and event.get("content"):
        block: dict[str, Any] = {
            "kind": "user_message",
            "content": event["content"],
        }
        from_id = event.get("from_agent_id")
        if from_id:
            block["from"] = from_id
        blocks.append(block)

    elif event_type == "TEAM_MESSAGE" and event.get("content"):
        blocks.append({
            "kind": "team_message",
            "from": event.get("from_agent_id", "unknown"),
            "content": event["content"],
        })

    return state


def snapshot_deep_copy(state: SnapshotState) -> SnapshotState:
    """Return a deep copy of the snapshot state for safe serialization.

    RFC-0045: 深拷贝快照以避免序列化时的并发修改

    The internal snapshot is mutated in-place, so a deep copy is needed
    before serializing to JSON for the HTTP response.
    """
    return copy.deepcopy(state)
