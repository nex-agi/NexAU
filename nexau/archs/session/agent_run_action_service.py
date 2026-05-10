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

"""Service for managing agent run actions (APPEND/UNDO/REPLACE)."""

from __future__ import annotations

import logging
from typing import NamedTuple

from nexau.core.messages import (
    Message,
    ReasoningBlock,
    Role,
    ToolResultBlock,
    ToolUseBlock,
)

from .models import AgentRunActionModel
from .models.agent_run_action_model import ReplaceVariantBase, RunActionType, RunStatus
from .orm import AndFilter, ComparisonFilter, DatabaseEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Defensive helpers — guard the persistence + read boundaries against
# stream-interrupted / partial-iteration artifacts that violate provider
# message contracts (DeepSeek "assistant must have thinking", Anthropic
# "tool_use must be paired with tool_result").
# ============================================================================


def _is_reasoning_only_assistant(msg: Message) -> bool:
    """True if ``msg`` is an assistant message whose content is ALL reasoning.

    Such messages come from LLM streams that produced reasoning chunks but
    were interrupted before any text/tool_use was emitted. They have no
    semantic value — feeding them back to the LLM produces nonsense.
    """
    if msg.role != Role.ASSISTANT:
        return False
    content = msg.content or []
    if not content:
        return False
    return all(isinstance(b, ReasoningBlock) for b in content)


_ORPHAN_TOOL_RESULT_PLACEHOLDER = (
    "Tool execution did not complete. Synthesized by NexAU at fold time to "
    "maintain the tool_use ↔ tool_result pairing invariant required by "
    "Anthropic / OpenAI APIs. The original tool failure is in the trace logs."
)


def _ensure_tool_use_paired(messages: list[Message]) -> list[Message]:
    """Append synthetic tool_result messages for any orphan tool_use blocks.

    Anthropic API rejects messages lists that contain a ``tool_use`` block
    without a matching ``tool_result`` (by ``tool_use_id``). NexAU sometimes
    persists an orphan when tool execution crashes between writing the
    assistant turn and writing the tool_result (process kill, OOM, etc.).
    Real-data forensic scan found 17 such orphans across two production DBs.

    This function is a read-side fix-up: after the canonical fold produces
    its messages list, scan for unpaired tool_use blocks and inject synthetic
    error tool_result messages right after the originating assistant message
    so the result list is always provider-safe.

    Returns a new list (does not mutate input).
    """
    # Step 1: collect all tool_use IDs and the index of their assistant message
    tool_use_at: dict[str, int] = {}  # tool_use_id → assistant message index
    for i, m in enumerate(messages):
        if m.role != Role.ASSISTANT:
            continue
        for b in m.content:
            if isinstance(b, ToolUseBlock):
                tool_use_at[str(b.id)] = i

    # Step 2: collect all tool_result tool_use_ids that DO have a result
    seen_results: set[str] = set()
    for m in messages:
        if m.role != Role.TOOL:
            continue
        for b in m.content:
            if isinstance(b, ToolResultBlock):
                seen_results.add(str(b.tool_use_id))

    # Step 3: identify orphans + their positions; insert synthetic tool_result
    # right after the assistant message containing the unpaired tool_use.
    orphans = [(tu_id, pos) for tu_id, pos in tool_use_at.items() if tu_id not in seen_results]
    if not orphans:
        return messages

    # Insert in reverse-position order so earlier insertions don't shift
    # later indices.
    orphans.sort(key=lambda x: -x[1])
    out = list(messages)
    for tu_id, pos in orphans:
        # Deterministic id: re-folding the same actions produces the same
        # synthetic message (idempotent). Prefix is intentionally non-UUID so
        # consumers can recognize synthetic results visually if needed.
        synthetic = Message(
            id=f"synth-tool-result-{tu_id}",
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id=tu_id,
                    content=_ORPHAN_TOOL_RESULT_PLACEHOLDER,
                    is_error=True,
                )
            ],
        )
        out.insert(pos + 1, synthetic)
    logger.info(
        "load_messages: synthesized %d tool_result message(s) for orphan tool_use IDs",
        len(orphans),
    )
    return out


class AgentRunActionKey(NamedTuple):
    """Key for identifying agent run actions."""

    user_id: str
    session_id: str
    agent_id: str


class AgentRunActionService:
    """Service for managing agent run action history.

    This service provides methods to:
    - Persist APPEND actions (normal run with messages)
    - Persist UNDO actions (revert to before a specific run)
    - Persist REPLACE actions (full state replacement for compaction)
    - Load and rebuild message history from action records

    The service uses event sourcing pattern where all operations are append-only.
    """

    def __init__(self, *, engine: DatabaseEngine) -> None:
        """Initialize the service.

        Args:
            engine: Database engine for persistence
        """
        self._engine = engine
        self._last_created_at_ns: dict[AgentRunActionKey, int] = {}

    async def _run_exists(self, *, key: AgentRunActionKey, run_id: str) -> bool:
        action_filter = AndFilter(
            filters=[
                ComparisonFilter.eq("user_id", key.user_id),
                ComparisonFilter.eq("session_id", key.session_id),
                ComparisonFilter.eq("agent_id", key.agent_id),
                ComparisonFilter.eq("run_id", run_id),
            ]
        )
        return await self._engine.find_first(AgentRunActionModel, filters=action_filter) is not None

    async def _first_action_ns_of_run(self, *, key: AgentRunActionKey, run_id: str) -> int | None:
        """Earliest ``created_at_ns`` of any action with the given ``run_id``.

        Used by UNDO resolution: ``UNDO before=X`` should remove every action
        whose timestamp is >= X's first action's timestamp. Returns None if
        the target run has no actions (silent no-op for the UNDO).
        """
        action_filter = AndFilter(
            filters=[
                ComparisonFilter.eq("user_id", key.user_id),
                ComparisonFilter.eq("session_id", key.session_id),
                ComparisonFilter.eq("agent_id", key.agent_id),
                ComparisonFilter.eq("run_id", run_id),
            ]
        )
        rows = await self._engine.find_many(
            AgentRunActionModel,
            filters=action_filter,
            order_by=("created_at_ns", "action_id"),
            limit=1,
        )
        if not rows:
            return None
        return int(rows[0].created_at_ns)

    async def persist_append(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        parent_run_id: str | None,
        agent_name: str,
        messages: list[Message],
    ) -> AgentRunActionModel:
        """Persist an APPEND action.

        Args:
            key: Agent run action key (user_id, session_id, agent_id)
            run_id: Current run ID
            root_run_id: Root run ID
            parent_run_id: Parent run ID (for sub-agents)
            agent_name: Agent name
            messages: List of messages produced in this run

        Returns:
            The created action record
        """
        # Filter out system messages
        messages = [m for m in messages if m.role != Role.SYSTEM]

        # Drop reasoning-only assistant messages (LLM stream interrupted before
        # text/tool_use was emitted). These have no semantic value — feeding
        # them back to the LLM produces nonsense follow-ups, and DeepSeek-style
        # providers reject "assistant message with no thinking" on resume too.
        # Real-data forensic scan found 27 such messages across two production
        # DBs; this filter prevents new ones at the write boundary.
        before = len(messages)
        messages = [m for m in messages if not _is_reasoning_only_assistant(m)]
        dropped = before - len(messages)
        if dropped:
            logger.warning(
                "persist_append dropped %d reasoning-only assistant message(s) (stream-interrupted artifacts) for key=%s run_id=%s",
                dropped,
                key,
                run_id,
            )

        if not messages:
            raise ValueError("Cannot persist APPEND action with no messages")

        logger.debug(
            "🔍 [HISTORY-DEBUG] persist_append key=%s run_id=%s: %d messages, roles=%s",
            key,
            run_id,
            len(messages),
            [m.role.value for m in messages],
        )
        for i, msg in enumerate(messages):
            block_types = [type(b).__name__ for b in msg.content]
            logger.debug(
                "🔍 [HISTORY-DEBUG]   persist msg[%d] role=%s blocks=%s text=%.80s",
                i,
                msg.role.value,
                block_types,
                msg.get_text_content()[:80] if msg.get_text_content() else "<empty>",
            )

        record = AgentRunActionModel.create_append(
            user_id=key.user_id,
            session_id=key.session_id,
            agent_id=key.agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            messages=messages,
        )
        return await self._engine.create(record)

    async def persist_undo(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        undo_before_run_id: str,
        parent_run_id: str | None = None,
        agent_name: str = "",
    ) -> AgentRunActionModel:
        """Persist an UNDO action.

        Semantics: undo(before=X) removes runs in range [X, undo_run),
        does not affect runs added after the undo.

        Args:
            key: Agent run action key (user_id, session_id, agent_id)
            run_id: Current run ID (for the undo operation itself)
            root_run_id: Root run ID
            undo_before_run_id: Target run to undo to (exclusive)
            parent_run_id: Parent run ID (for sub-agents)
            agent_name: Agent name

        Returns:
            The created action record
        """
        record = AgentRunActionModel.create_undo(
            user_id=key.user_id,
            session_id=key.session_id,
            agent_id=key.agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            undo_before_run_id=undo_before_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )
        return await self._engine.create(record)

    async def persist_replace(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        parent_run_id: str | None = None,
        agent_name: str = "",
        extra: ReplaceVariantBase | None = None,
    ) -> AgentRunActionModel:
        """Persist a REPLACE action.

        Used for context compaction - replaces all previous history with
        the provided compacted messages.

        Args:
            key: Agent run action key (user_id, session_id, agent_id)
            run_id: Current run ID
            root_run_id: Root run ID
            messages: Complete compacted message history
            parent_run_id: Parent run ID (for sub-agents)
            agent_name: Agent name
            extra: Optional typed ``ReplaceExtra`` variant
                (``CompactAutoVariant`` / ``CompactFocusedVariant`` / etc.)
                to annotate the REPLACE event with reason + stats. RFC-0022
                Phase 3: writers who know "why" — context compaction
                middleware, ``/clear`` handler, ``/compact`` handler — must
                pass a typed variant here so consumers (UI, replay, billing)
                can match on ``reason`` instead of inferring from message diffs.

        Returns:
            The created action record
        """
        # Filter out system messages
        messages = [m for m in messages if m.role != Role.SYSTEM]
        if not messages:
            raise ValueError("Cannot persist REPLACE action with no messages")

        if extra is not None:
            record = AgentRunActionModel.create_replace_with_variant(
                user_id=key.user_id,
                session_id=key.session_id,
                agent_id=key.agent_id,
                run_id=run_id,
                root_run_id=root_run_id,
                messages=messages,
                variant=extra,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
            )
        else:
            record = AgentRunActionModel.create_replace(
                user_id=key.user_id,
                session_id=key.session_id,
                agent_id=key.agent_id,
                run_id=run_id,
                root_run_id=root_run_id,
                messages=messages,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
            )
        created = await self._engine.create(record)

        # NOTE: REPLACE is append-only. We MUST NOT delete prior action rows
        # here even though REPLACE supersedes earlier history at the load
        # layer (load_messages stops at the first REPLACE seen during DESC
        # scan — see line 573). The previous "GC prior rows" branch
        # contradicted RFC-0022's event-sourcing model, broke RFC-0088's
        # ``persisted ⊇ event`` SSOT invariant for NAC, made /undo
        # impossible past a REPLACE boundary, dropped audit / billing
        # token records, and introduced a cross-task GC race when two
        # REPLACEs landed concurrently (each task's
        # ``delete WHERE action_id != self.id`` could wipe the peer's
        # row, leaving zero survivors). Retention is a separate concern —
        # implement it via a periodic job at the storage layer, NEVER as
        # a side-effect inside an event write.
        return created

    async def persist_run_start(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        agent_name: str,
        parent_run_id: str | None = None,
        trace_id: str | None = None,
    ) -> AgentRunActionModel | None:
        """Persist a RUN_START lifecycle marker (RFC-0022 Phase 2).

        RFC-0022: 在 run iteration 开始时写入 RUN_START 边界标记。

        RUN_START 是 **Class A** (Reader-NOOP) action — 它不改变 messages
        状态，只携带 ``trace_id`` 用于 observability 串联（RFC-0024）。

        ``idempotency_key=f"{run_id}:start"`` 保证重试 / 双写场景下只落一行
        （DB 唯一约束）；冲突时静默吞掉返回 None，调用方按已存在处理。

        Returns:
            The created action record, or ``None`` if the row already exists
            for this run_id (idempotency key collision — treat as success).
        """
        record = AgentRunActionModel.create_run_start(
            user_id=key.user_id,
            session_id=key.session_id,
            agent_id=key.agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            trace_id=trace_id,
            idempotency_key=f"{run_id}:start",
        )
        try:
            return await self._engine.create(record)
        except Exception as exc:
            # Idempotency: a RUN_START with this run_id already exists. Common
            # causes: caller-side retry, two concurrent runners observing the
            # same run_id (lock layer should prevent but we belt-and-suspender).
            # Anything else (true ORM/DB error) is surfaced as warning + None;
            # we do NOT raise because RUN_START failure must not abort the run.
            logger.warning(
                "persist_run_start: skipped (idempotent or db error) key=%s run_id=%s err=%s",
                key,
                run_id,
                exc,
            )
            return None

    async def persist_run_end(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        agent_name: str,
        status: RunStatus,
        parent_run_id: str | None = None,
        reason: str | None = None,
    ) -> AgentRunActionModel | None:
        """Persist a RUN_END lifecycle marker (RFC-0022 Phase 2).

        Same semantics as ``persist_run_start`` — Class A reader-NOOP,
        idempotent via ``idempotency_key=f"{run_id}:end"``, never raises.

        ``status`` is business-required (Literal['ok','error','cancelled']
        enforced by the factory).

        RFC-0024: trace_id is RUN_START only. The schema field on
        ``RunEndExtra`` is kept as a protobuf-compat parking lot for
        forward readers but the writer never populates it.
        """
        record = AgentRunActionModel.create_run_end(
            user_id=key.user_id,
            session_id=key.session_id,
            agent_id=key.agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            status=status,
            reason=reason,
            idempotency_key=f"{run_id}:end",
        )
        try:
            return await self._engine.create(record)
        except Exception as exc:
            logger.warning(
                "persist_run_end: skipped (idempotent or db error) key=%s run_id=%s status=%s err=%s",
                key,
                run_id,
                status,
                exc,
            )
            return None

    async def _scan_actions_desc(
        self,
        *,
        key: AgentRunActionKey,
        page_size: int,
        cursor: int | None,
    ) -> list[AgentRunActionModel]:
        filters = [
            ComparisonFilter.eq("user_id", key.user_id),
            ComparisonFilter.eq("session_id", key.session_id),
            ComparisonFilter.eq("agent_id", key.agent_id),
        ]
        if cursor is not None:
            filters.append(ComparisonFilter.lt("created_at_ns", cursor))

        action_filter = AndFilter(filters=filters)
        return await self._engine.find_many(
            AgentRunActionModel,
            filters=action_filter,
            order_by=("-created_at_ns", "-action_id"),
            limit=page_size,
        )

    async def load_messages(
        self,
        *,
        key: AgentRunActionKey,
    ) -> list[Message]:
        """Reconstruct the messages state by folding the action stream.

        Algorithm (RFC-0022 §Reduction 算法):

        - DESC scan paginated by ``created_at_ns``
        - REPLACE = self-contained anchor → record + early stop
        - UNDO before=X: resolve X's earliest ``created_at_ns`` once via
          ``_first_action_ns_of_run``, set ``cutoff_ns = min(cutoff_ns,
          target_first_ns)``. Any subsequent action with
          ``created_at_ns >= cutoff_ns`` is skipped (it was undone). This
          correctly handles target runs with multiple actions (RUN_START
          / multiple APPENDs / RUN_END), and runs that interleaved between
          the UNDO target and the UNDO row itself.
        - APPEND collected, then re-applied in chronological order at the end
        - RUN_START / RUN_END are reader-NOOPs (Class A)
        - Missing UNDO target → silent no-op (no cutoff update)
          (NOTE: RFC-0022 §不变量 #2 says fail-loud; production silently
          tolerates. To resolve in a follow-up.)
        """
        cursor: int | None = None
        page_size = 200
        # All actions with created_at_ns >= cutoff_ns are undone. None = no cutoff.
        cutoff_ns: int | None = None
        appends_desc: list[AgentRunActionModel] = []
        base_replace: AgentRunActionModel | None = None

        while True:
            page = await self._scan_actions_desc(
                key=key,
                page_size=page_size,
                cursor=cursor,
            )
            if not page:
                break

            stop = False
            for action in page:
                # Skip actions undone by an UNDO seen earlier in this DESC scan.
                if cutoff_ns is not None and int(action.created_at_ns) >= cutoff_ns:
                    continue

                if action.action_type == RunActionType.UNDO and action.undo_before_run_id:
                    target_first_ns = await self._first_action_ns_of_run(key=key, run_id=action.undo_before_run_id)
                    if target_first_ns is not None:
                        # Multiple UNDOs nest by taking the earliest cutoff.
                        cutoff_ns = target_first_ns if cutoff_ns is None else min(cutoff_ns, target_first_ns)
                    # else: missing target → silent no-op (current production semantics)
                    continue

                if action.action_type == RunActionType.REPLACE:
                    base_replace = action
                    stop = True
                    break

                if action.action_type == RunActionType.APPEND:
                    appends_desc.append(action)

            if stop:
                break

            cursor = int(page[-1].created_at_ns)

        logger.debug(
            "🔍 [HISTORY-DEBUG] load_messages key=%s: found %d APPEND actions, base_replace=%s",
            key,
            len(appends_desc),
            base_replace is not None,
        )

        messages: list[Message] = []
        message_by_id: dict[str, int] = {}

        def apply_messages(msgs: list[Message] | None, source: str) -> None:
            if not msgs:
                logger.debug("🔍 [HISTORY-DEBUG] apply_messages(%s): no messages", source)
                return
            logger.debug(
                "🔍 [HISTORY-DEBUG] apply_messages(%s): %d messages, roles=%s",
                source,
                len(msgs),
                [m.role.value for m in msgs],
            )
            for msg in msgs:
                if msg.role == Role.SYSTEM:
                    continue
                msg_id = str(msg.id)
                if msg_id not in message_by_id:
                    message_by_id[msg_id] = len(messages)
                    messages.append(msg)
                else:
                    messages[message_by_id[msg_id]] = msg

        if base_replace is not None:
            apply_messages(base_replace.replace_messages, "REPLACE")

        for i, action in enumerate(reversed(appends_desc)):
            apply_messages(action.append_messages, f"APPEND[{i}] run_id={action.run_id}")

        logger.debug(
            "🔍 [HISTORY-DEBUG] load_messages RESULT: %d messages, roles=%s",
            len(messages),
            [m.role.value for m in messages],
        )
        for i, msg in enumerate(messages):
            block_types = [type(b).__name__ for b in msg.content]
            logger.debug(
                "🔍 [HISTORY-DEBUG]   msg[%d] role=%s blocks=%s text_preview=%.100s",
                i,
                msg.role.value,
                block_types,
                msg.get_text_content()[:100] if msg.get_text_content() else "<empty>",
            )

        # Read-side defenses against stream-interrupted artifacts:
        # 1. Drop reasoning-only assistant messages — heals legacy rows
        #    written before persist_append got the same filter (DeepSeek
        #    rejects them; other providers get confused).
        # 2. Synthesize tool_result for orphan tool_use — Anthropic / OpenAI
        #    APIs reject messages lists with unpaired tool_use, would brick
        #    sessions on resume.
        before_filter = len(messages)
        messages = [m for m in messages if not _is_reasoning_only_assistant(m)]
        dropped = before_filter - len(messages)
        if dropped:
            logger.info(
                "load_messages: dropped %d legacy reasoning-only assistant message(s)",
                dropped,
            )
        return _ensure_tool_use_paired(messages)
