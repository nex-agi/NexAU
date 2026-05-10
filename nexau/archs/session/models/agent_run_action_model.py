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

"""Agent run action model — event-sourcing protocol (RFC-0022).

Event sourcing framing: each ``AgentRunActionModel`` row is a mutation on the
session's messages state; the messages list is the fold-derived state. See
``docs/rfcs/0022-agent-run-action-lifecycle-and-typed-blocks.md`` for the
full protocol design (RunActionType algebra, *Extra Pydantic typing,
fold algorithm, protobuf-philosophy compatibility).

action_type 值:
- APPEND      — incremental messages
- UNDO        — revert to before a specific run_id
- REPLACE     — full state replacement (user reset / debug / context compaction;
                see ReplaceExtra.reason canonical values)
- RUN_START   — run lifecycle begin marker (carries RunStartExtra)
- RUN_END     — run lifecycle end marker (carries RunEndExtra)
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter
from sqlalchemy import Column, Index
from sqlmodel import Field, SQLModel

from nexau.core.messages import Message

from ..id_generator import generate_action_id
from .types import PydanticJson


class RunActionType(StrEnum):
    """Run action operator tags. See RFC-0022 §6.1 for the reduction algebra.

    **Forward-compat classification (RFC-0022 §设计原则 §6)**:
    - APPEND / REPLACE / UNDO: existing types from pre-Phase-1 era (handled
      by all SDK versions)
    - RUN_START / RUN_END: **Class A** (Reader-NOOP) — pre-Phase-1 SDKs
      silently skip them; correct because they don't change messages state
    - **No Class C types** added in Phase 1. Compaction was originally
      drafted as ``COMPACT`` but reclassified as **Class B**: REPLACE +
      ``ReplaceExtra(reason="compact_auto" / "compact_manual" / "compact_focused", ...)``
      so pre-Phase-1 SDKs reading compacted history apply the correct
      reduction (state = payload). This is critical for NAC where agent
      runtime images aren't redeployed every PR — mixed-version SDKs
      coexist during upgrade windows, and a Class C compaction would
      cause silent context-overflow OOMs in old readers.
    """

    APPEND = "append"
    UNDO = "undo"
    REPLACE = "replace"
    RUN_START = "run_start"
    RUN_END = "run_end"


# ============================================================================
# Typed *Extra Pydantic classes (RFC-0022 §6.3)
#
# Protobuf evolution philosophy:
# - All fields ``T | None = None`` (no required fields at protocol level)
# - ``model_config = ConfigDict(extra='allow')`` — old reader ignores
#   future-added fields, new reader sees missing fields as None
# - Body uses ``str`` instead of ``Literal[...]`` so unknown enum values
#   don't break old SDKs reading new data; factory method sigs use
#   Literal for write-side strict validation (双层防御)
# - Business "required" enforced at factory layer, not protocol layer
# ============================================================================

PROTOBUF_PHILOSOPHY = ConfigDict(extra="allow")


class AppendExtra(BaseModel):
    """Extra metadata for APPEND actions.

    Phase 2 (iter 级持久化) fills these. Phase 1 batch APPEND keeps all None.
    """

    model_config = PROTOBUF_PHILOSOPHY

    iter_index: int | None = None
    iter_kind: str | None = None
    """canonical: 'tool_round' / 'final_response' / 'subagent_call'."""
    llm_call_id: str | None = None
    trace_id: str | None = None


class CompactStats(BaseModel):
    """Stats payload for ReplaceExtra.stats (compaction-flavored REPLACE).

    Defined before ReplaceExtra so the type reference resolves cleanly.
    """

    model_config = PROTOBUF_PHILOSOPHY

    pre_message_count: int | None = None
    post_message_count: int | None = None
    pre_tokens: int | None = None
    post_tokens: int | None = None


# ============================================================================
# ReplaceExtra — discriminated union over ``reason`` (protobuf oneof equivalent)
# ============================================================================
#
# Why discriminated union (vs flat all-optional)
# ----------------------------------------------
# A flat ``ReplaceExtra(reason, strategy, focus_instructions, stats, ...)`` lets
# any caller write ``ReplaceExtra(reason="user_clear", focus_instructions="...")``
# — semantically nonsense but type-checks. Each variant below carries ONLY the
# fields meaningful for its reason, so:
#
# - Compile-time: ``UserClearVariant`` doesn't have ``focus_instructions`` —
#   typo'd code fails mypy, IDE autocomplete only shows valid fields per branch.
# - Pattern matching with type narrowing: ``case CompactFocusedVariant(focus_
#   instructions=fi):`` — ``fi`` is guaranteed non-None at use site.
# - Mutual exclusion: a row can't be both user_clear and compact_focused.
#
# protobuf oneof correspondence
# -----------------------------
# - ``oneof variant { ... }`` → ``Annotated[Union[...], Discriminator(callable)]``
# - per-variant message → per-variant ``BaseModel`` subclass
# - unknown variant preserved as opaque field → ``UnknownReplaceVariant`` +
#   ``ConfigDict(extra='allow')``
# - mutually exclusive setter semantics → Pydantic discriminator dispatch
# - forward-compat (new variant, old reader) → callable Discriminator falls
#   back to ``UnknownReplaceVariant`` instead of raising ValidationError


class _ReplaceVariantBase(BaseModel):
    model_config = PROTOBUF_PHILOSOPHY  # extra='allow' for forward-compat
    trace_id: str | None = None


# Public alias — write-side callers (services, middlewares) accept this base
# class as the parameter type of typed REPLACE writes. The leading-underscore
# original stays for back-compat with internal references.
ReplaceVariantBase = _ReplaceVariantBase


class UserClearVariant(_ReplaceVariantBase):
    """User typed ``/clear`` or equivalent reset. No special metadata."""

    reason: Literal["user_clear"] = "user_clear"


class CompactAutoVariant(_ReplaceVariantBase):
    """Automatic context compaction triggered by token threshold.

    No ``focus_instructions`` field — auto compaction has no user intent
    to preserve. Compile-time guarantee that nobody accidentally fills it.
    """

    reason: Literal["compact_auto"] = "compact_auto"
    strategy: str | None = None
    """Which compaction algorithm ran (e.g. 'sliding_window' / 'tool_result').
    Diagnostic / telemetry; not part of fold semantics."""
    stats: CompactStats | None = None


class CompactManualVariant(_ReplaceVariantBase):
    """User invoked ``/compact`` without focus instructions.

    Same field set as ``CompactAutoVariant`` (just user-triggered instead of
    automatic). Kept as a distinct variant for view-deriver / UI / billing
    differentiation ("Auto-compacted" vs "Compacted by you" rendering).
    """

    reason: Literal["compact_manual"] = "compact_manual"
    strategy: str | None = None
    stats: CompactStats | None = None


class CompactFocusedVariant(_ReplaceVariantBase):
    """User invoked ``/compact <focus_instructions>``.

    The intent — what the user told the LLM to focus on while compacting —
    is preserved as machine-readable typed data. Codex CLI / Claude Code /
    Cursor all lose this at the storage layer (it lives only in the prompt
    sent to the LLM); we keep it for follow-up compactions, audit, and
    "show user what they asked for" UX.
    """

    reason: Literal["compact_focused"] = "compact_focused"
    strategy: str | None = None
    focus_instructions: str
    """REQUIRED — the whole point of this variant is preserving user intent.
    Empty / None should not produce a CompactFocusedVariant; use Manual instead."""
    stats: CompactStats | None = None


class UnknownReplaceVariant(_ReplaceVariantBase):
    """Fallback for ``reason`` values this SDK doesn't know.

    protobuf oneof's "unknown variant preserved" rule: a future SDK adds a
    new ``reason`` value (e.g. ``"fork"``, ``"import_from_cursor"``); old
    readers land here with the raw payload preserved verbatim via
    ``extra='allow'``. Fold algorithm continues correctly because REPLACE
    state semantics don't depend on extra.

    Without this fallback, callable Discriminator returning a non-existent
    tag would raise ValidationError, making any new ``reason`` a Class C
    silent-corruption hazard (RFC-0022 §设计原则 §6).
    """

    reason: str
    """Any string that didn't match a known discriminator."""


def _discriminate_replace(v: object) -> str:
    """Callable Discriminator with fallback — protobuf oneof unknown-field rule.

    Maps the ``reason`` value to a variant tag. Unknown / missing reason →
    ``__unknown__`` tag → ``UnknownReplaceVariant``.
    """
    reason: object
    if isinstance(v, dict):
        reason = cast("dict[str, Any]", v).get("reason")
    else:
        reason = getattr(v, "reason", None)
    if reason in ("user_clear", "compact_auto", "compact_manual", "compact_focused"):
        return cast(str, reason)
    return "__unknown__"


ReplaceExtra = Annotated[
    (
        Annotated[UserClearVariant, Tag("user_clear")]
        | Annotated[CompactAutoVariant, Tag("compact_auto")]
        | Annotated[CompactManualVariant, Tag("compact_manual")]
        | Annotated[CompactFocusedVariant, Tag("compact_focused")]
        | Annotated[UnknownReplaceVariant, Tag("__unknown__")]
    ),
    Discriminator(_discriminate_replace),
]
"""Discriminated union over REPLACE ``reason`` value. See variant docstrings.

Persistence layer stores raw ``dict[str, Any]``; this union is the typed
read-side view materialized via ``parse_extra()``.
"""

# Adapter for parse_extra dispatch
_REPLACE_EXTRA_ADAPTER: TypeAdapter[Any] = TypeAdapter(ReplaceExtra)


class UndoExtra(BaseModel):
    """Extra metadata for UNDO actions."""

    model_config = PROTOBUF_PHILOSOPHY

    reason: str | None = None
    """canonical: 'user_rewind' / 'user_edit' / 'system_recover'."""
    trace_id: str | None = None


class RunStartExtra(BaseModel):
    """Extra metadata for RUN_START lifecycle marker.

    **Class A (Reader-NOOP)** — RUN_START rows do NOT change messages state
    and carry NO messages payload (``append_messages`` / ``replace_messages``
    columns are NULL). Total DB cost is one row + this small metadata blob.

    Purpose: lifecycle boundary signal for view derivers / UI rendering /
    Phase 2 iter-level idempotency (``idempotency_key`` patterns like
    ``"{run_id}:start"``). Carries trace correlation metadata.

    **Not a snapshot anchor**: an earlier draft of the fold algorithm cached
    ``state`` here for an O(1) UNDO fast path, but bench showed that scheme
    paid O(N²) snapshot-copy cost across all RUN_STARTs on every fold pass —
    far worse than the O(N) filter-and-re-fold fallback that's only paid when
    UNDO actually fires. Snapshot caching was removed; UNDO is always
    filter-and-re-fold (RFC-0022 §Reduction 算法).

    **Field history**: an early draft also reserved ``user_message_blocks``
    (rendering hint) and ``fresh_context`` (compaction-boundary flag) here.
    Both were defined but never populated by ``Agent.run()``; the user
    message is recoverable from the next APPEND row, and no consumer for
    ``fresh_context`` materialised. Reserving unused schema slots misleads
    future maintainers (and AI assistants reading the code) into thinking
    they're load-bearing — removed in cleanup; YAGNI applies.
    """

    model_config = PROTOBUF_PHILOSOPHY

    trace_id: str | None = None
    """W3C trace id (32-hex middle segment of ``traceparent``). RFC-0024."""


class RunEndExtra(BaseModel):
    """Extra metadata for RUN_END lifecycle marker."""

    model_config = PROTOBUF_PHILOSOPHY

    status: str | None = None
    """canonical: 'ok' / 'error' / 'cancelled' — business required, factory enforces."""
    finished_at_ns: int | None = None
    reason: str | None = None
    """Free-form diagnostic when status is error / cancelled."""
    trace_id: str | None = None


# Top-level union dispatched via ``RunAction.action_type``. Note ``ReplaceExtra``
# itself is a (sub-)discriminated union over ``reason``; consumers ``match`` on
# the inner variant for compile-time field access (see UserClearVariant /
# CompactAutoVariant / CompactManualVariant / CompactFocusedVariant /
# UnknownReplaceVariant docstrings).
RunActionExtra = AppendExtra | ReplaceExtra | UndoExtra | RunStartExtra | RunEndExtra


# Convenience Literal aliases for factory method signatures (write-side strict).
# ``ReplaceReason`` is no longer a Literal — replaced by per-variant types.
# ``UndoReason`` stays free-form (no per-variant fields differentiate, so
# variant union doesn't pay off; document canonical values via field doc).
RunStatus = Literal["ok", "error", "cancelled"]
IterKind = Literal["tool_round", "final_response", "subagent_call"]


# TypeAdapter for list[Message] serialization
_messages_adapter: TypeAdapter[list[Message]] = TypeAdapter(list[Message])


class AgentRunActionModel(SQLModel, table=True):
    """Per-run history mutation record (append-only). See RFC-0022 §6.

    Each row represents one mutation on the session's messages state. The
    messages list is recovered by ``fold(actions)`` (see RFC-0022 §Reduction 算法).
    """

    __tablename__ = "agent_run_actions"  # type: ignore[assignment]
    __table_args__ = (
        Index(
            "idx_agent_run_actions_created_at_ns",
            "user_id",
            "session_id",
            "agent_id",
            "created_at_ns",
        ),
        Index(
            "idx_agent_run_actions_action_type_created_at_ns",
            "user_id",
            "session_id",
            "agent_id",
            "action_type",
            "created_at_ns",
        ),
    )

    action_id: str = Field(primary_key=True, default_factory=generate_action_id)
    user_id: str = Field(index=True)
    session_id: str = Field(index=True)
    agent_id: str = Field(index=True)
    run_id: str = Field(index=True)

    # === Run tracking ===
    root_run_id: str = Field(index=True)
    parent_run_id: str | None = Field(default=None, index=True)
    agent_name: str = ""

    # === Timestamp ===
    created_at: datetime = Field(default_factory=datetime.now)
    created_at_ns: int = Field(default_factory=time.time_ns, index=True)

    # === Action type (string column, NOT enum-typed at SQL layer) ===
    # CRITICAL forward-compat: column type is plain string (not SQL ENUM),
    # so older SDK readers without a future ``RunActionType`` member don't
    # crash on ``LookupError`` when reading rows written by newer SDK.
    # Production scenario: NAC agent runtime images aren't redeployed every
    # PR; mixed-version SDKs coexist during the upgrade window. An older SDK
    # reading a row with action_type='future_v3' MUST not crash.
    #
    # Same protobuf evolution philosophy as *Extra body fields:
    # - Storage: str (forward-compat, unknown values pass through)
    # - Write-side strict: factories construct via ``RunActionType.X``
    #   (StrEnum subclass of str → assignment auto-converts to str value)
    # - Read-side: consumers use string comparison or ``match`` (StrEnum
    #   equality with str works because ``RunActionType.APPEND == "append"``)
    action_type: str = Field(index=True)

    # === APPEND payload — incremental messages produced in this run ===
    append_messages: list[Message] | None = Field(default=None, sa_column=Column(PydanticJson(list[Message])))

    # === REPLACE payload — full self-contained messages ===
    # Used for user resets, debug imports, and context compaction
    # (compaction is Class B aliasing — see RunActionType / ReplaceExtra docstrings).
    replace_messages: list[Message] | None = Field(default=None, sa_column=Column(PydanticJson(list[Message])))

    # === UNDO payload — target run_id to undo before ===
    undo_before_run_id: str | None = None

    # === Streaming idempotency key ===
    # Convention: streaming consumers use "{run_id}:{iter_index}". Legacy batch APPEND keeps NULL.
    # SQL UNIQUE so retries (e.g. Redis Consumer Group redeliveries) collapse to one row.
    # NOTE: keep this as a plain ``Field`` (no ``sa_column=Column(...)``) so SQLModel infers
    # the column type from the ``str | None`` annotation. An explicit ``Column(name, unique=True,
    # nullable=True)`` without a ``String`` type would yield ``NullType`` in the DDL compiler.
    idempotency_key: str | None = Field(default=None, unique=True, nullable=True)

    # === Action-level catchall (typed *Extra serialized to JSONB) ===
    # Storage layer: loose ``dict[str, Any] | None`` so old readers ignore
    # unknown future fields gracefully (protobuf evolution philosophy).
    # Write-side: factories construct typed *Extra and serialize via model_dump.
    # Read-side: ``parse_extra()`` returns the typed *Extra instance.
    extra: dict[str, Any] | None = Field(default=None, sa_column=Column(PydanticJson(dict[str, Any])))

    def parse_extra(self) -> RunActionExtra | None:
        """Return the typed *Extra view of ``self.extra``, dispatched by action_type.

        For REPLACE: returns one of ``UserClearVariant`` / ``CompactAutoVariant``
        / ``CompactManualVariant`` / ``CompactFocusedVariant`` /
        ``UnknownReplaceVariant`` based on the ``reason`` field. ``match`` on
        the result for compile-time field access (e.g. ``focus_instructions``
        is only reachable on ``CompactFocusedVariant``).

        Returns ``None`` if ``self.extra`` is None OR ``self.action_type``
        is unknown to this SDK version (forward-compat: row was written
        by a newer SDK with an action_type not yet in our dispatch map —
        callers should treat this as "skip" or "raw dict").
        """
        if self.extra is None:
            return None
        # Explicit per-type dispatch keeps the return type narrowable by mypy
        # without ``# type: ignore``. Each branch's literal class makes the
        # ``model_validate`` return type concretely match the union member.
        if self.action_type == RunActionType.REPLACE:
            return _REPLACE_EXTRA_ADAPTER.validate_python(self.extra)
        if self.action_type == RunActionType.APPEND:
            return AppendExtra.model_validate(self.extra)
        if self.action_type == RunActionType.UNDO:
            return UndoExtra.model_validate(self.extra)
        if self.action_type == RunActionType.RUN_START:
            return RunStartExtra.model_validate(self.extra)
        if self.action_type == RunActionType.RUN_END:
            return RunEndExtra.model_validate(self.extra)
        # Unknown action_type — forward-compat: row written by newer SDK with
        # an action_type not yet known here. Caller must handle None.
        return None

    # ========================================================================
    # Factory methods (write-side strict — Literal types validate at IDE/mypy)
    # ========================================================================

    @classmethod
    def create_append(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        parent_run_id: str | None = None,
        agent_name: str = "",
        iter_index: int | None = None,
        iter_kind: IterKind | None = None,
        llm_call_id: str | None = None,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentRunActionModel:
        """Create an APPEND action — incremental messages in this run.

        Phase 2 iter 级持久化 fills ``iter_index`` / ``iter_kind`` / ``llm_call_id``;
        Phase 1 batch APPEND can leave them None.
        """
        extra = AppendExtra(
            iter_index=iter_index,
            iter_kind=iter_kind,
            llm_call_id=llm_call_id,
            trace_id=trace_id,
        )
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.APPEND,
            append_messages=messages,
            extra=extra.model_dump(exclude_none=True) or None,
            idempotency_key=idempotency_key,
        )

    @classmethod
    def create_undo(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        undo_before_run_id: str,
        reason: str | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
        trace_id: str | None = None,
    ) -> AgentRunActionModel:
        """Create an UNDO action — fold returns state from before ``undo_before_run_id``'s first action.

        ``reason`` is free-form; canonical values include ``"user_rewind"`` /
        ``"user_edit"`` / ``"system_recover"`` but UNDO doesn't need a
        discriminated union (no per-reason fields differ).
        """
        extra = UndoExtra(reason=reason, trace_id=trace_id)
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.UNDO,
            undo_before_run_id=undo_before_run_id,
            extra=extra.model_dump(exclude_none=True) or None,
        )

    @classmethod
    def create_replace_with_variant(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        variant: _ReplaceVariantBase | None,
        parent_run_id: str | None = None,
        agent_name: str = "",
    ) -> AgentRunActionModel:
        """Internal builder shared by typed ``create_replace_*`` factories."""
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.REPLACE,
            replace_messages=messages,
            extra=(variant.model_dump(exclude_none=True) or None) if variant is not None else None,
        )

    @classmethod
    def create_replace_user_clear(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        parent_run_id: str | None = None,
        agent_name: str = "",
        trace_id: str | None = None,
    ) -> AgentRunActionModel:
        """REPLACE for ``/clear`` (user-initiated reset). UserClearVariant — no fields beyond trace_id."""
        return cls.create_replace_with_variant(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            messages=messages,
            variant=UserClearVariant(trace_id=trace_id),
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )

    @classmethod
    def create_replace_compact_auto(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        strategy: str | None = None,
        stats: CompactStats | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
        trace_id: str | None = None,
    ) -> AgentRunActionModel:
        """REPLACE for automatic context compaction. CompactAutoVariant."""
        return cls.create_replace_with_variant(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            messages=messages,
            variant=CompactAutoVariant(strategy=strategy, stats=stats, trace_id=trace_id),
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )

    @classmethod
    def create_replace_compact_manual(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        strategy: str | None = None,
        stats: CompactStats | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
        trace_id: str | None = None,
    ) -> AgentRunActionModel:
        """REPLACE for ``/compact`` without focus instructions. CompactManualVariant."""
        return cls.create_replace_with_variant(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            messages=messages,
            variant=CompactManualVariant(strategy=strategy, stats=stats, trace_id=trace_id),
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )

    @classmethod
    def create_replace_compact_focused(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        focus_instructions: str,
        strategy: str | None = None,
        stats: CompactStats | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
        trace_id: str | None = None,
    ) -> AgentRunActionModel:
        """REPLACE for ``/compact <focus_instructions>`` — preserves user intent.

        ``focus_instructions`` is REQUIRED — that's the whole point of this
        variant (vs ``compact_manual``). Codex / Claude Code lose this at the
        storage layer; we keep it for follow-up compactions, audit, "show user
        what they asked for" UX.
        """
        if not focus_instructions:
            raise ValueError("create_replace_compact_focused requires non-empty focus_instructions")
        return cls.create_replace_with_variant(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            messages=messages,
            variant=CompactFocusedVariant(
                focus_instructions=focus_instructions,
                strategy=strategy,
                stats=stats,
                trace_id=trace_id,
            ),
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )

    @classmethod
    def create_replace(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        reason: str | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
        trace_id: str | None = None,
    ) -> AgentRunActionModel:
        """Generic REPLACE factory. Prefer the typed ``create_replace_*`` siblings.

        Used for two cases:
        1. Plain REPLACE without any reason annotation (``reason=None``) —
           writes ``extra=None``. This is the most common production path
           (``persist_replace`` from history_list calls here without reason).
        2. Forward-compat: caller knows a future reason value not yet wired
           into the typed factories. The reason string is preserved as
           ``UnknownReplaceVariant``-shaped extra.

        For known reasons, prefer the typed siblings:
          - ``create_replace_user_clear`` for /clear
          - ``create_replace_compact_auto`` / ``compact_manual`` / ``compact_focused``
        """
        if reason is None:
            variant: _ReplaceVariantBase | None = None
        else:
            # Build via discriminator dispatch — known reason → typed variant,
            # unknown → UnknownReplaceVariant. Same path parse_extra() takes.
            variant = _REPLACE_EXTRA_ADAPTER.validate_python({"reason": reason, "trace_id": trace_id})
        return cls.create_replace_with_variant(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            messages=messages,
            variant=variant,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )

    @classmethod
    def create_run_start(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        parent_run_id: str | None = None,
        agent_name: str = "",
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentRunActionModel:
        """Create a RUN_START lifecycle marker.

        reducer 在此处对当前 messages 状态拍快照(供未来 UNDO 使用)。
        """
        extra = RunStartExtra(
            trace_id=trace_id,
        )
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.RUN_START,
            extra=extra.model_dump(exclude_none=True) or None,
            idempotency_key=idempotency_key,
        )

    @classmethod
    def create_run_end(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        status: RunStatus,
        finished_at_ns: int | None = None,
        reason: str | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
        idempotency_key: str | None = None,
    ) -> AgentRunActionModel:
        """Create a RUN_END lifecycle marker.

        ``status`` is business-required (Literal enforced at factory). Defaults
        ``finished_at_ns`` to ``time.time_ns()`` if not provided.

        RFC-0024: trace_id is RUN_START only — not accepted here. The
        ``RunEndExtra.trace_id`` schema slot stays for forward-compat reads
        but no writer populates it.
        """
        if finished_at_ns is None:
            finished_at_ns = time.time_ns()
        extra = RunEndExtra(
            status=status,
            finished_at_ns=finished_at_ns,
            reason=reason,
        )
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.RUN_END,
            extra=extra.model_dump(exclude_none=True) or None,
            idempotency_key=idempotency_key,
        )
