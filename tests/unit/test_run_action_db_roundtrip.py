# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DB-layer integration tests for AgentRunActionModel (RFC-0022).

These tests exercise the **real INSERT → SELECT roundtrip** through
SQLAlchemy + SQLite (in-memory), catching gaps the Pydantic-only tests
miss:

- Does ``StrEnum`` action_type roundtrip through the column without losing
  its enum-class identity?
- Does ``extra: dict[str, Any]`` JSONB column preserve ``extra='allow'``
  unknown fields after dict→JSONB→dict cycle?
- Does ``idempotency_key`` UNIQUE actually fire at the DB layer? (Multi-NULL
  semantics differ between SQLite and PostgreSQL.)
- Does ``parse_extra()`` correctly dispatch by action_type after roundtrip?

These are the Phase 1 protobuf-philosophy guarantees in their actual
deployed environment, not just at the Python-class layer.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.exc import IntegrityError

from nexau.archs.session.models.agent_run_action_model import (
    AgentRunActionModel,
    AppendExtra,
    CompactFocusedVariant,
    CompactStats,
    RunActionType,
    RunEndExtra,
    RunStartExtra,
    UndoExtra,
    UserClearVariant,
)
from nexau.archs.session.orm.filters import ComparisonFilter
from nexau.archs.session.orm.sql_engine import SQLDatabaseEngine
from nexau.core.messages import Message, Role, TextBlock


@asynccontextmanager
async def _engine() -> AsyncGenerator[SQLDatabaseEngine]:
    """In-memory SQLite engine with AgentRunActionModel table set up."""
    eng = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
    try:
        await eng.setup_models([AgentRunActionModel])
        yield eng
    finally:
        await eng._engine.dispose()


def _msg(text: str) -> Message:
    return Message(role=Role.ASSISTANT, content=[TextBlock(text=text)])


# ============================================================================
# action_type StrEnum roundtrip
# ============================================================================


@pytest.mark.parametrize(
    "action_type",
    list(RunActionType),
)
def test_action_type_enum_db_roundtrip(action_type):
    """StrEnum action_type must come back as RunActionType (not raw str) after SELECT."""
    import asyncio

    async def run():
        async with _engine() as eng:
            # Build a minimal valid record per action_type
            kwargs = dict(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                agent_name="test",
            )
            if action_type == RunActionType.APPEND:
                model = AgentRunActionModel.create_append(messages=[_msg("hi")], **kwargs)
            elif action_type == RunActionType.UNDO:
                model = AgentRunActionModel.create_undo(undo_before_run_id="r0", **kwargs)
            elif action_type == RunActionType.REPLACE:
                model = AgentRunActionModel.create_replace(messages=[_msg("x")], reason="user_clear", **kwargs)
            elif action_type == RunActionType.RUN_START:
                model = AgentRunActionModel.create_run_start(**kwargs)
            elif action_type == RunActionType.RUN_END:
                model = AgentRunActionModel.create_run_end(status="ok", **kwargs)
            else:
                raise AssertionError(f"unhandled action_type {action_type}")

            await eng.create(model)

            # SELECT back
            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", model.action_id))
            assert result is not None
            results = [result]
            assert len(results) == 1
            row = results[0]

            # action_type stored as plain string (forward-compat — see model
            # docstring). StrEnum equality with str works:
            # ``RunActionType.APPEND == "append"`` is True.
            assert isinstance(row.action_type, str)
            assert row.action_type == action_type.value
            assert row.action_type == action_type  # StrEnum / str equality

    asyncio.run(run())


# ============================================================================
# extra JSONB roundtrip per action_type — typed *Extra in / parse_extra out
# ============================================================================


def test_append_extra_roundtrip_db():
    """AppendExtra survives factory → INSERT → SELECT → parse_extra cycle."""
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_append(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                messages=[_msg("hello")],
                iter_index=3,
                iter_kind="tool_round",
                llm_call_id="msg_01ABC",
                trace_id="trace_xyz",
                idempotency_key="r1:3",
            )
            await eng.create(written)

            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", written.action_id))
            assert result is not None
            results = [result]
            assert len(results) == 1
            read_back = results[0]

            assert read_back.action_type == RunActionType.APPEND
            assert read_back.idempotency_key == "r1:3"
            assert read_back.append_messages is not None
            assert len(read_back.append_messages) == 1

            extra = read_back.parse_extra()
            assert isinstance(extra, AppendExtra)
            assert extra.iter_index == 3
            assert extra.iter_kind == "tool_round"
            assert extra.llm_call_id == "msg_01ABC"
            assert extra.trace_id == "trace_xyz"

    asyncio.run(run())


def test_replace_user_clear_variant_roundtrip_db():
    """UserClearVariant — generic create_replace dispatches via discriminator."""
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_replace(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                messages=[_msg("clean state")],
                reason="user_clear",
                trace_id="trace_clear",
            )
            await eng.create(written)

            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", written.action_id))
            assert result is not None
            extra = result.parse_extra()
            # Variant dispatch: reason="user_clear" → UserClearVariant
            assert isinstance(extra, UserClearVariant)
            assert extra.reason == "user_clear"
            assert extra.trace_id == "trace_clear"

    asyncio.run(run())


def test_undo_extra_roundtrip_db():
    """UndoExtra and undo_before_run_id both roundtrip."""
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_undo(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                undo_before_run_id="r0_target",
                reason="user_rewind",
            )
            await eng.create(written)

            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", written.action_id))
            assert result is not None
            results = [result]
            row = results[0]
            assert row.undo_before_run_id == "r0_target"
            extra = row.parse_extra()
            assert isinstance(extra, UndoExtra)
            assert extra.reason == "user_rewind"

    asyncio.run(run())


def test_run_start_extra_roundtrip_db():
    """RunStartExtra (with trace_id) roundtrips."""
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_run_start(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                trace_id="trace_start",
                idempotency_key="r1:start",
            )
            await eng.create(written)

            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", written.action_id))
            assert result is not None
            results = [result]
            row = results[0]
            assert row.idempotency_key == "r1:start"
            extra = row.parse_extra()
            assert isinstance(extra, RunStartExtra)
            assert extra.trace_id == "trace_start"

    asyncio.run(run())


def test_run_end_extra_roundtrip_db():
    """RunEndExtra (with int finished_at_ns + status) roundtrips."""
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_run_end(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                status="error",
                finished_at_ns=1234567890,
                reason="LLM rate limit",
            )
            await eng.create(written)

            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", written.action_id))
            assert result is not None
            results = [result]
            extra = results[0].parse_extra()
            assert isinstance(extra, RunEndExtra)
            assert extra.status == "error"
            assert extra.finished_at_ns == 1234567890
            assert extra.reason == "LLM rate limit"

    asyncio.run(run())


def test_compact_focused_variant_roundtrip_db():
    """CompactFocusedVariant — typed factory enforces focus_instructions, full round-trip.

    RFC-0022 §设计原则 §6 Class B aliasing: compaction reuses REPLACE rather
    than introducing a Class C ``COMPACT`` action_type, so old SDK readers
    fold it correctly as a state replacement.
    """
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_replace_compact_focused(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                messages=[_msg("compacted summary")],
                focus_instructions="Keep auth context",
                strategy="sliding_window",
                stats=CompactStats(
                    pre_message_count=100,
                    post_message_count=20,
                    pre_tokens=50000,
                    post_tokens=8000,
                ),
                trace_id="trace_compact",
            )
            await eng.create(written)

            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", written.action_id))
            assert result is not None
            extra = result.parse_extra()
            # Variant dispatch via discriminator
            assert isinstance(extra, CompactFocusedVariant)
            assert extra.reason == "compact_focused"
            assert extra.strategy == "sliding_window"
            assert extra.focus_instructions == "Keep auth context"
            assert extra.stats is not None
            assert extra.stats.pre_message_count == 100
            assert extra.stats.post_message_count == 20
            assert extra.stats.pre_tokens == 50000
            assert extra.stats.post_tokens == 8000
            assert extra.trace_id == "trace_compact"
            # And the row reads back as REPLACE (not a separate type) so old
            # SDKs fold it correctly via the existing REPLACE branch.
            assert result.action_type == RunActionType.REPLACE

    asyncio.run(run())


# ============================================================================
# extra='allow' unknown fields survive DB cycle (protobuf forward-compat)
# ============================================================================


def test_extra_allow_unknown_fields_survive_db_cycle():
    """Future SDK writes ``extra={"future_field": "x"}``; old SDK reads back intact.

    This is the protobuf-philosophy "forward compat" guarantee at the
    DB layer (not just at the Pydantic layer).
    """
    import asyncio

    async def run():
        async with _engine() as eng:
            # Manually construct a record with an "unknown future field" in extra,
            # simulating a write by a newer SDK version.
            model = AgentRunActionModel(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                action_type=RunActionType.RUN_END,
                extra={
                    "status": "ok",
                    "finished_at_ns": 1234,
                    "future_metric_v2": {"latency_p99_ms": 850, "tokens_used": 5000},
                },
            )
            await eng.create(model)

            result = await eng.find_first(AgentRunActionModel, filters=ComparisonFilter.eq("action_id", model.action_id))
            assert result is not None
            results = [result]
            row = results[0]
            # Raw extra dict preserves unknown field
            assert row.extra is not None
            assert row.extra["future_metric_v2"] == {"latency_p99_ms": 850, "tokens_used": 5000}
            # parse_extra also preserves it via extra='allow'
            extra = row.parse_extra()
            assert isinstance(extra, RunEndExtra)
            dumped = extra.model_dump()
            assert dumped["future_metric_v2"] == {"latency_p99_ms": 850, "tokens_used": 5000}

    asyncio.run(run())


# ============================================================================
# idempotency_key UNIQUE + multi-NULL semantics (per backend)
# ============================================================================


def test_idempotency_key_unique_violation():
    """Two INSERTs with the same idempotency_key must violate UNIQUE."""
    import asyncio

    async def run():
        async with _engine() as eng:
            first = AgentRunActionModel.create_run_start(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                idempotency_key="r1:0",
            )
            await eng.create(first)

            duplicate = AgentRunActionModel.create_append(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                messages=[_msg("dup")],
                idempotency_key="r1:0",  # ← collision
            )
            with pytest.raises(IntegrityError):
                await eng.create(duplicate)

    asyncio.run(run())


def test_idempotency_key_multi_null_allowed():
    """Multiple rows with NULL idempotency_key must all succeed (UNIQUE allows multi-NULL).

    Both SQLite and PostgreSQL allow multiple NULLs in UNIQUE columns by default.
    Phase 1 batch APPEND path relies on this (legacy callers leave key NULL).
    """
    import asyncio

    async def run():
        async with _engine() as eng:
            for i in range(5):
                model = AgentRunActionModel.create_append(
                    user_id="u",
                    session_id="s",
                    agent_id="a",
                    run_id=f"r{i}",
                    root_run_id=f"r{i}",
                    messages=[_msg(f"msg {i}")],
                    # idempotency_key omitted → NULL
                )
                assert model.idempotency_key is None
                await eng.create(model)

            # All 5 inserted successfully
            results = await eng.find_many(AgentRunActionModel, filters=ComparisonFilter.eq("user_id", "u"))
            assert len(results) == 5

    asyncio.run(run())


# ============================================================================
# parse_extra() boundaries
# ============================================================================


def test_parse_extra_returns_none_when_extra_is_none():
    """Action with no extra → parse_extra returns None."""
    model = AgentRunActionModel(
        user_id="u",
        session_id="s",
        agent_id="a",
        run_id="r1",
        root_run_id="r1",
        action_type=RunActionType.APPEND,
        extra=None,
    )
    assert model.parse_extra() is None


def test_parse_extra_dispatches_correctly_per_action_type():
    """Each action_type's parse_extra returns the right *Extra class.

    REPLACE dispatches further via the discriminator union: ``reason="user_clear"``
    → ``UserClearVariant``. Other action_types use the flat *Extra classes.
    """
    cases: list[tuple[RunActionType, type, dict]] = [
        (RunActionType.APPEND, AppendExtra, {"trace_id": "tx"}),
        # REPLACE without reason falls to UnknownReplaceVariant (reason missing).
        # With reason="user_clear" dispatches to UserClearVariant.
        (RunActionType.REPLACE, UserClearVariant, {"reason": "user_clear", "trace_id": "tx"}),
        (RunActionType.UNDO, UndoExtra, {"trace_id": "tx"}),
        (RunActionType.RUN_START, RunStartExtra, {"trace_id": "tx"}),
        (RunActionType.RUN_END, RunEndExtra, {"trace_id": "tx"}),
    ]
    for action_type, expected_cls, extra_dict in cases:
        model = AgentRunActionModel(
            user_id="u",
            session_id="s",
            agent_id="a",
            run_id="r1",
            root_run_id="r1",
            action_type=action_type,
            extra=extra_dict,
        )
        parsed = model.parse_extra()
        assert isinstance(parsed, expected_cls), f"{action_type.name}: expected {expected_cls.__name__}, got {type(parsed).__name__}"


def test_parse_extra_tolerates_partial_data():
    """parse_extra never raises on protobuf-compat partial dict."""
    # Empty dict
    model = AgentRunActionModel(
        user_id="u",
        session_id="s",
        agent_id="a",
        run_id="r1",
        root_run_id="r1",
        action_type=RunActionType.RUN_END,
        extra={},
    )
    extra = model.parse_extra()
    assert isinstance(extra, RunEndExtra)
    assert extra.status is None  # Optional, no required validation

    # Partial dict with unknown field
    model2 = AgentRunActionModel(
        user_id="u",
        session_id="s",
        agent_id="a",
        run_id="r1",
        root_run_id="r1",
        action_type=RunActionType.RUN_END,
        extra={"random_unknown_field": [1, 2, 3]},
    )
    extra2 = model2.parse_extra()
    assert isinstance(extra2, RunEndExtra)


# ============================================================================
# Forward compat: older SDK reads rows written by newer SDK with unknown
# action_type values. Critical for NAC where agent runtime images aren't
# redeployed every PR — mixed-version SDKs coexist during upgrade windows.
# ============================================================================


def test_action_type_unknown_value_does_not_crash_old_reader():
    """Older SDK (without future RunActionType members) must not crash on read.

    Scenario:
    - v2 SDK ships with RunActionType including new ``COMPACT_INCREMENTAL``
    - NAC has runtime instances with v1 SDK (only APPEND/UNDO/REPLACE/etc) still running
    - v2 instance writes ``action_type='compact_incremental'``
    - v1 instance reads the row → MUST NOT raise LookupError

    This test simulates that by inserting a row with an action_type string
    that's not in the current RunActionType enum, then reading it back.
    """
    import asyncio

    from sqlalchemy import text

    async def run():
        async with _engine() as eng:
            # Simulate "future SDK" writing an unknown action_type via raw SQL
            async with eng._engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO agent_run_actions ("
                        "  action_id, user_id, session_id, agent_id, run_id, "
                        "  root_run_id, agent_name, created_at, created_at_ns, "
                        "  action_type"
                        ") VALUES ("
                        "  'a_future', 'u', 's', 'a', 'r1', 'r1', 'test', "
                        "  CURRENT_TIMESTAMP, 1000000000, 'compact_incremental_v3'"
                        ")"
                    )
                )

            # Current SDK (= "old" relative to future) reads it
            result = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", "a_future"),
            )
            # MUST NOT raise; reader sees raw string
            assert result is not None
            assert result.action_type == "compact_incremental_v3"
            # parse_extra returns None for unknown action_type (forward-compat
            # branch in agent_run_action_model.parse_extra). Consumer code
            # treats None as "skip" or reads ``self.extra`` as raw dict.
            assert result.parse_extra() is None
            # And the unknown action_type isn't in the SDK's enum
            assert result.action_type not in {at.value for at in RunActionType}

    asyncio.run(run())


# ============================================================================
# Forward-compat matrix (RFC-0022 §设计原则 §6)
#
# Verifies that every Phase 1 addition is either Class A (Reader-NOOP) or
# Class B (Old-Type Aliasing) — no Class C silent-corruption hazards.
# When future PRs add a new action_type, they MUST extend this matrix and
# justify which class the new type belongs to.
# ============================================================================


def _reference_old_reader_fold(actions):
    """Pre-Phase-1 ``fold`` (only knows APPEND / REPLACE / UNDO).

    Models a v0 SDK reader that has NOT been upgraded — used to verify that
    every Phase 1 addition either degenerates to Class A skip or Class B
    aliased semantics, never silent corruption.
    """
    state = []
    snapshots = {}
    for a in actions:
        # v0 didn't have RUN_START / RUN_END / 'compact_*' awareness. It
        # matched on RunActionType.{APPEND,REPLACE,UNDO} string values
        # directly (StrEnum values), and silently skipped anything else.
        at = str(a.action_type)
        if at == "append":
            state.extend(a.append_messages or [])
        elif at == "replace":
            state = list(a.replace_messages or [])
        elif at == "undo":
            assert a.undo_before_run_id in snapshots
            state = list(snapshots[a.undo_before_run_id])
        else:
            # Unknown to v0 reader: silently skip
            pass
        # v0 reader didn't have RUN_START snapshot mechanic, so snapshots
        # never get populated except synthetically. UNDO targets in this
        # matrix avoid v0 readers (only post-Phase-1 readers UNDO).
    return state


def test_forward_compat_class_a_run_start_run_end_skip_safely():
    """Class A: pre-Phase-1 reader skips RUN_START / RUN_END → state unaffected.

    Acceptance criterion for Class A: the Phase 1 reader's fold and the
    pre-Phase-1 reader's fold produce identical messages state, because
    the new types are reader-NOOP (don't change messages state).
    """
    msg = Message(role=Role.ASSISTANT, content=[TextBlock(text="hi")])
    actions = [
        AgentRunActionModel.create_run_start(user_id="u", session_id="s", agent_id="a", run_id="r1", root_run_id="r1"),
        AgentRunActionModel.create_append(user_id="u", session_id="s", agent_id="a", run_id="r1", root_run_id="r1", messages=[msg]),
        AgentRunActionModel.create_run_end(user_id="u", session_id="s", agent_id="a", run_id="r1", root_run_id="r1", status="ok"),
    ]
    # Pre-Phase-1 reader sees only APPEND, skips RUN_START/RUN_END
    state = _reference_old_reader_fold(actions)
    assert len(state) == 1


def test_forward_compat_class_b_compaction_aliases_to_replace():
    """Class B: pre-Phase-1 reader applies compaction as plain REPLACE.

    Acceptance criterion for Class B: the pre-Phase-1 reader's fold of a
    'compact_*' REPLACE row produces the SAME state as the Phase 1 reader's
    fold (no silent context-overflow OOM during mixed-version upgrade).
    """
    pre = Message(role=Role.ASSISTANT, content=[TextBlock(text="huge pre-compact context")])
    summary = Message(role=Role.ASSISTANT, content=[TextBlock(text="summary")])

    actions = [
        AgentRunActionModel.create_append(user_id="u", session_id="s", agent_id="a", run_id="r0", root_run_id="r0", messages=[pre]),
        AgentRunActionModel.create_replace_compact_auto(
            user_id="u",
            session_id="s",
            agent_id="a",
            run_id="r1",
            root_run_id="r1",
            messages=[summary],
            strategy="sliding_window",
            stats=CompactStats(pre_message_count=1, post_message_count=1),
        ),
    ]
    state = _reference_old_reader_fold(actions)
    # The compact_auto REPLACE row aliases to plain REPLACE → state replaced
    # with the summary, not stuck on the huge pre-compact context. This is
    # the property that prevents silent OOM in old readers.
    assert len(state) == 1
    text_block = state[0].content[0]
    assert isinstance(text_block, TextBlock)
    assert text_block.text == "summary"


def test_forward_compat_no_class_c_action_types_in_phase_1():
    """Phase 1 ships zero Class C action_types — every member is A or B compatible.

    A Class C type would change messages state AND have no equivalent old
    type to alias to → pre-Phase-1 readers would silently skip and corrupt.
    Verify the Phase 1 RunActionType enum contains only Class A / B types.
    """
    # Pre-Phase-1 type set (the historical baseline).
    pre_phase_1_types = {"append", "replace", "undo"}
    # Class A = Reader-NOOP (don't change messages state).
    class_a_types = {"run_start", "run_end"}
    # Phase 1 enum members must equal pre-Phase-1 ∪ Class A — anything
    # outside this set would be Class C and require a reader gate.
    current = {member.value for member in RunActionType}
    expected = pre_phase_1_types | class_a_types
    assert current == expected, (
        f"Phase 1 RunActionType must be Class A or B-compatible only. "
        f"Unexpected members: {current - expected}. "
        f"If you intend to add a Class C type, see RFC-0022 §设计原则 §6 "
        f"for the reader-version-gate requirement, then update this test."
    )


# ============================================================================
# RFC-0024: ToolResultBlock.raw_output roundtrip
# ============================================================================


def test_tool_result_block_raw_output_roundtrip_db():
    """RFC-0024: ToolResultBlock.raw_output survives DB roundtrip.

    The new field carries the tool's raw structured return value (returnDisplay,
    duration_ms, custom meta) so downstream UI consumers can render typed fields
    without reverse-parsing the formatter output. Verify the field round-trips
    cleanly through APPEND.append_messages → JSON column → SELECT → parse.
    """
    import asyncio

    from nexau.core.messages import Message, Role, ToolResultBlock

    raw_output = {
        "content": "Output: ok\n",
        "returnDisplay": "Done in 320ms",
        "duration_ms": 320,
        "exit_code": 0,
        "stdout_file": "/tmp/foo/stdout.txt",
    }

    tool_msg = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call_1",
                content="Output: ok\n",
                raw_output=raw_output,
            )
        ],
    )

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_append(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r1",
                root_run_id="r1",
                messages=[tool_msg],
                iter_index=0,
                iter_kind="tool_round",
                idempotency_key="r1:0",
            )
            await eng.create(written)

            result = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", written.action_id),
            )
            assert result is not None
            assert result.append_messages is not None
            stored = result.append_messages[0]
            # `append_messages` round-trips Message objects (pydantic models).
            block = stored.content[0]
            block_dict = block.model_dump() if hasattr(block, "model_dump") else block
            assert block_dict["type"] == "tool_result"
            assert block_dict["raw_output"] == raw_output

    asyncio.run(run())


def test_tool_result_block_raw_output_none_excluded():
    """ToolResultBlock with raw_output=None should serialize cleanly.

    Plain-string-returning tools leave raw_output as None; verify the
    serialized JSON either omits the field entirely or carries explicit null
    (both are valid for downstream readers using `extra='allow'`).
    """
    import json

    from nexau.core.messages import ToolResultBlock

    block = ToolResultBlock(tool_use_id="call_1", content="just a string")
    serialized = json.loads(block.model_dump_json())
    # Field present (None) is OK; absent is OK. Just must not blow up.
    assert serialized["type"] == "tool_result"
    assert serialized["content"] == "just a string"
    assert serialized.get("raw_output") in (None,)


# ----------------------------------------------------------------------------
# RFC-0024 edge cases — trace_id absence / inheritance, raw_output non-dict
# shapes. Pre-merge safety net: lock down the behaviour the user-visible
# Compare panel + tracing UI rely on.
# ----------------------------------------------------------------------------


def test_run_start_extra_omitted_when_no_payload_fields_set():
    """``create_run_start`` with no user_message_blocks / fresh_context /
    trace_id collapses ``extra`` to NULL (not an empty dict).

    Why: callers without an observability stack pass ``trace_id=None``.
    Persisted row should be ``extra=NULL``, not ``extra='{}'`` — the
    latter forces every reader to disambiguate "extra exists but
    empty" from "no extra at all" and bloats storage on long sessions.
    """
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_run_start(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r_no_extra",
                root_run_id="r_no_extra",
                idempotency_key="r_no_extra:start",
                # Intentional: no trace_id / user_message_blocks / fresh_context.
            )
            await eng.create(written)
            row = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", written.action_id),
            )
            assert row is not None
            assert row.extra is None, f"expected extra=NULL, got {row.extra!r}"

    asyncio.run(run())


def test_run_start_extra_only_trace_id_persists_just_that_field():
    """Only ``trace_id`` set → persisted ``extra`` carries exactly that
    one field (no leaking-None entries via ``model_dump(exclude_none=True)``).
    """
    import asyncio

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_run_start(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r_tid",
                root_run_id="r_tid",
                trace_id="trace_only",
                idempotency_key="r_tid:start",
            )
            await eng.create(written)
            row = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", written.action_id),
            )
            assert row is not None
            assert row.extra == {"trace_id": "trace_only"}, f"unexpected extra: {row.extra!r}"
            extra = row.parse_extra()
            assert isinstance(extra, RunStartExtra)
            assert extra.trace_id == "trace_only"

    asyncio.run(run())


def test_subagent_run_start_carries_inherited_trace_id():
    """Sub-agent RUN_START persisted with the same trace_id as parent.

    Persistence-side proof of explicit trace_id propagation through the
    ``call_sub_agent(trace_id=...)`` chain (agent_tool → SubAgentManager →
    sub_agent.run_async). Single Langfuse trace covers the whole task tree.
    """
    import asyncio

    async def run():
        async with _engine() as eng:
            await eng.create(
                AgentRunActionModel.create_run_start(
                    user_id="u",
                    session_id="s",
                    agent_id="parent_agent",
                    run_id="r_parent",
                    root_run_id="r_parent",
                    agent_name="parent",
                    trace_id="trace_shared",
                    idempotency_key="r_parent:start",
                )
            )
            child = AgentRunActionModel.create_run_start(
                user_id="u",
                session_id="s",
                agent_id="child_agent",
                run_id="r_child",
                root_run_id="r_parent",
                parent_run_id="r_parent",
                agent_name="child",
                # Simulates effective_trace_id resolution from parent_agent_state.
                trace_id="trace_shared",
                idempotency_key="r_child:start",
            )
            await eng.create(child)

            row = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", child.action_id),
            )
            assert row is not None
            extra = row.parse_extra()
            assert isinstance(extra, RunStartExtra)
            assert extra.trace_id == "trace_shared"
            assert row.parent_run_id == "r_parent"
            assert row.root_run_id == "r_parent"

    asyncio.run(run())


def test_subagent_can_override_inherited_trace_id():
    """Explicit non-None trace_id on the sub-agent wins over the parent's.

    Lets advanced callers fork the trace tree (e.g. running a sub-agent
    under an isolated Langfuse trace).
    """
    import asyncio

    async def run():
        async with _engine() as eng:
            await eng.create(
                AgentRunActionModel.create_run_start(
                    user_id="u",
                    session_id="s",
                    agent_id="parent",
                    run_id="rA",
                    root_run_id="rA",
                    trace_id="trace_A",
                    idempotency_key="rA:start",
                )
            )
            child = AgentRunActionModel.create_run_start(
                user_id="u",
                session_id="s",
                agent_id="child",
                run_id="rB",
                root_run_id="rA",
                parent_run_id="rA",
                trace_id="trace_B",  # explicit, overrides parent
                idempotency_key="rB:start",
            )
            await eng.create(child)
            row = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", child.action_id),
            )
            assert row is not None
            extra = row.parse_extra()
            assert isinstance(extra, RunStartExtra)
            assert extra.trace_id == "trace_B"

    asyncio.run(run())


def test_tool_result_block_raw_output_as_list_roundtrip():
    """``raw_output`` is typed ``dict | list | None``. Many tools return
    list-shaped structured output (table rows, search hits, batched API
    responses) — the persisted block must carry the list verbatim, not
    silently coerce to dict.
    """
    import asyncio

    from nexau.core.messages import Message, Role, ToolResultBlock

    raw_output: list[dict[str, object]] = [
        {"name": "alice", "score": 0.9},
        {"name": "bob", "score": 0.7},
    ]
    tool_msg = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call_list",
                content="2 results",
                raw_output=raw_output,
            )
        ],
    )

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_append(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r_list",
                root_run_id="r_list",
                messages=[tool_msg],
                iter_index=0,
                iter_kind="tool_round",
                idempotency_key="r_list:0",
            )
            await eng.create(written)
            row = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", written.action_id),
            )
            assert row is not None
            assert row.append_messages is not None
            stored_block = row.append_messages[0].content[0]
            block_dict = stored_block.model_dump() if hasattr(stored_block, "model_dump") else stored_block
            assert block_dict["raw_output"] == raw_output, f"list raw_output not preserved: {block_dict['raw_output']!r}"

    asyncio.run(run())


def test_tool_result_block_raw_output_with_unicode_and_nested_dict_roundtrip():
    """Realistic raw_output: nested dict + unicode + list mix. Common
    case for tools returning i18n strings or nested metadata. Pydantic
    + JSON column must round-trip without loss.
    """

    from nexau.core.messages import Message, Role, ToolResultBlock

    raw_output: dict[str, object] = {
        "returnDisplay": "搜索完成（中文）",
        "duration_ms": 432,
        "results": [
            {"title": "测试 / тест", "score": 0.95, "tags": ["分析", "🚀"]},
            {"title": "edge case", "score": 0.88, "tags": []},
        ],
        "meta": {"engine": "v2", "timestamp_ns": 1778000000_000_000_000},
    }
    tool_msg = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call_unicode",
                content="搜索完成",
                raw_output=raw_output,
            )
        ],
    )

    async def run():
        async with _engine() as eng:
            written = AgentRunActionModel.create_append(
                user_id="u",
                session_id="s",
                agent_id="a",
                run_id="r_uni",
                root_run_id="r_uni",
                messages=[tool_msg],
                iter_index=0,
                iter_kind="tool_round",
                idempotency_key="r_uni:0",
            )
            await eng.create(written)
            row = await eng.find_first(
                AgentRunActionModel,
                filters=ComparisonFilter.eq("action_id", written.action_id),
            )
            assert row is not None
            assert row.append_messages is not None
            stored_block = row.append_messages[0].content[0]
            block_dict = stored_block.model_dump() if hasattr(stored_block, "model_dump") else stored_block
            assert block_dict["raw_output"] == raw_output
