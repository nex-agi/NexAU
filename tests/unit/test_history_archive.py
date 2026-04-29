# Copyright (c) Nex-AGI. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Tests for RFC-0021 history archive on context compaction (single-file design)."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from nexau.archs.main_sub.execution.middleware.context_compaction.history_archive import (
    ARCHIVE_SUBDIR,
    ARCHIVED_IMAGE_URL_PREFIX,
    BOUNDARY_KEY,
    IMAGES_SUBDIR,
    TRANSCRIPT_FILENAME,
    HistoryArchiveWriter,
    build_archive_hint,
)
from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock


@pytest.fixture
def temp_workdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sandbox(temp_workdir):
    return LocalSandbox(sandbox_id="test", work_dir=temp_workdir)


class _FakeAgentState:
    """Minimal stand-in matching the get_sandbox() contract used by from_sandbox."""

    def __init__(self, sandbox: Any, agent_id: str = "agent_test", run_id: str = "run_test") -> None:
        self._sandbox = sandbox
        self.agent_id = agent_id
        self.run_id = run_id

    def get_sandbox(self) -> Any:
        return self._sandbox


def _msg(role: Role, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def _tool_result_msg(text: str) -> Message:
    return Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="tu_1", content=text, is_error=False)],
    )


def _summary_msg(text: str) -> Message:
    return Message(
        role=Role.USER,
        content=[TextBlock(text=text)],
        metadata={"isSummary": True},
    )


def _read_transcript_lines(archive_dir: Path) -> list[str]:
    transcript = archive_dir / TRANSCRIPT_FILENAME
    if not transcript.exists():
        return []
    return [ln for ln in transcript.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _split_lines(lines: list[str]) -> tuple[list[Message], list[dict[str, Any]]]:
    """Split transcript lines into (messages, boundaries)."""
    messages: list[Message] = []
    boundaries: list[dict[str, Any]] = []
    for ln in lines:
        obj = json.loads(ln)
        if isinstance(obj, dict) and BOUNDARY_KEY in obj:
            boundaries.append(obj[BOUNDARY_KEY])
        else:
            messages.append(Message.model_validate_json(ln))
    return messages, boundaries


# ---------------------------------------------------------------------------
# from_sandbox: graceful degradation
# ---------------------------------------------------------------------------


class TestFromSandbox:
    def test_returns_none_when_agent_state_none(self):
        assert HistoryArchiveWriter.from_sandbox(agent_state=None) is None

    def test_returns_none_when_no_get_sandbox(self):
        class _Bare:
            pass

        assert HistoryArchiveWriter.from_sandbox(agent_state=_Bare()) is None

    def test_returns_none_when_get_sandbox_returns_none(self):
        agent_state = _FakeAgentState(sandbox=None)
        assert HistoryArchiveWriter.from_sandbox(agent_state=agent_state) is None

    def test_creates_archive_dir(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None
        assert Path(temp_workdir, ARCHIVE_SUBDIR).is_dir()
        assert writer.total_rounds == 0
        assert writer.total_archived == 0


# ---------------------------------------------------------------------------
# Single-round write: transcript.jsonl gets removed messages + 1 boundary line
# ---------------------------------------------------------------------------


class TestSingleRound:
    def test_writes_messages_then_boundary_line(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        removed = [
            _msg(Role.USER, "what is the capital of France"),
            _msg(Role.ASSISTANT, "Paris"),
            _tool_result_msg("search results: ..."),
        ]
        record = writer.write_round(
            removed=removed,
            tokens_before=1000,
            tokens_after=200,
            trigger_reason="token_threshold",
            strategy_name="SlidingWindowCompaction",
            run_id="run_1",
            agent_id="agent_1",
        )
        assert record is not None
        assert record.round == 1
        assert record.removed_message_count == 3

        archive_dir = Path(temp_workdir, ARCHIVE_SUBDIR)
        transcript_path = archive_dir / TRANSCRIPT_FILENAME
        assert transcript_path.is_file()
        # 没有其他文件 (单文件设计)
        assert {p.name for p in archive_dir.iterdir()} == {TRANSCRIPT_FILENAME}

        lines = _read_transcript_lines(archive_dir)
        assert len(lines) == 4  # 3 messages + 1 boundary

        messages, boundaries = _split_lines(lines)
        assert len(messages) == 3
        assert len(boundaries) == 1

        b = boundaries[0]
        assert b["round"] == 1
        assert b["removed_message_count"] == 3
        assert b["strategy"] == "SlidingWindowCompaction"
        assert b["run_id"] == "run_1"
        assert b["agent_id"] == "agent_1"
        assert "capital of France" in b["preview"]

    def test_empty_removed_returns_none_no_write(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        record = writer.write_round(
            removed=[],
            tokens_before=0,
            tokens_after=0,
            trigger_reason="x",
            strategy_name="x",
            run_id=None,
            agent_id=None,
        )
        assert record is None
        assert writer.total_rounds == 0
        archive_dir = Path(temp_workdir, ARCHIVE_SUBDIR)
        assert not (archive_dir / TRANSCRIPT_FILENAME).exists()


# ---------------------------------------------------------------------------
# Multiple rounds: transcript grows append-only
# ---------------------------------------------------------------------------


class TestMultipleRounds:
    def test_round_numbers_increment(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        for i in range(1, 4):
            record = writer.write_round(
                removed=[_msg(Role.USER, f"round {i}"), _msg(Role.ASSISTANT, f"reply {i}")],
                tokens_before=100 * i,
                tokens_after=10,
                trigger_reason="t",
                strategy_name="S",
                run_id=f"r{i}",
                agent_id="a",
            )
            assert record is not None
            assert record.round == i

        lines = _read_transcript_lines(Path(temp_workdir, ARCHIVE_SUBDIR))
        messages, boundaries = _split_lines(lines)
        assert len(boundaries) == 3
        assert [b["round"] for b in boundaries] == [1, 2, 3]
        assert len(messages) == 6  # 2 per round × 3
        assert writer.total_rounds == 3
        assert writer.total_archived == 6

    def test_transcript_is_append_only(self, temp_workdir, sandbox):
        """Round N's write must not mutate bytes from rounds 1..N-1."""
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        writer.write_round(
            removed=[_msg(Role.USER, "first round content")],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r1",
            agent_id="a",
        )
        transcript = Path(temp_workdir, ARCHIVE_SUBDIR, TRANSCRIPT_FILENAME)
        prefix = transcript.read_bytes()

        time.sleep(0.01)
        writer.write_round(
            removed=[_msg(Role.USER, "second round content")],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r2",
            agent_id="a",
        )
        # 整个 prefix 不可被改动
        new_bytes = transcript.read_bytes()
        assert new_bytes.startswith(prefix), "transcript prefix must remain byte-identical"
        assert len(new_bytes) > len(prefix), "transcript must grow"


# ---------------------------------------------------------------------------
# Resume: restart writer picks up max round from existing transcript
# ---------------------------------------------------------------------------


class TestResumeAcrossInstances:
    def test_resume_after_existing_transcript(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        w1 = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert w1 is not None
        w1.write_round(
            removed=[_msg(Role.USER, "first"), _msg(Role.ASSISTANT, "ok")],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r1",
            agent_id="a",
        )

        # Re-instantiate writer (simulate restart)
        w2 = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert w2 is not None
        assert w2.total_archived == 2
        assert w2.total_rounds == 1

        record = w2.write_round(
            removed=[_msg(Role.USER, "second")],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r2",
            agent_id="a",
        )
        assert record is not None
        assert record.round == 2

        lines = _read_transcript_lines(Path(temp_workdir, ARCHIVE_SUBDIR))
        _, boundaries = _split_lines(lines)
        assert [b["round"] for b in boundaries] == [1, 2]


# ---------------------------------------------------------------------------
# Boundary record content
# ---------------------------------------------------------------------------


class TestBoundaryContent:
    def test_summary_id_recorded(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        s_msg = _summary_msg("summary content")
        record = writer.write_round(
            removed=[_msg(Role.USER, "u"), s_msg, _msg(Role.ASSISTANT, "a")],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.summary_message_id == str(s_msg.id)

    def test_no_summary_id_when_absent(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        record = writer.write_round(
            removed=[_msg(Role.USER, "u"), _msg(Role.ASSISTANT, "a")],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.summary_message_id is None


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


class TestPreview:
    def test_preview_skips_system_role(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        sys_msg = _msg(Role.SYSTEM, "system instructions here")
        user_msg = _msg(Role.USER, "actual user question text")
        record = writer.write_round(
            removed=[sys_msg, user_msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert "user question text" in record.preview
        assert "system instructions" not in record.preview

    def test_preview_truncated_to_300(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        long_text = "x" * 1000
        record = writer.write_round(
            removed=[_msg(Role.USER, long_text)],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert len(record.preview) == 300


# ---------------------------------------------------------------------------
# Hint text
# ---------------------------------------------------------------------------


class TestArchiveHint:
    def test_hint_mentions_path_and_search_tools(self):
        text = build_archive_hint(
            total_archived=42,
            total_rounds=3,
            latest_round=3,
        )
        assert ARCHIVE_SUBDIR in text  # 路径走常量, 不再可配
        assert TRANSCRIPT_FILENAME in text
        assert "search_file_content" in text
        assert "read_file" in text
        assert "42" in text


# ---------------------------------------------------------------------------
# Diff invariant: the messages NOT in messages_after end up in transcript
# ---------------------------------------------------------------------------


class TestMessageIdDiff:
    def test_diff_excludes_kept_messages(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        m1 = _msg(Role.USER, "msg 1")
        m2 = _msg(Role.ASSISTANT, "msg 2")
        m3 = _msg(Role.USER, "msg 3")
        m4_kept = _msg(Role.ASSISTANT, "msg 4 - kept")

        messages_before = [m1, m2, m3, m4_kept]
        new_summary = _summary_msg("summary of m1,m2,m3")
        messages_after = [new_summary, m4_kept]

        after_ids = {m.id for m in messages_after}
        removed = [m for m in messages_before if m.id not in after_ids]
        assert [m.id for m in removed] == [m1.id, m2.id, m3.id]

        record = writer.write_round(
            removed=removed,
            tokens_before=400,
            tokens_after=100,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.first_message_id == str(m1.id)
        assert record.last_message_id == str(m3.id)


class TestNoCircularImport:
    def test_history_archive_module_imports(self):
        from nexau.archs.main_sub.execution.middleware.context_compaction import history_archive  # noqa: F401


# ---------------------------------------------------------------------------
# Middleware integration: _maybe_archive_compaction + _inject_archive_hint
# ---------------------------------------------------------------------------


@pytest.fixture
def archive_middleware():
    from nexau.archs.main_sub.execution.middleware.context_compaction import (
        ContextCompactionMiddleware,
    )

    return ContextCompactionMiddleware(
        max_context_tokens=10000,
        auto_compact=True,
        save_history=True,
    )


class TestMiddlewareArchiveIntegration:
    def test_maybe_archive_writes_diff_and_injects_hint(self, temp_workdir, sandbox, archive_middleware):
        agent_state = _FakeAgentState(sandbox=sandbox, agent_id="ag", run_id="rn")

        m1 = _msg(Role.USER, "lost message 1")
        m2 = _msg(Role.ASSISTANT, "lost message 2")
        kept = _msg(Role.ASSISTANT, "kept tail")
        new_summary = _summary_msg("compressed summary")

        messages_before = [m1, m2, kept]
        messages_after = [new_summary, kept]

        result = archive_middleware._maybe_archive_compaction(
            agent_state=agent_state,
            messages_before=messages_before,
            messages_after=messages_after,
            tokens_before=500,
            tokens_after=100,
            trigger_reason="test_trigger",
            strategy_name="TestStrategy",
        )

        # transcript.jsonl 单文件
        arch_dir = Path(temp_workdir, ARCHIVE_SUBDIR)
        transcript = arch_dir / TRANSCRIPT_FILENAME
        assert transcript.is_file()
        assert {p.name for p in arch_dir.iterdir()} == {TRANSCRIPT_FILENAME}

        # 2 removed messages + 1 boundary
        lines = _read_transcript_lines(arch_dir)
        messages, boundaries = _split_lines(lines)
        assert len(messages) == 2
        assert len(boundaries) == 1

        # Hint appended to summary message
        summary_after = next(m for m in result if (m.metadata or {}).get("isSummary") is True)
        all_text = " ".join(b.text for b in summary_after.content if isinstance(b, TextBlock))
        assert ".nexau_history_archive" in all_text
        assert TRANSCRIPT_FILENAME in all_text

        # 多数 LLM provider 把同一消息的多个 TextBlock 拼成单字符串而不加分隔。
        # hint 必须是独立 TextBlock 且自带 ``\n\n`` 前缀, 否则在 prompt / Langfuse
        # trace 里会粘在前一段文本末尾, 用户/agent 都看不见。
        text_blocks = [b for b in summary_after.content if isinstance(b, TextBlock)]
        hint_block = next(b for b in text_blocks if "📁 [Archive]" in b.text)
        assert hint_block.text.startswith("\n\n"), (
            "hint TextBlock must start with '\\n\\n' to render as its own paragraph after sibling TextBlocks (provider-side concat)"
        )

    def test_maybe_archive_skipped_when_flag_false(self, temp_workdir, sandbox):
        from nexau.archs.main_sub.execution.middleware.context_compaction import (
            ContextCompactionMiddleware,
        )

        mw = ContextCompactionMiddleware(
            max_context_tokens=10000,
            auto_compact=True,
            save_history=False,
        )
        agent_state = _FakeAgentState(sandbox=sandbox)
        m1 = _msg(Role.USER, "x")
        kept = _msg(Role.ASSISTANT, "y")

        result = mw._maybe_archive_compaction(
            agent_state=agent_state,
            messages_before=[m1, kept],
            messages_after=[kept],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
        )
        assert result == [kept]
        assert not Path(temp_workdir, ARCHIVE_SUBDIR).exists()

    def test_no_summary_falls_back_to_framework_message(self, temp_workdir, sandbox, archive_middleware):
        """ToolResultCompaction 路径: 无 summary 消息, hint 应作为 framework 消息追加。"""
        agent_state = _FakeAgentState(sandbox=sandbox)

        m1 = _msg(Role.USER, "u1")
        m2 = _msg(Role.ASSISTANT, "a1")
        kept = _msg(Role.ASSISTANT, "kept")

        result = archive_middleware._maybe_archive_compaction(
            agent_state=agent_state,
            messages_before=[m1, m2, kept],
            messages_after=[kept],  # 无 summary
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
        )

        assert len(result) == 2  # kept + framework hint
        last = result[-1]
        assert last.role == Role.FRAMEWORK
        text = last.content[0].text  # type: ignore[union-attr]
        assert ".nexau_history_archive" in text
        assert TRANSCRIPT_FILENAME in text

    def test_multiple_rounds_append_to_same_file(self, temp_workdir, sandbox, archive_middleware):
        agent_state = _FakeAgentState(sandbox=sandbox)

        # Round 1
        m1 = _msg(Role.USER, "round 1 lost")
        kept = _msg(Role.ASSISTANT, "kept")
        archive_middleware._maybe_archive_compaction(
            agent_state=agent_state,
            messages_before=[m1, kept],
            messages_after=[kept],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
        )

        # Round 2
        m2 = _msg(Role.USER, "round 2 lost")
        archive_middleware._maybe_archive_compaction(
            agent_state=agent_state,
            messages_before=[kept, m2],
            messages_after=[m2],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
        )

        arch_dir = Path(temp_workdir, ARCHIVE_SUBDIR)
        # 仍然只有一个 transcript.jsonl
        assert {p.name for p in arch_dir.iterdir()} == {TRANSCRIPT_FILENAME}

        lines = _read_transcript_lines(arch_dir)
        messages, boundaries = _split_lines(lines)
        assert len(boundaries) == 2
        assert [b["round"] for b in boundaries] == [1, 2]


# ---------------------------------------------------------------------------
# ImageBlock support
# ---------------------------------------------------------------------------


def _png_1x1() -> bytes:
    """A real, minimal 1x1 PNG (67 bytes). Validates b64-decode produces decodable bytes."""
    import base64

    return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


class TestImageBlockRoundTrip:
    """ImageBlock 必须能完整 round-trip 通过 transcript archive (URL 和 base64 两种)。"""

    def test_url_image_unchanged_in_archive(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        url_img = ImageBlock(url="https://example.com/photo.jpg", mime_type="image/jpeg", detail="high")
        msg = Message(role=Role.USER, content=[TextBlock(text="see this:"), url_img])
        record = writer.write_round(
            removed=[msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.extracted_images == 0  # URL 图片不外置

        # 没有 images/ 目录被创建 (因为 needs_processing=False)
        assert not (Path(temp_workdir, ARCHIVE_SUBDIR, IMAGES_SUBDIR)).exists()

        # 反序列化回来 URL 完全一致
        lines = _read_transcript_lines(Path(temp_workdir, ARCHIVE_SUBDIR))
        messages, _ = _split_lines(lines)
        assert len(messages) == 1
        restored_blocks = messages[0].content
        url_block = next(b for b in restored_blocks if isinstance(b, ImageBlock))
        assert url_block.url == "https://example.com/photo.jpg"
        assert url_block.base64 is None
        assert url_block.mime_type == "image/jpeg"
        assert url_block.detail == "high"

    def test_base64_image_externalized_to_file(self, temp_workdir, sandbox):
        """关键: base64 数据被剥离到独立文件, transcript.jsonl 里只剩 file: 引用。"""
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        png_bytes = _png_1x1()
        import base64 as _b64

        b64_str = _b64.b64encode(png_bytes).decode()
        b64_img = ImageBlock(base64=b64_str, mime_type="image/png", detail="auto")
        msg = Message(role=Role.USER, content=[TextBlock(text="ref:"), b64_img])

        record = writer.write_round(
            removed=[msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.extracted_images == 1

        # images/ 目录已创建, 含一个 .png
        images_dir = Path(temp_workdir, ARCHIVE_SUBDIR, IMAGES_SUBDIR)
        assert images_dir.is_dir()
        png_files = list(images_dir.glob("*.png"))
        assert len(png_files) == 1
        # 文件名 = {msg_id}-{idx}.png
        assert png_files[0].name == f"{msg.id}-1.png"
        # 字节级一致 (PNG 完整还原)
        assert png_files[0].read_bytes() == png_bytes

        # transcript.jsonl 里 base64 字段清空, url 改为 file: 引用
        lines = _read_transcript_lines(Path(temp_workdir, ARCHIVE_SUBDIR))
        messages, _ = _split_lines(lines)
        img_block = next(b for b in messages[0].content if isinstance(b, ImageBlock))
        assert img_block.base64 is None
        assert img_block.url is not None
        assert img_block.url.startswith(ARCHIVED_IMAGE_URL_PREFIX)
        assert img_block.url == f"{ARCHIVED_IMAGE_URL_PREFIX}{IMAGES_SUBDIR}/{msg.id}-1.png"
        assert img_block.mime_type == "image/png"

        # transcript 文件本身不再含原始 base64 大块 (体积控制核心目的)
        raw = (Path(temp_workdir, ARCHIVE_SUBDIR) / TRANSCRIPT_FILENAME).read_text()
        assert b64_str not in raw

    def test_mixed_blocks_preserve_order(self, temp_workdir, sandbox):
        """text + url-img + text + b64-img 混合, 所有 block 顺序与类型不变。"""
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        import base64 as _b64

        b64_str = _b64.b64encode(_png_1x1()).decode()
        msg = Message(
            role=Role.USER,
            content=[
                TextBlock(text="prelude"),
                ImageBlock(url="https://cdn.example.com/x.png", mime_type="image/png"),
                TextBlock(text="and another"),
                ImageBlock(base64=b64_str, mime_type="image/png"),
            ],
        )
        record = writer.write_round(
            removed=[msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.extracted_images == 1  # 只有 b64 那个被外置

        # round-trip
        lines = _read_transcript_lines(Path(temp_workdir, ARCHIVE_SUBDIR))
        messages, _ = _split_lines(lines)
        blocks = messages[0].content
        assert len(blocks) == 4
        assert isinstance(blocks[0], TextBlock) and blocks[0].text == "prelude"
        assert isinstance(blocks[1], ImageBlock) and blocks[1].url == "https://cdn.example.com/x.png"
        assert isinstance(blocks[2], TextBlock) and blocks[2].text == "and another"
        assert isinstance(blocks[3], ImageBlock)
        assert blocks[3].base64 is None
        assert (blocks[3].url or "").startswith(ARCHIVED_IMAGE_URL_PREFIX)

    def test_multiple_b64_images_get_unique_filenames(self, temp_workdir, sandbox):
        """同一条 message 多张 b64 图: 文件名按 block index 区分, 不冲突。"""
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        import base64 as _b64

        b64_str = _b64.b64encode(_png_1x1()).decode()
        msg = Message(
            role=Role.USER,
            content=[
                ImageBlock(base64=b64_str, mime_type="image/png"),  # idx=0
                TextBlock(text="separator"),
                ImageBlock(base64=b64_str, mime_type="image/jpeg"),  # idx=2 (不同 mime)
            ],
        )
        record = writer.write_round(
            removed=[msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.extracted_images == 2

        images_dir = Path(temp_workdir, ARCHIVE_SUBDIR, IMAGES_SUBDIR)
        names = sorted(p.name for p in images_dir.iterdir())
        assert names == [f"{msg.id}-0.png", f"{msg.id}-2.jpg"]

    def test_unknown_mime_falls_back_to_bin(self, temp_workdir, sandbox):
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        import base64 as _b64

        b64_str = _b64.b64encode(b"\x00\x01\x02").decode()
        msg = Message(
            role=Role.USER,
            content=[ImageBlock(base64=b64_str, mime_type="image/heif")],
        )
        record = writer.write_round(
            removed=[msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.extracted_images == 1

        files = list(Path(temp_workdir, ARCHIVE_SUBDIR, IMAGES_SUBDIR).iterdir())
        assert len(files) == 1
        assert files[0].name.endswith(".bin")  # 未知 mime → .bin

    def test_no_images_no_dir_created(self, temp_workdir, sandbox):
        """没有图片的 message 不应触发 images/ 目录创建 (零开销)。"""
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        record = writer.write_round(
            removed=[_msg(Role.USER, "plain text only")],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.extracted_images == 0
        assert not Path(temp_workdir, ARCHIVE_SUBDIR, IMAGES_SUBDIR).exists()

    def test_b64_image_nested_in_tool_result_externalized(self, temp_workdir, sandbox):
        """ImageBlock 嵌套在 ToolResultBlock.content 里 (multimodal 工具返回图)
        必须也被外置 — 否则 transcript 仍会被 base64 撑爆。"""
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        import base64 as _b64

        png_bytes = _png_1x1()
        b64_str = _b64.b64encode(png_bytes).decode()

        # 一个 tool result, content 是 list[Text|Image] 形式
        tool_msg = Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[
                        TextBlock(text="here is the frame:"),
                        ImageBlock(base64=b64_str, mime_type="image/png"),
                        TextBlock(text="(end)"),
                    ],
                    is_error=False,
                )
            ],
        )
        record = writer.write_round(
            removed=[tool_msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )
        assert record is not None
        assert record.extracted_images == 1

        # 嵌套图片用 {msg_id}-{outer_idx}-{inner_idx}.{ext} 命名 (避免与顶层冲突)
        images_dir = Path(temp_workdir, ARCHIVE_SUBDIR, IMAGES_SUBDIR)
        files = sorted(images_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == f"{tool_msg.id}-0-1.png"  # outer block idx=0, inner image idx=1
        assert files[0].read_bytes() == png_bytes

        # transcript 里 ToolResultBlock 内的 ImageBlock 已被换成 file: ref
        lines = _read_transcript_lines(Path(temp_workdir, ARCHIVE_SUBDIR))
        messages, _ = _split_lines(lines)
        tool_block = messages[0].content[0]
        assert isinstance(tool_block, ToolResultBlock)
        assert isinstance(tool_block.content, list)
        nested_img = next(b for b in tool_block.content if isinstance(b, ImageBlock))
        assert nested_img.base64 is None
        assert nested_img.url is not None
        assert nested_img.url.startswith(ARCHIVED_IMAGE_URL_PREFIX)

        # 原 base64 不在 transcript 文件
        raw = (Path(temp_workdir, ARCHIVE_SUBDIR) / TRANSCRIPT_FILENAME).read_text()
        assert b64_str not in raw

    def test_original_messages_not_mutated(self, temp_workdir, sandbox):
        """归档不应破坏 active context: 原 message 的 base64 必须仍在。"""
        agent_state = _FakeAgentState(sandbox=sandbox)
        writer = HistoryArchiveWriter.from_sandbox(agent_state=agent_state)
        assert writer is not None

        import base64 as _b64

        b64_str = _b64.b64encode(_png_1x1()).decode()
        original_img = ImageBlock(base64=b64_str, mime_type="image/png")
        msg = Message(role=Role.USER, content=[original_img])

        writer.write_round(
            removed=[msg],
            tokens_before=100,
            tokens_after=10,
            trigger_reason="t",
            strategy_name="S",
            run_id="r",
            agent_id="a",
        )

        # 原 ImageBlock 仍带完整 base64 (没被改掉)
        assert original_img.base64 == b64_str
        assert original_img.url is None
