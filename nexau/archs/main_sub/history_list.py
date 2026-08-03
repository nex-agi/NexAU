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

"""HistoryList: A list that automatically persists modifications to SessionManager."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Coroutine, Iterable
from typing import TYPE_CHECKING, Any, SupportsIndex

from nexau.archs.main_sub.utils.image_probe import (
    OVERSIZED_IMAGE_PLACEHOLDER,
    image_exceeds_hard_limit,
    resize_base64_image_if_oversized,
)
from nexau.core.messages import (
    ImageBlock,
    Message,
    Role,
    TextBlock,
)

if TYPE_CHECKING:
    from concurrent.futures import Future as ConcurrentFuture

    from nexau.archs.session import AgentRunActionKey, SessionManager
    from nexau.archs.session.models.agent_run_action_model import ReplaceVariantBase

logger = logging.getLogger(__name__)


def _resize_image_block(block: ImageBlock) -> None:
    """就地把一个超预算 ImageBlock 降采样到面积封顶内。

    url-only（``base64`` 为空）图跳过 —— 没有本地字节可处理。界内小图、
    解码/降采样失败都是 no-op（``resize_base64_image_if_oversized`` 返回 None），
    原样保留。
    """
    if not block.base64:
        return
    resized = resize_base64_image_if_oversized(block.base64, block.mime_type)
    if resized is None:
        return
    block.base64, block.mime_type = resized


def _omit_or_resize_image_block(block: ImageBlock) -> TextBlock | None:
    """超大图 → 返回占位 ``TextBlock``（调用方用它替换该 ``ImageBlock``）；否则就地
    降采样该图并返回 ``None``（原位置的 block 不动）。

    ``image_exceeds_hard_limit`` 必须在 ``_resize_image_block`` 之前判：后者会全解码，
    而硬 gate 正是要在解码前拦下 decode 炸弹 + 巨大 payload。url-only（``base64``
    为空）图既不触发硬 gate（无字节可测）也没有本地字节可 resize，返回 ``None`` 原样
    保留。
    """
    if block.base64 and image_exceeds_hard_limit(block.base64):
        return TextBlock(text=OVERSIZED_IMAGE_PLACEHOLDER)
    _resize_image_block(block)
    return None


# 设计边界(#601):持久化闸只处理**用户直传**的顶层 ImageBlock。工具产出
# (ToolResultBlock.content 嵌套图与 raw_output 副本)一律不碰 —— builtin 读
# 工具的图已被读路径强制压缩到界内;用户自定义工具/MCP 返回什么、多大、要不
# 要压缩,是工具作者自己的责任,框架不做任何限制或改写(此前的无差别
# resize/omit 曾误杀 >20MiB 的非图 base64、并使 image_token_budget=0 的
# escape hatch 形同虚设)。


class HistoryList(list[Message]):
    """A list that intercepts modifications and persists to SessionManager.

    This class provides transparent persistence for agent history with run-level semantics:
    - append/extend operations persist new messages immediately (APPEND action)
    - replace_all() intelligently detects append vs replace operations
    - Index assignment (__setitem__) only updates locally, does NOT persist
      (run-level API doesn't support message-level updates)

    The class maintains backward compatibility with plain list[Message] usage
    while adding automatic persistence when SessionManager is available.

    Note: The run-level API is designed for immutable history. Each agent.run()
    produces new messages that are appended. To modify history, use replace_all()
    which creates a REPLACE action.
    """

    def __init__(
        self,
        messages: list[Message] | None = None,
        *,
        session_manager: SessionManager | None = None,
        history_key: AgentRunActionKey | None = None,
        run_id: str | None = None,
        root_run_id: str | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
    ):
        """Initialize HistoryList.

        Args:
            messages: Initial messages (optional)
            session_manager: SessionManager for persistence (optional)
            history_key: Key for history storage (optional)
            run_id: Current run ID (optional)
            root_run_id: Root run ID (optional)
            parent_run_id: Parent run ID (optional)
            agent_name: Agent name for logging (optional)
        """
        super().__init__(messages or [])

        self._session_manager = session_manager
        self._history_key = history_key
        self._run_id = run_id
        self._root_run_id = root_run_id
        self._parent_run_id = parent_run_id
        self._agent_name = agent_name

        # Capture the owning event loop for cross-thread async scheduling.
        # When flush() is called from a worker thread (e.g. executor via
        # asyncio.to_thread), we use run_coroutine_threadsafe to dispatch
        # persistence I/O to the main loop instead of creating a nested loop.
        try:
            self._owner_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._owner_loop = None

        # Flag to enable/disable persistence
        self._persistence_enabled = session_manager is not None and history_key is not None

        # Accumulate messages added in current run (for batch persistence)
        self._pending_messages: list[Message] = []
        self._baseline_fingerprints: list[str] = self._compute_fingerprints([m for m in self if m.role != Role.SYSTEM])

        # 保持对 fire-and-forget persistence tasks 的引用，防止 GC 在
        # task 完成前回收它们（回收后 done-callback 不会触发）。
        self._background_tasks: set[asyncio.Task[None]] = set()

    def update_history_key(self, history_key: AgentRunActionKey) -> None:
        """Update the persistence routing key.

        Called by Agent.create() after async init resolves the real agent_id.
        """
        self._history_key = history_key
        self._persistence_enabled = self._session_manager is not None

    @property
    def has_pending_messages(self) -> bool:
        """Check if there are unflushed pending messages."""
        return bool(self._pending_messages)

    def _resize_oversized_images(self, messages: Iterable[Message]) -> None:
        """就地把超预算图片降采样到 ``DEFAULT_IMAGE_MAX_PIXELS`` 面积封顶内，超大图
        （``image_exceeds_hard_limit``：base64 > 20 MiB 或像素 > 60 MP）则直接 omit
        成占位（不 resize、不全解码，避免 decode 炸弹 + 巨大 payload）。

        在 append / extend / replace_all 把消息落到 ``_pending_messages`` 之前
        调用 —— 这是生产代码唯一的 persist 汇聚点，用户消息、工具结果、权限恢复
        重跑、压缩 REPLACE 都经此，但本闸只处理用户直传的顶层 ``ImageBlock``。
        #599 只在 ``read_visual_file`` 把图返回给 LLM 时降采样；这里补上用户直传
        图片的持久化兜底，避免原图 base64 进入 SQL / JSONL / memory / remote
        backend。自定义工具/MCP 输出保持框架透明，由工具作者治理：

        只处理 ``content`` 里**顶层** ``ImageBlock``(用户直传;超硬限 → 占位,
        超预算 → 就地 resize)。工具产出(``ToolResultBlock`` 的嵌套图与
        ``raw_output``)刻意不碰:builtin 读工具已在读路径压缩,自定义工具/
        MCP 的输出由工具作者自己负责。

        原地改同一份对象 —— 既缩小落库体积，也让下一轮喂给 LLM 的历史省 token。
        graceful：逐条消息 try/except，任何异常吞掉并记日志，绝不让历史写入因
        图片处理失败而失败（与 read_visual_file 降采样“降级不 fail”一致）。
        """
        for message in messages:
            try:
                self._resize_message_images(message)
            except Exception:
                logger.warning(
                    "Image resize on persist failed for message %s; keeping original",
                    message.id,
                    exc_info=True,
                )

    @staticmethod
    def _resize_message_images(message: Message) -> None:
        # 顶层 block：超大 ImageBlock → 就地按索引换成 TextBlock 占位（TextBlock 是
        # DiscriminatedBlock 的合法成员，类型安全）；界内超预算图就地 resize。只替换
        # 当前索引、不增删元素，因此边遍历边改是安全的。
        for index, block in enumerate(message.content):
            if isinstance(block, ImageBlock):
                placeholder = _omit_or_resize_image_block(block)
                if placeholder is not None:
                    message.content[index] = placeholder

    def append(self, item: Message) -> None:
        """Append a message (will be persisted on flush)."""
        self._resize_oversized_images((item,))
        super().append(item)

        if self._persistence_enabled and item.role != Role.SYSTEM:
            self._pending_messages.append(item)

    def extend(self, items: list[Message] | tuple[Message, ...]) -> None:  # type: ignore[override]
        """Extend with messages (will be persisted on flush)."""
        self._resize_oversized_images(items)
        super().extend(items)

        if self._persistence_enabled:
            non_system = [msg for msg in items if msg.role != Role.SYSTEM]
            self._pending_messages.extend(non_system)

    def __setitem__(self, key: SupportsIndex | slice, value: Message | list[Message]) -> None:  # type: ignore[override]
        """Intercept item assignment (local update only, no persistence).

        Note: The run-level API doesn't support message-level updates.
        This method only updates the local list. To persist changes,
        use replace_all() which creates a REPLACE action.

        Args:
            key: Index or slice
            value: New message(s)

        Examples:
            >>> history[0] = Message.user("modified")  # Local only
            >>> history[0:2] = [msg1, msg2]  # Local only
            >>> history.replace_all([msg1, msg2])  # Persists
        """
        if isinstance(key, slice):
            if isinstance(value, list):
                super().__setitem__(key, value)
            else:
                assert isinstance(value, Message)
                super().__setitem__(key, [value])
        else:
            if isinstance(value, list):
                super().__setitem__(key, value[0])
            else:
                assert isinstance(value, Message)
                super().__setitem__(key, value)

    def replace_all(
        self,
        new_messages: list[Message],
        *,
        update_baseline: bool = False,
        replace_extra: ReplaceVariantBase | None = None,
    ) -> None:
        """Replace all messages with smart detection.

        This method intelligently detects whether the operation is:
        1. A simple append (old messages + new messages) -> persist only new
        2. A true replacement (different messages) -> create replace record

        Args:
            new_messages: New message list
            update_baseline: If True, update baseline fingerprints to match new messages.
                           Use this when loading history from storage to set initial state.
                           Default is False to allow flush() to detect changes.
            replace_extra: RFC-0026 — when provided, treats this call as a
                           typed REPLACE event (compaction / ``/clear`` / etc.)
                           and synchronously schedules the persist write with
                           the typed variant. Implies ``update_baseline=True``
                           so subsequent ``flush()`` doesn't double-write an
                           untyped REPLACE inferred from fingerprint diff.
        """
        logger.debug(
            "🔍 [HISTORY-DEBUG] replace_all: incoming=%d roles=%s, update_baseline=%s, replace_extra=%s",
            len(new_messages),
            [m.role.value for m in new_messages],
            update_baseline,
            type(replace_extra).__name__ if replace_extra is not None else None,
        )
        self._resize_oversized_images(new_messages)
        self.clear()
        super().extend(new_messages)
        if self._persistence_enabled:
            self._pending_messages.clear()
            # RFC-0026: replace_extra implies update_baseline — the typed
            # write below sets the post-REPLACE ground truth, so flush()
            # must NOT compute a fingerprint diff against the old baseline.
            if update_baseline or replace_extra is not None:
                # Update baseline fingerprints to match the new message list
                # This ensures flush() only persists truly new messages added after this call
                current_non_system = [m for m in self if m.role != Role.SYSTEM]
                self._baseline_fingerprints = self._compute_fingerprints(current_non_system)
            if replace_extra is not None:
                self._schedule_typed_replace(new_messages, replace_extra)

    def _schedule_typed_replace(
        self,
        new_messages: list[Message],
        extra: ReplaceVariantBase,
    ) -> None:
        """RFC-0026: schedule a typed REPLACE persist write.

        Fire-and-forget on the owner event loop, mirroring :meth:`flush`.
        Failures surface via :meth:`_on_task_done` → logger.error. No-op
        when ``session_manager`` / ``history_key`` are missing (e.g.
        in-process tests).
        """
        if not self._session_manager or not self._history_key:
            return
        run_id = self._run_id or "unknown"
        root_run_id = self._root_run_id or "unknown"
        self._schedule_async(
            self._session_manager.agent_run_action.persist_replace(
                key=self._history_key,
                run_id=run_id,
                root_run_id=root_run_id,
                parent_run_id=self._parent_run_id,
                agent_name=self._agent_name,
                messages=new_messages,
                extra=extra,
            )
        )

    @staticmethod
    def _fingerprint_message(msg: Message) -> str:
        payload = msg.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _compute_fingerprints(cls, messages: list[Message]) -> list[str]:
        return [cls._fingerprint_message(m) for m in messages]

    def _schedule_async(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule an async coroutine to run (thread-safe).

        Uses the event loop captured at construction time to safely dispatch
        persistence I/O regardless of which thread calls flush().

        - Same thread as owner loop → asyncio.create_task (fastest path)
        - Different thread (e.g. executor worker) → run_coroutine_threadsafe
        - No owner loop → best-effort asyncio.run (sync-only entry points)

        所有 create_task / run_coroutine_threadsafe 返回的 task/future 都通过
        done-callback 记录错误，避免持久化失败被静默吞掉。

        Args:
            coro: Coroutine to schedule
        """
        owner = self._owner_loop

        # 1. 尝试获取当前线程的 running loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None and running is owner:
            # 同一事件循环线程，直接 create_task（最快路径）
            task = asyncio.create_task(coro)  # type: ignore[arg-type]
            task.add_done_callback(self._on_task_done)
            self._background_tasks.add(task)
        elif owner is not None and owner.is_running():
            # 跨线程：通过 run_coroutine_threadsafe 调度到主循环
            future = asyncio.run_coroutine_threadsafe(coro, owner)  # type: ignore[arg-type]
            future.add_done_callback(self._on_task_done)
        elif running is not None:
            # 在不同的事件循环线程上（不是 owner），用当前循环的 create_task
            task = asyncio.create_task(coro)  # type: ignore[arg-type]
            task.add_done_callback(self._on_task_done)
            self._background_tasks.add(task)
        else:
            # 无任何 running loop（纯 sync 入口，如 CLI 脚本）
            # asyncio.run() 创建临时 loop 执行一次性持久化
            try:
                asyncio.run(coro)  # type: ignore[arg-type]
            except RuntimeError:
                logger.error(
                    "HistoryList: failed to schedule persistence — no event loop available. "
                    "Messages added in this run may not be persisted to storage."
                )

    def _on_task_done(self, task: asyncio.Task[object] | asyncio.Future[object] | ConcurrentFuture[object]) -> None:
        """Done-callback for fire-and-forget persistence tasks.

        记录持久化失败，避免异常被静默吞掉（Python 仅在 Task 被 GC 时
        打印 'Task exception was never retrieved' 警告，很容易遗漏）。
        """
        self._background_tasks.discard(task)  # type: ignore[arg-type]
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.error("❌ HistoryList background persistence failed: %s", exc, exc_info=exc)

    async def _persist_flush_async(
        self,
        *,
        append_messages: list[Message] | None,
        replace_messages: list[Message] | None,
        iter_index: int | None = None,
    ) -> None:
        if not self._session_manager or not self._history_key:
            return
        run_id = self._run_id or "unknown"
        root_run_id = self._root_run_id or "unknown"

        try:
            if replace_messages is not None:
                if not replace_messages:
                    return
                await self._session_manager.agent_run_action.persist_replace(
                    key=self._history_key,
                    run_id=run_id,
                    root_run_id=root_run_id,
                    messages=replace_messages,
                    parent_run_id=self._parent_run_id,
                    agent_name=self._agent_name,
                )
            elif append_messages is not None:
                if not append_messages:
                    return
                # RFC-0022 Phase 2 plumbing: when caller passes iter_index, also
                # compose idempotency_key="{run_id}:{iter_index}" so retries /
                # Consumer Group redeliveries collapse on UNIQUE. iter_index=None
                # (Phase 1 batch / end-of-run trailing flush) keeps key=NULL.
                idempotency_key = f"{run_id}:{iter_index}" if iter_index is not None else None
                await self._session_manager.agent_run_action.persist_append(
                    key=self._history_key,
                    run_id=run_id,
                    root_run_id=root_run_id,
                    parent_run_id=self._parent_run_id,
                    agent_name=self._agent_name,
                    messages=append_messages,
                    iter_index=iter_index,
                    idempotency_key=idempotency_key,
                )
        except Exception as e:
            logger.error(f"❌ Failed to flush history: {e}")

    def _prepare_flush(self) -> tuple[list[Message] | None, list[Message] | None, list[Message]]:
        """Compute append/replace payloads and return (append, replace, non_system).

        Shared flush logic: compares baseline fingerprints to determine whether
        the change is append-only or a full replace. The third element of the
        returned tuple is the current non-system message list, used to update
        the baseline after persistence.
        """
        current_non_system = [m for m in self if m.role != Role.SYSTEM]

        logger.debug(
            "🔍 [HISTORY-DEBUG] flush: total=%d, non_system=%d, baseline_len=%d, roles=%s",
            len(self),
            len(current_non_system),
            len(self._baseline_fingerprints),
            [m.role.value for m in current_non_system],
        )
        for i, msg in enumerate(current_non_system):
            block_types = [type(b).__name__ for b in msg.content]
            logger.debug(
                "🔍 [HISTORY-DEBUG]   flush msg[%d] role=%s blocks=%s text=%.80s",
                i,
                msg.role.value,
                block_types,
                msg.get_text_content()[:80] if msg.get_text_content() else "<empty>",
            )

        is_append_only = True
        if len(current_non_system) < len(self._baseline_fingerprints):
            is_append_only = False
        else:
            for i in range(len(self._baseline_fingerprints)):
                if self._fingerprint_message(current_non_system[i]) != self._baseline_fingerprints[i]:
                    is_append_only = False
                    break

        append_messages: list[Message] | None = None
        replace_messages: list[Message] | None = None
        if is_append_only:
            if len(current_non_system) > len(self._baseline_fingerprints):
                append_messages = current_non_system[len(self._baseline_fingerprints) :]
        else:
            replace_messages = current_non_system

        return append_messages, replace_messages, current_non_system

    def flush(self, *, iter_index: int | None = None) -> None:
        """Flush pending messages to persistence (fire-and-forget).

        This should be called at the end of each run to persist all accumulated messages.
        The actual I/O is scheduled as a background task via ``create_task`` and is
        **not** awaited.  Use :meth:`flush_async` when you need to guarantee the
        write has completed before continuing (e.g. on error paths where the event
        loop may shut down shortly after).

        Args:
            iter_index: RFC-0022 Phase 2. When provided, the resulting APPEND row
                carries ``AppendExtra.iter_index`` and ``idempotency_key=
                f"{run_id}:{iter_index}"``. Pass the just-completed iter index
                from per-iter flush sites; leave None for end-of-run trailing
                flushes (batch semantics, key NULL).
        """
        if not self._persistence_enabled:
            return

        append_messages, replace_messages, current_non_system = self._prepare_flush()

        if append_messages is not None or replace_messages is not None:
            self._schedule_async(
                self._persist_flush_async(
                    append_messages=append_messages,
                    replace_messages=replace_messages,
                    iter_index=iter_index,
                )
            )

        self._pending_messages.clear()
        self._baseline_fingerprints = self._compute_fingerprints(current_non_system)

    async def flush_async(self, *, iter_index: int | None = None) -> None:
        """Flush pending messages to persistence (awaitable).

        Same semantics as :meth:`flush` but **awaits** the persistence I/O instead
        of scheduling it as a fire-and-forget task.  Use this on error / cleanup
        paths where the caller needs to ensure data is written before the coroutine
        or event loop exits.

        Args:
            iter_index: See :meth:`flush`.
        """
        if not self._persistence_enabled:
            return

        append_messages, replace_messages, current_non_system = self._prepare_flush()

        if append_messages is not None or replace_messages is not None:
            await self._persist_flush_async(
                append_messages=append_messages,
                replace_messages=replace_messages,
                iter_index=iter_index,
            )

        self._pending_messages.clear()
        self._baseline_fingerprints = self._compute_fingerprints(current_non_system)

    def update_context(
        self,
        *,
        run_id: str | None = None,
        root_run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        """Update the run context for persistence.

        This is called when starting a new run to update the run IDs.
        Automatically flushes pending messages from previous run before updating context.

        Args:
            run_id: New run ID
            root_run_id: New root run ID
            parent_run_id: New parent run ID
        """
        # Flush pending messages from previous run
        self.flush()

        if run_id is not None:
            self._run_id = run_id
        if root_run_id is not None:
            self._root_run_id = root_run_id
        if parent_run_id is not None:
            self._parent_run_id = parent_run_id

        if self._persistence_enabled:
            current_non_system = [m for m in self if m.role != Role.SYSTEM]
            self._baseline_fingerprints = self._compute_fingerprints(current_non_system)
