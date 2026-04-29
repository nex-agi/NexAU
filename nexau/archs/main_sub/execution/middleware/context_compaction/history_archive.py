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

"""History archive writer for context compaction.

RFC-0021: 上下文压缩时归档历史消息到 sandbox

每次压缩把"被移除的原始消息" + 一行 boundary 元数据 append 到单文件
``transcript.jsonl``。Agent 后续用 ``read_file`` / ``search_file_content`` 自助召回。

设计要点：

- **单文件 append-only**: ``{sandbox}/.nexau_history_archive/transcript.jsonl``
- **一行一记录**, 两种类型:
    - 序列化 ``Message`` (含 id/role/content/...)
    - boundary 元数据: ``{"_boundary": {round, compacted_at, ...}}``
- **Resume**: 扫 transcript 找最大 boundary round, 下一轮 = max+1
- **异常静默**: sandbox 不可用 / 写失败时跳过, 不抛异常 (归档失败绝不应中断压缩)

为什么单文件: agent 直接 ``search_file_content`` grep transcript.jsonl 就能召回
任意历史细节, 不需要先看索引再开 round 文件; append-only 自然崩溃安全。
"""

from __future__ import annotations

import base64 as _b64
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from nexau.archs.sandbox.base_sandbox import BaseSandbox, SandboxStatus
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock

logger = logging.getLogger(__name__)

ARCHIVE_SUBDIR = ".nexau_history_archive"
"""sandbox 根目录下的归档目录名 (RFC-0021)。

写死为常量而不暴露成 config: 避免用户传 ``"../foo"`` 之类的路径穿越输入,
且默认值就是设计上的唯一选择, 没有真实 use case 需要改名。
"""

TRANSCRIPT_FILENAME = "transcript.jsonl"
IMAGES_SUBDIR = "images"
ARCHIVED_IMAGE_URL_PREFIX = "file:"
BOUNDARY_KEY = "_boundary"
PREVIEW_MAX_CHARS = 300

# 常见 mime → 后缀; 未知用 .bin
_MIME_EXT: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}


def _ext_from_mime(mime: str) -> str:
    return _MIME_EXT.get(mime.lower(), "bin")


@dataclass(frozen=True)
class BoundaryRecord:
    """RFC-0021: 一轮压缩归档的 boundary 元数据, 在 transcript.jsonl 中以
    ``{"_boundary": {...}}`` 形式占一行。"""

    round: int
    compacted_at: str
    agent_id: str | None
    run_id: str | None
    trigger_reason: str
    strategy: str
    tokens_before: int | None
    tokens_after: int | None
    removed_message_count: int
    first_message_id: str
    last_message_id: str
    summary_message_id: str | None
    preview: str
    extracted_images: int = 0

    def to_line_dict(self) -> dict[str, Any]:
        return {
            BOUNDARY_KEY: {
                "round": self.round,
                "compacted_at": self.compacted_at,
                "agent_id": self.agent_id,
                "run_id": self.run_id,
                "trigger_reason": self.trigger_reason,
                "strategy": self.strategy,
                "tokens_before": self.tokens_before,
                "tokens_after": self.tokens_after,
                "removed_message_count": self.removed_message_count,
                "first_message_id": self.first_message_id,
                "last_message_id": self.last_message_id,
                "summary_message_id": self.summary_message_id,
                "preview": self.preview,
                "extracted_images": self.extracted_images,
            }
        }


class HistoryArchiveWriter:
    """RFC-0021: 把每轮被压缩移除的原始消息 + boundary append 到 sandbox 内的
    transcript.jsonl。

    通过 sandbox 抽象 IO, 兼容 LocalSandbox / E2BSandbox 等所有后端。
    """

    def __init__(
        self,
        *,
        sandbox: BaseSandbox,
        archive_dir: str,
        next_round: int = 1,
        total_archived: int = 0,
    ) -> None:
        self._sandbox: BaseSandbox = sandbox
        self._archive_dir = archive_dir
        self._transcript_path = str(Path(archive_dir) / TRANSCRIPT_FILENAME)
        self._next_round = next_round
        self._total_archived = total_archived

    @classmethod
    def from_sandbox(
        cls,
        *,
        agent_state: Any,
    ) -> HistoryArchiveWriter | None:
        """从 agent_state 构造 writer; sandbox 不可用时静默返回 None。

        归档子目录固定为 ``ARCHIVE_SUBDIR`` (``.nexau_history_archive``), 不暴露成
        config — 避免用户输入路径穿越; 默认值是设计上的唯一选择。

        try 边界故意收紧: 只裹 sandbox 交互 (get_sandbox / work_dir / create_directory),
        把"sandbox 不可用"和"内部 bug"分开。``_scan_transcript`` 自己有 try, 信它。
        """
        if agent_state is None:
            return None
        get_sb = getattr(agent_state, "get_sandbox", None)  # noqa: B009 — duck-typing
        if not callable(get_sb):
            logger.debug("[HistoryArchiveWriter] agent_state has no get_sandbox; skip.")
            return None

        try:
            sandbox = get_sb()
            if not isinstance(sandbox, BaseSandbox):
                logger.debug("[HistoryArchiveWriter] Sandbox unavailable; skip archiving.")
                return None
            work_dir = str(sandbox.work_dir) if sandbox.work_dir else ""
            if not work_dir:
                logger.debug("[HistoryArchiveWriter] Empty sandbox.work_dir; skip archiving.")
                return None
            archive_dir = str(Path(work_dir) / ARCHIVE_SUBDIR)
            sandbox.create_directory(archive_dir, parents=True)
        except Exception as exc:
            logger.warning("[HistoryArchiveWriter] sandbox setup failed: %s", exc)
            return None

        next_round, total_archived = cls._scan_transcript(sandbox, archive_dir)
        logger.info(
            "[HistoryArchiveWriter] Initialized at %s (next_round=%d, prior_archived=%d)",
            archive_dir,
            next_round,
            total_archived,
        )
        return cls(
            sandbox=sandbox,
            archive_dir=archive_dir,
            next_round=next_round,
            total_archived=total_archived,
        )

    @staticmethod
    def _scan_transcript(sandbox: BaseSandbox, archive_dir: str) -> tuple[int, int]:
        """扫现有 transcript.jsonl, 返回 (next_round, total_archived)。"""
        path = str(Path(archive_dir) / TRANSCRIPT_FILENAME)
        try:
            if not sandbox.file_exists(path):
                return 1, 0
            res = sandbox.read_file(path, encoding="utf-8", binary=False)
            if res.status != SandboxStatus.SUCCESS or not isinstance(res.content, str):
                return 1, 0
        except Exception as exc:
            logger.warning("[HistoryArchiveWriter] scan_transcript failed: %s", exc)
            return 1, 0

        max_round = 0
        total = 0
        for raw_line in res.content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and BOUNDARY_KEY in obj:
                obj_dict: dict[str, Any] = cast("dict[str, Any]", obj)
                b_raw = obj_dict[BOUNDARY_KEY]
                if isinstance(b_raw, dict):
                    bd: dict[str, Any] = cast("dict[str, Any]", b_raw)
                    r = bd.get("round")
                    if isinstance(r, int) and r > max_round:
                        max_round = r
                    removed = bd.get("removed_message_count")
                    if isinstance(removed, int):
                        total += removed
        return max_round + 1, total

    def write_round(
        self,
        *,
        removed: list[Message],
        tokens_before: int | None,
        tokens_after: int | None,
        trigger_reason: str,
        strategy_name: str,
        run_id: str | None,
        agent_id: str | None,
    ) -> BoundaryRecord | None:
        """写一轮归档: 把 removed 消息 + boundary 行 append 到 transcript.jsonl。

        base64 ImageBlock 会被外置到 ``images/{msg_id}-{idx}.{ext}`` 单独文件,
        transcript.jsonl 里只留 ``url=file:images/...`` 引用 (避免 transcript 膨胀)。
        URL ImageBlock 不动。

        失败时记日志并返回 None, 绝不抛异常 —— 归档失败不应中断压缩。
        """
        if not removed:
            return None
        try:
            round_num = self._next_round

            # 1. base64 图片外置: 不修改原 removed messages, 只 dump 给归档用的拷贝
            processed, extracted_images = self._externalize_images(removed)

            # 2. 构造 boundary record
            preview = self._build_preview(removed)
            summary_id = self._find_summary_id(removed)
            record = BoundaryRecord(
                round=round_num,
                compacted_at=datetime.now(UTC).isoformat(),
                agent_id=agent_id,
                run_id=run_id,
                trigger_reason=trigger_reason,
                strategy=strategy_name,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                removed_message_count=len(removed),
                first_message_id=str(removed[0].id),
                last_message_id=str(removed[-1].id),
                summary_message_id=summary_id,
                preview=preview,
                extracted_images=extracted_images,
            )

            # 3. 拼装本轮新增内容: removed messages + boundary 行
            new_lines: list[str] = [m.model_dump_json() for m in processed]
            new_lines.append(json.dumps(record.to_line_dict(), ensure_ascii=False))
            new_content = "\n".join(new_lines) + "\n"

            # 3. read+rewrite append (sandbox API 无 append 模式, 兼容所有后端)
            existing = ""
            try:
                if self._sandbox.file_exists(self._transcript_path):
                    res = self._sandbox.read_file(
                        self._transcript_path,
                        encoding="utf-8",
                        binary=False,
                    )
                    if res.status == SandboxStatus.SUCCESS and isinstance(res.content, str):
                        existing = res.content
                        if existing and not existing.endswith("\n"):
                            existing += "\n"
            except Exception as exc:
                logger.warning("[HistoryArchiveWriter] read transcript failed: %s", exc)

            self._sandbox.write_file(
                self._transcript_path,
                existing + new_content,
                create_directories=True,
            )

            self._next_round = round_num + 1
            self._total_archived += len(removed)
            logger.info(
                "[HistoryArchiveWriter] Round %d archived: %d messages -> %s",
                round_num,
                len(removed),
                TRANSCRIPT_FILENAME,
            )
            return record
        except Exception as exc:
            logger.warning("[HistoryArchiveWriter] write_round failed: %s", exc)
            return None

    @property
    def archive_dir(self) -> str:
        return self._archive_dir

    @property
    def archive_subdir_name(self) -> str:
        return ARCHIVE_SUBDIR

    @property
    def transcript_path(self) -> str:
        return self._transcript_path

    @property
    def total_rounds(self) -> int:
        return self._next_round - 1

    @property
    def total_archived(self) -> int:
        return self._total_archived

    def _externalize_images(self, removed: list[Message]) -> tuple[list[Message], int]:
        """把 base64 ImageBlock 外置到 ``images/{msg_id}-{path}.{ext}`` 单文件,
        返回 (新的消息列表, 外置图片数量)。

        覆盖两层:
        - **顶层** ImageBlock (USER / ASSISTANT 等消息直接挂的图)
        - **嵌套** ImageBlock 在 ``ToolResultBlock.content`` 里 (multimodal 工具返回图)

        - 仅当 ``ImageBlock.base64`` 非空时才外置; URL-only 不动
        - 外置失败 (写盘 / decode error) 时该 block 保持原样 (回退到内联存储)
        - 不修改原 messages (用 ``model_copy(update=...)`` 浅拷); 原始消息继续在
          active context 用, 字段值按引用共享
        """

        def _has_b64_image(block: Any) -> bool:
            if isinstance(block, ImageBlock) and block.base64:
                return True
            if isinstance(block, ToolResultBlock) and isinstance(block.content, list):
                return any(isinstance(c, ImageBlock) and c.base64 for c in block.content)
            return False

        # 先扫一遍是否有需要外置的图片, 避免无谓的复制
        needs_processing = any(_has_b64_image(b) for msg in removed for b in msg.content)
        if not needs_processing:
            return removed, 0

        images_dir_abs = str(Path(self._archive_dir) / IMAGES_SUBDIR)
        try:
            self._sandbox.create_directory(images_dir_abs, parents=True)
        except Exception as exc:
            logger.warning("[HistoryArchiveWriter] cannot create images dir: %s", exc)
            return removed, 0

        # 计数器在 nested function 引用前先初始化, mypy 才能解析 nonlocal
        extracted = 0

        def _externalize_one(msg_id: str, path_label: str, img: ImageBlock) -> ImageBlock:
            """写盘并返回替换 block; 失败则原样返回。"""
            nonlocal extracted
            ext = _ext_from_mime(img.mime_type)
            rel = f"{IMAGES_SUBDIR}/{msg_id}-{path_label}.{ext}"
            abs_path = str(Path(self._archive_dir) / rel)
            try:
                img_bytes = _b64.b64decode(img.base64 or "")
                self._sandbox.write_file(
                    abs_path,
                    img_bytes,
                    binary=True,
                    create_directories=True,
                )
                extracted += 1
                return img.model_copy(
                    update={
                        "base64": None,
                        "url": f"{ARCHIVED_IMAGE_URL_PREFIX}{rel}",
                    }
                )
            except Exception as exc:
                logger.warning(
                    "[HistoryArchiveWriter] failed to externalize image (msg=%s, path=%s): %s",
                    msg_id,
                    path_label,
                    exc,
                )
                return img  # fallback: 原 block 内联存储

        result: list[Message] = []
        for msg in removed:
            if not any(_has_b64_image(b) for b in msg.content):
                result.append(msg)
                continue

            # 只深拷被修改的 block; 其他 block 共享引用 — 避免无谓的整 message 深拷贝。
            new_content: list[Any] = []
            for idx, block in enumerate(msg.content):
                if isinstance(block, ImageBlock) and block.base64:
                    new_content.append(_externalize_one(str(msg.id), str(idx), block))
                elif (
                    isinstance(block, ToolResultBlock)
                    and isinstance(block.content, list)
                    and any(isinstance(c, ImageBlock) and c.base64 for c in block.content)
                ):
                    # 递归处理 tool result 内嵌的 ImageBlock
                    new_inner: list[Any] = []
                    for jdx, inner in enumerate(block.content):
                        if isinstance(inner, ImageBlock) and inner.base64:
                            new_inner.append(_externalize_one(str(msg.id), f"{idx}-{jdx}", inner))
                        else:
                            new_inner.append(inner)
                    new_content.append(block.model_copy(update={"content": new_inner}))
                else:
                    new_content.append(block)

            # message 自身浅拷 (Pydantic update= 创建新 instance, 但其他字段按引用复制
            # — 比 deep=True 快得多, 大 message 尤其明显)
            result.append(msg.model_copy(update={"content": new_content}))
        return result, extracted

    @staticmethod
    def _build_preview(removed: list[Message]) -> str:
        """选第一条非 system/framework 角色消息做 preview。"""
        for msg in removed:
            if msg.role in (Role.SYSTEM, Role.FRAMEWORK):
                continue
            text = _extract_text_for_preview(msg)
            if text:
                return text[:PREVIEW_MAX_CHARS]
        return _extract_text_for_preview(removed[0])[:PREVIEW_MAX_CHARS]

    @staticmethod
    def _find_summary_id(removed: list[Message]) -> str | None:
        """如果本轮归档包含上一轮产生的 summary 消息, 记录其 id。"""
        for msg in removed:
            md = msg.metadata or {}
            if md.get("isSummary") is True or md.get("is_compacted") is True:
                return str(msg.id)
        return None


def _extract_text_for_preview(msg: Message) -> str:
    """从 Message 抽取文本用于 preview。"""
    parts: list[str] = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                parts.append(block.content)
    return " ".join(p.strip() for p in parts if p).strip()


def build_archive_hint(
    *,
    total_archived: int,
    total_rounds: int,
    latest_round: int,
) -> str:
    """RFC-0021: 构造归档路径提示文本 (不带前后空白)。

    告诉 agent 单文件 transcript.jsonl 在哪里, 以及如何用 search_file_content / read_file 召回。
    路径走常量 ``ARCHIVE_SUBDIR``。

    **返回的 hint 不带任何前后空白** —— 由调用方决定怎么分隔。新 TextBlock 是天然
    边界, 不需要 ``\\n\\n`` 前缀; 如果调用方要拼到现有文本末尾, 自己加。
    """
    return (
        f"📁 [Archive] {total_archived} earlier message(s) archived across "
        f"{total_rounds} compaction round(s) (latest: round {latest_round}).\n"
        f"To recall earlier conversation, use your file tools on "
        f"`{ARCHIVE_SUBDIR}/{TRANSCRIPT_FILENAME}`:\n"
        f"  • `search_file_content` to grep for keywords (each matched line is a serialized Message)\n"
        f"  • `read_file` for full chronological view\n"
        f'  • Boundary lines `{{"{BOUNDARY_KEY}": ...}}` mark each compaction round'
    )
