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

"""Sliding window compaction strategy."""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anthropic
import openai
from openai.types.chat import ChatCompletion

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
from nexau.core.messages import Message, Role, TextBlock, ToolUseBlock

from .....utils.token_counter import TokenCounter
from ..llm_config_utils import normalize_summary_llm_overrides, resolve_summary_llm_config

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_context import GlobalStorage

logger = logging.getLogger(__name__)

# Backward-compatibility: older tests patch `sliding_window.OpenAI`.
OpenAI = openai.OpenAI
Anthropic = anthropic.Anthropic

_HANDOFF_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of the conversation so far. "
    "You also have access to the state of the tools that were used by that language model. "
    "Use this to build on the work that has already been done and avoid duplicating work. "
    "Here is the summary produced by the other language model; use the information in this summary to assist with your own analysis:"
)


def with_handoff_prefix(summary: str) -> str:
    """Prepend the fixed handoff prefix to compacted summary text."""
    stripped_summary = summary.strip()
    return f"{_HANDOFF_SUMMARY_PREFIX}\n\n{stripped_summary}" if stripped_summary else _HANDOFF_SUMMARY_PREFIX


def _load_compact_prompt(prompt_path: str) -> str:
    """Load the compact prompt template from file.

    Args:
        prompt_path: Path to compact prompt file (already resolved by config).

    Returns:
        The compact prompt content as a string.

    Raises:
        FileNotFoundError: If the template file is not found.
    """
    template_file = Path(prompt_path)

    try:
        with open(template_file, encoding="utf-8") as f:
            content = f.read()
            return content
    except FileNotFoundError:
        logger.error(f"Compact prompt template not found at {template_file}")
        raise
    except Exception as e:
        logger.error(f"Failed to load compact prompt template: {e}")
        raise


class SlidingWindowCompaction:
    """Sliding window compaction strategy - keeps recent conversation iterations.

    An "iteration" is bounded by ASSISTANT messages:
    [USER or FRAMEWORK](optional) -> [ASSISTANT] -> [TOOL results](optional)

    - Each ASSISTANT message starts a new iteration
    - USER or FRAMEWORK messages before the ASSISTANT are part of that iteration (optional)
    - TOOL results after the ASSISTANT are part of that iteration (optional)

    This strategy:
    1. Groups messages into conversation iterations
    2. Keeps the most recent N iterations in full
    3. Compresses older iterations using LLM summarization
    4. Handles large inputs safely via chunked summarization
    """

    # Reserved tokens for compact_prompt + LLM output overhead
    _SUMMARY_RESERVED_TOKENS = 4096

    # Hard truncation fallback 的最大输出 token 数，避免 fallback "摘要"
    # 与 context window 等大，导致压缩→失败→注入大文本→再压缩的死循环。
    _HARD_TRUNCATION_MAX_TOKENS = 10240

    # 连续 fallback 次数上限；超过后跳过压缩，避免无限循环。
    _MAX_CONSECUTIVE_FALLBACKS = 3

    def __init__(
        self,
        keep_system: bool = True,
        keep_iterations: int = 3,
        keep_user_rounds: int = 0,
        summary_llm_config: dict[str, Any] | None = None,
        summary_model: str | None = None,
        summary_base_url: str | None = None,
        summary_api_key: str | None = None,
        summary_api_type: str | None = None,
        max_context_tokens: int | None = None,
        compact_prompt_path: str | None = None,
        token_counter: TokenCounter | None = None,
        retry_attempts: int = 3,
    ):
        """Initialize sliding window compaction.

        Args:
            keep_system: Whether to preserve the system message.
            keep_iterations: Number of recent iterations to keep. Default: 3.
            keep_user_rounds: Number of recent user rounds to keep. Default: 0 (disabled).
                When > 0, uses user rounds mode instead of iterations mode.
            summary_llm_config: Optional nested LLMConfig-style overrides for summarization.
            summary_model: Optional legacy model override for summarization.
            summary_base_url: Optional legacy base URL override for summarization.
            summary_api_key: Optional legacy API key override for summarization.
            summary_api_type: Optional legacy API type override for summarization.
            token_counter: Token counter instance for counting tokens. If None, a default TokenCounter is used.
            max_context_tokens: Context window size of the summary LLM. None = inherit from agent config.
            retry_attempts: Number of retry attempts for summary LLM calls. Default: 3.
            compact_prompt_path: Path to compact prompt file (already resolved by config). Required.

        Raises:
            ValueError: If both keep_iterations != 3 and keep_user_rounds > 0 are set.
            ValueError: If keep_iterations < 1 or keep_user_rounds < 0.
            ValueError: If runtime LLM configuration is missing when needed.
        """
        if keep_iterations != 3 and keep_user_rounds > 0:
            raise ValueError("Cannot set both keep_iterations and keep_user_rounds")

        if keep_iterations < 1:
            raise ValueError(f"keep_iterations must be >= 1, got {keep_iterations}")
        if keep_user_rounds < 0:
            raise ValueError(f"keep_user_rounds must be >= 0, got {keep_user_rounds}")

        # Validate compact prompt path
        if not compact_prompt_path:
            raise ValueError("compact_prompt_path is required for SlidingWindowCompaction.")

        self.keep_system = keep_system
        self.keep_iterations = keep_iterations
        self.keep_user_rounds = keep_user_rounds
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter or TokenCounter()
        self.retry_attempts = retry_attempts

        # Initialize summary runtime configuration.
        self.summary_llm_config_overrides = normalize_summary_llm_overrides(
            summary_llm_config,
            summary_model=summary_model,
            summary_base_url=summary_base_url,
            summary_api_key=summary_api_key,
            summary_api_type=summary_api_type,
        )
        self.summary_model = self.summary_llm_config_overrides.get("model")
        self.summary_base_url = self.summary_llm_config_overrides.get("base_url")
        self.summary_api_key = self.summary_llm_config_overrides.get("api_key")
        self.summary_api_type = self.summary_llm_config_overrides.get("api_type")
        self.summary_llm_config: LLMConfig | None = None
        self._base_llm_config: LLMConfig | None = None
        self._base_openai_client: Any | None = None
        self._llm_caller: LLMCaller | None = None
        self._summary_client: Any | None = None
        self._session_id: str | None = None
        self._global_storage: GlobalStorage | None = None

        # 连续 hard truncation fallback 计数器
        self._consecutive_fallback_count: int = 0

        # Load compact prompt using the resolved path.
        self.compact_prompt = _load_compact_prompt(compact_prompt_path)

        if all(self.summary_llm_config_overrides.get(key) for key in ("model", "base_url", "api_key")):
            self._refresh_llm_runtime()

        logger.info(
            "[SlidingWindowCompaction] Initialized: summary_overrides=%s, keep_iterations=%s, keep_user_rounds=%s, max_context_tokens=%s",
            sorted(self.summary_llm_config_overrides.keys()),
            self.keep_iterations,
            self.keep_user_rounds,
            self.max_context_tokens,
        )

    @property
    def _summary_input_limit(self) -> int:
        """Max input tokens allowed when calling the summary LLM."""
        if self.max_context_tokens is None:
            raise RuntimeError("max_context_tokens not resolved; configure_llm_runtime must be called first")
        return self.max_context_tokens - self._SUMMARY_RESERVED_TOKENS

    def configure_llm_runtime(
        self,
        llm_config: LLMConfig,
        openai_client: Any | None = None,
        *,
        session_id: str | None = None,
        global_storage: Any | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        """Inject the agent/runtime LLM config used as the default summary model."""
        self._base_llm_config = llm_config.copy()
        self._base_openai_client = openai_client
        self._session_id = session_id
        self._global_storage = global_storage
        # Inherit from agent config when not explicitly set
        if self.max_context_tokens is None and max_context_tokens is not None:
            self.max_context_tokens = max_context_tokens
        self._refresh_llm_runtime()

    def _resolve_summary_llm_config(self) -> LLMConfig:
        try:
            return resolve_summary_llm_config(
                base_llm_config=self._base_llm_config,
                summary_overrides=self.summary_llm_config_overrides,
            )
        except ValueError as exc:
            if self.summary_llm_config_overrides:
                raise
            raise ValueError(
                "LLM configuration is required for SlidingWindowCompaction. Provide agent llm_config or set summary_llm_config."
            ) from exc

    def _build_client(self, llm_config: LLMConfig) -> Any | None:
        if llm_config.api_type == "gemini_rest":
            return None
        client_kwargs = llm_config.to_client_kwargs()
        if llm_config.api_type == "anthropic_chat_completion":
            return Anthropic(**client_kwargs)
        if llm_config.api_type in ["openai_responses", "openai_chat_completion"]:
            return OpenAI(**client_kwargs)
        raise ValueError(f"Invalid API type: {llm_config.api_type}")

    def _refresh_llm_runtime(self) -> None:
        summary_llm_config = self._resolve_summary_llm_config()
        reuse_base_client = (
            self._base_llm_config is not None
            and self._base_openai_client is not None
            and self._base_llm_config.api_type == summary_llm_config.api_type
            and self._base_llm_config.to_client_kwargs() == summary_llm_config.to_client_kwargs()
        )
        summary_client = self._base_openai_client if reuse_base_client else self._build_client(summary_llm_config)

        self.summary_llm_config = summary_llm_config
        self._summary_client = summary_client
        self._llm_caller = LLMCaller(
            summary_client,
            summary_llm_config,
            retry_attempts=self.retry_attempts,
            session_id=self._session_id,
            global_storage=self._global_storage,
        )

    def _ensure_llm_caller(self) -> LLMCaller:
        if self._llm_caller is None:
            self._refresh_llm_runtime()
        if self._llm_caller is None:
            raise RuntimeError("SlidingWindowCompaction LLM caller initialization failed")
        return self._llm_caller

    def compact(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Compact messages by keeping recent iterations or user rounds.

        Uses incremental summarization: each compaction only summarizes
        the messages that have slid out of the window since last time.
        Previous summaries are naturally included because they were injected
        into a user message that is now part of the messages to compress.

        For large inputs that exceed the summary LLM's context window,
        falls back to chunked summarization (split → summarize each → merge).

        If all summarization attempts fail, uses hard truncation as a last
        resort to guarantee the output fits within context limits.
        """
        logger.info(f"[SlidingWindowCompaction] Starting compaction on {len(messages)} messages")

        result: list[Message] = []
        start_idx = 0

        # Keep system message if present
        if self.keep_system and messages and messages[0].role == Role.SYSTEM:
            result.append(messages[0])
            start_idx = 1

        # Group messages based on keep_user_rounds or keep_iterations
        if self.keep_user_rounds > 0:
            groups = self._group_into_user_rounds(messages[start_idx:])
            keep_count = self.keep_user_rounds
            group_name = "user_rounds"
        else:
            groups = self._group_into_iterations(messages[start_idx:])
            keep_count = self.keep_iterations
            group_name = "iterations"
        if len(groups) <= keep_count:
            logger.info(f"[SlidingWindowCompaction] Skipping: {len(groups)} {group_name} <= {keep_count}")
            return messages.copy()

        # Calculate how many groups to compress
        groups_to_compress = groups[:-keep_count]
        groups_to_keep = groups[-keep_count:]

        # Collect all messages to compress (include system for context)
        all_compressed_messages: list[Message] = []
        if self.keep_system and messages and messages[0].role == Role.SYSTEM:
            all_compressed_messages.append(messages[0])
        for group_msgs in groups_to_compress:
            all_compressed_messages.extend(group_msgs)

        # Generate summary safely (handles oversized input)
        summary = self._generate_summary_safe(all_compressed_messages)

        # Inject summary into the first USER message of kept groups
        self._inject_summary(result, groups_to_keep, summary)
        input_tokens = self.token_counter.count_tokens(messages)
        summary_tokens = self.token_counter.count_tokens(result)
        logger.info(
            f"[SlidingWindowCompaction] Compaction complete: "
            f"{input_tokens} tokens -> {summary_tokens} tokens "
            f"({len(groups_to_compress)} {group_name} compressed, {len(groups_to_keep)} {group_name} kept)"
        )
        return result

    async def compact_async(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Async version of compact().

        P2 async/sync 技术债修复: 异步 compact 链

        使用 LLMCaller.call_llm_async() 替代 sync call_llm，
        在主事件循环上执行 LLM 摘要调用，避免阻塞 event loop。
        分组和窗口逻辑与 sync 版本相同。
        """
        logger.info(f"[SlidingWindowCompaction] Starting async compaction on {len(messages)} messages")

        result: list[Message] = []
        start_idx = 0

        if self.keep_system and messages and messages[0].role == Role.SYSTEM:
            result.append(messages[0])
            start_idx = 1

        if self.keep_user_rounds > 0:
            groups = self._group_into_user_rounds(messages[start_idx:])
            keep_count = self.keep_user_rounds
            group_name = "user_rounds"
        else:
            groups = self._group_into_iterations(messages[start_idx:])
            keep_count = self.keep_iterations
            group_name = "iterations"

        if len(groups) <= keep_count:
            logger.info(f"[SlidingWindowCompaction] Skipping async: {len(groups)} {group_name} <= {keep_count}")
            return messages.copy()

        groups_to_compress = groups[:-keep_count]
        groups_to_keep = groups[-keep_count:]

        all_compressed_messages: list[Message] = []
        if self.keep_system and messages and messages[0].role == Role.SYSTEM:
            all_compressed_messages.append(messages[0])
        for group_msgs in groups_to_compress:
            all_compressed_messages.extend(group_msgs)

        summary = await self._generate_summary_safe_async(all_compressed_messages)

        self._inject_summary(result, groups_to_keep, summary)
        input_tokens = self.token_counter.count_tokens(messages)
        summary_tokens = self.token_counter.count_tokens(result)
        logger.info(
            f"[SlidingWindowCompaction] Async compaction complete: "
            f"{input_tokens} tokens -> {summary_tokens} tokens "
            f"({len(groups_to_compress)} {group_name} compressed, {len(groups_to_keep)} {group_name} kept)"
        )
        return result

    async def _generate_summary_safe_async(self, messages: list[Message]) -> str:
        """Async version of _generate_summary_safe."""
        if self._consecutive_fallback_count >= self._MAX_CONSECUTIVE_FALLBACKS:
            logger.error(
                "[SlidingWindowCompaction] Summary LLM has failed %d consecutive times (async), "
                "skipping compaction to prevent fallback accumulation loop",
                self._consecutive_fallback_count,
            )
            raise RuntimeError(
                f"Summary LLM persistently unavailable ({self._consecutive_fallback_count} consecutive failures), skipping compaction"
            )

        input_tokens = self.token_counter.count_tokens(messages)
        input_limit = self._summary_input_limit

        logger.info(f"[SlidingWindowCompaction] Async summary input: {input_tokens} tokens, limit: {input_limit} tokens")
        if input_tokens <= input_limit:
            try:
                summary = await self._generate_summary_async(messages)
                self._consecutive_fallback_count = 0
                return summary
            except Exception as e:
                logger.error(f"[SlidingWindowCompaction] Async direct summary failed: {e}")
                return self._record_fallback(messages)

        logger.info(f"[SlidingWindowCompaction] Input exceeds limit ({input_tokens} > {input_limit}), using async chunked summarization")
        try:
            summary = await self._chunked_summary_async(messages, input_limit)
            self._consecutive_fallback_count = 0
            return summary
        except Exception as e:
            logger.error(f"[SlidingWindowCompaction] Async chunked summary failed: {e}")
            return self._record_fallback(messages)

    async def _chunked_summary_async(self, messages: list[Message], chunk_token_limit: int) -> str:
        """Async version of _chunked_summary."""
        chunks = self._split_into_chunks(messages, chunk_token_limit)
        logger.info(f"[SlidingWindowCompaction] Async split into {len(chunks)} chunks")

        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            try:
                summary = await self._generate_summary_async(chunk)
                chunk_summaries.append(summary)
                logger.info(f"[SlidingWindowCompaction] Async chunk {i + 1}/{len(chunks)} summarized")
            except Exception as e:
                logger.warning(f"[SlidingWindowCompaction] Async chunk {i + 1}/{len(chunks)} failed: {e}, skipping")

        if not chunk_summaries:
            raise RuntimeError("All async chunk summaries failed")

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        merged_text = "\n\n".join(f"[Part {i + 1}]: {s}" for i, s in enumerate(chunk_summaries))
        merged_msg = Message(role=Role.USER, content=[TextBlock(text=merged_text)])
        merged_tokens = self.token_counter.count_tokens([merged_msg])

        if merged_tokens <= chunk_token_limit:
            try:
                return await self._generate_summary_async([merged_msg])
            except Exception:
                logger.warning("[SlidingWindowCompaction] Async final merge summary failed, using concatenated summaries")
                return merged_text
        else:
            logger.info("[SlidingWindowCompaction] Async merged summaries still too large, recursing")
            return await self._chunked_summary_async([merged_msg], chunk_token_limit)

    async def _generate_summary_async(self, messages: list[Message]) -> str:
        """Async version of _generate_summary using LLMCaller.call_llm_async.

        P2 async/sync 技术债修复: 异步 LLM 摘要生成

        使用 LLMCaller.call_llm_async() 调用 LLM 生成摘要，
        在主事件循环上执行，避免阻塞 event loop。
        """
        llm_caller = self._ensure_llm_caller()
        summary_model_name = self.summary_llm_config.model if self.summary_llm_config is not None else self.summary_model
        logger.info(f"[SlidingWindowCompaction] Async calling LLM to generate summary (model: {summary_model_name})")

        llm_messages = messages.copy()
        llm_messages.append(Message(role=Role.USER, content=[TextBlock(text=self.compact_prompt)]))

        summary_api_type = (
            self.summary_llm_config.api_type if self.summary_llm_config is not None else (self.summary_api_type or "openai_chat_completion")
        )
        tool_call_mode = "structured"

        try:
            model_response = await llm_caller.call_llm_async(
                llm_messages,
                tool_call_mode=tool_call_mode,
            )
            summary = (model_response.content or "").strip() if model_response else ""
            logger.info("[SlidingWindowCompaction] Async LLM summary generated successfully")
            return summary
        except Exception as exc:
            if summary_api_type != "openai_chat_completion":
                raise

            logger.warning(
                "[SlidingWindowCompaction] LLMCaller async summary generation failed; falling back to sync direct client call: %s",
                exc,
            )
            # 回退到 sync direct client call（在线程中执行）
            summary = await asyncio.to_thread(self._generate_summary_direct_fallback, llm_messages)
            logger.info("[SlidingWindowCompaction] Async LLM summary generated successfully via sync fallback")
            return summary

    def _generate_summary_safe(self, messages: list[Message]) -> str:
        """Generate summary with automatic chunking for oversized inputs.

        Strategy:
        1. If input fits within summary LLM's context → direct summarization
        2. If input is too large → split into chunks by iteration boundaries,
           summarize each chunk, then merge summaries
        3. If all LLM calls fail → return a hard truncation placeholder
        4. If fallback has been used consecutively too many times, raise to
           signal the caller to skip compaction entirely.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary text (never raises on recoverable failures).

        Raises:
            RuntimeError: If consecutive fallback count exceeds the limit,
                indicating the summary LLM is persistently unavailable.
        """
        # 连续 fallback 过多时中断，避免压缩→失败→注入大文本→再压缩的死循环
        if self._consecutive_fallback_count >= self._MAX_CONSECUTIVE_FALLBACKS:
            logger.error(
                "[SlidingWindowCompaction] Summary LLM has failed %d consecutive times, "
                "skipping compaction to prevent fallback accumulation loop",
                self._consecutive_fallback_count,
            )
            raise RuntimeError(
                f"Summary LLM persistently unavailable ({self._consecutive_fallback_count} consecutive failures), skipping compaction"
            )

        input_tokens = self.token_counter.count_tokens(messages)
        input_limit = self._summary_input_limit

        logger.info(f"[SlidingWindowCompaction] Summary input: {input_tokens} tokens, limit: {input_limit} tokens")
        if input_tokens <= input_limit:
            # Normal path: input fits, summarize directly
            try:
                summary = self._generate_summary(messages)
                self._consecutive_fallback_count = 0
                return summary
            except Exception as e:
                logger.error(f"[SlidingWindowCompaction] Direct summary failed: {e}")
                return self._record_fallback(messages)

        # Input too large: chunked summarization
        logger.info(f"[SlidingWindowCompaction] Input exceeds limit ({input_tokens} > {input_limit}), using chunked summarization")
        try:
            summary = self._chunked_summary(messages, input_limit)
            self._consecutive_fallback_count = 0
            return summary
        except Exception as e:
            logger.error(f"[SlidingWindowCompaction] Chunked summary failed: {e}")
            return self._record_fallback(messages)

    def _record_fallback(self, messages: list[Message]) -> str:
        """Increment the consecutive fallback counter and return a truncated summary."""
        self._consecutive_fallback_count += 1
        logger.warning(
            "[SlidingWindowCompaction] Using hard truncation fallback (consecutive fallback count: %d/%d)",
            self._consecutive_fallback_count,
            self._MAX_CONSECUTIVE_FALLBACKS,
        )
        return self._hard_truncation_fallback(messages)

    def _chunked_summary(self, messages: list[Message], chunk_token_limit: int) -> str:
        """Split messages into chunks, summarize each, then merge.

        Args:
            messages: Messages to summarize.
            chunk_token_limit: Max tokens per chunk.

        Returns:
            Merged summary text.

        Raises:
            RuntimeError: If all chunk summaries fail.
        """
        chunks = self._split_into_chunks(messages, chunk_token_limit)
        logger.info(f"[SlidingWindowCompaction] Split into {len(chunks)} chunks")

        # Summarize each chunk independently
        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            try:
                summary = self._generate_summary(chunk)
                chunk_summaries.append(summary)
                logger.info(f"[SlidingWindowCompaction] Chunk {i + 1}/{len(chunks)} summarized")
            except Exception as e:
                logger.warning(f"[SlidingWindowCompaction] Chunk {i + 1}/{len(chunks)} failed: {e}, skipping")

        if not chunk_summaries:
            raise RuntimeError("All chunk summaries failed")

        # Merge chunk summaries
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        merged_text = "\n\n".join(f"[Part {i + 1}]: {s}" for i, s in enumerate(chunk_summaries))

        # Check if merged summaries need a final consolidation pass
        merged_msg = Message(role=Role.USER, content=[TextBlock(text=merged_text)])
        merged_tokens = self.token_counter.count_tokens([merged_msg])

        if merged_tokens <= chunk_token_limit:
            # Merge is small enough, do one final consolidation
            try:
                return self._generate_summary([merged_msg])
            except Exception:
                logger.warning("[SlidingWindowCompaction] Final merge summary failed, using concatenated summaries")
                return merged_text
        else:
            # Merged summaries still too large, recurse
            logger.info("[SlidingWindowCompaction] Merged summaries still too large, recursing")
            return self._chunked_summary([merged_msg], chunk_token_limit)

    def _split_into_chunks(self, messages: list[Message], chunk_token_limit: int) -> list[list[Message]]:
        """Split messages into chunks respecting iteration boundaries.

        Keeps complete iterations together within each chunk.
        If a single iteration exceeds the limit, it goes alone in its own chunk.

        Args:
            messages: Messages to split.
            chunk_token_limit: Max tokens per chunk.

        Returns:
            List of message chunks.
        """
        # Re-group into iterations to avoid splitting mid-iteration
        iterations = self._group_into_iterations(messages)

        chunks: list[list[Message]] = []
        current_chunk: list[Message] = []
        current_tokens = 0

        for iteration_msgs in iterations:
            iter_tokens = self.token_counter.count_tokens(iteration_msgs)

            if current_tokens + iter_tokens > chunk_token_limit and current_chunk:
                # Current chunk is full, start a new one
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.extend(iteration_msgs)
            current_tokens += iter_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _hard_truncation_fallback(self, messages: list[Message]) -> str:
        """Fallback by keeping the newest messages within a compact token budget.

        Uses _HARD_TRUNCATION_MAX_TOKENS (not _summary_input_limit) to ensure
        the fallback output is genuinely compact and won't bloat the conversation
        when summary LLM calls repeatedly fail.
        """
        max_tokens = self._HARD_TRUNCATION_MAX_TOKENS
        tokens_used = 0
        retained_messages: list[Message] = []

        # Keep system message if present
        if self.keep_system and messages and messages[0].role == Role.SYSTEM:
            retained_messages.append(messages[0])
            tokens_used += self.token_counter.count_tokens([messages[0]])

        # Iterate messages from newest to oldest
        for msg in reversed(messages):
            msg_tokens = self.token_counter.count_tokens([msg])
            if tokens_used + msg_tokens > max_tokens:
                break  # stop adding older messages
            retained_messages.insert(1 if self.keep_system else 0, msg)
            tokens_used += msg_tokens

        # Convert retained messages to concatenated text
        context_snippets = [msg.get_text_content().strip() for msg in retained_messages if msg.get_text_content()]
        fallback_text = "\n".join(context_snippets)
        logger.info(
            "[SlidingWindowCompaction] Hard truncation fallback: %d/%d messages retained (%d tokens, limit %d)",
            len(retained_messages),
            len(messages),
            tokens_used,
            max_tokens,
        )
        return fallback_text

    def _inject_summary(
        self,
        result: list[Message],
        groups_to_keep: list[list[Message]],
        summary: str,
    ) -> None:
        """Inject summary into the message list, always near the beginning.

        When the first message of the kept groups is a USER message, the
        summary is merged into it (preserving the original user content).
        Otherwise a standalone summary USER message is inserted before the
        kept groups so that the LLM always sees the context summary early in
        the conversation — not buried at the end after a long tool-call chain.

        Args:
            result: Result message list to append to (modified in place).
            groups_to_keep: Groups of messages to keep.
            summary: Summary text to inject.
        """
        summary_prefix = with_handoff_prefix(summary)

        # 1. 判断保留组的第一条消息是否为真正的 USER 消息
        first_kept_msg: Message | None = None
        if groups_to_keep and groups_to_keep[0]:
            first_kept_msg = groups_to_keep[0][0]

        if first_kept_msg is not None and first_kept_msg.role == Role.USER:
            # 2a. 第一条就是 USER → 合并摘要到该消息（位置已正确）
            merged = False
            for group_msgs in groups_to_keep:
                for msg in group_msgs:
                    if msg is first_kept_msg and not merged:
                        original_content = msg.get_text_content()
                        modified_content = f"{summary_prefix} The user request for this round is: {original_content}"
                        modified_msg = msg.model_copy(update={"content": [TextBlock(text=modified_content)]})
                        modified_msg.metadata["isSummary"] = True
                        if self._session_id is not None:
                            modified_msg.metadata["session_id"] = self._session_id
                        result.append(modified_msg)
                        merged = True
                    else:
                        result.append(msg)
        else:
            # 2b. 保留组以 ASSISTANT/TOOL 开头（典型的工具调用链场景）
            #     → 在所有保留消息之前插入独立的摘要 USER 消息
            summary_msg = Message(role=Role.USER, content=[TextBlock(text=summary_prefix)])
            summary_msg.metadata["isSummary"] = True
            if self._session_id is not None:
                summary_msg.metadata["session_id"] = self._session_id
            result.append(summary_msg)
            for group_msgs in groups_to_keep:
                for msg in group_msgs:
                    result.append(msg)

    def _group_into_iterations(self, messages: list[Message]) -> list[list[Message]]:
        """Group messages into conversation iterations.

        An iteration is bounded by ASSISTANT messages:
        [USER or FRAMEWORK](optional) -> [ASSISTANT] -> [TOOL results](optional)

        - Each ASSISTANT message starts a new iteration
        - USER or FRAMEWORK messages before the ASSISTANT are part of that iteration (optional)
        - TOOL results after the ASSISTANT are part of that iteration (optional)
        """
        iterations: list[list[Message]] = []
        current_iteration: list[Message] = []

        for msg in messages:
            if msg.role == Role.ASSISTANT:
                # ASSISTANT starts a new iteration
                # Move any preceding USER and FRAMEWORK messages to the new iteration
                prefix_msgs: list[Message] = []
                while current_iteration and current_iteration[-1].role in (Role.USER, Role.FRAMEWORK):
                    prefix_msgs.insert(0, current_iteration.pop())

                if current_iteration:
                    iterations.append(current_iteration)

                current_iteration = prefix_msgs + [msg]
            else:
                # Continue current iteration (user, framework, or tool)
                current_iteration.append(msg)

        # Add the last iteration
        if current_iteration:
            iterations.append(current_iteration)

        return iterations

    def _group_into_user_rounds(self, messages: list[Message]) -> list[list[Message]]:
        """Group messages into user rounds.

        A UserRound starts with a USER message and ends with an ASSISTANT message
        that has no tool calls (final response).
        """
        user_rounds: list[list[Message]] = []
        current_round: list[Message] = []

        for msg in messages:
            current_round.append(msg)

            if msg.role == Role.ASSISTANT:
                # Check if this is a final response (no tool calls)
                has_tool_use = any(isinstance(block, ToolUseBlock) for block in msg.content)
                if not has_tool_use and current_round:
                    user_rounds.append(current_round)
                    current_round = []

        # Handle incomplete round at the end
        if current_round:
            user_rounds.append(current_round)

        return user_rounds

    def _generate_summary_direct_fallback(self, llm_messages: list[Message]) -> str:
        """Fallback summary path for simple OpenAI-compatible mocked clients."""
        summary_api_type = (
            self.summary_llm_config.api_type if self.summary_llm_config is not None else (self.summary_api_type or "openai_chat_completion")
        )
        if summary_api_type != "openai_chat_completion":
            raise RuntimeError("Direct summary fallback only supports openai_chat_completion")
        if self._summary_client is None:
            raise RuntimeError("Summary client is not initialized")

        create_kwargs: dict[str, Any] = {
            "model": self.summary_llm_config.model if self.summary_llm_config is not None else self.summary_model,
            "messages": messages_to_legacy_openai_chat(llm_messages),
        }
        if self.summary_llm_config is not None and self.summary_llm_config.max_tokens is not None:
            create_kwargs["max_tokens"] = self.summary_llm_config.max_tokens

        response = cast(
            ChatCompletion,
            self._summary_client.chat.completions.create(**create_kwargs),
        )
        if not response.choices:
            return ""

        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        return ""

    def _generate_summary(self, messages: list[Message]) -> str:
        """Generate summary using LLM.

        Raises:
            Exception: If LLM call fails.
        """
        llm_caller = self._ensure_llm_caller()
        summary_model_name = self.summary_llm_config.model if self.summary_llm_config is not None else self.summary_model
        logger.info(f"[SlidingWindowCompaction] Calling LLM to generate summary (model: {summary_model_name})")

        # Prepare messages for LLM.
        llm_messages = messages.copy()
        llm_messages.append(Message(role=Role.USER, content=[TextBlock(text=self.compact_prompt)]))

        summary_api_type = (
            self.summary_llm_config.api_type if self.summary_llm_config is not None else (self.summary_api_type or "openai_chat_completion")
        )
        tool_call_mode = "structured"

        try:
            model_response = llm_caller.call_llm(
                llm_messages,
                tool_call_mode=tool_call_mode,
            )
            summary = (model_response.content or "").strip() if model_response else ""
            logger.info("[SlidingWindowCompaction] LLM summary generated successfully")
            return summary
        except Exception as exc:
            if summary_api_type != "openai_chat_completion":
                raise

            logger.warning(
                "[SlidingWindowCompaction] LLMCaller summary generation failed; falling back to direct OpenAI-compatible client call: %s",
                exc,
            )
            summary = self._generate_summary_direct_fallback(llm_messages)
            logger.info("[SlidingWindowCompaction] LLM summary generated successfully via direct client fallback")
            return summary
