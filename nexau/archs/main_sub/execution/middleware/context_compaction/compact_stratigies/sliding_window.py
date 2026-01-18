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

import logging
from pathlib import Path

import openai

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.core.messages import Message, Role, TextBlock, ToolUseBlock

logger = logging.getLogger(__name__)

# Backward-compatibility: older tests patch `sliding_window.OpenAI`.
OpenAI = openai.OpenAI


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
    3. Compresses older iterations (generates summary or placeholder)
    """

    def __init__(
        self,
        keep_system: bool = True,
        keep_iterations: int = 3,
        keep_user_rounds: int = 0,
        summary_model: str | None = None,
        summary_base_url: str | None = None,
        summary_api_key: str | None = None,
        compact_prompt_path: str | None = None,
    ):
        """Initialize sliding window compaction.

        Args:
            keep_system: Whether to preserve the system message.
            keep_iterations: Number of recent iterations to keep. Default: 3.
            keep_user_rounds: Number of recent user rounds to keep. Default: 0 (disabled).
                When > 0, uses user rounds mode instead of iterations mode.
            summary_model: LLM model for summarization. Required.
            summary_base_url: LLM API base URL for summarization. Required.
            summary_api_key: LLM API key for summarization. Required.
            compact_prompt_path: Path to compact prompt file (already resolved by config). Required.

        Raises:
            ValueError: If both keep_iterations != 3 and keep_user_rounds > 0 are set.
            ValueError: If keep_iterations < 1 or keep_user_rounds < 0.
            ValueError: If LLM configuration is missing.
        """
        if keep_iterations != 3 and keep_user_rounds > 0:
            raise ValueError("Cannot set both keep_iterations and keep_user_rounds")

        if keep_iterations < 1:
            raise ValueError(f"keep_iterations must be >= 1, got {keep_iterations}")
        if keep_user_rounds < 0:
            raise ValueError(f"keep_user_rounds must be >= 0, got {keep_user_rounds}")

        # Validate LLM configuration
        if not summary_model or not summary_base_url or not summary_api_key:
            raise ValueError(
                "LLM configuration is required for SlidingWindowCompaction. "
                "Please provide summary_model, summary_base_url, and summary_api_key."
            )

        # Validate compact prompt path
        if not compact_prompt_path:
            raise ValueError("compact_prompt_path is required for SlidingWindowCompaction.")

        self.keep_system = keep_system
        self.keep_iterations = keep_iterations
        self.keep_user_rounds = keep_user_rounds

        # Initialize LLM caller for summarization (route API calls through LLMCaller).
        self.summary_model = summary_model
        self.summary_base_url = summary_base_url
        self.summary_api_key = summary_api_key

        # Load compact prompt using the resolved path
        self.compact_prompt = _load_compact_prompt(compact_prompt_path)

        summary_llm_config = LLMConfig(
            model=self.summary_model,
            base_url=self.summary_base_url,
            api_key=self.summary_api_key,
            api_type="openai_chat_completion",
        )
        summary_client = OpenAI(**summary_llm_config.to_client_kwargs())
        self._llm_caller = LLMCaller(summary_client, summary_llm_config, retry_attempts=3)
        logger.info(
            f"[SlidingWindowCompaction] Initialized: model={self.summary_model}, "
            f"keep_iterations={self.keep_iterations}, keep_user_rounds={self.keep_user_rounds}"
        )

    def compact(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Compact messages by keeping recent iterations or user rounds."""
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

        # Process kept groups - check if we need to add summary
        if groups_to_compress:
            # Generate summary for compressed groups
            # Include system message if present
            all_compressed_messages: list[Message] = []
            if self.keep_system and messages and messages[0].role == Role.SYSTEM:
                all_compressed_messages.append(messages[0])

            for group_msgs in groups_to_compress:
                all_compressed_messages.extend(group_msgs)

            try:
                summary = self._generate_summary(all_compressed_messages)
            except Exception as e:
                logger.error(f"[SlidingWindowCompaction] Failed to generate summary, returning original messages: {e}")
                return messages.copy()

            # Find the first user message in kept groups and prepend the summary
            first_group_modified = False
            for group_msgs in groups_to_keep:
                for msg in group_msgs:
                    if msg.role == Role.USER and not first_group_modified:
                        # Get original user query
                        original_content = msg.get_text_content()
                        # Prepend summary context
                        modified_content = (
                            f"This session is being continued from a previous conversation that ran out of context. "
                            f"The previous conversation is summarized as follows: {summary}. "
                            f"The user request for this round is: {original_content}"
                        )
                        modified_msg = msg.model_copy(update={"content": [TextBlock(text=modified_content)]})
                        modified_msg.metadata["isSummary"] = True
                        result.append(modified_msg)
                        first_group_modified = True
                    else:
                        result.append(msg)

        else:
            # No compression needed, just add all kept groups
            for group_msgs in groups_to_keep:
                result.extend(group_msgs)

        logger.info(
            f"[SlidingWindowCompaction] Compaction complete: "
            f"{len(messages)} messages -> {len(result)} messages "
            f"({len(groups_to_compress)} {group_name} compressed, {len(groups_to_keep)} {group_name} kept)"
        )
        return result

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

    def _generate_summary(self, messages: list[Message]) -> str:
        """Generate summary using LLM.

        Raises:
            Exception: If LLM call fails.
        """
        logger.info(f"[SlidingWindowCompaction] Calling LLM to generate summary (model: {self.summary_model})")

        # Prepare messages for LLM: original messages + compact prompt as final user message
        llm_messages = messages.copy()
        llm_messages.append(Message(role=Role.USER, content=[TextBlock(text=self.compact_prompt)]))

        model_response = self._llm_caller.call_llm(
            llm_messages,
            max_tokens=2048,
        )
        summary = (model_response.content or "").strip() if model_response else ""
        logger.info("[SlidingWindowCompaction] LLM summary generated success")
        return summary
