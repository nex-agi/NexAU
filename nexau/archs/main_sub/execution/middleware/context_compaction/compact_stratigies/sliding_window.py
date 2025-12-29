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
from nexau.core.messages import Message, Role, TextBlock

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

    A "iteration" is defined as: user message + assistant response + tool results (if any).

    This strategy:
    1. Groups messages into conversation iterations
    2. Keeps the most recent N iterations in full
    3. Compresses older iterations (generates summary or placeholder)
    """

    def __init__(
        self,
        keep_system: bool = True,
        window_size: int = 2,
        summary_model: str | None = None,
        summary_base_url: str | None = None,
        summary_api_key: str | None = None,
        compact_prompt_path: str | None = None,
    ):
        """Initialize sliding window compaction.

        Args:
            keep_system: Whether to preserve the system message.
            window_size: Number of recent iterations to keep. Must be >= 1.
            summary_model: LLM model for summarization. Required.
            summary_base_url: LLM API base URL for summarization. Required.
            summary_api_key: LLM API key for summarization. Required.
            compact_prompt_path: Path to compact prompt file (already resolved by config). Required.

        Raises:
            ValueError: If window_size < 1 or LLM configuration is missing.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

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
        self.window_size = window_size

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
            f"[SlidingWindowCompaction] Initialized with LLM summarization: model={self.summary_model}, window_size={self.window_size}"
        )

    def compact(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Compact messages by keeping recent iterations."""
        logger.info(f"[SlidingWindowCompaction] Starting compaction on {len(messages)} messages")

        result: list[Message] = []
        start_idx = 0

        # Keep system message if present
        if self.keep_system and messages and messages[0].role == Role.SYSTEM:
            result.append(messages[0])
            start_idx = 1

        # Group messages into iterations, iteration means user message + assistant response + tool results (if any)
        iterations = self._group_into_iterations(messages[start_idx:])

        if len(iterations) <= self.window_size:
            logger.info(f"[SlidingWindowCompaction] Skipping compaction: {len(iterations)} iterations <= {self.window_size} (window size)")
            return messages.copy()

        # Calculate how many iterations to compress
        iterations_to_compress = iterations[: -self.window_size]
        iterations_to_keep = iterations[-self.window_size :]

        # Process kept iterations - check if we need to add summary
        if iterations_to_compress:
            # Generate summary for compressed iterations
            # Include system message if present
            all_compressed_messages: list[Message] = []
            if self.keep_system and messages and messages[0].role == Role.SYSTEM:
                all_compressed_messages.append(messages[0])

            for iteration_msgs in iterations_to_compress:
                all_compressed_messages.extend(iteration_msgs)

            try:
                summary = self._generate_summary(all_compressed_messages)
            except Exception as e:
                logger.error(f"[SlidingWindowCompaction] Failed to generate summary, returning original messages: {e}")
                return messages.copy()

            # Find the first user message in kept iterations and prepend the summary
            first_iteration_modified = False
            for iteration_msgs in iterations_to_keep:
                for msg in iteration_msgs:
                    if msg.role == Role.USER and not first_iteration_modified:
                        # Get original user query
                        original_content = msg.get_text_content()
                        # Prepend summary context
                        modified_content = (
                            f"This session is being continued from a previous conversation that ran out of context. "
                            f"The conversation is summarized as follows: {summary}. "
                            f"The user request for this round is: {original_content}"
                        )
                        modified_msg = msg.model_copy(update={"content": [TextBlock(text=modified_content)]})
                        modified_msg.metadata["isSummary"] = True
                        result.append(modified_msg)
                        first_iteration_modified = True
                    else:
                        result.append(msg)

        else:
            # No compression needed, just add all kept iterations
            for iteration_msgs in iterations_to_keep:
                result.extend(iteration_msgs)

        logger.info(
            f"[SlidingWindowCompaction] Compaction complete: "
            f"{len(messages)} messages -> {len(result)} messages "
            f"({len(iterations_to_compress)} iterations compressed, {len(iterations_to_keep)} iterations kept)"
        )
        return result

    def _group_into_iterations(self, messages: list[Message]) -> list[list[Message]]:
        """Group messages into conversation iterations.

        A iteration consists of: [user] -> [assistant] -> [tool, tool, ...]
        """
        iterations: list[list[Message]] = []
        current_iteration: list[Message] = []

        for msg in messages:
            if msg.role == Role.USER:
                # Start a new iteration
                if current_iteration:
                    iterations.append(current_iteration)
                current_iteration = [msg]
            else:
                # Continue current iteration (assistant or tool)
                current_iteration.append(msg)

        # add the last iteration
        if current_iteration:
            iterations.append(current_iteration)

        return iterations

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
