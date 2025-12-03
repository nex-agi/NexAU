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

"""Factory functions for creating compaction and trigger strategies."""

from .compact_stratigies.base import CompactionStrategy
from .compact_stratigies.compact_tool_result import ToolResultCompaction
from .compact_stratigies.sliding_window import SlidingWindowCompaction
from .config import CompactionConfig
from .trigger_strategies.base import TriggerStrategy
from .trigger_strategies.token_threshold import TokenThresholdTrigger


def create_compaction_strategy(config: CompactionConfig) -> CompactionStrategy:
    """Builds the specific compaction strategy based on config.

    Args:
        config: Validated configuration object (with resolved paths)

    Returns:
        Initialized compaction strategy instance

    Raises:
        ValueError: If strategy type is unknown
    """
    if config.compaction_strategy == "sliding_window":
        return SlidingWindowCompaction(
            window_size=config.window_size,
            summary_model=config.summary_model,
            summary_base_url=config.summary_base_url,
            summary_api_key=config.summary_api_key,
            compact_prompt_path=config.compact_prompt_path,  # Already resolved by config
        )

    elif config.compaction_strategy == "tool_result_compaction":
        return ToolResultCompaction()

    raise ValueError(f"Unknown compaction strategy: {config.compaction_strategy}")


def create_trigger_strategy(config: CompactionConfig) -> TriggerStrategy:
    """Builds the trigger strategy based on config.

    Args:
        config: Validated configuration object

    Returns:
        Initialized trigger strategy instance
    """
    return TokenThresholdTrigger(threshold=config.threshold)
