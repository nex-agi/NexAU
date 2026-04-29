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

"""Configuration schema for context compaction middleware."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .llm_config_utils import normalize_summary_llm_overrides


class CompactionConfig(BaseModel):
    """Parses and validates the flat YAML configuration."""

    model_config = ConfigDict(extra="forbid")

    # General Settings
    # None = inherit from AgentConfig.max_context_tokens at runtime via set_llm_runtime
    max_context_tokens: int | None = None
    auto_compact: bool = True
    emergency_compact_enabled: bool = True
    threshold: float = 0.75

    # Trigger Selection (micro-compact: 新增 time_based 触发器)
    trigger: Literal["token_threshold", "time_based"] = "token_threshold"
    gap_threshold_minutes: float = 5  # micro-compact: time_based trigger 专用

    # Strategy Selection
    compaction_strategy: Literal["llm_summary", "sliding_window", "tool_result_compaction"] = "tool_result_compaction"

    # Shared Settings (used by both strategies)
    keep_iterations: int = 3  # Number of recent iterations to keep uncompacted
    keep_user_rounds: int = 0  # Number of recent user rounds to keep uncompacted (0 = disabled)

    # micro-compact: tool_result_compaction 工具类型过滤
    compactable_tools: list[str] | None = None

    # Summary LLM overrides
    summary_llm_config: dict[str, Any] | None = None

    # Legacy flat summary settings (still supported for backward compatibility)
    summary_model: str | None = None
    summary_base_url: str | None = None
    summary_api_key: str | None = None
    summary_api_type: str | None = None
    compact_prompt_path: str | None = None
    retry_attempts: int = 3

    # RFC-0021: 压缩时归档被移除的原始消息到 sandbox
    save_history: bool = True
    """开启历史消息文件归档（Opt-out: 启用压缩即归档）。

    归档目录位置固定为 ``{sandbox.work_dir}/.nexau_history_archive/``,
    不暴露成 config —— 避免用户输入路径穿越, 默认值就是设计上的唯一选择。

    启用归档时, summary 末尾会自动注入"如何用 search_file_content / read_file
    召回"的提示文本, 让 agent 知道归档存在并能使用 —— 这是归档的关键价值,
    不再单独提供 opt-out 开关 (没有"归档但不告诉 agent"的真实用例)。

    命名说明: 对外字段叫 ``save_history`` (用户视角: 是否保存历史), 内部模块
    用 ``archive`` 词汇 (``HistoryArchiveWriter`` / ``ARCHIVE_SUBDIR`` /
    ``.nexau_history_archive/`` / ``_boundary``) —— 因为内部强调"归档"语义
    (write-once + per-round + grep-friendly), 而对外只是"开关历史是否落盘"。
    """

    @model_validator(mode="after")
    def validate_and_resolve_paths(self) -> CompactionConfig:
        """Validate config and resolve prompt paths."""
        if self.compaction_strategy == "sliding_window":
            warnings.warn(
                "`compaction_strategy=sliding_window` is deprecated; use `llm_summary` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.compaction_strategy = "llm_summary"

        self.summary_llm_config = (
            normalize_summary_llm_overrides(
                self.summary_llm_config,
                summary_model=self.summary_model,
                summary_base_url=self.summary_base_url,
                summary_api_key=self.summary_api_key,
                summary_api_type=self.summary_api_type,
            )
            or None
        )

        if self.compact_prompt_path:
            resolved_path = Path(self.compact_prompt_path)
        else:
            config_dir = Path(__file__).parent
            prompts_dir = config_dir / "prompts"
            resolved_path = prompts_dir / "compact_prompt.md"

        self.compact_prompt_path = str(resolved_path)
        return self
