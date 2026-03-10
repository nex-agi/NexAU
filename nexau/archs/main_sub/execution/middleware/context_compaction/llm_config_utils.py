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

"""Helpers for deriving summary LLM runtime config from middleware settings."""

from __future__ import annotations

from typing import Any

from nexau.archs.llm.llm_config import LLMConfig


def normalize_summary_llm_overrides(
    summary_llm_config: dict[str, Any] | None,
    *,
    summary_model: str | None = None,
    summary_base_url: str | None = None,
    summary_api_key: str | None = None,
    summary_api_type: str | None = None,
) -> dict[str, Any]:
    """Merge nested and legacy summary config fields into one override mapping."""
    overrides = dict(summary_llm_config or {})
    if summary_model is not None and "model" not in overrides:
        overrides["model"] = summary_model
    if summary_base_url is not None and "base_url" not in overrides:
        overrides["base_url"] = summary_base_url
    if summary_api_key is not None and "api_key" not in overrides:
        overrides["api_key"] = summary_api_key
    if summary_api_type is not None and "api_type" not in overrides:
        overrides["api_type"] = summary_api_type
    return overrides


def resolve_summary_llm_config(
    *,
    base_llm_config: LLMConfig | None,
    summary_overrides: dict[str, Any],
) -> LLMConfig:
    """Resolve the effective summary runtime config.

    Semantics:
    - If ``summary_llm_config`` (or legacy flat summary fields) is provided, use it
      as the full summary runtime without inheriting fields from the agent model.
    - If no summary override is provided, reuse the agent/runtime LLM config.
    """
    if summary_overrides:
        missing = [key for key in ("model", "base_url", "api_key") if not summary_overrides.get(key)]
        if missing:
            raise ValueError(
                f"summary_llm_config must be a complete standalone LLM config when provided. Missing keys: {', '.join(missing)}"
            )

        explicit_kwargs = dict(summary_overrides)
        explicit_kwargs.setdefault("api_type", "openai_chat_completion")
        return LLMConfig(**explicit_kwargs)

    if base_llm_config is None:
        raise ValueError(
            "LLM configuration is required for context compaction summarization. Provide agent llm_config or set summary_llm_config."
        )

    return LLMConfig(
        model=base_llm_config.model,
        base_url=base_llm_config.base_url,
        api_key=base_llm_config.api_key,
        temperature=base_llm_config.temperature,
        max_tokens=base_llm_config.max_tokens,
        top_p=base_llm_config.top_p,
        frequency_penalty=base_llm_config.frequency_penalty,
        presence_penalty=base_llm_config.presence_penalty,
        timeout=base_llm_config.timeout,
        max_retries=base_llm_config.max_retries,
        debug=base_llm_config.debug,
        stream=base_llm_config.stream,
        additional_drop_params=base_llm_config.additional_drop_params,
        api_type=base_llm_config.api_type,
        **base_llm_config.extra_params,
    )
