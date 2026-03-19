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

"""Tool-call mode normalization and provider resolution helpers.

RFC-0006: xml / structured 外部语义与 provider 延迟适配

集中管理 ``tool_call_mode`` 的标准值、legacy alias 兼容提示，以及
structured 模式下 ``llm_config.api_type -> provider target`` 的边界解析。
"""

from __future__ import annotations

import logging
import threading
from typing import Literal

logger = logging.getLogger(__name__)

ToolCallMode = Literal["xml", "structured"]
StructuredProviderTarget = Literal["openai", "anthropic", "gemini"]

VALID_TOOL_CALL_MODES: set[str] = {"xml", "structured", "openai", "anthropic"}
STRUCTURED_TOOL_CALL_MODES: set[str] = {"structured"}
LEGACY_STRUCTURED_TOOL_CALL_MODE_ALIASES: set[str] = {"openai", "anthropic"}
DEFAULT_TOOL_CALL_MODE = "structured"

_OPENAI_FAMILY_API_TYPES = {"openai_chat_completion", "openai_responses", "generate_with_token"}
_ANTHROPIC_API_TYPES = {"anthropic_chat_completion"}
_GEMINI_API_TYPES = {"gemini_rest"}

_legacy_tool_call_mode_alias_lock = threading.Lock()
_warned_legacy_tool_call_mode_aliases: set[str] = set()


def _warn_legacy_tool_call_mode_alias_once(alias: str) -> None:
    """Emit a one-time compatibility warning for legacy structured aliases.

    RFC-0006: 兼容 alias 日志与提示策略

    ``openai`` / ``anthropic`` 继续可用，但只表示 structured tool calling；
    真正的 provider wire format 由 ``llm_config.api_type`` 决定。
    """

    with _legacy_tool_call_mode_alias_lock:
        if alias in _warned_legacy_tool_call_mode_aliases:
            return
        _warned_legacy_tool_call_mode_aliases.add(alias)

    logger.warning(
        "[RFC-0006] tool_call_mode=%r is a legacy compatibility alias and now behaves as "
        "'structured'. Please migrate to 'structured'; provider selection is determined by "
        "llm_config.api_type.",
        alias,
    )


def normalize_tool_call_mode(mode: str | None) -> str:
    """Normalize a ``tool_call_mode`` value and validate it.

    RFC-0006: tool_call_mode 语义收敛与兼容入口

    对外标准值收敛为 ``xml`` / ``structured``；``openai`` / ``anthropic``
    保留为兼容 alias，但会输出一次弃用提示并统一归一化到 ``structured``。
    """

    # 1. 统一默认值与大小写处理，确保 Python / YAML 两个入口语义一致。
    normalized = (mode or DEFAULT_TOOL_CALL_MODE).lower()

    # 2. 拒绝 RFC-0006 之外的对外模式值，避免继续扩散 provider 语义。
    if normalized not in VALID_TOOL_CALL_MODES:
        raise ValueError(
            "tool_call_mode must be one of 'xml', 'structured', 'openai', or 'anthropic'",
        )

    # 3. 对 legacy alias 输出一次兼容提示，并统一收敛到 neutral structured 模式。
    if normalized in LEGACY_STRUCTURED_TOOL_CALL_MODE_ALIASES:
        _warn_legacy_tool_call_mode_alias_once(normalized)
        return "structured"
    return normalized


def resolve_structured_provider_target(api_type: str | None) -> StructuredProviderTarget:
    """Resolve the concrete structured-tool provider target from ``api_type``.

    RFC-0006: Provider 延迟适配

    structured 模式下，provider wire format 不再由 ``tool_call_mode`` 推断，
    而是在真正发请求前根据 ``llm_config.api_type`` 解析为边界 adapter target。
    """

    # 1. 仅以 api_type 作为 structured provider target 的权威来源。
    normalized_api_type = (api_type or "").strip()
    if normalized_api_type in _OPENAI_FAMILY_API_TYPES:
        return "openai"
    if normalized_api_type in _ANTHROPIC_API_TYPES:
        return "anthropic"
    if normalized_api_type in _GEMINI_API_TYPES:
        return "gemini"

    # 2. 对不支持的 provider fail fast，并给出明确的 RFC-0006 语义提示。
    raise ValueError(
        "Structured tool calling is not supported for "
        f"api_type='{normalized_api_type or 'unknown'}'. Supported api_type values are: "
        "openai_chat_completion, openai_responses, anthropic_chat_completion, gemini_rest.",
    )
