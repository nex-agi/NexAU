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

"""Tool registry with deferred loading support.

RFC-0005: Tool Search — 工具按需动态注入

ToolRegistry 管理工具的注册、分类和按需注入。
工具来源（source）是不可变的，注入（inject）仅在运行时生效，不修改任何 source。
"""

import logging
import re
import threading
from collections.abc import Sequence

from .tool import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry that separates eager and deferred tools.

    RFC-0005: 不可变来源 + computed tools + inject

    Core interface:
    - add_source(name, tools): 注册工具来源
    - compute_eager_tools(): 当前应传给 LLM 的工具列表
    - compute_deferred_tools(): ToolSearch 搜索池
    - compute_serial_tool_names(): 当前串行工具名称列表
    - inject(tool_name): 运行时注入 deferred tool
    - get_all(): 完整注册表（用于工具执行）
    - search(query, max_results): 搜索 deferred 工具并注入
    """

    def __init__(self) -> None:
        self._sources: dict[str, list[Tool]] = {}
        self._injected: set[str] = set()
        self._lock = threading.RLock()

    def add_source(self, name: str, tools: Sequence[Tool]) -> None:
        """Register a tool source (append-only, does not modify existing entries).

        RFC-0005: 注册工具来源（只追加，不修改已有条目）

        Args:
            name: Source identifier (e.g. 'config', 'mcp', 'builtin')
            tools: List of tools from this source
        """
        with self._lock:
            if name in self._sources:
                self._sources[name].extend(tools)
            else:
                self._sources[name] = list(tools)

    def compute_eager_tools(self) -> list[Tool]:
        """Compute tools that should be sent to LLM this turn.

        RFC-0005: defer_loading=false 的工具 + 已注入的 deferred 工具

        Returns:
            List of tools to include in LLM tools parameter
        """
        with self._lock:
            result: list[Tool] = []
            for tools in self._sources.values():
                for tool in tools:
                    if not tool.defer_loading or tool.name in self._injected:
                        result.append(tool)
            return result

    def compute_deferred_tools(self) -> list[Tool]:
        """Compute tools available for ToolSearch (not yet injected).

        RFC-0005: defer_loading=true 且尚未注入的工具

        Returns:
            List of deferred tools not yet injected
        """
        with self._lock:
            result: list[Tool] = []
            for tools in self._sources.values():
                for tool in tools:
                    if tool.defer_loading and tool.name not in self._injected:
                        result.append(tool)
            return result

    def compute_serial_tool_names(self) -> list[str]:
        """Compute tool names that must execute serially.

        Returns:
            List of tool names whose execution should block subsequent tool submission.
        """
        with self._lock:
            result: list[str] = []
            for tools in self._sources.values():
                for tool in tools:
                    if tool.disable_parallel:
                        result.append(tool.name)
            return result

    def inject(self, tool_name: str) -> bool:
        """Inject a deferred tool so it appears in eager tools next turn.

        RFC-0005: 运行时注入 deferred tool（不修改任何 source）

        Args:
            tool_name: Name of the tool to inject

        Returns:
            True if tool was found and injected, False otherwise
        """
        with self._lock:
            if tool_name in self._injected:
                return True
            for tools in self._sources.values():
                for tool in tools:
                    if tool.name == tool_name and tool.defer_loading:
                        self._injected.add(tool_name)
                        logger.info("Injected deferred tool '%s'", tool_name)
                        return True
            return False

    def get_all(self) -> dict[str, Tool]:
        """Get complete registry (for tool execution lookup).

        RFC-0005: 获取完整注册表（用于工具执行）

        Returns:
            Dict mapping tool name to Tool object
        """
        with self._lock:
            registry: dict[str, Tool] = {}
            for tools in self._sources.values():
                for tool in tools:
                    registry[tool.name] = tool
            return registry

    def get_tool(self, name: str) -> Tool | None:
        """Look up a single tool by name (any source, eager or deferred)."""
        with self._lock:
            for tools in self._sources.values():
                for tool in tools:
                    if tool.name == name:
                        return tool
            return None

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[Tool]:
        """Search deferred tools and inject matches.

        RFC-0005: 搜到即注入，无需额外 activate 步骤

        Weighted keyword search on name + description + search_hint.

        Args:
            query: Search query string
            max_results: Maximum number of tools to inject

        Returns:
            List of matched and injected tools
        """
        if max_results <= 0:
            return []

        return self._search_keyword(query, max_results)

    def _search_keyword(self, query: str, limit: int) -> list[Tool]:
        """Keyword search with weighted scoring.

        RFC-0005: 加权关键词搜索

        Scoring:
        - Tool name exact match with token: +10
        - Tool name partial match: +5
        - Tool name full contains: +3
        - search_hint match: +4
        - Description match: +2

        Supports +keyword for required filtering.
        """
        deferred = self.compute_deferred_tools()
        if not deferred:
            return []

        # 1. 分离 required tokens (+keyword) 和普通 tokens
        raw_tokens = query.lower().split()
        required_tokens: list[str] = []
        search_tokens: list[str] = []
        for token in raw_tokens:
            if token.startswith("+") and len(token) > 1:
                required_tokens.append(token[1:])
            else:
                search_tokens.append(token)

        all_tokens = required_tokens + search_tokens

        if not all_tokens:
            return []

        # 2. 前置过滤（required tokens 必须出现在 name 中）
        candidates = deferred
        if required_tokens:
            filtered: list[Tool] = []
            for tool in candidates:
                name_lower = tool.name.lower()
                if all(req in name_lower for req in required_tokens):
                    filtered.append(tool)
            candidates = filtered

        # 3. 加权评分
        scored: list[tuple[float, Tool]] = []
        for tool in candidates:
            score = self._score_tool(tool, all_tokens)
            if score > 0:
                scored.append((score, tool))

        # 4. 排序并截断
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [tool for _, tool in scored[:limit]]

        # 5. 注入
        with self._lock:
            for tool in results:
                self._injected.add(tool.name)

        return results

    @staticmethod
    def _score_tool(tool: Tool, tokens: list[str]) -> float:
        """Calculate weighted relevance score for a tool.

        RFC-0005: 对名字 + description + search_hint 做加权关键词匹配
        """
        score = 0.0
        name_lower = tool.name.lower()
        desc_lower = (tool.description or "").lower()
        hint_lower = (tool.search_hint or "").lower()

        # Phase 1.5: CamelCase 分词支持
        # 两步 re.sub 处理连续大写：HTTPClient → HTTP_Client → http_client → ["http", "client"]
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", tool.name)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        name_parts = [p for p in re.split(r"[_\-]", s.lower()) if p]

        for token in tokens:
            # name 精确匹配某个 token（按 _ / CamelCase 分词）
            if token in name_parts:
                score += 10
            elif token in name_lower:
                score += 5
            elif name_lower in token or token in name_lower:
                score += 3

            # search_hint 匹配
            if hint_lower and token in hint_lower:
                score += 4

            # description 匹配
            if token in desc_lower:
                score += 2

        return score

    def build_deferred_index(self) -> str:
        """Build a compact index of deferred tools for ToolSearch description.

        RFC-0005: description 中附带 deferred tools 的简短索引（名字 + 一句话描述）

        Returns:
            Formatted string listing available deferred tools
        """
        deferred = self.compute_deferred_tools()
        if not deferred:
            return ""
        lines = [f"- {tool.name}: {str(tool.description or '')[:80]}" for tool in deferred]
        return "\n".join(lines)

    @property
    def injected_count(self) -> int:
        with self._lock:
            return len(self._injected)

    @property
    def deferred_count(self) -> int:
        return len(self.compute_deferred_tools())

    @property
    def eager_count(self) -> int:
        return len(self.compute_eager_tools())
