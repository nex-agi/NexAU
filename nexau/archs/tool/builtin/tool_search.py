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

"""ToolSearch builtin tool implementation.

RFC-0005: Tool Search — 工具按需动态注入
RFC-0006: 通过 FrameworkContext.tools.search() 访问

ToolSearch 作为 eager tool 注册，LLM 可调用。
搜到即注入，下一轮 LLM 直接 function call。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexau.archs.main_sub.framework_context import FrameworkContext


def tool_search(query: str, ctx: FrameworkContext, *, max_results: int = 5) -> str:
    """Search for deferred tools and inject them for use.

    RFC-0005: 搜到即注入，无需额外 activate 步骤
    RFC-0006: 通过 ctx.tools.search() 访问，不直接接触 ToolRegistry
    """
    matched = ctx.tools.search(query=query, max_results=max_results)

    if not matched:
        return "No matching tools found."

    names = ", ".join(tool.name for tool in matched)
    return f"Found {len(matched)} tool(s): {names}. They are now available for use."
