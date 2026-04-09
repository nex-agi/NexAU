"""Agent builtin tool — unified entry point for creating and resuming sub-agents.

RFC-0015: 合并 Sub-agent 工具为统一 Agent 工具

Provides a single `Agent` tool that handles both creating new sub-agents
and resuming existing ones. When `sub_agent_id` is omitted, a new sub-agent
is created; when provided, an existing sub-agent run is resumed.
"""

from __future__ import annotations

import re
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution import SubAgentManager


def _extract_sub_agent_id(text: str) -> str | None:
    """Extract sub_agent_id from the [sub_agent_id: ...] prefix.

    RFC-0015: SubAgentManager 在返回字符串开头插入 [sub_agent_id: <id>]
    以确保主代理 LLM 无论成功或失败都能第一时间看到 sub_agent_id。
    """
    match = re.match(r"\[sub_agent_id:\s*([^\]]+)\]", text)
    return match.group(1) if match else None


def call_sub_agent(
    sub_agent_name: str,
    message: str,
    sub_agent_id: str | None = None,
    agent_state: AgentState | None = None,
) -> dict[str, Any]:
    """Delegate work to a sub-agent.

    RFC-0015: Agent 统一工具实现

    当 sub_agent_id 为空时创建新子代理，非空时恢复已有子代理。
    执行层路由到 SubAgentManager.call_sub_agent()。

    Args:
        sub_agent_name: Name of the sub-agent as configured on the parent agent.
        message: Task/question for the sub-agent.
        sub_agent_id: Optional identifier of a previously finished sub-agent run.
        agent_state: Injected by the framework; provides access to the sub-agent manager.

    Returns:
        Dict with `status` and either `result` or `error`.
    """
    # 0. 规范化 sub_agent_id：LLM 常将可选字符串参数发送为空字符串 ""，需统一为 None
    if not sub_agent_id:
        sub_agent_id = None

    # 1. 验证 agent_state 可用
    if agent_state is None:
        return {"status": "error", "error": "Agent state not available"}

    # 2. 获取 SubAgentManager
    subagent_manager: SubAgentManager | None = agent_state.subagent_manager
    if subagent_manager is None:
        return {
            "status": "error",
            "error": "Sub-agent manager not available on agent_state",
        }

    # 3. 路由到 SubAgentManager.call_sub_agent()
    try:
        result = subagent_manager.call_sub_agent(
            sub_agent_name,
            message,
            sub_agent_id,
            parent_agent_state=agent_state,
        )
        # RFC-0015: 从返回字符串开头提取实际 sub_agent_id（新建子代理时输入参数为 None）
        actual_sub_agent_id = _extract_sub_agent_id(result)
        return {
            "status": "success",
            "sub_agent_name": sub_agent_name,
            "sub_agent_id": actual_sub_agent_id,
            "message": message,
            "result": result,
        }
    except Exception as exc:
        # RFC-0015: 异常消息也包含 [sub_agent_id: ...] 前缀
        actual_sub_agent_id = _extract_sub_agent_id(str(exc))
        return {
            "status": "error",
            "sub_agent_name": sub_agent_name,
            "sub_agent_id": actual_sub_agent_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


__all__ = ["call_sub_agent"]
