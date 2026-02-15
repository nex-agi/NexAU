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

"""Stop result data model.

RFC-0001: Agent 中断时状态持久化

StopResult 封装 agent.stop() 的返回值，
包含停止时的消息快照、停止原因和可选的部分 LLM 响应。
"""

from dataclasses import dataclass, field

from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.core.messages import Message


@dataclass
class StopResult:
    """Result returned by Agent.stop().

    RFC-0001: Agent 停止结果数据模型

    Attributes:
        messages: 停止时的完整消息历史快照
        stop_reason: 停止原因（USER_INTERRUPTED）
        interrupted_at_iteration: 停止发生时的迭代编号
        partial_response: 部分 LLM 响应（流式中断时可用）
    """

    messages: list[Message] = field(default_factory=lambda: list[Message]())
    stop_reason: AgentStopReason = AgentStopReason.USER_INTERRUPTED
    interrupted_at_iteration: int = 0
    partial_response: str | None = None
