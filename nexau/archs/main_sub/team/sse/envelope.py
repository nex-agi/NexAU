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

"""Team SSE stream envelope.

RFC-0002: 多 Agent SSE 事件封装
"""

from __future__ import annotations

from pydantic import BaseModel

from nexau.archs.llm.llm_aggregators.events import Event


class TeamStreamEnvelope(BaseModel):
    """Envelope for multi-agent SSE events.

    RFC-0002: 多 Agent 流式事件封装

    每个 SSE 事件包含 agent 来源信息，客户端可按 agent_id 分栏显示。
    """

    team_id: str
    agent_id: str
    role_name: str | None = None
    run_id: str | None = None
    event: Event
