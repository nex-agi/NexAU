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

"""ContextValue model for structured runtime parameter passing."""

from pydantic import BaseModel, Field


class ContextValue(BaseModel):
    """Structured runtime parameters passed from external callers.

    Fields:
        template: Jinja2 system prompt template variables, merged into AgentContext
            for prompt rendering. Accessible via agent_state.get_context_value().
        runtime_vars: Runtime variables not injected into prompts. Accessible via
            agent_state.get_variable(). Suitable for API keys and other secrets.
        sandbox_env: Sandbox environment variables, injected into BaseSandbox.envs
            at Agent initialization time. Accessible inside sandbox as $KEY.
    """

    template: dict[str, str] = Field(default_factory=dict)
    runtime_vars: dict[str, str] = Field(default_factory=dict)
    sandbox_env: dict[str, str] = Field(default_factory=dict)
