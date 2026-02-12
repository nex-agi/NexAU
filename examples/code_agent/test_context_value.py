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

"""Example: verify all three ContextValue fields (template, runtime_vars, sandbox_env)."""

import logging
import os
from datetime import datetime
from pathlib import Path

from nexau import Agent
from nexau.archs.main_sub.context_value import ContextValue

logging.basicConfig(level=logging.INFO)


def get_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> bool:
    """Create an Agent and verify ContextValue injection end-to-end."""
    print("ContextValue Full Verification Test")
    print("=" * 60)

    try:
        script_dir = Path(__file__).parent
        agent = Agent.from_yaml(config_path=script_dir / "code_agent.yaml")
        print("Agent loaded successfully\n")

        variables = ContextValue(
            template={
                "date": get_date(),
                "username": os.getenv("USER", "unknown"),
                "working_directory": "/tmp/test-injected-cwd",
            },
            runtime_vars={
                "api_key": "sk-test-12345",
                "tenant_id": "tenant-abc-001",
            },
            sandbox_env={
                "MY_PROJECT_NAME": "nexau-demo",
                "MY_SECRET_TOKEN": "token-abc-999",
                "MY_DEPLOY_ENV": "staging",
            },
        )

        # --- Step 1: Verify template (Jinja2 system prompt vars) ---
        print("[Step 1] template - Jinja2 system prompt variables")
        print(f"  working_directory = {variables.template['working_directory']}")
        print(f"  date              = {variables.template['date']}")
        print(f"  username          = {variables.template['username']}")
        print("  (system prompt renders {{working_directory}} -> check Langfuse)\n")

        # --- Step 2: Verify variables (runtime, not in prompt) ---
        print("[Step 2] runtime_vars - runtime vars (not exposed to LLM)")
        print(f"  api_key   = {variables.runtime_vars['api_key']}")
        print(f"  tenant_id = {variables.runtime_vars['tenant_id']}")
        print("  (accessible via agent_state.get_variable() -> check Langfuse)\n")

        # --- Step 3: Verify sandbox_env (injected into sandbox) ---
        print("[Step 3] sandbox_env - injected into sandbox process env")
        for k, v in variables.sandbox_env.items():
            print(f"  {k}={v}")
        print()

        # Run 1: verify sandbox_env via echo
        print("-" * 40)
        print("Run 1: echo sandbox env vars")
        print("-" * 40)
        resp1 = agent.run(
            message=(
                "Run this command and show me the output:\n"
                "echo \"PROJECT=$MY_PROJECT_NAME TOKEN=$MY_SECRET_TOKEN ENV=$MY_DEPLOY_ENV\""
            ),
            variables=variables,
        )
        print(resp1)

        # Run 2: verify template by asking agent about its working directory
        print("\n" + "-" * 40)
        print("Run 2: check template rendering in system prompt")
        print("-" * 40)
        resp2 = agent.run(
            message="What is your current working directory? Just tell me the path.",
            variables=variables,
        )
        print(resp2)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
