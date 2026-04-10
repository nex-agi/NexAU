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

"""Test the code agent with an E2B sandbox."""

import logging
import os
from datetime import datetime
from pathlib import Path

from nexau import Agent, AgentConfig
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware, Event
from nexau.archs.sandbox.base_sandbox import E2BSandboxConfig

logging.basicConfig(level=logging.INFO)


def get_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def handler(event: Event) -> None:
    print(event)


def build_e2b_sandbox_config() -> E2BSandboxConfig:
    """Build E2B sandbox config from environment variables."""
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is required to run the code agent with an E2B sandbox.")

    timeout_raw = os.getenv("E2B_TIMEOUT", "300")
    try:
        timeout = int(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"E2B_TIMEOUT must be an integer, got: {timeout_raw}") from exc

    template = os.getenv("E2B_TEMPLATE", "base")
    work_dir = os.getenv("E2B_WORK_DIR", "/home/user")
    api_url = os.getenv("E2B_API_URL") or None

    metadata = {
        "example": "code_agent",
        "launcher": "start_e2b.py",
    }
    if user := os.getenv("USER"):
        metadata["user"] = user

    return E2BSandboxConfig(
        type="e2b",
        api_key=api_key,
        api_url=api_url,
        template=template,
        timeout=timeout,
        work_dir=work_dir,
        metadata=metadata,
    )


def main() -> bool:
    """Test the code agent with YAML config and an E2B sandbox override."""
    print("Testing Code Agent (YAML-based, E2B sandbox)")
    print("=" * 60)

    try:
        print("Loading Code agent from YAML configuration...")

        script_dir = Path(__file__).parent
        config = AgentConfig.from_yaml(config_path=script_dir / "code_agent.yaml")
        config.sandbox_config = build_e2b_sandbox_config()
        if config.sub_agents:
            for sub_agent_name in config.sub_agents:
                config.sub_agents[sub_agent_name].sandbox_config = build_e2b_sandbox_config()

        event_middleware = AgentEventsMiddleware(session_id="test", on_event=handler)
        if config.middlewares:
            config.middlewares.append(event_middleware)
        else:
            config.middlewares = [event_middleware]

        claude_code_agent = Agent(config=config)
        print("✓ Agent loaded successfully from YAML")
        print(
            "✓ E2B sandbox configured: template={}, work_dir={}, timeout={}s".format(
                config.sandbox_config.template,
                config.sandbox_config.work_dir,
                config.sandbox_config.timeout,
            )
        )

        sandbox = claude_code_agent.sandbox_manager.instance
        if sandbox is None:
            print("✗ Sandbox failed to start")
            return False

        print("✓ E2B sandbox started successfully: {}".format(sandbox.sandbox_id))

        print("\nTesting Code Agent...")
        user_message = input("Enter your task: ")
        print(f"\nUser: {user_message}")
        print("\nAgent Response:")
        print("-" * 30)

        response = claude_code_agent.run(
            message=user_message,
            context={
                "date": get_date(),
                "username": os.getenv("USER"),
                "working_directory": str(sandbox.work_dir),
                "env_content": {
                    "date": get_date(),
                    "username": os.getenv("USER"),
                    "working_directory": str(sandbox.work_dir),
                },
            },
        )
        print(response)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
