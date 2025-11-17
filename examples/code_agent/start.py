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

"""Test the code agent loading from YAML configuration."""

import logging
import os
from datetime import datetime
from pathlib import Path

import langfuse

from nexau.archs.config.config_loader import load_agent_config

logging.basicConfig(level=logging.INFO)


def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Test the code agent with YAML-based agent configuration."""
    print("Testing Code Agent (YAML-based)")
    print("=" * 60)

    try:
        # Load agent from YAML configuration
        print("Loading Code agent from YAML configuration...")

        script_dir = Path(__file__).parent
        claude_code_agent = load_agent_config(str(script_dir / "cc_agent.yaml"))
        print("✓ Agent loaded successfully from YAML")

        print("\nTesting Code Agent...")
        user_message = input("Enter your task: ")
        print(f"\nUser: {user_message}")
        print("\nAgent Response:")
        print("-" * 30)

        response = claude_code_agent.run(
            user_message,
            context={
                "date": get_date(),
                "username": os.getenv("USER"),
                "working_directory": os.getcwd(),
                "env_content": {
                    "date": get_date(),
                    "username": os.getenv("USER"),
                    "working_directory": os.getcwd(),
                },
            },
        )
        print(response)

        if claude_code_agent.langfuse_trace_id:
            print("Langfuse trace ID:", claude_code_agent.langfuse_trace_id)
            trace_url = langfuse.get_client().get_trace_url(trace_id=claude_code_agent.langfuse_trace_id)
            print("Langfuse trace URL:", trace_url)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
