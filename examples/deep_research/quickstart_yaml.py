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

"""Test the quick start example loading agent from YAML configuration."""

import logging
import os
from datetime import datetime
from pathlib import Path

from nexau import Agent, AgentConfig, LLMConfig

logging.basicConfig(level=logging.INFO)


def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Test the quick start example with YAML-based agent configuration."""
    print("Testing NexAU Framework Quick Start Example (YAML-based)")
    print("=" * 60)

    try:
        # Load agent from YAML configuration
        print("Loading deep research agent from YAML configuration...")

        # Build LLM configuration from environment variables

        llm_config = LLMConfig(
            model=os.getenv("LLM_MODEL"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=0.7,
            max_tokens=4096,
        )

        script_dir = Path(__file__).parent
        deep_research_agent_config = AgentConfig.from_yaml(
            script_dir / "deep_research_agent.yaml",
        )
        deep_research_agent_config.llm_config = llm_config
        deep_research_agent = Agent(config=deep_research_agent_config)
        print("✓ Agent loaded successfully from YAML")

        print("\nTesting delegation with web research...")
        # web_message = '做一个孙悟空介绍的的html网页'
        # web_message = "List all commits in https://github.com/nex-agi/bp-sandbox"
        # web_message = "Show me details of the skill `algorithmic-art`"
        web_message = "请使用sub_agent 查一下今天上海的天气"
        print(f"\nUser: {web_message}")
        print("\nAgent Response:")
        print("-" * 30)

        response = deep_research_agent.run(
            message=web_message,
            context={
                "date": get_date(),
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
