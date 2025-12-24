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

"""Test the quick start example from the specification."""

import os
from datetime import datetime
from pathlib import Path

from nexau import Agent, AgentConfig
from nexau.archs.llm import LLMConfig
from nexau.archs.tool import Tool
from nexau.archs.tool.builtin.todo_write import todo_write
from nexau.archs.tool.builtin.web_tool import web_read, web_search


def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Test the quick start example."""
    print("Testing NexAU Framework Quick Start Example")
    print("=" * 50)

    try:
        # Create tools
        script_dir = Path(__file__).parent
        print("Creating tools...")
        web_search_tool = Tool.from_yaml(
            str(script_dir / "tools/WebSearch.yaml"),
            binding=web_search,
        )
        web_read_tool = Tool.from_yaml(
            str(script_dir / "tools/WebRead.yaml"),
            binding=web_read,
        )
        todo_write_tool = Tool.from_yaml(
            str(script_dir / "tools/TodoWrite.tool.yaml"),
            binding=todo_write,
        )
        print("✓ Tools created successfully")

        # Create sub-agents
        llm_config = LLMConfig(
            model=os.getenv("LLM_MODEL"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
        )
        print("\nCreating sub-agents...")

        system_prompt = """
Date: {{date}}. You are a deep research agent.
You are given a message and you need to search for the information that matches the message.
Use the web_search and web_read tools to get the information.
Wait for the web_read tool to finish before you continue your response.
Before searching, you need to use todo_write_tool to write a todo list to track the research progress.
When completing a task, you need to update the todo list. Todo list: {{current_todos}}
"""
        # It is not allowed for a sub-agent to list itself as one of its own sub-agents.
        sub_agent_config = AgentConfig(
            description="Delegate a research task to this agent.",
            name="sub_deep_research_agent",
            tools=[web_search_tool, web_read_tool, todo_write_tool],
            llm_config=llm_config,
            system_prompt=system_prompt,
        )
        main_agent_config = AgentConfig(
            name="deep_research_agent",
            tools=[web_search_tool, web_read_tool, todo_write_tool],
            llm_config=llm_config,
            system_prompt=system_prompt,
            sub_agents={"sub_agent": sub_agent_config},
        )
        deep_research_agent = Agent(config=main_agent_config)
        print("✓ Sub-agents created successfully")
        print("\nTesting delegation with web research...")
        web_message = "调研一下腾讯，拆解成多个调研子任务，并让多个 subagent 分别并行执行这些子任务"
        print(f"\nUser: {web_message}")
        print("\nAgent Response:")
        print("-" * 30)

        response = deep_research_agent.run(
            web_message,
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
