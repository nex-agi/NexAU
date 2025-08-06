#!/usr/bin/env python3
"""Test the quick start example from the specification."""

import os
from datetime import datetime
from pathlib import Path

from northau.archs.main_sub import create_agent
from northau.archs.tool import Tool
from northau.archs.tool.builtin.todo_write import todo_write
from northau.archs.tool.builtin.web_tool import web_search, web_read
from northau.archs.llm import LLMConfig

def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Test the quick start example."""
    print("Testing Northau Framework Quick Start Example")
    print("=" * 50)
    
    try:
        # Create tools
        script_dir = Path(__file__).parent
        print("Creating tools...")
        web_search_tool = Tool.from_yaml(str(script_dir / "tools/WebSearch.yaml"), binding=web_search)
        web_read_tool = Tool.from_yaml(str(script_dir / "tools/WebRead.yaml"), binding=web_read)
        todo_write_tool = Tool.from_yaml(str(script_dir / "tools/TodoWrite.tool.yaml"), binding=todo_write)
        print("✓ Tools created successfully")
        
        # Create sub-agents
        llm_config = LLMConfig(
            model=os.getenv("LLM_MODEL"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
        )
        print("\nCreating sub-agents...")
        
        deep_research_agent = create_agent(
            name="deep_research_agent",
            tools=[web_search_tool, web_read_tool, todo_write_tool],
            llm_config=llm_config,
            system_prompt="Date: {{date}}. You are a deep research agent. You are given a message and you need to search for the information that matches the message. Use the web_search and web_read tools to get the information. Wait for the web_read tool to finish before you continue your response. Before searching, you need to use todo_write_tool to write a todo list to track the research progress. When completing a task, you need to update the todo list. Todo list: {{current_todos}}",
        )
        print("✓ Sub-agents created successfully")

        print("\nTesting delegation with web research...")
        web_message = "What day is it today? and the stock price of Tencent on the day?"
        print(f"\nUser: {web_message}")
        print("\nAgent Response:")
        print("-" * 30)
        
        response = deep_research_agent.run(web_message, context={
            "date": get_date(),
        })
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