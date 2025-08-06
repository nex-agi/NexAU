#!/usr/bin/env python3
"""Test the quick start example from the specification."""

import os
from datetime import datetime

from northau.archs.main_sub import create_agent
from northau.archs.tool import Tool
from northau.archs.tool.builtin.file_tools.file_edit_tool import file_edit_tool
from northau.archs.tool.builtin.file_tools.file_read_tool import file_read_tool
from northau.archs.tool.builtin.file_tools.file_write_tool import file_write_tool
from northau.archs.tool.builtin.file_tools.grep_tool import grep_tool
from northau.archs.tool.builtin.file_tools.glob_tool import glob_tool
from northau.archs.tool.builtin.todo_write import todo_write
from northau.archs.tool.builtin.web_tool import web_search, web_read
from northau.archs.tool.builtin.bash_tool import bash_tool
from northau.archs.tool.builtin.ls_tool import ls_tool
from northau.archs.tool.builtin.multiedit_tool import multiedit_tool
from northau.archs.llm import LLMConfig

def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Test the quick start example."""
    print("Testing Northau Framework Quick Start Example")
    print("=" * 50)
    
    try:
        # Create tools
        print("Creating tools...")
        # Bind YAML configurations to existing tools
        grep_tool_configured = Tool.from_yaml("tools/claude_code/Grep.tool.yaml", binding=grep_tool)
        glob_tool_configured = Tool.from_yaml("tools/claude_code/Glob.tool.yaml", binding=glob_tool)
        todo_write_tool = Tool.from_yaml("tools/claude_code/TodoWrite.tool.yaml", binding=todo_write)
        web_search_tool = Tool.from_yaml("tools/claude_code/WebSearch.tool.yaml", binding=web_search)
        web_read_tool = Tool.from_yaml("tools/claude_code/WebFetch.tool.yaml", binding=web_read)
        file_read_tool_configured = Tool.from_yaml("tools/claude_code/Read.tool.yaml", binding=file_read_tool)
        file_write_tool_configured = Tool.from_yaml("tools/claude_code/Write.tool.yaml", binding=file_write_tool)
        file_edit_tool_configured = Tool.from_yaml("tools/claude_code/Edit.tool.yaml", binding=file_edit_tool)
        bash_tool_configured = Tool.from_yaml("tools/claude_code/Bash.tool.yaml", binding=bash_tool)
        ls_tool_configured = Tool.from_yaml("tools/claude_code/Ls.tool.yaml", binding=ls_tool)
        multiedit_tool_configured = Tool.from_yaml("tools/claude_code/MultiEdit.tool.yaml", binding=multiedit_tool)
        
        print("✓ Tools created successfully")
        
        # Create sub-agents
        llm_config = LLMConfig(
            model=os.getenv("LLM_MODEL"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
        )
        print("\nCreating sub-agents...")
        
        claude_code_agent = create_agent(
            name="claude_code_agent",
            tools=[
                web_search_tool, 
                web_read_tool, 
                todo_write_tool, 
                grep_tool_configured, 
                glob_tool_configured,
                file_read_tool_configured,
                file_write_tool_configured,
                file_edit_tool_configured,
                bash_tool_configured,
                ls_tool_configured,
                multiedit_tool_configured
            ],
            llm_config=llm_config,
            system_prompt=open("agents/claude_code/system-workflow.md").read(),
        )
        print("✓ Sub-agents created successfully")

        print("\nTesting delegation with web research...")
        web_message = "Create a simple Python web server that serves static files"
        print(f"\nUser: {web_message}")
        print("\nAgent Response:")
        print("-" * 30)
        
        response = claude_code_agent.run(web_message, context={
            "env_content": {
                "date": get_date(),
                "username": os.getenv("USER"),
                "pwd": os.getcwd(),
            },
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