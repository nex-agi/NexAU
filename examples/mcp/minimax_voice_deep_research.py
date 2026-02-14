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

"""Example demonstrating MiniMax MCP server integration with NexAU agents."""

import logging
import os

from nexau import Agent, AgentConfig
from nexau.archs.llm import LLMConfig
from nexau.archs.main_sub.execution.hooks import create_tool_after_approve_hook
from nexau.archs.tool import Tool
from nexau.archs.tool.builtin.shell_tools import run_shell_command
from nexau.archs.tool.builtin.feishu import (
    get_feishu_chat_list,
    send_feishu_message,
    upload_feishu_file,
)
from nexau.archs.tool.builtin.web_tools import google_web_search, web_fetch

# Configure logging for hooks to ensure they appear
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
hook_logger = logging.getLogger("minimax_agent_hooks")
hook_logger.setLevel(logging.INFO)


def main():
    """Demonstrate MiniMax MCP"""

    # GitHub MCP server configuration
    # This uses the stdio protocol with the official GitHub MCP server
    mcp_servers = [
        {
            "name": "MiniMax",
            "type": "stdio",
            "command": "uvx",
            "args": [
                "minimax-mcp",
                "-y",
            ],
            "env": {
                "MINIMAX_API_KEY": os.getenv("MINIMAX_API_KEY"),
                "MINIMAX_MCP_BASE_PATH": os.getcwd(),
                "MINIMAX_API_HOST": "https://api.minimax.chat",
                "MINIMAX_API_RESOURCE_MODE": "url",
            },
            "timeout": 30,
        },
    ]

    src_dir = os.path.dirname(os.path.abspath(__file__))

    feishu_upload_file_tool = Tool.from_yaml(
        os.path.join(
            src_dir,
            "tools/feishu_upload_file.yaml",
        ),
        binding=upload_feishu_file,
    )
    feishu_send_message_tool = Tool.from_yaml(
        os.path.join(
            src_dir,
            "tools/feishu_send_message.yaml",
        ),
        binding=send_feishu_message,
    )
    get_feishu_chat_list_tool = Tool.from_yaml(
        os.path.join(
            src_dir,
            "tools/get_feishu_chat_list.yaml",
        ),
        binding=get_feishu_chat_list,
    )
    configured_bash_tool = Tool.from_yaml(
        os.path.join(
            src_dir,
            "tools/Bash.tool.yaml",
        ),
        binding=run_shell_command,
    )
    web_search_tool = Tool.from_yaml(
        os.path.join(
            src_dir,
            "tools/WebSearch.yaml",
        ),
        binding=google_web_search,
    )
    web_read_tool = Tool.from_yaml(
        os.path.join(
            src_dir,
            "tools/WebRead.yaml",
        ),
        binding=web_fetch,
    )
    tools = [
        feishu_upload_file_tool,
        feishu_send_message_tool,
        get_feishu_chat_list_tool,
        configured_bash_tool,
        web_search_tool,
        web_read_tool,
    ]

    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    print("📋 Configured MiniMax MCP Server:")
    for server in mcp_servers:
        print(f"   - {server['name']}")
        print(f"     Type: {server['type']}")
        if server.get("command"):
            args = server.get("args", [])
            args_str = " ".join(args) if isinstance(args, list) else str(args)
            print(f"     Command: {server['command']} {args_str}")
        if server.get("url"):
            print(f"     URL: {server['url']}")
        if server.get("headers"):
            print(f"     Headers: {server['headers']}")
        print(f"     Timeout: {server.get('timeout', 30)}s")
    print()

    try:
        # Create an agent with Github MCP server
        print("🤖 Creating agent with MiniMax MCP server...")
        agent_config = AgentConfig(
            name="minimax_agent",
            system_prompt="""You are an AI agent with access to MiniMax MCP, feishu tools, bash tool, and web search tool.

You can also use feishu tools to upload files and send messages to feishu.

You can use the following tools:
- feishu_upload_file
- feishu_send_message
- get_feishu_chat_list
- web_search
- web_read
- bash_tool
And the MCP tools from MiniMax.

You can use Bash tool to execute bash commands, e.g., convert a mp3 file to opus file:
```bash
ffmpeg -i SourceFile.mp3 -acodec libopus -ac 1 -ar 16000 TargetFile.opus
```

When use video generation of MiniMax:
1. Always use asynchronous mode for video generation
2. Monitor ongoing video generation tasks throughout the conversation until it is Success or FAILED.

Do not ask user to provide any information, just use the tools to get the information.
Note that if you response with multiple tools, they will run in parallel.
If you need to use the result of one tool in another tool,
you need to wait for the first tool to complete and then use the result in the second tool.

Today is {{date}}.
""",
            mcp_servers=mcp_servers,
            llm_config=llm_config,
            tools=tools,
            after_model_hooks=[create_tool_after_approve_hook("WebSearch")],
        )
        agent = Agent(config=agent_config)

        print("✅ Agent created successfully!")
        print(f"   Agent name: {agent.config.name}")
        print(f"   Total tools available: {len(agent.config.tools)}")

        # List available tools
        if agent.config.tools:
            print("\n🗺️  Available MiniMax tools:")
            for tool in agent.config.tools:
                print(
                    f"   - {tool.name}: {getattr(tool, 'description', 'No description')}",
                )
        else:
            print("\n⚠️  No tools available")
        response = agent.run(
            "WebSearch 和 WebRead 工具，搜索并整理今天的最新的 LLM 相关的资讯，然后用MiniMax的生成语音，并发语音消息到飞书的 bot测试群 里",
            context={
                "date": "2025年 8 月 27 日",
            },
        )

        print(response)

    except Exception as e:
        print(f"❌ Error creating agent with GitHub MCP server: {e}")
        print("\nNote: This error might occur if:")
        print("- Node.js/npm is not installed")
        print("- Network connectivity issues")
        print("- Invalid GitHub Personal Access Token")
        print("- GitHub server is not accessible")


if __name__ == "__main__":
    main()
