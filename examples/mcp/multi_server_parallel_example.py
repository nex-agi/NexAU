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

"""Example demonstrating parallel MCP server initialization with multiple servers."""

import os
import time

from nexau import Agent, AgentConfig
from nexau.archs.llm import LLMConfig


def main():
    """Demonstrate parallel initialization of multiple MCP servers."""
    print("🚀 Multi-Server Parallel MCP Initialization Example\n")

    # Configure multiple MCP servers to test parallel initialization
    mcp_servers = [
        # GitHub MCP server (stdio)
        {
            "name": "github",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
            },
            "timeout": 60,
        },
        # Filesystem MCP server (stdio)
        {
            "name": "filesystem",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "timeout": 60,
        },
        # Memory MCP server (stdio)
        {
            "name": "memory",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "timeout": 60,
        },
    ]

    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )

    print(f"📋 Configured {len(mcp_servers)} MCP Servers:")
    for server in mcp_servers:
        print(f"   - {server['name']} ({server['type']})")
        if server.get("command"):
            args = server.get("args", [])
            args_str = " ".join(args) if isinstance(args, list) else str(args)
            print(f"     Command: {server['command']} {args_str}")
    print()

    try:
        # Create an agent with multiple MCP servers
        print("🤖 Creating agent with multiple MCP servers...")
        print("   (Servers should initialize in PARALLEL)\n")

        start_time = time.time()

        agent_config = AgentConfig(
            name="multi_mcp_agent",
            system_prompt="""You are an AI agent with access to multiple services through MCP:

1. GitHub - for repository operations
2. Filesystem - for file operations in /tmp
3. Memory - for storing and retrieving key-value data

Use these tools to help users with their tasks.""",
            mcp_servers=mcp_servers,
            llm_config=llm_config,
        )
        agent = Agent(config=agent_config)

        init_time = time.time() - start_time

        print(f"✅ Agent created successfully in {init_time:.2f}s")
        print(f"   Agent name: {agent.config.name}")

        # Get all tools from config.tools (MCP tools are added here during init)
        all_tools = agent.config.tools
        print(f"   Total tools available: {len(all_tools)}")

        # Group tools by server (MCP tools have server_config attribute)
        tools_by_server: dict[str, list[str]] = {}
        for tool in all_tools:
            # MCPTool has server_config, regular tools don't
            server_config = getattr(tool, "server_config", None)
            server_name = server_config.name if server_config else "builtin"
            if server_name not in tools_by_server:
                tools_by_server[server_name] = []
            tools_by_server[server_name].append(tool.name)

        print("\n📦 Tools by server:")
        for server_name, tool_names in tools_by_server.items():
            print(f"   {server_name}: {len(tool_names)} tools")
            for tool_name in tool_names[:3]:  # Show first 3 tools
                print(f"      - {tool_name}")
            if len(tool_names) > 3:
                print(f"      ... and {len(tool_names) - 3} more")

        # Performance analysis
        print("\n⏱️  Performance Analysis:")
        print(f"   Total initialization time: {init_time:.2f}s")
        print(f"   Number of servers: {len(mcp_servers)}")
        print(f"   Average per server (if serial): {init_time / len(mcp_servers):.2f}s")

        if init_time < 30:  # If total time is less than 30s for 3 servers
            print("   ✅ Parallel initialization appears to be working!")
        else:
            print("   ⚠️  Initialization took longer than expected")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure Node.js/npm is installed")
        print("- Check network connectivity")
        print("- Verify environment variables are set")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
