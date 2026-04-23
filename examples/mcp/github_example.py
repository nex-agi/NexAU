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

"""Example demonstrating Amap Maps MCP server integration with NexAU agents."""

import os

from nexau import Agent, AgentConfig
from nexau.archs.llm import LLMConfig


def main():
    """Demonstrate Github MCP"""

    # GitHub MCP server configuration
    # This uses the stdio protocol with the official GitHub MCP server
    mcp_servers = [
        {
            "name": "github",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
            },
            "timeout": 30,
        },
    ]

    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    print("📋 Configured Github MCP Server:")
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
        print("🤖 Creating agent with Github MCP server...")
        agent_config = AgentConfig(
            name="github_agent",
            system_prompt="""You are an AI agent with access to Github services through MCP.

You can use Github tools to:
Toolset	Description
context	Strongly recommended: Tools that provide context about the current user and GitHub context you are operating in
actions	GitHub Actions workflows and CI/CD operations
code_security	Code security related tools, such as GitHub Code Scanning
dependabot	Dependabot tools
discussions	GitHub Discussions related tools
experiments	Experimental features that are not considered stable yet
gists	GitHub Gist related tools
issues	GitHub Issues related tools
notifications	GitHub Notifications related tools
orgs	GitHub Organization related tools
pull_requests	GitHub Pull Request related tools
repos	GitHub Repository related tools
secret_protection	Secret protection related tools, such as GitHub Secret Scanning
users	GitHub User related tools

When using Github tools, always provide clear and helpful information to users.
Explain what you're doing and provide context for the results.""",
            mcp_servers=mcp_servers,
            llm_config=llm_config,
        )
        agent = Agent(config=agent_config)

        print("✅ Agent created successfully!")
        print(f"   Agent name: {agent.config.name}")
        all_tools = agent.tool_registry
        print(f"   Total tools available: {len(all_tools)}")

        # List available tools (tool_registry includes MCP tools)
        if all_tools:
            print("\n🗺️  Available Github tools:")
            for tool in all_tools.values():
                print(
                    f"   - {tool.name}: {getattr(tool, 'description', 'No description')}",
                )
        else:
            print("\n⚠️  No tools available")

        response = agent.run(
            message="https://github.com/nex-agi/NexAU/tree/main 的代码结构是什么样的？",
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
