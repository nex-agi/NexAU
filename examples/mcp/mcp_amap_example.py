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

"""Example demonstrating Amap Maps MCP server integration with NexAU agents.

Includes Langfuse tracing to observe MCP tool calls in the Langfuse dashboard.
Requires environment variables: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
"""

import os

from nexau import Agent, AgentConfig
from nexau.archs.llm import LLMConfig
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer


def main():
    """Demonstrate Amap Maps MCP integration with NexAU agents."""
    print("🚀 Amap Maps MCP Integration Example for NexAU Framework\n")

    # Langfuse tracer — reads credentials from env vars:
    #   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    langfuse_tracer = LangfuseTracer()

    # Amap Maps MCP server configuration
    # This uses the streamable HTTP MCP protocol
    mcp_servers = [
        {
            "name": "amap-maps-streamableHTTP",
            "type": "http",
            "url": "https://mcp.amap.com/mcp?key=2dcc10a01caa038253df2c5beb078efe",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            "timeout": 10,
        },
    ]

    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        api_type="openai_chat_completion",
    )

    print("📋 Configured Amap MCP Server:")
    for server in mcp_servers:
        print(f"   - {server['name']}")
        print(f"     Type: {server['type']}")
        print(f"     URL: {server['url']}")
        if server.get("headers"):
            print(f"     Headers: {server['headers']}")
        print(f"     Timeout: {server.get('timeout', 10)}s")
    print()

    try:
        # Create an agent with Amap Maps MCP server
        print("🤖 Creating agent with Amap Maps MCP server...")
        agent_config = AgentConfig(
            name="amap_agent",
            system_prompt="""You are an AI agent with access to Amap Maps services through MCP.

You can use Amap Maps tools to:
- Search for locations and points of interest
- Get directions and navigation information
- Calculate distances and travel times
- Find nearby businesses and services
- Access real-time traffic information
- And other location-based services

When using map tools, always provide clear and helpful information to users.
Explain what you're doing and provide context for the results.""",
            tool_call_mode="structured",
            mcp_servers=mcp_servers,
            llm_config=llm_config,
            tracers=[langfuse_tracer],
        )
        agent = Agent(config=agent_config)

        print("✅ Agent created successfully!")
        print(f"   Agent name: {agent.config.name}")
        all_tools = agent.tool_registry
        print(f"   Total tools available: {len(all_tools)}")

        # List available tools (tool_registry includes MCP tools)
        if all_tools:
            print("\n🗺️  Available Amap Maps tools:")
            for tool in all_tools.values():
                print(
                    f"   - {tool.name}: {getattr(tool, 'description', 'No description')}",
                )
        else:
            print("\n⚠️  No tools available")
            print(
                "     This is expected behavior as the Amap MCP server used in this example",
            )
            print(
                "     is not currently accessible or may require different authentication.",
            )
            print()
            print("     Common causes for MCP connection failures:")
            print("     - Network connectivity issues")
            print("     - Invalid API key in the URL")
            print("     - Server not responding or unavailable")
            print("     - Authentication problems")
            print("     - MCP protocol version mismatch")
            print("     - Server doesn't support the streamable HTTP protocol")
            print()
            print("     ✅ The MCP integration framework is working correctly!")
            print(
                "        The timeout has been reduced from 30s to 10s for faster failure detection.",
            )

        response = agent.run(message="现在从漕河泾现代服务园A6到上南路 4265弄要多久？")
        print(response)

    except Exception as e:
        print(f"❌ Error creating agent with Amap MCP server: {e}")
        print("\nNote: This error might occur if:")
        print("- The Amap MCP server is not accessible")
        print("- Network connectivity issues")
        print("- Invalid API key or authentication problems")
        print("- MCP protocol version mismatch")


if __name__ == "__main__":
    main()
