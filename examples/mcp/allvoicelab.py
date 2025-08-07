#!/usr/bin/env python3
"""Example demonstrating AllvoiceLab MCP server integration with Northau agents."""

import os

from northau.archs.main_sub.agent import create_agent
from northau.archs.llm import LLMConfig


def main():
    """Demonstrate AllvoiceLab MCP"""
    
    # GitHub MCP server configuration
    # This uses the stdio protocol with the official GitHub MCP server
    mcp_servers = [
        {
            "name": "AllVoceLab",
            "type": "stdio",
            "command": "uvx",
            "args": ["allvoicelab-mcp"],
            "env": {
                "ALLVOICELAB_API_KEY": "xxxx",
                "ALLVOICELAB_API_DOMAIN": "https://api.allvoicelab.cn",
            },
            "timeout": 30
        }
    ]
    
    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )
    
    print("📋 Configured AllvoiceLab MCP Server:")
    for server in mcp_servers:
        print(f"   - {server['name']}")
        print(f"     Type: {server['type']}")
        if server.get('command'):
            args = server.get('args', [])
            args_str = ' '.join(args) if isinstance(args, list) else str(args)
            print(f"     Command: {server['command']} {args_str}")
        if server.get('url'):
            print(f"     URL: {server['url']}")
        if server.get('headers'):
            print(f"     Headers: {server['headers']}")
        print(f"     Timeout: {server.get('timeout', 30)}s")
    print()
    
    try:
        # Create an agent with Github MCP server
        print("🤖 Creating agent with AllvoiceLab MCP server...")
        agent = create_agent(
            name="allvoicelab_agent",
            system_prompt="""You are an AI agent with access to AllvoiceLab services through MCP.
    Methods	Brief description
text_to_speech	Convert text to speech
speech_to_speech	Convert audio to another voice while preserving the speech content
isolate_human_voice	Extract clean human voice by removing background noise and non-speech sounds
clone_voice	Create a custom voice profile by cloning from an audio sample
remove_subtitle	Remove hardcoded subtitles from a video using OCR
video_translation_dubbing	Translate and dub video speech into different languages ​​
text_translation	Translate a text file into another language
subtitle_extraction	Extract subtitles from a video using OCR""",
            mcp_servers=mcp_servers,
            llm_config=llm_config,
        )
        
        print("✅ Agent created successfully!")
        print(f"   Agent name: {agent.name}")
        print(f"   Total tools available: {len(agent.tools)}")
        
        # List available tools
        if agent.tools:
            print("\n🗺️  Available AllvoiceLab tools:")
            for tool in agent.tools:
                print(f"   - {tool.name}: {getattr(tool, 'description', 'No description')}")
        else:
            print("\n⚠️  No tools available")
        

        response = agent.run("帮我生成一段语音，内容是：大家好我是小北，我是一个AI助手，我可以帮助你完成各种任务。")
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