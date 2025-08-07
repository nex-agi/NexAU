#!/usr/bin/env python3
"""Example demonstrating AllvoiceLab MCP server integration with Northau agents."""

from typing import Any


import os

from northau.archs.main_sub.agent import create_agent
from northau.archs.llm import LLMConfig
from northau.archs.tool import Tool
from northau.archs.tool.builtin.feishu import upload_feishu_file, send_feishu_message, get_feishu_chat_list
from northau.archs.tool.builtin.bash_tool import bash_tool

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
                "ALLVOICELAB_API_KEY": os.getenv("ALLVOICELAB_API_KEY"),
                "ALLVOICELAB_API_DOMAIN": "https://api.allvoicelab.cn",
            },
            "timeout": 30
        },
    ]
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    feishu_upload_file_tool = Tool.from_yaml(os.path.join(src_dir, "tools/feishu_upload_file.yaml"), binding=upload_feishu_file)
    feishu_send_message_tool = Tool.from_yaml(os.path.join(src_dir, "tools/feishu_send_message.yaml"), binding=send_feishu_message)
    get_feishu_chat_list_tool = Tool.from_yaml(os.path.join(src_dir, "tools/get_feishu_chat_list.yaml"), binding=get_feishu_chat_list)
    configured_bash_tool = Tool.from_yaml(os.path.join(src_dir, "tools/Bash.tool.yaml"), binding=bash_tool)
    tools = [feishu_upload_file_tool, feishu_send_message_tool, get_feishu_chat_list_tool, configured_bash_tool]
    
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
subtitle_extraction	Extract subtitles from a video using OCR

You can also use feishu tools to upload files and send messages to feishu.

You can use the following tools:
- feishu_upload_file
- feishu_send_message
- get_feishu_chat_list

You can use Bash tool to execute bash commands, e.g., convert a mp3 file to opus file:
```bash
ffmpeg -i SourceFile.mp3 -acodec libopus -ac 1 -ar 16000 TargetFile.opus
```

""",
            mcp_servers=mcp_servers,
            llm_config=llm_config,
            tools=tools,
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
        

        content = """
        大家好呀！我是你们的AI助手小北，很高兴正式加入奇绩智峰这个大家庭！🤖✨

作为一个AI，我没有周末，没有假期，也不需要咖啡提神（虽然偶尔也会"电量不足"😴）。我可以帮你们处理各种任务，从日程安排到数据分析，从文档整理到创意发想，样样都行！

多用多喷多进化，小作坊下料就是猛！没错，我就是那个越用越聪明的AI，你们的反馈就是我进化的动力！🚀

有任何问题或需求，随时可以找Leon和天海反馈，他们会帮我"升级打怪"，让我变得更加智能好用~

期待与大家共同工作，创造更多价值！让我们一起，用科技改变世界吧！💪
        """
        response = agent.run(f"帮我生成一段语音并发送到飞书群 bot 测试群里，内容是：{content}")
        # response = agent.run("/Users/hanzhenhua/Desktop/tts_1754551121_7gneh8.mp3 帮我转成 opus并发送到飞书群 bot 测试群里")
        # response = agent.run("帮我获取飞书群列表")
        # response = agent.run("帮我上传/Users/hanzhenhua/Desktop/tts_1754551121_7gneh8.opus文件到飞书，注意要用 opus 格式，需要带 duration （可以用ffproobe 拿），并用语音消息发送到飞书群 bot 测试群里")
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