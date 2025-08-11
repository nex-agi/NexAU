#!/usr/bin/env python3
"""Example demonstrating MiniMax MCP server integration with Northau agents."""

from typing import Any


import os

from northau.archs.main_sub.agent import create_agent
from northau.archs.llm import LLMConfig
from northau.archs.tool import Tool
from northau.archs.tool.builtin.feishu import upload_feishu_file, send_feishu_message, get_feishu_chat_list
from northau.archs.tool.builtin.bash_tool import bash_tool
from northau.archs.tool.builtin.web_tool import web_search, web_read



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
                "-y"
            ],
            "env": {
                "MINIMAX_API_KEY": os.getenv("MINIMAX_API_KEY"),
                "MINIMAX_MCP_BASE_PATH": os.getcwd(),
                "MINIMAX_API_HOST": "https://api.minimax.chat",
                "MINIMAX_API_RESOURCE_MODE": "url"
            },
            "timeout": 30
        },
    ]

    src_dir = os.path.dirname(os.path.abspath(__file__))

    feishu_upload_file_tool = Tool.from_yaml(os.path.join(
        src_dir, "tools/feishu_upload_file.yaml"), binding=upload_feishu_file)
    feishu_send_message_tool = Tool.from_yaml(os.path.join(
        src_dir, "tools/feishu_send_message.yaml"), binding=send_feishu_message)
    get_feishu_chat_list_tool = Tool.from_yaml(os.path.join(
        src_dir, "tools/get_feishu_chat_list.yaml"), binding=get_feishu_chat_list)
    configured_bash_tool = Tool.from_yaml(os.path.join(
        src_dir, "tools/Bash.tool.yaml"), binding=bash_tool)
    web_search_tool = Tool.from_yaml(os.path.join(
        src_dir, "tools/WebSearch.yaml"), binding=web_search)
    web_read_tool = Tool.from_yaml(os.path.join(
        src_dir, "tools/WebRead.yaml"), binding=web_read)
    tools = [feishu_upload_file_tool, feishu_send_message_tool,
             get_feishu_chat_list_tool, configured_bash_tool, web_search_tool, web_read_tool]

    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    print("📋 Configured MiniMax MCP Server:")
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
        print("🤖 Creating agent with MiniMax MCP server...")
        agent = create_agent(
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
            print("\n🗺️  Available MiniMax tools:")
            for tool in agent.tools:
                print(
                    f"   - {tool.name}: {getattr(tool, 'description', 'No description')}")
        else:
            print("\n⚠️  No tools available")

#         content = """
#         早安呀，亲爱的小伙伴们！我是小北~ 

# 【今日上海天气小播报 - 8月8日】

# 今天上海的天空依旧是多云飘飘的呢~ 不过呢，小北要提醒大家，今天真的是热辣辣的一天哦！

# 温度情况：最高温36℃，最低温30℃
#       当前温度33℃，但体感温度高达39.9℃！（是不是感觉自己在桑拿房里呀~）

# 湿度风力：相对湿度61%，西南风小于3级（约2.9米/秒）

# 空气质量：AQI 57，等级"良"，首要污染物PM2.5

# 降水概率：0%（今天不用带伞啦~）

# 特别注意：上海中心气象台已发布高温黄色预警！预计全市最高气温将超过35℃

# 小北温馨提示：
# 1. 今天外出记得防晒、补水、戴墨镜！
# 2. 尽量避开中午高温时段户外活动
# 3. 多喝水，少吃辛辣食物
# 4. 办公室的小伙伴们别忘了适当起身活动，避免久坐哦~

# 祝大家今天工作顺利，保持清凉好心情！
# —— 爱你们的小北
#         """
#         response = agent.run(f"帮我生成一段语音并发送到飞书群 bot测试群 里，内容是：{content}")
        response = agent.run("今天是 2025年 8 月 10 日，现使用 WebSearch 和 WebRead 工具，搜索并整理今天的最新的 LLM 相关的资讯，然后用MiniMax的生成语音功能制造一个语音，并发语音消息到飞书的 bot测试群 里")
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
