# RFC-0001 验证脚本: Agent 中断时状态持久化
#
# 场景：
#   1. 创建一个带 sleep_tool 的 Agent（流式输出）
#   2. 让模型调用 sleep_tool 20 次
#   3. 在第 5 次工具调用完成后发送 stop
#   4. 中断后，用新指令问模型已经调用了多少次
#
# 用法：
#   uv run python examples/test_interrupt.py          # 直接调用 agent.stop()
#   uv run python examples/test_interrupt.py direct    # 同上
#   uv run python examples/test_interrupt.py http      # 通过 HTTP transport POST /stop

import asyncio
import json
import logging
from typing import Any
import sys
import threading
import time

import httpx

from nexau import Agent, AgentConfig
from nexau.archs.llm.llm_aggregators.events import Event, ToolCallResultEvent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
)
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.tool import Tool
from nexau.archs.transports.http import HTTPConfig, SSETransportServer

logging.basicConfig(level=logging.WARNING)

# ── 工具定义 ──────────────────────────────────────────────

tool_call_count = 0


def sleep_tool(seconds: int = 1) -> str:
    """Sleep for the given number of seconds and return a confirmation."""
    global tool_call_count
    tool_call_count += 1
    current = tool_call_count
    print(f"  🔧 sleep_tool 第 {current} 次调用，休眠 {seconds}s ...")
    time.sleep(seconds)
    return f"Slept for {seconds} second(s). This is call #{current}."


sleep = Tool(
    name="sleep_tool",
    description=(
        "Sleep for a given number of seconds. "
        "Use this tool when asked to sleep or wait. "
        "IMPORTANT: You can only call this tool ONCE per response. "
        "Wait for the result before calling it again."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "seconds": {
                "type": "integer",
                "description": "Number of seconds to sleep",
                "default": 1,
            },
        },
        "required": [],
    },
    implementation=sleep_tool,
    disable_parallel=True,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "CRITICAL RULE: You must NEVER call more than ONE tool per response. "
    "After calling a tool, you MUST stop and wait for the result. "
    "Only after receiving the result can you call the tool again in your next response. "
    "Calling multiple tools in a single response is STRICTLY FORBIDDEN."
)

USER_MESSAGE = "请帮我调用 sleep_tool 共 20 次，每次休眠 1 秒。规则：每次回复只能调用一次 sleep_tool，等收到结果后再调用下一次。"

FOLLOWUP_MESSAGE = "你之前调用了多少次 sleep_tool？请回顾对话历史并告诉我具体次数。"


# ── Mode 1: 直接调用 agent.stop() ─────────────────────────


async def main_direct() -> None:
    """Test stop via direct agent.stop() call."""
    global tool_call_count

    print("=" * 60)
    print("RFC-0001 验证: Agent 中断时状态持久化 (direct mode)")
    print("=" * 60)

    # 事件回调
    tool_result_count = 0
    interrupt_triggered = threading.Event()

    def on_event(event: Event) -> None:
        nonlocal tool_result_count
        if isinstance(event, ToolCallResultEvent):
            tool_result_count += 1
            print(f"  📩 收到第 {tool_result_count} 次工具调用结果")
            if tool_result_count >= 5:
                print("  ⚡ 达到 5 次，准备发送 stop ...")
                interrupt_triggered.set()

    # 1. 创建 Agent（流式输出）
    middleware = AgentEventsMiddleware(
        session_id="interrupt_test",
        on_event=on_event,
    )

    config = AgentConfig(
        name="interrupt_test_agent",
        system_prompt=SYSTEM_PROMPT,
        llm_config=LLMConfig(stream=True),
        tools=[sleep],
        middlewares=[middleware],
        max_iterations=25,
    )

    agent = Agent(config=config)

    # 2. 第一轮：让模型调用 sleep_tool 20 次
    print("\n📤 第一轮指令: 请调用 sleep_tool 20 次，每次休眠 1 秒")
    print("-" * 60)

    async def run_agent() -> str:
        resp = await agent.run_async(message=USER_MESSAGE)
        return resp if isinstance(resp, str) else resp[0]

    agent_task = asyncio.create_task(run_agent())

    # 等待第 5 次工具调用完成
    while not interrupt_triggered.is_set():
        await asyncio.sleep(0.2)
    await asyncio.sleep(0.5)

    # 3. 发送 stop
    print("\n🛑 发送 stop(force=True) ...")
    result = await agent.stop(force=True)
    print(f"  ✅ stop 完成")
    print(f"  📊 stop_reason = {result.stop_reason.name}")
    print(f"  📊 历史消息数 = {len(result.messages)}")
    print(f"  📊 实际工具调用次数 = {tool_call_count}")

    # 等待 agent_task 完成
    try:
        response = await asyncio.wait_for(agent_task, timeout=5.0)
        print(f"\n📥 第一轮响应: {response[:200]}...")
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
        print(f"\n📥 第一轮因中断结束: {type(e).__name__}")

    # 4. 第二轮：问模型已经调用了多少次
    print("\n" + "=" * 60)
    print("📤 第二轮指令: 你之前调用了多少次 sleep_tool？")
    print("-" * 60)

    response2_raw = await agent.run_async(message=FOLLOWUP_MESSAGE)
    response2 = response2_raw if isinstance(response2_raw, str) else response2_raw[0]
    print(f"\n📥 第二轮响应:\n{response2}")

    # 5. 验证结果
    _print_summary(tool_call_count, tool_result_count, len(result.messages), response2)

    # 清理
    await agent.stop(force=True)


# ── Mode 2: 通过 HTTP transport POST /stop ────────────────

HTTP_PORT = 18765
BASE_URL = f"http://127.0.0.1:{HTTP_PORT}"
SESSION_ID = "http_stop_test"
USER_ID = "test_user"


async def main_http() -> None:
    """Test stop via HTTP transport POST /stop endpoint."""
    global tool_call_count

    print("=" * 60)
    print("RFC-0001 验证: Agent 中断时状态持久化 (http mode)")
    print("=" * 60)

    # 1. 创建 SSE Transport Server
    engine = InMemoryDatabaseEngine()
    agent_config = AgentConfig(
        name="interrupt_test_agent",
        system_prompt=SYSTEM_PROMPT,
        llm_config=LLMConfig(stream=True),
        tools=[sleep],
        max_iterations=25,
    )

    server = SSETransportServer(
        engine=engine,
        config=HTTPConfig(port=HTTP_PORT),
        default_agent_config=agent_config,
    )

    # 2. 启动 uvicorn 后台线程
    import uvicorn

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(
            server.app,
            host="127.0.0.1",
            port=HTTP_PORT,
            log_level="warning",
        ),
        daemon=True,
    )
    server_thread.start()

    # 等待服务器就绪
    await _wait_for_server(BASE_URL)
    print("✅ HTTP 服务器已启动")

    # 3. 发送流式请求并在第 5 次工具调用后 stop
    tool_result_count = 0
    stop_result_data: dict[str, Any] | None = None

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        # 启动流式请求（后台任务）
        stream_done = asyncio.Event()

        async def consume_stream() -> None:
            nonlocal tool_result_count
            try:
                async with client.stream(
                    "POST",
                    "/stream",
                    json={
                        "messages": USER_MESSAGE,
                        "user_id": USER_ID,
                        "session_id": SESSION_ID,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        event_type = data.get("type", "")
                        if event_type == "TOOL_CALL_RESULT":
                            tool_result_count += 1
                            print(f"  📩 [SSE] 收到第 {tool_result_count} 次工具调用结果")
            except httpx.RemoteProtocolError:
                # 服务端因 stop 关闭连接
                pass
            except Exception as e:
                print(f"  ⚠️ 流式请求异常: {type(e).__name__}: {e}")
            finally:
                stream_done.set()

        stream_task = asyncio.create_task(consume_stream())

        # 等待第 5 次工具调用完成
        print("\n📤 第一轮指令 (via POST /stream): 请调用 sleep_tool 20 次")
        print("-" * 60)

        while tool_result_count < 5 and not stream_done.is_set():
            await asyncio.sleep(0.2)

        if tool_result_count >= 5:
            await asyncio.sleep(0.5)  # 确保第 5 次结果已处理

            # 4. 发送 POST /stop
            print(f"\n🛑 发送 POST /stop (force=True) ...")
            stop_resp = await client.post(
                "/stop",
                json={
                    "user_id": USER_ID,
                    "session_id": SESSION_ID,
                    "force": True,
                    "timeout": 30.0,
                },
            )
            stop_result_data = stop_resp.json()
            print(f"  ✅ /stop 响应: {stop_result_data}")
        else:
            print("  ⚠️ 流式请求提前结束，未达到 5 次工具调用")

        # 等待流式请求结束
        await asyncio.wait_for(stream_task, timeout=10.0)

        # 5. 第二轮：通过 POST /query 验证上下文恢复
        print("\n" + "=" * 60)
        print("📤 第二轮指令 (via POST /query): 你之前调用了多少次 sleep_tool？")
        print("-" * 60)

        query_resp = await client.post(
            "/query",
            json={
                "messages": FOLLOWUP_MESSAGE,
                "user_id": USER_ID,
                "session_id": SESSION_ID,
            },
        )
        query_data = query_resp.json()
        response2 = query_data.get("response", "")
        print(f"\n📥 第二轮响应:\n{response2}")

    # 6. 验证结果
    message_count = int(stop_result_data.get("message_count", 0)) if stop_result_data else 0
    _print_summary(tool_call_count, tool_result_count, message_count, response2)


# ── 辅助函数 ──────────────────────────────────────────────


async def _wait_for_server(base_url: str, retries: int = 20) -> None:
    """Poll server until it responds to health check."""
    async with httpx.AsyncClient() as client:
        for _ in range(retries):
            try:
                resp = await client.get(f"{base_url}/docs")
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.3)
    raise RuntimeError(f"Server at {base_url} did not start in time")


def _print_summary(calls: int, results: int, message_count: int, response2: str) -> None:
    """Print verification summary."""
    print("\n" + "=" * 60)
    print("📊 验证结果:")
    print(f"  - 实际工具调用次数: {calls}")
    print(f"  - 工具结果事件数: {results}")
    print(f"  - 中断后历史消息数: {message_count}")
    print(f"  - 第二轮能恢复上下文: {'是' if response2 else '否'}")
    print("=" * 60)


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "direct"
    if mode == "http":
        asyncio.run(main_http())
    else:
        asyncio.run(main_direct())
