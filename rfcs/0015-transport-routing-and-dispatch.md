# RFC-0015: 传输层路由与请求调度

- **状态**: implemented
- **优先级**: P2
- **标签**: `transport`, `http`, `sse`, `stdio`, `jsonrpc`, `architecture`
- **影响服务**: `nexau/archs/transports/base.py`, `nexau/archs/transports/http/sse_server.py`, `nexau/archs/transports/http/sse_client.py`, `nexau/archs/transports/http/team_routes.py`, `nexau/archs/transports/http/team_registry.py`, `nexau/archs/transports/http/models.py`, `nexau/archs/transports/http/config.py`, `nexau/archs/transports/stdio/stdio_transport.py`, `nexau/archs/transports/stdio/config.py`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

NexAU 把"对外暴露 agent"抽象成 `TransportBase` 单一基类：HTTP+SSE / Stdio (JSON-RPC 2.0-stream) 两种已实现，WebSocket / gRPC 占位。所有 transport 共享同一套请求路径——`handle_request` (同步) 与 `handle_streaming_request` (流式) 由基类提供，子类只负责协议解码、路由分发与响应编码。基类内部统一处理：(1) `session_id` 自动生成、(2) `AgentEventsMiddleware` 注入、(3) `_recursively_apply_middlewares` 把 middleware 与 `enable_stream` flag 深拷贝下沉到所有 sub-agents、(4) `Agent.__init__` 在 `asyncio.to_thread` 中执行避免事件循环嵌套、(5) `_running_agents` 字典登记运行中 agent 供 `/stop` 端点查找。本 RFC 描述这条管线、SSE 与 JSON-RPC 两种事件帧编码、`/stream` `/query` `/stop` 三个 HTTP 端点与 `agent.stream` `agent.query` `agent.stop` 三个 stdio 方法的对应关系，以及 transport 与 RFC-0010 (LLM 聚合器事件流) 的边界。

## 动机

agent 必须能被多种宿主消费：浏览器走 SSE、CLI 走 stdio、IDE 插件走 WebSocket、跨进程走 gRPC。设计问题：

1. **session 管理逻辑不能各 transport 复制**：`session_id` 生成、agent 注册、并发 lock、stop 端点这些都跨 transport 通用；分散在每个 transport 子类里会出现行为不一致（如 SSE 自动生成 session 但 stdio 要求手填）。
2. **streaming vs sync 不能让用户选错入口**：HTTP `/stream` 流式、`/query` 同步；stdio `agent.stream` 流式、`agent.query` 同步。两套入口必须语义对称——都接受同一份 `AgentRequest` 模型、都返回同样事件类型。
3. **middleware 必须递归下沉到 sub-agents**：`AgentEventsMiddleware` 只挂在 root agent 上则 sub-agent 的事件丢失。但 deep-copy `AgentConfig` 时若 tracer / middleware 含 thread lock / httpx client 会 pickling fail。
4. **`asyncio` 事件循环嵌套**：`Agent.__init__` 内部用 `asyncio.run()` 做同步 session 初始化；如果在 async transport handler 里直接 `Agent(...)` 会 RuntimeError "asyncio.run() cannot be called from a running event loop"。
5. **stdio 必须保护 stdout**：JSON-RPC 协议要求 stdout 只输出 JSON 帧；任何第三方库 `print()` 会污染解析。
6. **stop 必须能找到运行中的 agent**：`/stop` 端点收到 `(user_id, session_id, agent_id)` 后必须定位到正在跑的 `Agent` 实例并调用 `agent.stop()`，而 agent 实例不持久化。

把这些抽象到 `TransportBase` 后，新增 transport 只需实现 `start()` / `stop()` 与协议层编解码——业务逻辑全部复用。

## 设计

### 概述

```
                    ┌─────────────────────────────────────────────┐
                    │ TransportBase[TConfig] (base.py:32)         │
                    │  - _session_manager (base.py:80)            │
                    │  - _running_agents (base.py:88)             │
                    │  - _recursively_apply_middlewares           │
                    │    (base.py:91-92)                          │
                    │  + handle_request (base.py:154)             │
                    │  + handle_streaming_request (base.py:228)   │
                    │  + handle_stop_request (base.py:319)        │
                    └─────┬──────────────────────────┬────────────┘
                          │                          │
                          ▼                          ▼
        ┌──────────────────────────────┐ ┌──────────────────────────┐
        │ SSETransportServer           │ │ StdioTransport           │
        │ (sse_server.py:35)           │ │ (stdio_transport.py:57)  │
        │  POST /stream  →             │ │  agent.stream  →         │
        │    handle_streaming_request  │ │    handle_streaming_     │
        │  POST /query   →             │ │      request             │
        │    handle_request            │ │  agent.query   →         │
        │  POST /stop    →             │ │    handle_request        │
        │    handle_stop_request       │ │  agent.stop    →         │
        │                              │ │    handle_stop_request   │
        └──────────────────────────────┘ └──────────────────────────┘
                          │                          │
                          ▼                          ▼
                 ┌────────────────────────────────────────────┐
                 │ AgentRequest (http/models.py:17)           │
                 │  messages / user_id / session_id /         │
                 │  context / variables                       │
                 └────────────────────────────────────────────┘
```

### 详细设计

#### 1. `TransportBase` 单一基类

`TransportBase[TTransportConfig]`（base.py:32）是泛型 ABC，子类指定 `TTransportConfig` 类型（如 `HTTPConfig` / `StdioConfig`）。`__init__`（base.py:57-89）接收：

```python
engine: DatabaseEngine            # 共享给所有 transport
config: TTransportConfig          # 协议特定
default_agent_config: AgentConfig # 用户没传 agent_config 时的兜底
lock_ttl: float = 30.0            # SessionManager lock 配置
heartbeat_interval: float = 10.0  # SessionManager 心跳
```

内部固定建立两个共享对象：

- `self._session_manager = SessionManager(engine=..., lock_ttl=..., heartbeat_interval=...)`（base.py:80-84）——所有请求复用同一 SessionManager 与底层 DB。
- `self._running_agents: dict[(user_id, session_id, agent_id), Agent]`（base.py:88-89）——运行表，由 `asyncio.Lock` 保护，供 `/stop` 端点 O(1) 查找。

抽象方法只有两个：`start()`（base.py:146-148）与 `stop()`（base.py:150-152）。实现 transport 只需写协议解码 + 调用基类的三个 handler。

#### 2. `_recursively_apply_middlewares`：把 middleware 下沉到所有 sub-agents

代码（base.py:91-144，`@staticmethod` 装饰器在 L91，def 在 L92）做四件事：

1. **保留 stateful 对象**（base.py:108-110）：`tracers / resolved_tracer / middlewares` 含 thread lock 与 httpx client，无法 pickle。先存起来。
2. **临时清空 + 深拷贝**（base.py:112-118）：把 stateful 字段清掉，深拷贝整个 config，再恢复原对象。
3. **把 stateful 对象赋给 copy（浅引用）**（base.py:126-128）：tracer/middleware 应该被原版与 copy 共享，不能复制——thread lock 复制后语义不一致。
4. **追加新 middleware + 可选 enable_stream**（base.py:131-135）：把传入的 `events_mw` 加进 copy 的 middleware 列表；若 `enable_stream=True` 则 `cfg_copy.llm_config.stream = True`。
5. **递归到 sub_agents**（base.py:137-142）：每个 sub-agent 的 config 同样走一遍这个流程，确保子 agent 也有 events middleware。

**关键约束**：必须 deep-copy AgentConfig 而不是 mutate 原对象——因为 transport 是长期运行的服务，多个并发请求共享 `default_agent_config`，mutate 会污染下个请求。但 deep-copy 不能直接做（pickling fail），所以走"临时摘出 + copy + 装回"三步。

#### 3. `handle_request` / `handle_streaming_request` 主流程

两者结构对称（base.py:154-226 / 228-317）：

```python
# 1. 自动生成 session_id（base.py:184-188 / 258-262）
if session_id is None:
    session_id = generate_session_id()

# 2. 注入 AgentEventsMiddleware（流式时绑定 event_queue.put_nowait）
events_mw = AgentEventsMiddleware(session_id=session_id, on_event=...)
config_with_middlewares = self._recursively_apply_middlewares(
    agent_config or self._default_agent_config, events_mw,
    enable_stream=True,  # 仅 streaming 设为 True (base.py:264-271)
)

# 3. asyncio.to_thread 创建 agent（base.py:197-208 / 273-284）
def create_agent() -> Agent:
    return Agent(config=..., session_manager=..., user_id=..., session_id=..., variables=...)
agent = await asyncio.to_thread(create_agent)

# 4. 注册到运行表（base.py:210-213 / 286-289）
agent_key = (user_id, session_id, agent.agent_id)
async with self._running_agents_lock:
    self._running_agents[agent_key] = agent

# 5. 跑 agent / 流式时还要起 background task + drain queue（base.py:217 / 291-310）

# 6. finally 清出运行表（base.py:218-221 / 311-314）
```

**事件循环嵌套防御**（base.py:197-208）：`Agent.__init__` 内部会调 `asyncio.run()` 做同步 session 初始化（持久化 / 加载历史等）。直接在 async handler 里 `Agent(...)` 会 `RuntimeError`。`asyncio.to_thread` 把它扔到线程池，线程内可独立 `asyncio.run`。

**流式 drain 模式**（base.py:296-310）：

```python
agent_task = asyncio.create_task(run_agent())
try:
    while not agent_task.done():
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
            yield event
        except TimeoutError:
            continue

    # Drain remaining events after agent completes
    while not event_queue.empty():
        yield event_queue.get_nowait()
    await agent_task  # 抛 agent 内异常
```

`timeout=0.1` 而非 `await event_queue.get()` 是为了周期检查 `agent_task.done()`——agent 完成时不会主动 close queue，必须轮询。

#### 4. `handle_stop_request`：运行表查找 + 优雅停机

代码（base.py:319-376）：

```python
async with self._running_agents_lock:
    if agent_id:
        agent = self._running_agents.get((user_id, session_id, agent_id))
    else:
        # 未指定 agent_id 时取该 session 下任意 agent
        for key, running_agent in self._running_agents.items():
            if key[0] == user_id and key[1] == session_id:
                agent = running_agent
                break

if agent is None:
    raise ValueError(...)
result = await agent.stop(force=force, timeout=timeout)  # base.py:370
```

**`agent_id=None` 的 fallback 语义**（base.py:360-365）：取该 session 下第一个匹配的 agent。在单 root-agent 场景下这正是用户想要的；在多并发 agent 场景下用户必须指定 `agent_id`。

#### 5. SSE Transport（HTTP）

`SSETransportServer`（sse_server.py:35）是 `TransportBase[HTTPConfig]` 的实现，封装 FastAPI app。

**4 个端点**（sse_server.py:154-245）：

| Endpoint | Method | 调用 |
|----------|--------|------|
| `/` | GET | service info（sse_server.py:161） |
| `/health` | GET | health check（sse_server.py:177） |
| `/stream` | POST | `_stream_agent_response` → `handle_streaming_request`（sse_server.py:184-202） |
| `/query` | POST | `handle_request` 直接返回 JSON（sse_server.py:204-219） |
| `/stop` | POST | `handle_stop_request` 返回 `StopResponse`（sse_server.py:222-245） |

**SSE 帧编码**（sse_server.py:267）：`yield f"data: {event.model_dump_json()}\n\n"`——标准 SSE `data:` 前缀 + 双换行结束帧。

**team 路由 mount**（sse_server.py:137-150）：在 `_create_app` 末尾 mount `team_router`（RFC-0002 团队模式）。这让 SSE server 同时承担"单 agent 流式"与"team 多 agent 流式"两种模式，端点前缀区分。

**runtime_context 注入**（sse_server.py:97-101）：服务器启动时构造 `working_directory / username / date`，但**当前未在 handler 中传入** `Agent.run_async`——这是个未连通的桥，未来 RFC 或 patch 完成。

#### 6. Stdio Transport（JSON-RPC 2.0-stream）

`StdioTransport`（stdio_transport.py:57）是 `TransportBase[StdioConfig]` 的实现。

**自定义 JSON-RPC 变体**（stdio_transport.py:24）：

```python
JSONRPC_VERSION: Literal["2.0-stream"] = "2.0-stream"
```

不是标准 JSON-RPC 2.0；扩展是支持"一个请求 → 多帧响应（事件帧）+ 终结帧（success/error）"的流式语义。

**4 个帧类型**（stdio_transport.py:27-54）：

| 模型 | 含义 |
|------|------|
| `JsonRpcRequest`（:27） | client → server 请求：`{jsonrpc, method, params, id}` |
| `JsonRpcSuccessResponse`（:45） | server → client 终结成功：`{jsonrpc, id, result}` |
| `JsonRpcErrorResponse`（:49） | server → client 终结错误：`{jsonrpc, id, error}` |
| `JsonRpcEventFrame`（:53） | server → client 中间事件帧：`{jsonrpc, id, event}` |

**3 个 method**（stdio_transport.py:171-182）：

| method | 调用 |
|--------|------|
| `agent.stream` | `_handle_streaming_request_model` → `handle_streaming_request`，每个事件 → `JsonRpcEventFrame`，结束 → `JsonRpcSuccessResponse(result=None)`（stdio_transport.py:191-225） |
| `agent.query` | `_handle_sync_request_model` → `handle_request`，结果 → `JsonRpcSuccessResponse(result=str)`（stdio_transport.py:227-252） |
| `agent.stop` | `_handle_stop_request_model` → `handle_stop_request`（stdio_transport.py:254-312） |

**stdout 保护**（stdio_transport.py:114-129）：

```python
self._original_stdout = sys.stdout
self._real_stdout = sys.stdout
self._running = True
try:
    sys.stdout = sys.stderr   # 第三方库 print 全部走 stderr
    asyncio.run(self._run_loop())
finally:
    if self._original_stdout is not None:
        sys.stdout = self._original_stdout  # 恢复
```

`_write_line`（stdio_transport.py:314-325）写入 `self._real_stdout`（保留的真 stdout 引用），绕过 redirect。这套机制保证了：JSON-RPC 帧只走真 stdout，第三方库的 `print` 全部进 stderr，client 解析永不会被污染。

#### 7. AgentRequest 统一请求模型

`AgentRequest`（http/models.py:17）：

```python
class AgentRequest(BaseModel):
    messages: str | list[Message]
    user_id: str = "default-user"
    session_id: str | None = None
    context: dict[str, Any] | None = None
    variables: ContextValue | None = None
```

被 HTTP `/query` `/stream` 和 stdio `agent.query` `agent.stream` **共用**——stdio 在 `_handle_line`（stdio_transport.py:166-169）把 `params` dict 直接 `AgentRequest.model_validate({**params})`。这保证了 client 切 transport 时业务字段语义不变。

`StopRequest` / `StopResponse`（http/models.py:50-72）类似，由 `/stop` 与 `agent.stop` 共用。

### 示例

启动 SSE server：

```python
from nexau.archs.transports.http import SSETransportServer, HTTPConfig
from nexau.archs.session.orm import SQLDatabaseEngine

engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///sessions.db")
server = SSETransportServer(
    engine=engine,
    config=HTTPConfig(host="0.0.0.0", port=8000, cors_origins=["*"]),
    default_agent_config=agent_config,
)
server.run()  # uvicorn.run blocking
```

client SSE 流式请求：

```
POST /stream
{"messages": "Hello", "user_id": "u1", "session_id": "s1"}

→ data: {"type":"TEXT_MESSAGE_CONTENT","content":"Hi"}\n\n
→ data: {"type":"RUN_FINISHED",...}\n\n
```

stdio 流式请求（一行 JSON）：

```
{"jsonrpc":"2.0-stream","method":"agent.stream","params":{"messages":"Hello","user_id":"u1"},"id":"1"}

→ {"jsonrpc":"2.0-stream","id":"1","event":{...}}
→ {"jsonrpc":"2.0-stream","id":"1","event":{...}}
→ {"jsonrpc":"2.0-stream","id":"1","result":null}
```

CLI 启动：

```bash
uv run nexau serve http --config agent.yaml --port 8000
uv run nexau serve stdio --config agent.yaml
```

## 权衡取舍

### 考虑过的替代方案

1. **每个 transport 独立写 session/agent 管理**：拒绝原因：会出现 `/stream` 与 `agent.stream` 行为不一致（如 session 自动生成规则不同），用户从浏览器迁移到 CLI 会踩坑。基类提取保证语义一致。
2. **不递归下沉 middleware，让用户在 sub-agent 里手挂**：拒绝原因：YAML 中 sub-agent 是声明式定义，用户不写 Python 代码，无法手动注入 `AgentEventsMiddleware`。框架自动下沉是唯一可行。
3. **`Agent(...)` 直接 await 而非 `asyncio.to_thread`**：拒绝原因：`Agent.__init__` 含 `asyncio.run()` 同步初始化（session/history 加载），在 async handler 里直接调用必报 RuntimeError。重构 `Agent.__init__` 为完全 async 是另一个 RFC 范围。
4. **stdio 走标准 JSON-RPC 2.0**：拒绝原因：标准 JSON-RPC 2.0 一请求一响应，无法表达"流式中间事件帧 + 终结帧"语义。自定 `2.0-stream` 是 superset，与标准客户端不兼容但内部一致。
5. **不保护 stdout，要求第三方库不要 `print`**：拒绝原因：依赖链里第三方库（如某些 LLM SDK）会偶发 print debug 输出，无法 audit 全部依赖。redirect 是 defense-in-depth。
6. **`/stop` 走 long-polling 或 GraphQL subscription**：拒绝原因：stop 是命令式动作（"现在停"），用 POST `/stop` 同步返回 `StopResult` 比订阅更直接。
7. **WebSocket 取代 SSE**：拒绝原因：SSE 单向（server→client）正好符合 streaming agent 响应的语义；WebSocket 双向但需要 client 实现复杂的 ping/pong 与重连。SSE 在浏览器 EventSource 原生支持。WebSocket transport 已留 placeholder，未来需要时再启用。

### 缺点

1. **`asyncio.to_thread` 包装 `Agent.__init__` 是 workaround**：根因是 `Agent.__init__` 含同步 session init。长期应重构为完全 async constructor。
2. **`runtime_context` 在 SSE server 收集但未传给 agent**（sse_server.py:97-101）：`working_directory` 等只在 `__init__` 时收集，未在 `_stream_agent_response` 中传入 `Agent.run_async`——这是个 dead bridge。RFC-0014 的 NEXAU.md 注入依赖 `runtime_context["working_directory"]`，目前是从 `sandbox_config.work_dir` 兜底。
3. **`_running_agents` 是进程内 dict**：多进程部署（gunicorn workers）下 `/stop` 找不到其他 worker 上的 agent。需要分布式 lock + 跨进程注册（如 Redis）才能修复，目前单 worker 假设。
4. **JSON-RPC `2.0-stream` 是私有方言**：通用 JSON-RPC client 无法直接消费——event frame 没有 standard `result` / `error` 字段。第三方集成必须实现自定义 client。
5. **`_recursively_apply_middlewares` 的 stateful 对象浅引用**：tracer / middleware 在父 config 与所有 sub-config 之间共享。若 middleware 有 per-agent 状态（如 token counter），所有 agent 会共用同一计数器——目前的 middleware 都是 stateless 或 session-scoped，未踩坑，但是隐性约束。
6. **WebSocket / gRPC transport 仅占位无实现**（websocket/__init__.py / grpc/__init__.py）：声明了模块路径但 `__all__ = []`，给未来扩展预留入口而已。

## 实现计划

### 阶段划分

- [x] Phase 1: `TransportBase` 抽象 + session/agent/middleware 通用流程（base.py 全部）
- [x] Phase 2: `SSETransportServer` + `/query` `/stream` `/health` 端点（sse_server.py:35-275）
- [x] Phase 3: `StdioTransport` + `2.0-stream` 协议 + 三个 method（stdio_transport.py 全部）
- [x] Phase 4: `AgentRequest` / `StopRequest` 统一模型（http/models.py:17-72）
- [x] Phase 5: `_running_agents` + `/stop` 与 `agent.stop`（RFC-0001 Phase 4 集成）
- [x] Phase 6: `_recursively_apply_middlewares` 的 stateful 摘出/装回保护（base.py:104-128）
- [x] Phase 7: SSE team 路由 mount（sse_server.py:137-150，RFC-0002）
- [ ] Phase 8（未来）：WebSocket transport 实装（双向通信、ping/pong、断线重连）
- [ ] Phase 9（未来）：gRPC transport 实装（高性能 RPC，proto 定义）
- [ ] Phase 10（未来）：multi-worker 部署下的分布式 `_running_agents`（Redis-backed）
- [ ] Phase 11（未来）：`runtime_context` 在 SSE handler 中传给 `agent.run_async`，连通 NEXAU.md 注入桥
- [ ] Phase 12（未来）：`Agent.__init__` async-native，移除 `asyncio.to_thread` workaround

### 相关文件

- `nexau/archs/transports/base.py` - `TransportBase` 抽象基类与三大 handler
- `nexau/archs/transports/http/sse_server.py` - SSE server + FastAPI app + 4 个端点
- `nexau/archs/transports/http/sse_client.py` - SSE client 参考实现
- `nexau/archs/transports/http/team_routes.py` - RFC-0002 团队路由（与 SSE server 共享 app）
- `nexau/archs/transports/http/team_registry.py` - team 注册表
- `nexau/archs/transports/http/models.py` - `AgentRequest` / `AgentResponse` / `StopRequest` / `StopResponse`
- `nexau/archs/transports/http/config.py` - `HTTPConfig`（host / port / CORS）
- `nexau/archs/transports/stdio/stdio_transport.py` - stdio transport 与 JSON-RPC 2.0-stream 实现
- `nexau/archs/transports/stdio/config.py` - `StdioConfig`
- `nexau/archs/transports/websocket/__init__.py` - WebSocket transport 占位
- `nexau/archs/transports/grpc/__init__.py` - gRPC transport 占位
- `nexau/archs/transports/CLAUDE.md` - transport 开发者指南（架构总览 + 代码示例）

## 测试方案

### 单元测试

- **`_recursively_apply_middlewares` 不污染原 config**：构造含 tracer 的 `AgentConfig`，调用后断言原 config 的 `tracers / middlewares` 与 `id()` 不变；copy 的 middleware 列表含新加入的 `events_mw`。
- **`_recursively_apply_middlewares` 递归到 sub-agent**：含 2 层嵌套 sub-agent 的 config，断言每层 sub-agent 的 copy 都含 `events_mw`。
- **`_recursively_apply_middlewares` `enable_stream` 透传**：`enable_stream=True` 后，所有层级的 `cfg_copy.llm_config.stream == True`。
- **`handle_request` 自动生成 session_id**：`session_id=None` 调用，断言返回前 `_running_agents` 中有一个 key 包含 `session_id != None`。
- **`handle_request` 完成后清出运行表**：调用前后比较 `_running_agents.keys()`，无残留。
- **`handle_stop_request` 找不到 agent → ValueError**：空运行表时调用，断言抛 `ValueError`，message 含 `user_id` / `session_id`。
- **`handle_stop_request` `agent_id=None` fallback**：运行表里 mock 一个 `(u1, s1, a1)` agent，调用 `handle_stop_request(user_id="u1", session_id="s1")`（agent_id 不给），断言找到 a1。
- **stdio JSON-RPC `2.0-stream` 帧格式**：`JsonRpcEventFrame.model_dump_json()` 含 `"jsonrpc":"2.0-stream"` 与 `"event"` 字段，无 `"result"` 字段。
- **stdio stdout 保护**：mock `sys.stdout` 为 `StringIO`，`StdioTransport.start` 后断言 `sys.stdout` 已被改为 stderr；调用 `_write_line` 写入的是 `_real_stdout`（原 stdout）。

### 集成测试

- **SSE `/stream` 端到端**：用 httpx AsyncClient POST `/stream`，断言返回 `text/event-stream`，多帧 `data: {...}\n\n` 解析后含 `TEXT_MESSAGE_CONTENT` + 终结的 `RUN_FINISHED`。
- **SSE `/query` vs `/stream` 一致性**：相同 messages，`/query` 返回 `response` 字段；`/stream` 把所有 TEXT_MESSAGE_CONTENT.content 拼接 == `response`。
- **SSE `/stop` 中断**：起一个长 task → POST `/stream` → 0.5s 后 POST `/stop` → 断言流提前结束 + `StopResponse.status="success"`。
- **stdio 端到端**：subprocess 启动 stdio transport，stdin 写一行 `agent.stream` 请求，stdout 收事件帧 + 终结帧；断言事件类型 + `id` 一致。
- **stdio stdout 不被第三方污染**：在 agent_config 里挂一个故意 `print("dirty")` 的 middleware，跑 `agent.query`，断言 stdout 仅含 JSON 帧，"dirty" 在 stderr 而非 stdout。
- **多并发 session 不串扰**：并发 5 个 `/stream` 不同 session_id，断言每路收到的事件 session_id 与请求一致。

### 手动验证

1. 浏览器 EventSource 连 `/stream`，发字符串问题，看实时打字效果。
2. `uv run nexau serve stdio --config agent.yaml`，手输 JSON 行，看流式响应。
3. `curl -X POST http://localhost:8000/stop -d '{"user_id":"u1","session_id":"s1"}'` 中断长任务，确认 SSE client 连接立刻收终止帧。

## 未解决的问题

1. **WebSocket / gRPC 何时实装**：当前仅占位。需要明确触发条件——是有 IDE 集成需要双向 push 时，还是高性能场景（如多 agent 跨进程协调）？
2. **multi-worker 部署的分布式运行表**：单进程 dict 在 `gunicorn -w 4` 下 `/stop` 命中率 25%。是否引入 Redis 注册表 + cross-worker stop 转发？
3. **`runtime_context` 桥接**：当前 SSE server 收集 `working_directory / username / date` 但未传给 agent。如何在 RFC-0014（NEXAU.md 注入）+ 本 RFC 之间打通？
4. **JSON-RPC `2.0-stream` 标准化**：是否值得起草一个 mini-spec，让第三方 client 库可以兼容实现？
5. **`Agent.__init__` 同步 init 重构**：移除 `asyncio.to_thread` workaround 涉及 Agent 构造器全面重写，工作量大；何时启动？
6. **rate limiting / authn / authz**：当前 transport 完全开放，靠 reverse proxy 拦截。是否在 `TransportBase` 提供 hook，让 middleware 可以拒绝请求？

## 参考资料

- `nexau/archs/transports/base.py:32` — `class TransportBase[TTransportConfig](ABC)`
- `nexau/archs/transports/base.py:57` — `__init__` 签名
- `nexau/archs/transports/base.py:80-84` — `SessionManager` 实例化（共享 engine）
- `nexau/archs/transports/base.py:88` — `_running_agents: dict[(user_id, session_id, agent_id), Agent]`
- `nexau/archs/transports/base.py:89` — `_running_agents_lock = asyncio.Lock()`
- `nexau/archs/transports/base.py:91-92` — `_recursively_apply_middlewares` (L91 `@staticmethod`, L92 def)
- `nexau/archs/transports/base.py:108-110` — 保留 stateful (tracer / middleware) 引用
- `nexau/archs/transports/base.py:112-118` — 临时清空 + 深拷贝 + finally 恢复
- `nexau/archs/transports/base.py:126-128` — 把 stateful 浅赋给 copy
- `nexau/archs/transports/base.py:131` — 追加新 middleware
- `nexau/archs/transports/base.py:134-135` — `enable_stream` 设 `llm_config.stream`
- `nexau/archs/transports/base.py:137-142` — 递归到 sub_agents
- `nexau/archs/transports/base.py:146-148` — `start()` abstract (L146 `@abstractmethod`, L147 def)
- `nexau/archs/transports/base.py:150-152` — `stop()` abstract (L150 `@abstractmethod`, L151 def)
- `nexau/archs/transports/base.py:154` — `handle_request` def
- `nexau/archs/transports/base.py:185-188` — session_id 自动生成
- `nexau/archs/transports/base.py:191-195` — `AgentEventsMiddleware` 注入（同步路径）
- `nexau/archs/transports/base.py:197-208` — `asyncio.to_thread(create_agent)`
- `nexau/archs/transports/base.py:210-213` — 注册到 `_running_agents`
- `nexau/archs/transports/base.py:217` — `agent.run_async` 调用
- `nexau/archs/transports/base.py:218-221` — finally 清出运行表
- `nexau/archs/transports/base.py:228` — `handle_streaming_request` def
- `nexau/archs/transports/base.py:265-271` — event_queue + middleware + `enable_stream=True`
- `nexau/archs/transports/base.py:294` — `agent_task = asyncio.create_task(...)`
- `nexau/archs/transports/base.py:298-303` — drain loop with `wait_for(timeout=0.1)`
- `nexau/archs/transports/base.py:306-307` — post-task drain remaining queue
- `nexau/archs/transports/base.py:319` — `handle_stop_request` def
- `nexau/archs/transports/base.py:357-365` — agent_id 给定 vs fallback 查找
- `nexau/archs/transports/base.py:370` — `agent.stop(force=force, timeout=timeout)`
- `nexau/archs/transports/http/sse_server.py:35` — `class SSETransportServer(TransportBase[HTTPConfig])`
- `nexau/archs/transports/http/sse_server.py:61` — `__init__` with optional team callbacks
- `nexau/archs/transports/http/sse_server.py:88` — `_team_registry` 字段
- `nexau/archs/transports/http/sse_server.py:94` — `self.app = self._create_app()`
- `nexau/archs/transports/http/sse_server.py:97-101` — `_runtime_context`
- `nexau/archs/transports/http/sse_server.py:103` — `_create_app`
- `nexau/archs/transports/http/sse_server.py:117-122` — `FastAPI(...)` 实例化
- `nexau/archs/transports/http/sse_server.py:125-131` — CORS middleware
- `nexau/archs/transports/http/sse_server.py:137-150` — team router mount
- `nexau/archs/transports/http/sse_server.py:154` — `_add_routes`
- `nexau/archs/transports/http/sse_server.py:161` — `GET /`
- `nexau/archs/transports/http/sse_server.py:177` — `GET /health`
- `nexau/archs/transports/http/sse_server.py:184` — `POST /stream`
- `nexau/archs/transports/http/sse_server.py:204` — `POST /query`
- `nexau/archs/transports/http/sse_server.py:222` — `POST /stop`
- `nexau/archs/transports/http/sse_server.py:247` — `_stream_agent_response`
- `nexau/archs/transports/http/sse_server.py:267` — SSE `data: {...}\n\n` 帧
- `nexau/archs/transports/http/sse_server.py:344` — `run()` (uvicorn blocking)
- `nexau/archs/transports/http/models.py:17` — `class AgentRequest(BaseModel)`
- `nexau/archs/transports/http/models.py:32-36` — `AgentRequest` 字段
- `nexau/archs/transports/http/models.py:39` — `class AgentResponse(BaseModel)`
- `nexau/archs/transports/http/models.py:50` — `class StopRequest(BaseModel)`
- `nexau/archs/transports/http/models.py:63` — `class StopResponse(BaseModel)`
- `nexau/archs/transports/http/config.py:11-12` — `@dataclass` + `class HTTPConfig`
- `nexau/archs/transports/http/team_registry.py:38` — `class TeamRegistry`
- `nexau/archs/transports/stdio/stdio_transport.py:24` — `JSONRPC_VERSION: Literal["2.0-stream"]`
- `nexau/archs/transports/stdio/stdio_transport.py:27` — `class JsonRpcRequest(BaseModel)`
- `nexau/archs/transports/stdio/stdio_transport.py:45` — `class JsonRpcSuccessResponse`
- `nexau/archs/transports/stdio/stdio_transport.py:49` — `class JsonRpcErrorResponse`
- `nexau/archs/transports/stdio/stdio_transport.py:53` — `class JsonRpcEventFrame`
- `nexau/archs/transports/stdio/stdio_transport.py:57` — `class StdioTransport(TransportBase[StdioConfig])`
- `nexau/archs/transports/stdio/stdio_transport.py:105` — `def start`
- `nexau/archs/transports/stdio/stdio_transport.py:114-129` — stdout 保护 + redirect + finally 恢复
- `nexau/archs/transports/stdio/stdio_transport.py:131` — `def stop`
- `nexau/archs/transports/stdio/stdio_transport.py:135` — `_run_loop`（async stdin reader）
- `nexau/archs/transports/stdio/stdio_transport.py:160` — `_handle_line`（解码 + dispatch）
- `nexau/archs/transports/stdio/stdio_transport.py:166-169` — `JsonRpcRequest.model_validate_json` + `AgentRequest.model_validate`
- `nexau/archs/transports/stdio/stdio_transport.py:171` — `agent.stream` dispatch
- `nexau/archs/transports/stdio/stdio_transport.py:175` — `agent.query` dispatch
- `nexau/archs/transports/stdio/stdio_transport.py:180` — `agent.stop` dispatch
- `nexau/archs/transports/stdio/stdio_transport.py:191` — `_handle_streaming_request_model`
- `nexau/archs/transports/stdio/stdio_transport.py:227` — `_handle_sync_request_model`
- `nexau/archs/transports/stdio/stdio_transport.py:254` — `_handle_stop_request_model`
- `nexau/archs/transports/stdio/stdio_transport.py:314` — `_write_line` 写真 stdout
- `nexau/archs/transports/websocket/__init__.py` — WebSocket transport 占位
- `nexau/archs/transports/grpc/__init__.py` — gRPC transport 占位
- `nexau/archs/transports/CLAUDE.md` — transport 开发者指南

### 相关 RFC

- `rfcs/0001-state-persistence-on-stop.md` — RFC-0001 Phase 4 提供 `/stop` `agent.stop` 与 `_running_agents` 注册表
- `rfcs/0002-agent-team.md` — team 路由通过 `app.include_router` mount 在 SSE server 上
- `rfcs/0003-llm-failover-middleware.md` — LLM failover middleware 与本 RFC 的 transport 重试为不同层
- `rfcs/0006-rfc-catalog-completion-master-plan.md` — 本 RFC 的任务 T9
- `rfcs/0008-session-persistence-and-history.md` — `SessionManager` 由 transport 实例化并传给 agent
- `rfcs/0010-llm-aggregator-event-stream.md` — transport 流式产出的 `Event` 由 LLM 聚合器定义
- `rfcs/0011-middleware-hook-composition.md` — `AgentEventsMiddleware` 是被 transport 自动注入的 middleware
- `rfcs/0014-prompt-engineering-and-templating.md` — 未连通的 `runtime_context` 桥与 NEXAU.md 注入相关
