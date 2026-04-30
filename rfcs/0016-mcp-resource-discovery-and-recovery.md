# RFC-0016: MCP 资源发现与错误恢复

- **状态**: implemented
- **优先级**: P2
- **标签**: `mcp`, `tool`, `discovery`, `recovery`, `architecture`, `concurrency`
- **影响服务**: `nexau/archs/tool/builtin/mcp_client.py`, `examples/mcp/*.py`, `examples/deep_research/deep_research_with_mcp.yaml`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

NexAU 把 MCP（Model Context Protocol）服务器抽象成"可发现工具集合 + 可重连会话"。`MCPManager` 是顶层入口，下辖 `MCPClient`（多 server 注册表 + tool registry）；每个 MCP server 用 `MCPServerConfig` 描述（stdio 或 streamable HTTP / HTTP+SSE 三种 transport）；每个发现到的 MCP tool 包装为 `MCPTool(Tool)`，自动加入框架 tool registry。本 RFC 描述：(1) MCP server 类型分发（stdio subprocess / `HTTPMCPSession` 双 transport）；(2) HTTP 自动 fallback 链（先 streamable HTTP，4xx 时降级到 HTTP+SSE）；(3) 工具命名 `<server_name>.<tool_name>` 防冲突；(4) `MCPManager.initialize_servers` 并行 `asyncio.gather(return_exceptions=True)` 实现"部分失败不阻断"；(5) `MCPTool._execute_sync` 每次新建 event loop + 每线程 `_get_thread_local_session` 解决跨线程 / 跨 loop 复用问题；(6) `disable_parallel` 字段对接 RFC-0007 工具级并行控制；(7) 单 inline `DirectMCPSession` 重复声明（`_get_thread_local_session` + `connect_to_server` 各一份）的现状与历史成因。

## 动机

MCP 是 Anthropic 推动的"标准化 LLM 外部工具协议"——理论上一个 MCP server 跑起来，所有支持 MCP 的 client 都能消费它的工具。NexAU 必须解决的设计问题：

1. **MCP 标准客户端（`mcp.ClientSession`）只支持单 transport**：`stdio` 与 `http` 走不同 SDK 类，业务侧不愿手动二选一；需要 NexAU 自己抽出统一接口。
2. **Streamable HTTP 是较新协议（MCP 2024-11-05）**：旧 MCP server 只支持 HTTP+SSE 老协议。client 必须先尝试新协议、失败时降级，而不是要求 server 端升级。
3. **MCP tool 必须接入 NexAU `Tool` 接口**：`Tool` 期望 sync `implementation` 接口（`Tool(name, description, input_schema, implementation)`），但 MCP `call_tool` 是 async。需要 `_execute_sync` 包一层 `asyncio.run`——还得避开"event loop 已关闭" / "Future attached to different loop" 这类多线程地雷。
4. **多 server 启动不能串行**：每个 stdio MCP server 启动 subprocess 可能 1-3 秒，串行 5 个就 10 秒延迟。必须并行 + "某个 server 挂了不阻断其他"。
5. **跨 server tool 名冲突**：两个 server 都有 `read_file` tool 时框架必须区分。统一加 `<server_name>.` 前缀。
6. **per-tool parallel 控制**：某些 MCP tool 不允许并发调用（如有状态的会话工具），需要 `disable_parallel` flag 透传到 NexAU 工具调度层（RFC-0007）。
7. **MCP tool 不能携带框架参数**：NexAU 内部会注入 `agent_state` / `global_storage` 给所有 tool，但 MCP server 完全不认识这些字段，会 JSON 序列化失败——必须在 wrapper 层过滤。
8. **HTTP session 跨线程不安全**：`HTTPMCPSession` 持有 `httpx.AsyncClient` 与 SSE listener task，bound 到创建它的 event loop；其他线程上的 ThreadPoolExecutor 不能复用。

把这些抽象到 `MCPManager` + `MCPTool` 后，用户在 YAML 里写 `mcp_servers: [{name, type, command/url, ...}]`，框架自动发现 + 注册 + 容错。

## 设计

### 概述

```
                    ┌────────────────────────────────────────────┐
                    │ MCPManager (mcp_client.py:1384)            │
                    │  - get_mcp_manager() singleton (:1485)     │
                    │  - initialize_servers (:1417)              │
                    │    asyncio.gather(return_exceptions=True)  │
                    └────────────┬───────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────────────┐
                    │ MCPClient (mcp_client.py:977)              │
                    │  - servers: {name → MCPServerConfig}       │
                    │  - sessions: {name → ClientSession-like}   │
                    │  - tools: {<server>.<tool> → MCPTool}      │
                    │  - connect_to_server (:990) → 创建 session │
                    │  - discover_tools (:1304) → list_tools     │
                    │    + 包装为 MCPTool                        │
                    └─────┬─────────────────────┬────────────────┘
                          │                     │
              type=stdio  ▼                type=http ▼
       ┌──────────────────────┐    ┌────────────────────────────┐
       │ DirectMCPSession     │    │ HTTPMCPSession (:50)       │
       │ (inline class :1029) │    │  - _initialize_streamable_ │
       │ subprocess + JSON-   │    │    http (:204)             │
       │ RPC over stdin/out   │    │  - 4xx fallback →          │
       │                      │    │  - _initialize_http_sse    │
       │                      │    │    (:247)                  │
       └──────────────────────┘    └────────────────────────────┘
                          │                     │
                          └─────────┬───────────┘
                                    ▼
                    ┌────────────────────────────────────────────┐
                    │ MCPTool(Tool) (mcp_client.py:635)          │
                    │  - name = mcp_tool.name                    │
                    │  - tool_key = "<server>.<tool>" (:1332)    │
                    │  - _execute_sync (:903): new loop / call   │
                    │  - _get_thread_local_session (:671)        │
                    │  - execute (:924): filter agent_state /    │
                    │    global_storage                          │
                    └────────────────────────────────────────────┘
```

### 详细设计

#### 1. `MCPServerConfig`：单一描述模型

`@dataclass MCPServerConfig`（mcp_client.py:617-632）字段：

```python
name: str
type: str = "stdio"          # "stdio" or "http"
# stdio
command: str | None = None
args: list[str] | None = None
env: dict[str, str] | None = None
# http
url: str | None = None
headers: dict[str, str] | None = None
timeout: float | None = 30
# 工具调度
disable_parallel: bool = False
```

`type` 字段决定 transport——`MCPClient.connect_to_server`（mcp_client.py:990）按它分发到 stdio / http 分支。`disable_parallel` 透传给 `MCPTool` 的 `super().__init__(disable_parallel=...)`（mcp_client.py:668），由 RFC-0007 工具调度层消费。

#### 2. 双 transport：stdio subprocess vs HTTP（含 fallback）

**stdio 分支**：`MCPClient.connect_to_server`（mcp_client.py:999-1011）启动 subprocess：

```python
server_params = StdioServerParameters(command=..., args=..., env=...)
# inline class DirectMCPSession (mcp_client.py:1029)
# - asyncio.create_subprocess_exec with stdin=PIPE, stdout=PIPE, stderr=PIPE
# - _initialize_connection 发送 jsonrpc initialize → 等 response
# - 然后 send "notifications/initialized" → tools/list verify
session = DirectMCPSession(config.command, config.args or [], merged_env)
await session.initialize()
```

`DirectMCPSession` 是**绕开 mcp SDK** 自己实现的 stdio session——SDK 的 `ClientSession` 与 `stdio_client` 在某些环境下事件循环管理有问题，自实现可控。

**HTTP 分支**：使用 `HTTPMCPSession`（mcp_client.py:50），在 `_initialize_session`（mcp_client.py:84-102）做 fallback：

```python
try:
    await self._initialize_streamable_http()  # mcp_client.py:204
except httpx.HTTPStatusError as exc:
    if 400 <= exc.response.status_code < 500:
        # 旧 server 不支持 streamable HTTP → 降级
        await self._initialize_http_sse()  # mcp_client.py:247
    else:
        raise  # 5xx 不降级，直接报错
```

**三个 Accept header 常量**（mcp_client.py:45-47）：

| 常量 | 值 |
|------|-----|
| `STREAMABLE_HTTP_ACCEPT` | `application/json, text/event-stream` |
| `SSE_ACCEPT` | `text/event-stream` |
| `DEFAULT_CONTENT_TYPE` | `application/json` |

streamable HTTP 必须接受两种 content type（同一 endpoint 既返回 JSON 又返回 SSE 流）；老 SSE 协议只接受 SSE。

**MCP 协议版本**（mcp_client.py:110）：固定 `"protocolVersion": "2024-11-05"`——硬编码，未来升级时改这一处。

#### 3. `MCPTool`：包装为 NexAU Tool

`MCPTool(Tool)`（mcp_client.py:635）构造时（mcp_client.py:638-669）：

```python
self.mcp_tool = mcp_tool                       # MCP SDK 的 Tool
self.client_session = client_session           # MCP session
self.server_config = server_config             # 用于线程间重建 session
self._session_type = type(client_session).__name__
self._session_params = ...                     # HTTP 用 dict / stdio 用 MCPServerConfig
self._sync_executor = self._execute_sync

# 关键：把 MCP tool 字段映射到 NexAU Tool 接口
super().__init__(
    name=mcp_tool.name,
    description=mcp_tool.description or "",
    input_schema=mcp_tool.inputSchema,
    implementation=self._sync_executor,
    disable_parallel=server_config.disable_parallel if server_config else False,
)
```

**`_session_params` 双类型**（mcp_client.py:648, 652-660）：

- HTTP session → 存 `_HTTPSessionParams` TypedDict（`{config, headers, timeout}`），供 `_get_thread_local_session` 在新线程里 `HTTPMCPSession(config, headers, timeout)` 重建。
- stdio session → 存 `MCPServerConfig`，供新线程里启 subprocess。

#### 4. 跨线程 / 跨 loop 安全：`_get_thread_local_session` + `_execute_sync`

NexAU 工具调度可能把 tool 调用扔到 ThreadPoolExecutor 并发跑（RFC-0007）。MCP session 持有的 `httpx.AsyncClient` 与 SSE listener task bound 到创建它的 event loop，跨线程复用会 `RuntimeError: Future attached to a different loop`。

**`_execute_sync`（mcp_client.py:903-922）每次创建新 event loop**：

```python
loop = asyncio.new_event_loop()
try:
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(self._execute_async(**kwargs))
finally:
    try: loop.close()
    except Exception: pass
    try: asyncio.get_event_loop()
    except RuntimeError: pass  # 没 loop 也 OK
```

**`_get_thread_local_session`（mcp_client.py:671-901）每次重建 session**：

- HTTP type → `HTTPMCPSession(config, headers, timeout)` 全新实例，新 loop 内 `await session._initialize_session()` 时建立连接
- stdio type → 新建 subprocess + `DirectMCPSession`（**inline class** 在 mcp_client.py:699-897 里第二次定义）
- 其他类型 → fallback 到 `self.client_session`（原始 session，仅在单线程场景安全）

**性能 tradeoff**：每次 tool 调用都新建 session（含 stdio subprocess）开销大。当前选择是"安全 > 性能"——MCP tool 调用本身远比 session 建立慢（网络 / 业务逻辑），相对可接受。未来若需优化，可引入"线程局部 session 缓存"。

#### 5. NexAU 框架参数过滤

`MCPTool.execute`（mcp_client.py:924-936）：

```python
filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("agent_state", "global_storage")}
filtered_kwargs = dict(sorted(filtered_kwargs.items(), key=lambda x: x[0]))
return self._sync_executor(**filtered_kwargs)
```

`agent_state` / `global_storage` 是 NexAU 框架自动注入的"上下文参数"（RFC-0012），MCP server 不认识；过滤后才能 JSON 序列化。`sorted` 是为了稳定 hashing / cache key 一致。

#### 6. 多 server 并行初始化

`MCPManager.initialize_servers`（mcp_client.py:1417-1470）：

```python
async def init_server(server_name):
    if await self.client.connect_to_server(server_name):
        tools = await self.client.discover_tools(server_name)
        return server_name, tools
    return server_name, []

results = await asyncio.gather(
    *[init_server(name) for name in server_names],
    return_exceptions=True,  # 关键：单个失败不影响其他
)

for i, result in enumerate(results):
    if isinstance(result, BaseException):
        failed_servers.append(server_names[i])
        logger.error(...)
        continue
    # 否则收集 tools
```

**`return_exceptions=True` 是核心容错点**：默认 `gather` 一旦某个 task 抛异常会取消所有其他 task；改成 `True` 后异常作为返回值传回，主流程过滤即可。这让"5 个 server 中 1 个挂掉"场景下另外 4 个仍能用。

#### 7. tool 命名与 registry

`discover_tools`（mcp_client.py:1304-1348）拿到 `tools_result.tools` 后，对每个 tool：

```python
tool = MCPTool(mcp_tool, session, server_config)
tool_key = f"{server_name}.{mcp_tool.name}"  # mcp_client.py:1332
self.tools[tool_key] = tool
```

`disconnect_server`（mcp_client.py:1358-1376）反向操作：删 session + 按前缀批量删 tool。

**注意**：`MCPTool.name`（NexAU Tool 接口的 name）是 **不带前缀**的 `mcp_tool.name`；前缀仅在 registry key 上。这意味着 LLM 在 tool call 时可以直接说 "read_file"（无前缀），框架按 registry 解析时若有冲突需上层处理——目前未见冲突时的明确仲裁逻辑（潜在 bug，列入未解决问题）。

#### 8. 入口函数

- `get_mcp_manager()`（mcp_client.py:1485-1490）：模块级单例（`_mcp_manager: MCPManager | None = None`，mcp_client.py:1482）。
- `initialize_mcp_tools(server_configs)`（mcp_client.py:1493-1532）：从 dict-of-configs 一键初始化所有 server 并返回 tool 列表。
- `sync_initialize_mcp_tools`（mcp_client.py:1535-1554）：同步包装（`asyncio.new_event_loop` + `run_until_complete`），供非 async caller 用。

YAML 配置侧通常通过 `examples/deep_research/deep_research_with_mcp.yaml` 这种声明式声明 mcp_servers，框架引导时自动调 `initialize_mcp_tools`。

### 示例

YAML 配置：

```yaml
mcp_servers:
  - name: amap
    type: http
    url: https://mcp.amap.com/sse
    headers: {}
    timeout: 30
  - name: github
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: ${env.GITHUB_TOKEN}
    disable_parallel: false
```

调用一次 `initialize_mcp_tools(server_configs)`：

1. `MCPManager.add_server` 把每个 config 注册到 `MCPClient.servers`
2. `initialize_servers` 并行 `gather` 两个 `init_server` task
3. amap：`HTTPMCPSession` → 先试 streamable HTTP → 如失败降级 SSE
4. github：subprocess 启 npx → JSON-RPC initialize → tools/list
5. 任一失败：`return_exceptions=True` 让另一边继续
6. 成功端的 tool 被 wrap 成 `MCPTool` 加入 registry：`amap.search_poi` / `github.create_issue` 等

LLM 调用 `github.create_issue`：

1. 框架找到 `MCPTool` 实例 → 调 `execute(**kwargs)`
2. `execute` 过滤 `agent_state` / `global_storage`
3. `_execute_sync` 新建 event loop
4. `_get_thread_local_session` 在当前线程重建 stdio subprocess + DirectMCPSession
5. `call_tool` 走 JSON-RPC 到 subprocess → 返回 `{"content": [...]}`
6. `_execute_async` 把 content 拼接为 `{"result": "<text>"}`

## 权衡取舍

### 考虑过的替代方案

1. **完全用 mcp SDK 的 `ClientSession` + `stdio_client`**：拒绝原因：SDK 的 stdio client 在某些环境下 event loop 管理有问题（多线程 / 重入 loop），会偶发 "Future attached to different loop"。`DirectMCPSession` 自实现绕过这个坑。
2. **不做 streamable HTTP / SSE 自动 fallback，要求用户在 YAML 选择**：拒绝原因：用户多数不知道 server 是哪个版本协议；fallback 让"先试新的、不行降级"对用户透明。
3. **为每个线程缓存 session 而不是每次新建**：拒绝原因：实现复杂（需要线程局部 dict + 引用计数 + 失活清理），且 session lifetime 与 thread lifetime 解耦时易泄漏。当前每次新建简单可控；如成为瓶颈再优化。
4. **不过滤 `agent_state` / `global_storage`，让 MCP server 自己忽略未知字段**：拒绝原因：MCP server 的 JSON schema validation 通常严格，未知字段直接报错；过滤是必需的。
5. **`asyncio.gather(return_exceptions=False)` 让初始化失败立即中断**：拒绝原因：MCP server 是外部依赖，挂掉是常态；让其他 server 仍可用比"全有或全无"更实用。
6. **tool name 全局加前缀（包括 `MCPTool.name`）**：拒绝原因：LLM 看到 `github.create_issue` 比 `create_issue` 更冗余，prompt token 浪费；当前用 registry key 加前缀、tool 实例 name 不加前缀，让 LLM 看到的是裸名。
7. **`DirectMCPSession` 抽出来作为顶层类**：拒绝原因（实际是历史欠债）：当前两处 inline 重复（mcp_client.py:699-897 与 mcp_client.py:1029-1219 几乎一致）。重构是低优先级整理，不影响功能。

### 缺点

1. **inline class 重复定义**：`DirectMCPSession` 在 `_get_thread_local_session`（mcp_client.py:699）与 `connect_to_server`（mcp_client.py:1029）各有一份 ≈200 行，几乎相同。任何 stdio JSON-RPC 协议变更必须改两处，易漏。
2. **`call_tool` 不区分 retry vs hard fail**：`_execute_async` 任何异常都返回 `{"error": str(e)}`（mcp_client.py:973-974）。LLM 看到 error 后无法判断是该重试（网络抖动）还是该放弃（参数错）。
3. **`tools` registry 跨 server 同名冲突未仲裁**：两个 server 各有 `read_file` 时，`MCPTool.name` 都是 `read_file`；上游若按 `name` 而非 `tool_key` 路由会随机取一个。
4. **`disable_parallel` 是 server-level 不是 tool-level**：`MCPServerConfig.disable_parallel=True` 会把该 server 的所有 tool 都标记禁并行，过粗。
5. **每次 tool 调用新建 session**（stdio）= 启 subprocess。开销 100ms+ 数量级，在高频 tool 场景明显。
6. **`HTTPMCPSession` SSE listener 重连**：`_consume_sse_stream`（mcp_client.py:385）+ `_fail_pending_requests`（mcp_client.py:494）只做单次失败处理，无重连。SSE 长连接断了之后 session 不可用，需重建。
7. **MCP 协议版本硬编码**：`"2024-11-05"` 写死。新版本协议若有 breaking change，client 需手动升级。
8. **`ToolCallResult.content` 解析 ad-hoc**：`_execute_async`（mcp_client.py:945-970）按 `hasattr(item, "text")` / `isinstance(item, dict)` 多分支猜结构。如果 server 返回新 content type（image / resource_link）会落到 `str(item)` fallback，丢信息。

## 实现计划

### 阶段划分

- [x] Phase 1: `MCPServerConfig` + `MCPClient` 注册表 + stdio `DirectMCPSession` (mcp_client.py:617-632, 977, 1029)
- [x] Phase 2: HTTP 双 transport `HTTPMCPSession` + streamable HTTP/SSE fallback (mcp_client.py:50, 84-102, 204, 247)
- [x] Phase 3: `MCPTool(Tool)` 包装 + 框架参数过滤 + thread-local session (mcp_client.py:635, 671, 903, 924)
- [x] Phase 4: `MCPManager` 顶层 + 并行 `initialize_servers` (mcp_client.py:1384, 1417)
- [x] Phase 5: `tools/list` 发现 + `<server>.<tool>` 命名 (mcp_client.py:1304-1338)
- [x] Phase 6: `disconnect_server` / `disconnect_all` 生命周期 (mcp_client.py:1358-1381)
- [x] Phase 7: 入口函数 `initialize_mcp_tools` / `sync_initialize_mcp_tools` (mcp_client.py:1493, 1535)
- [ ] Phase 8（未来）：把 `DirectMCPSession` 抽到顶层模块，消除重复
- [ ] Phase 9（未来）：错误分类（retryable / fatal）让 LLM 能正确反应
- [ ] Phase 10（未来）：tool name 冲突仲裁（按 server 优先级 / fallback 给 fully-qualified name）
- [ ] Phase 11（未来）：tool-level `disable_parallel` 而非 server-level
- [ ] Phase 12（未来）：HTTP SSE 自动重连 + 心跳
- [ ] Phase 13（未来）：thread-local session 缓存（弱引用 + LRU），降低 stdio subprocess 启停频率

### 相关文件

- `nexau/archs/tool/builtin/mcp_client.py` - 全部 MCP 实现（1554 行单文件）
- `nexau/archs/tool/tool.py` - `Tool` 基类（被 `MCPTool` 继承）
- `examples/mcp/mcp_amap_example.py` - HTTP MCP server 示例
- `examples/mcp/multi_server_parallel_example.py` - 多 server 并行初始化示例
- `examples/mcp/github_example.py` - stdio MCP server 示例
- `examples/deep_research/deep_research_with_mcp.yaml` - YAML 声明 mcp_servers 示例

## 测试方案

### 单元测试

- **`MCPServerConfig` 默认值**：构造 `MCPServerConfig(name="x")`，断言 `type=="stdio"`, `timeout==30`, `disable_parallel==False`。
- **`HTTPMCPSession._initialize_session` fallback**：mock `_initialize_streamable_http` 抛 `httpx.HTTPStatusError(status_code=404)`，断言 `_initialize_http_sse` 被调用；status_code=500 时不降级、原异常重抛。
- **`MCPTool.execute` 过滤框架字段**：构造 `MCPTool` + mock `_sync_executor`，调 `execute(agent_state="x", global_storage="y", real_arg="v")`；断言 `_sync_executor` 收到的 kwargs 仅含 `real_arg`。
- **`MCPTool._get_thread_local_session` HTTP 重建**：mock `_session_type="HTTPMCPSession"` + 合法 `_session_params`，断言返回新 `HTTPMCPSession(config, headers, timeout)` 实例（与 `self.client_session` 不同对象）。
- **`MCPTool._execute_sync` 新建 event loop**：mock `_execute_async`，断言每次调用前 `asyncio.new_event_loop` 被调；调用后 loop.close 被调。
- **`MCPClient.discover_tools` tool key 加前缀**：mock `session.list_tools()` 返回 `[Tool(name="read")]`；断言 `client.tools` 含 `"server1.read"` key。
- **`MCPClient.disconnect_server` 删除前缀工具**：先 discover 后 disconnect，断言 `tools` 中 `"server1.*"` 全删，其他 server tool 不动。
- **`MCPManager.initialize_servers` 部分失败**：mock 两个 server，第一个 `connect_to_server` 抛异常；断言返回 dict 含第二个 server 的 tools，failed log 含第一个 server name。

### 集成测试

- **真实 stdio MCP server 启动**：用 `npx @modelcontextprotocol/server-everything` 起 stdio server，跑 `initialize_mcp_tools`，断言返回的 tool 列表含 server 公开的工具且 `name` 不带前缀。
- **真实 HTTP MCP server 调用**：起 streamable HTTP MCP server（如 mcp-amap），跑 `initialize_mcp_tools`，断言能 `list_tools` + 调用一个 tool 拿到 result。
- **HTTP fallback 路径**：mock 一个只支持老 SSE 协议的 server（streamable HTTP 端点返回 404），跑初始化，断言降级到 HTTP+SSE 后能 list_tools。
- **多 server 并行 + 部分失败**：配置 3 个 server（其中 1 个故意 command 写错），跑 `initialize_mcp_tools`，断言返回 2 个 server 的 tools，错误 server 出现在 warning log。

### 手动验证

1. `python examples/mcp/mcp_amap_example.py` 跑 amap MCP，验证一次完整调用链。
2. `python examples/mcp/multi_server_parallel_example.py` 起多 server 并行；杀掉一个 server 进程，看主流程是否仍能用其他 tool。
3. 在 YAML 里写错 `command`（如 `npx-typo`），看 agent 启动时是否仅 warning 而不 fail-fast。

## 未解决的问题

1. **`DirectMCPSession` 重复 inline class 重构时机**：900 行重复代码什么时候拆出来？拆出来时是否同时升级 mcp SDK ClientSession 让自实现可以退役？
2. **Tool name 冲突的仲裁规则**：跨 server 同名 tool，应按 server 注册顺序优先？还是要求 LLM 必须用 `<server>.<tool>` 全名？
3. **error 分类**：什么样的 MCP error 应让 LLM 重试，什么样的应让 LLM 放弃？需要标注 `retryable: bool` 之类的元数据。
4. **HTTP SSE 长连接断线重连**：当前实现一断即死。是否引入 exponential backoff + heartbeat？还是切到 streamable HTTP 短连接？
5. **MCP 协议版本演进**：`"2024-11-05"` 硬编码；新版本的 capabilities negotiation 是否需要在 client 端表达？
6. **`disable_parallel` 粒度**：server-level 是否够？是否需要 tool-level（在 MCP `tools/list` 返回的元数据里读）？
7. **stdio subprocess 启停频率**：高频 tool 调用下 subprocess 启停成本如何摊销？线程局部缓存如何兼顾正确性？
8. **MCP server 的 prompts / resources 接口**：当前只接入了 `tools/list` + `tools/call`；MCP 协议还包含 `prompts/list`、`resources/list` 等。这些是否值得接入？接入后 prompt 会与 RFC-0014 的本地模板系统冲突吗？

## 参考资料

- `nexau/archs/tool/builtin/mcp_client.py:25` — `from mcp import ClientSession, StdioServerParameters`
- `nexau/archs/tool/builtin/mcp_client.py:26` — `from mcp.types import Tool as MCPToolType`
- `nexau/archs/tool/builtin/mcp_client.py:37` — `class _HTTPSessionParams(TypedDict)`
- `nexau/archs/tool/builtin/mcp_client.py:45` — `STREAMABLE_HTTP_ACCEPT = "application/json, text/event-stream"`
- `nexau/archs/tool/builtin/mcp_client.py:46` — `SSE_ACCEPT = "text/event-stream"`
- `nexau/archs/tool/builtin/mcp_client.py:47` — `DEFAULT_CONTENT_TYPE = "application/json"`
- `nexau/archs/tool/builtin/mcp_client.py:50` — `class HTTPMCPSession`
- `nexau/archs/tool/builtin/mcp_client.py:84` — `_initialize_session` def
- `nexau/archs/tool/builtin/mcp_client.py:91-102` — try streamable_http → 4xx fallback to http_sse
- `nexau/archs/tool/builtin/mcp_client.py:104` — `_build_initialize_request`
- `nexau/archs/tool/builtin/mcp_client.py:110` — `"protocolVersion": "2024-11-05"`
- `nexau/archs/tool/builtin/mcp_client.py:204` — `_initialize_streamable_http`
- `nexau/archs/tool/builtin/mcp_client.py:247` — `_initialize_http_sse`
- `nexau/archs/tool/builtin/mcp_client.py:296` — `_send_initialized_notification`
- `nexau/archs/tool/builtin/mcp_client.py:385` — `_consume_sse_stream`
- `nexau/archs/tool/builtin/mcp_client.py:494` — `_fail_pending_requests`
- `nexau/archs/tool/builtin/mcp_client.py:562` — `list_tools`
- `nexau/archs/tool/builtin/mcp_client.py:593` — `call_tool`
- `nexau/archs/tool/builtin/mcp_client.py:617-618` — `@dataclass class MCPServerConfig`
- `nexau/archs/tool/builtin/mcp_client.py:622` — `type: str = "stdio"  # "stdio" or "http"`
- `nexau/archs/tool/builtin/mcp_client.py:632` — `disable_parallel: bool = False`
- `nexau/archs/tool/builtin/mcp_client.py:635` — `class MCPTool(Tool)`
- `nexau/archs/tool/builtin/mcp_client.py:638` — `MCPTool.__init__`
- `nexau/archs/tool/builtin/mcp_client.py:648` — `_session_params` 双类型字段
- `nexau/archs/tool/builtin/mcp_client.py:652-660` — HTTP / stdio session params 分支
- `nexau/archs/tool/builtin/mcp_client.py:663-669` — `super().__init__` 把 MCP tool 字段映射到 NexAU `Tool`
- `nexau/archs/tool/builtin/mcp_client.py:668` — `disable_parallel` 透传
- `nexau/archs/tool/builtin/mcp_client.py:671` — `_get_thread_local_session`
- `nexau/archs/tool/builtin/mcp_client.py:699` — inline `class DirectMCPSession`（thread-local 重建路径）
- `nexau/archs/tool/builtin/mcp_client.py:903` — `_execute_sync` def（new event loop per call）
- `nexau/archs/tool/builtin/mcp_client.py:907` — `loop = asyncio.new_event_loop()`
- `nexau/archs/tool/builtin/mcp_client.py:924` — `def execute`（filter agent_state/global_storage）
- `nexau/archs/tool/builtin/mcp_client.py:928` — filter dict comprehension
- `nexau/archs/tool/builtin/mcp_client.py:938` — `_execute_async`
- `nexau/archs/tool/builtin/mcp_client.py:945-970` — `result.content` ad-hoc 解析
- `nexau/archs/tool/builtin/mcp_client.py:973-974` — error → `{"error": str(e)}`
- `nexau/archs/tool/builtin/mcp_client.py:977` — `class MCPClient`
- `nexau/archs/tool/builtin/mcp_client.py:990` — `connect_to_server`
- `nexau/archs/tool/builtin/mcp_client.py:1029` — inline `class DirectMCPSession`（首次定义路径）
- `nexau/archs/tool/builtin/mcp_client.py:1304` — `discover_tools`
- `nexau/archs/tool/builtin/mcp_client.py:1332` — `tool_key = f"{server_name}.{mcp_tool.name}"`
- `nexau/archs/tool/builtin/mcp_client.py:1350` — `get_tool`
- `nexau/archs/tool/builtin/mcp_client.py:1354` — `get_all_tools`
- `nexau/archs/tool/builtin/mcp_client.py:1358` — `disconnect_server`
- `nexau/archs/tool/builtin/mcp_client.py:1378` — `disconnect_all`
- `nexau/archs/tool/builtin/mcp_client.py:1384` — `class MCPManager`
- `nexau/archs/tool/builtin/mcp_client.py:1391` — `add_server`
- `nexau/archs/tool/builtin/mcp_client.py:1417` — `initialize_servers` def
- `nexau/archs/tool/builtin/mcp_client.py:1428` — `init_server` 内部协程
- `nexau/archs/tool/builtin/mcp_client.py:1436-1439` — `asyncio.gather(*..., return_exceptions=True)`
- `nexau/archs/tool/builtin/mcp_client.py:1472` — `get_available_tools`
- `nexau/archs/tool/builtin/mcp_client.py:1476` — `shutdown`
- `nexau/archs/tool/builtin/mcp_client.py:1482` — `_mcp_manager: MCPManager | None = None`
- `nexau/archs/tool/builtin/mcp_client.py:1485` — `def get_mcp_manager`
- `nexau/archs/tool/builtin/mcp_client.py:1493` — `async def initialize_mcp_tools`
- `nexau/archs/tool/builtin/mcp_client.py:1535` — `def sync_initialize_mcp_tools`
- `examples/mcp/mcp_amap_example.py` — HTTP MCP server 示例
- `examples/mcp/multi_server_parallel_example.py` — 多 server 并行示例
- `examples/mcp/github_example.py` — stdio MCP server 示例
- `examples/deep_research/deep_research_with_mcp.yaml` — YAML 声明示例

### 相关 RFC

- `rfcs/0006-rfc-catalog-completion-master-plan.md` — 本 RFC 的任务 T10
- `rfcs/0007-tool-system-architecture-and-binding.md` — `MCPTool` 继承 `Tool` 基类，`disable_parallel` 透传给工具调度层
- `rfcs/0011-middleware-hook-composition.md` — MCP tool 调用链上 middleware 钩子如何介入
- `rfcs/0012-global-storage-and-session-mutations.md` — `agent_state` / `global_storage` 是被 `MCPTool.execute` 过滤的框架字段
- `rfcs/0014-prompt-engineering-and-templating.md` — MCP `prompts/list` 接入后与本地 prompt 模板的潜在冲突（未解决问题 #8）
- `rfcs/0015-transport-routing-and-dispatch.md` — MCP HTTP 与 NexAU 自身 transport 的差异：MCP 是 client → 外部 server；本 RFC-0015 是 NexAU server → 用户 client
