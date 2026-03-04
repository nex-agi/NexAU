# RFC-0004: Context 超限治理与 TokenCounter Block 原生计数迁移

- **状态**: draft
- **优先级**: P1
- **标签**: `architecture`, `runtime`, `dx`, `observability`, `breaking-change`
- **影响服务**: `nexau/archs/main_sub/execution/`, `nexau/archs/main_sub/utils/`, `nexau/archs/main_sub/config/`, `nexau/archs/main_sub/`
- **创建日期**: 2026-02-26
- **更新日期**: 2026-03-04

## 摘要

本 RFC 统一记录以下改造：

1. Token 计数从 legacy dict 口径迁移到 UMP block 原生口径（breaking）。
2. 上下文超限控制拆分为两个显式开关：是否硬停、是否启用 emergency fallback。
3. `wrap_model_call` 在 provider 明确上下文超限时报错时，触发固定“两段式 emergency 压缩”并重试一次。
4. 新增 compaction 生命周期事件（开始/结束），并打通到 SSE 解析层。
5. 将 `CONTEXT_TOKEN_LIMIT` 统一归类为 `RunErrorEvent` 终态语义。
6. 增强 LLM 空响应诊断日志（`finish_reason` + `raw_message` 预览）。

---

## 动机

### 1) Token 估算偏差明显

- legacy dict 展平丢失 block 结构语义；
- 图片/tool_use/tool_result/reasoning 的计数口径不一致；
- 编码器与真实模型不对齐时偏差会系统性累积。

### 2) 超限治理路径不透明

- 需要明确“本地预检查是否硬停”与“provider overflow 时是否 fallback”是两件事；
- 需要可观测事件来确认压缩是否触发、成功与失败原因。

### 3) 线上排障信息不足

- 出现 `No response content or tool calls` 时，需要看到 provider 实际返回结构，便于区分空响应与上下文溢出。

---

## 设计与实现（当前代码状态）

### A. TokenCounter：UMP block 原生计数（Breaking）

当前实现（`nexau/archs/main_sub/utils/token_counter.py`）：

- `count_tokens(messages, tools=None)` 仅接受 `Sequence[Message]`；
- 传入 dict-like 消息会直接抛 `TypeError`（附迁移提示）；
- 覆盖 block 类型：`text` / `tool_use` / `tool_result` / `image` / `reasoning`；
- 图片固定估算 `85 tokens`；
- 计入消息结构与工具 schema 的固定开销；
- 模型编码 fallback 顺序：
  1. `encoding_for_model(model)`
  2. `get_encoding("o200k_base")`
  3. `get_encoding("cl100k_base")`
  4. 本地字符估算 fallback。

### B. 配置与执行层超限开关

当前实现拆分为两个公共开关：

1. `overflow_max_tokens_stop_enabled`（Agent/Executor 层）  
   - `true`：本地预检查超限硬停；  
   - `false`：即使本地预算不足也继续发起模型请求。

2. `emergency_compact_enabled`（ContextCompactionMiddleware 层）  
   - 控制 `wrap_model_call` 的 emergency fallback 是否可触发；  
   - 不控制常规 `before_model` / `after_model` 压缩。

### C. 常规压缩与 emergency fallback 边界

1. 常规路径（regular）
   - `before_model`：按 trigger + compaction_strategy 执行；
   - `after_model`：按 trigger + compaction_strategy 执行；
   - 与 `emergency_compact_enabled` 解耦。

2. emergency 路径（wrap fallback）
   - 仅在以下条件同时满足时触发：
     - `auto_compact=true`
     - `emergency_compact_enabled=true`
     - 捕获到 provider context overflow 错误标识（字符串匹配）
   - 触发后执行 `UserModelFullTraceAdaptiveCompaction`，完成后重试一次模型调用。

### D. 固定“两段式” emergency 压缩策略

当前实现文件：`nexau/archs/main_sub/execution/middleware/context_compaction/compact_stratigies/user_model_full_trace_adaptive.py`

关键流程：

1. 先保留最小安全区：
   - system message（若存在）
   - 最近 `1` 个 iteration
   - 未闭合 `tool_use -> tool_result` 链
   - 最后一条 user message
2. 将可压缩区按 token 近似 50/50 切两段；
3. 两段都使用 emergency prompt 进行摘要；
4. 将两段摘要再次 merge 成单条摘要；
5. 重建消息为 `system + merged_summary(framework) + keep_region`；
6. 做 token gate（包含 tools），仍超限则报错收敛。

提示词文件：`nexau/archs/main_sub/execution/middleware/context_compaction/prompts/emergency_compact_prompt.md`

### E. 事件与可观测性

新增事件类型（`nexau/archs/llm/llm_aggregators/events.py`）：

- `COMPACTION_STARTED`
- `COMPACTION_FINISHED`

并在以下阶段发射：

- `before_model`
- `after_model`
- `wrap_model_call`

SSE 解析已支持这两个新事件（`nexau/archs/transports/http/sse_client.py`）。

### F. 终态语义统一

`AgentEventsMiddleware` 已将 `CONTEXT_TOKEN_LIMIT` 纳入错误终态，统一走 `RunErrorEvent`。

### G. 空响应诊断增强

`llm_caller.py` 已新增：

- 将 OpenAI `finish_reason` 透传到 `usage`；
- 当响应无内容且无 tool_calls 时，记录：
  - `finish_reason`
  - `role`
  - `content_len`
  - `tool_calls`
  - `usage`
  - `raw_message` 序列化预览（截断后输出）。

---

## 公共接口变更（Breaking + 新开关）

### 1) TokenCounter 调用约束（Breaking）

- `TokenCounter.count_tokens(...)` 输入类型从“可接受 legacy dict”变更为“仅 `Sequence[Message]`”。
- dict 输入直接 `TypeError`。

### 2) 自定义 token_counter 签名迁移（Breaking）

旧：

```python
def counter(messages: list[dict[str, Any]]) -> int: ...
```

新：

```python
from nexau.core.messages import Message

def counter(messages: list[Message], tools: list[dict[str, Any]] | None = None) -> int: ...
```

### 3) 新增/明确开关

- `overflow_max_tokens_stop_enabled: bool = True`（Agent/Executor 配置域）
- `emergency_compact_enabled: bool = True`（ContextCompactionMiddleware 配置域）

---

## 相对 origin/main 差异清单（截至 2026-03-04）

> 下列内容是当前工作区与 `origin/main` 的真实差异汇总。

### 已跟踪文件修改（M/A/D）

1. 文档
   - `docs/advanced-guides/context_compaction.md`
   - `rfcs/README.md`
   - `rfcs/0004-context-overflow-emergency-compaction-and-error-events.md`（新增）

2. 运行时与配置
   - `nexau/archs/main_sub/config/base.py`
   - `nexau/archs/main_sub/config/config.py`
   - `nexau/archs/main_sub/agent.py`
   - `nexau/archs/main_sub/execution/executor.py`
   - `nexau/archs/main_sub/execution/llm_caller.py`
   - `nexau/archs/main_sub/utils/token_counter.py`

3. compaction 中间件
   - `nexau/archs/main_sub/execution/middleware/context_compaction/config.py`
   - `nexau/archs/main_sub/execution/middleware/context_compaction/factory.py`
   - `nexau/archs/main_sub/execution/middleware/context_compaction/middleware.py`
   - `nexau/archs/main_sub/execution/middleware/context_compaction/__init__.py`
   - `nexau/archs/main_sub/execution/middleware/context_compaction/compact_stratigies/__init__.py`

4. 事件与传输
   - `nexau/archs/llm/llm_aggregators/events.py`
   - `nexau/archs/main_sub/execution/middleware/agent_events_middleware.py`
   - `nexau/archs/transports/http/sse_client.py`

5. 测试
   - `tests/unit/test_context_compaction.py`（保留）
   - `tests/e2e/test_emergency_compaction_e2e.py`（删除）

### 当前工作区新增但未跟踪（??）

- `nexau/archs/main_sub/execution/middleware/context_compaction/compact_stratigies/user_model_full_trace_adaptive.py`
- `nexau/archs/main_sub/execution/middleware/context_compaction/prompts/emergency_compact_prompt.md`
- `tests/integration/test_wrap_emergency_compaction_integration.py`

> 这三项属于本 RFC 的核心实现/验证文件，后续应纳入版本控制。

---

## 测试策略（按当前决策）

### 保留

- unit test（`tests/unit/`）
- integration test（`tests/integration/`）

### 已移除

- 非 unit/integration 的测试资产（含 e2e 与示例压测脚本）不纳入当前范围。

---

## 已知限制与未解决问题

1. `wrap_model_call` fallback 目前只在识别到“provider 上下文溢出”错误文本时触发；  
   对“结构合法但空内容响应”不会自动判定为 overflow。

2. emergency 压缩是固定两段+merge 的单次流程，不含多 attempt 递增强度循环；  
   若压缩后仍超限，当前直接走失败收敛。

3. 图片 token 固定值（85）仍是工程近似，后续可考虑按模型分桶。

4. 历史 legacy dict shim 的全仓清理尚未完成（本 RFC 仅覆盖 TokenCounter 主链路）。

---

## 验收标准（文档层）

1. 能直接回答“为什么改、改了什么、和 `origin/main` 差在哪里、后续怎么落地”。
2. 对 breaking change 给出明确迁移方式与失败表现。
3. 对超限治理明确两开关语义与触发边界。
4. 对测试范围明确：仅 unit/integration。

---

## 参考资料

- [RFC-0001: Agent 中断时状态持久化](./0001-state-persistence-on-stop.md)
- [RFC-0002: AgentTeam — 多 Agent 协作框架](./0002-agent-team.md)
- [RFC 撰写指南](./WRITING_GUIDE.md)
