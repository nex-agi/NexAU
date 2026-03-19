# RFC-0009: Token Generate Provider And Trace Memory

- **状态**: implemented
- **优先级**: P1
- **标签**: `architecture`, `dx`
- **影响服务**: `nexau` agent runtime, `generate_with_token` provider integration
- **创建日期**: 2026-03-09
- **更新日期**: 2026-03-14

## 摘要

为 NexAU 增加 `api_type="generate_with_token"` 调用模式，并在 agent 主循环中维护一个可跨轮次、可跨 run 复用的 token buffer。当前实现采用“本地 HuggingFace tokenizer 编码消息 + client.generate_with_token 生成 + 可选 detokenize 回退”的方案，同时在 `trace_memory` 中保留消息级 trace 与 token 级 trace。

## 动机

当前 NexAU 的主调用链路以消息为中心，适合标准 Chat/Responses API，但不适合直接以 `input_ids` 驱动生成的后端。目标是：

- 首轮仍从 `Message` 历史开始，不额外暴露新的运行入口
- 后续轮次维持持续增长的 token buffer，而不是每轮全量重建
- 模型直接生成的 token 记为 `1`，其他上下文 token 记为 `0`
- 不重写现有 tool call / tool result 主循环
- 同时保留 `message_trace` 与 token trace，方便调试、回放和训练分析

## 设计

### 概述

实现由四部分组成：

- `LLMConfig` 暴露 `api_type="generate_with_token"` 与 `tokenizer_path`
- `TokenTraceSession` 负责 token buffer、`response_mask`、round 级 trace 和 provider usage
- `Executor` 在 `LLM -> parse -> tool -> next round` 主循环中接入 token session
- `Agent` 复用 session 并把消息级 trace 写入 `global_storage["trace_memory"]`

核心原则：

- 首轮把完整 `messages` 编码成 token ids
- 模型回复 token 追加到 buffer，mask 记为 `1`
- tool result / synthetic feedback 等非模型 token 追加到 buffer，mask 记为 `0`
- token session 不做 context compaction；超过上限时直接报错终止

### 详细设计

#### 1. Provider 配置

`LLMConfig` 支持：

- `api_type="generate_with_token"`
- `tokenizer_path="<hf model or local path>"`

其中：

- `tokenizer_path` 为必填项，用于本地加载 HuggingFace tokenizer
- `base_url` / `api_key` 仍用于 detokenize HTTP 请求及下游 client 初始化
- 额外参数仍通过 `LLMConfig.extra_params` 透传

当前实现不再依赖远端 `/tokenize` 接口。消息到 token 的转换在本地通过 `AutoTokenizer.apply_chat_template(...)` 完成。

#### 2. Generate 调用协议

真正的生成调用不是在 `TokenTraceSession` 里直接发 HTTP，而是由 `LLMCaller` 调用运行时 client 的：

```python
client.generate_with_token(...)
```

`TokenTraceSession.build_generate_with_token_kwargs(...)` 会基于当前 buffer 组装：

- `model`
- `input_ids`
- `sampling_params.max_new_tokens`
- 以及一组透传给 provider 的采样/控制参数

已支持的参数分两类：

- `sampling_params` 内部参数，如 `temperature`、`top_p`、`top_k`、`stop`、`stop_token_ids`
- 顶层参数，如 `stream`、`rid`、`return_logprob`、`return_hidden_states`

其中 `stream=True` 目前不会真正启用流式输出，运行时会降级为非流式并打印 warning。

#### 3. Response 约定

当前实现要求 `client.generate_with_token(...)` 返回一个类似 OpenAI Chat Completion 的 payload，并额外携带 `nexrl_train`：

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "input_tokens": 3,
    "completion_tokens": 2
  },
  "nexrl_train": {
    "prompt_tokens": [1, 2, 3],
    "response_tokens": [4, 5]
  }
}
```

归一化规则如下：

- `choices[0].message` 用于构造 `ModelResponse`
- `nexrl_train.response_tokens` 是 token trace 的真实模型输出来源
- `usage` 优先读取 provider 原值；缺失时回退到 `nexrl_train.prompt_tokens` / `response_tokens` 的长度
- `finish_reason` 支持字符串或带 `type` 字段的对象

如果 `message.content` 缺失，但 `response_tokens` 存在，则通过 detokenize 回填文本内容。

#### 4. Tokenize / Detokenize 责任划分

当前实现的责任边界是：

- tokenize：本地 HF tokenizer
- generate：运行时 client 的 `generate_with_token(...)`
- detokenize：`TokenTraceSession.detokenize(...)` 直接请求后端 HTTP 接口

detokenize 默认走：

- `POST {base_url}/detokenize`

也支持通过 `llm_config.detokenize_path` 覆盖路径，并通过 `extra_headers` 注入额外请求头。

#### 5. TokenTraceSession 数据模型

`TokenTraceSession` 维护：

- `token_ids`
- `response_mask`
- `round_traces`
- `token_provider_usage`
- `synced_message_count`

导出到 `trace_memory` 的结构为：

- `final_token_list`
- `response_mask`
- `round_traces`
- `token_provider_usage`

语义约定：

- `response_mask[i] == 1` 表示该 token 来自模型直接生成
- `response_mask[i] == 0` 表示该 token 来自 system / user / history / tool result / synthetic feedback
- 模型生成出的 tool call token 仍然属于模型输出，因此记为 `1`

`round_traces` 当前只记录最小必要信息：

- `request_tokens`
- `response_tokens`
- `response_text`
- `tool_calls`

provider usage 以独立列表 `token_provider_usage` 保存，不直接内嵌到 `round_traces` 中。

#### 6. Executor 接入方式

`Executor.execute(...)` 中的接入点如下：

1. 从 `AgentState` 读取已有 `token_trace_session`
2. 若当前 provider 为 `generate_with_token` 且 session 不存在，则懒创建
3. 首次调用时使用 `initialize_from_messages(messages)` 初始化 token buffer
4. 每轮 LLM 调用前用 `sync_external_messages(messages)` 同步外部新增的非模型消息
5. LLM 返回后，把 assistant message 追加到消息历史
6. 再通过 `append_model_response(...)` 把模型输出 token 追加到 token buffer
7. tool 执行完成后，把 tool result message 或 synthetic feedback message 追加到 token buffer，mask 为 `0`
8. 结束、异常、overflow 等所有退出路径都会把 token trace 写回 `trace_memory`

这保证了 token trace 与消息主循环保持同一份执行时序。

#### 7. Tool 回环细节

当前实现兼容两类 tool result 回写方式：

- OpenAI/structured tool call 模式：
  - tool 执行结果作为 `role=tool` 的 `Message` 追加
  - token trace 中对应 token 记为 `0`
- XML/tool-feedback 模式：
  - 工具执行摘要会包装成一个 synthetic `role=user` 消息
  - 文本形如 `Tool execution results:\n...`
  - token trace 中同样记为 `0`

因此 RFC 不再假设“tool result 一定表现为标准 tool message”；真实行为取决于当前 tool call mode。

#### 8. 跨 run 复用

`Agent` 会把 `_token_trace_session` 挂在实例上，并在后续 run 中继续传给新的 `AgentState`。这意味着：

- 同一个 `Agent` 实例多次运行时，可以延续已有 token buffer
- `trace_memory["message_trace"]` 由 `Agent._update_trace_memory()` 单独维护
- `trace_memory` 中的 token trace 由 `Executor._store_token_trace()` 合并写入

两者互不覆盖，共同构成最终调试视图。

#### 9. Context Overflow 策略

token trace session 与普通消息上下文管理不同：

- 普通消息仍会先做 `TokenCounter` 级别的 prompt token 预算检查
- 但 token trace 自身不支持 context compaction
- 当 `append_model_response(...)` 或 `append_messages(...)` 追加后将超过 `max_context_tokens` 时，直接抛出 `TokenTraceContextOverflowError`
- `Executor` 捕获该异常后，以 `CONTEXT_TOKEN_LIMIT` 停止执行，并保留当前 trace

拒绝 compaction 的原因是：一旦消息历史被压缩，token buffer 与 message trace 的一一对应关系就可能失真。

### 示例

```python
agent = Agent(
    config=AgentConfig(
        name="token-agent",
        system_prompt="You are helpful.",
        llm_config=LLMConfig(
            model="my-model",
            base_url="http://gateway.internal",
            api_key="token",
            api_type="generate_with_token",
            tokenizer_path="meta-llama/Llama-3.1-8B-Instruct",
        ),
    )
)

result = agent.run(message="帮我查一下天气")

trace_memory = agent.global_storage.get("trace_memory", {})
final_token_list = trace_memory["final_token_list"]
response_mask = trace_memory["response_mask"]
message_trace = trace_memory["message_trace"]
```

## 权衡取舍

### 考虑过的替代方案

- 新增独立的 token run 入口：
  - 拒绝原因：破坏现有 `Agent.run()` / `run_async()` 使用方式
- 每轮都从 `messages` 全量重建 token list：
  - 拒绝原因：无法满足持续 append token buffer 的需求，也不利于保留精确的 `response_mask`
- 仅保留 token trace，不保留消息 trace：
  - 拒绝原因：人类调试、tool 回放和训练复盘成本过高
- 对 token trace 做 context compaction：
  - 拒绝原因：压缩后很难保证 token buffer 与消息历史仍严格一致

### 缺点

- 依赖本地可加载的 HuggingFace tokenizer
- provider 返回格式需要同时满足 `choices[0].message` 与 `nexrl_train.response_tokens`
- streaming 目前未实现，`stream=True` 会退化为非流式
- token trace 与 message trace 并存，会增加额外内存占用

## 实现计划

### 阶段划分

- [x] Phase 1: 增加 RFC、`TokenTraceSession` 与 `generate_with_token` provider 分支
- [x] Phase 2: 把 token session 接入 `Executor` 主循环与 `trace_memory`
- [x] Phase 3: 补齐单元测试并覆盖多轮 tool trace / overflow 场景

### 相关文件

- `nexau/archs/llm/llm_config.py` - 新增 `api_type="generate_with_token"` 与 `tokenizer_path`
- `nexau/archs/main_sub/token_trace_session.py` - 维护 token buffer、mask、round trace、detokenize
- `nexau/archs/main_sub/execution/llm_caller.py` - 调用 `client.generate_with_token(...)` 并归一化响应
- `nexau/archs/main_sub/execution/executor.py` - 在多轮 tool call 循环中维护 token trace
- `nexau/archs/main_sub/agent.py` - 跨 run 复用 token session，并写入 `message_trace`
- `nexau/archs/main_sub/agent_state.py` - 透传 `token_trace_session`

## 测试方案

### 单元测试

- `build_generate_with_token_kwargs(...)` 能正确生成 `input_ids` 与 `sampling_params`
- `final_token_list` 与 `response_mask` 长度始终一致
- 模型输出 token 标记为 `1`
- tool result / synthetic feedback token 标记为 `0`
- provider 能从 `nexrl_train.response_tokens` 读取真实输出 token
- 当返回文本为空时，能够走 detokenize 回填
- `message_trace` 与 token trace 能共同写入 `trace_memory`
- 超过 `max_context_tokens` 时抛出 `TokenTraceContextOverflowError`

### 集成测试

- `generate_with_token` 返回 tool call，验证完整的多轮 `LLM -> tool -> LLM` 回环
- 验证 structured tool call 与 XML tool feedback 两条路径都能正确追加 token trace

### 手动验证

1. 配置 `api_type="generate_with_token"` 和 `tokenizer_path`
2. 运行一个带工具的 agent
3. 检查 `trace_memory.message_trace`
4. 检查 `trace_memory.final_token_list`
5. 检查 `trace_memory.response_mask`
6. 检查 `trace_memory.round_traces`
7. 检查 `trace_memory.token_provider_usage`

## 未解决的问题

- 是否需要把 `generate_with_token` 的 request path / transport 也像 `detokenize_path` 一样做成显式配置
- 是否需要在 `round_traces` 中记录更多 provider 原始字段，便于后续训练和调试
- 后续是否需要为该模式补齐真正的 streaming 支持

## 参考资料

- `rfcs/0000-template.md`
- `nexau/archs/main_sub/token_trace_session.py`
- `nexau/archs/main_sub/execution/executor.py`
- `nexau/archs/main_sub/execution/llm_caller.py`
- `tests/unit/test_token_trace_session.py`
- `tests/unit/test_llm_caller.py`
