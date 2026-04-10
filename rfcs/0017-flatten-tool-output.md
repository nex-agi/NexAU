# RFC-0017: 工具输出扁平化（Flatten Tool Output）

- **状态**: implemented
- **优先级**: P1
- **标签**: `architecture`, `dx`, `llm`
- **影响服务**: nexau core (tool yaml, tool executor, middleware, messages)
- **创建日期**: 2026-04-10
- **更新日期**: 2026-04-10

## 摘要

当工具返回 Dict 结构结果时（尤其是 Sub-agent、MCP、复杂 builtin tool），当前 NexAU 最终常常通过 `coerce_tool_result_content()` 退化为 `str(dict)`，导致 LLM 看到的是 Python repr 包裹的嵌套结构，而不是可直接阅读的主体内容。本 RFC 提议：**在 tool YAML 中支持配置 custom formatter；若未配置，则统一使用 NexAU 内置的 XML formatter**。同时，formatter 必须在 `after_tool` middleware 之前执行，`LongToolOutputMiddleware` 不再只处理 raw tool output，而是优先处理 formatter 产出的 LLM-facing output，从而保证“被截断的内容”和“最终发给 LLM 的内容”是同一个对象。

## 动机

### 当前问题

以 Sub-agent 工具为例，`call_sub_agent()` 当前返回：

```python
{
    "status": "success",
    "sub_agent_name": "explore",
    "sub_agent_id": "d6a23025ed0c",
    "message": "In the nexau-rs project...",
    "result": "## Answer\n\n...数千字分析..."
}
```

在当前链路里，LLM 最终常看到类似内容：

```
{'status': 'success', 'sub_agent_name': 'explore', 'sub_agent_id': 'd6a23025ed0c',
 'message': 'In the nexau-rs project...', 'result': '## Answer\n\n...数千字分析...'}
```

问题有四个：

1. **输出格式错误**：LLM 接收到的是 Python repr，而不是稳定、可读的渲染文本
2. **主体内容被包裹**：真正重要的长正文在 `result` / `content` 字段内部，被 Dict 壳包住
3. **截断对象不对**：如果 middleware 截断的是 raw dict，但最终发给 LLM 的是另一个重新组装后的字符串，那么截断策略和最终模型输入不是同一个对象
4. **元数据污染上下文**：`status`、`message`、`sub_agent_name` 等短字段与正文混在一起，浪费 token

### 为什么 formatter 必须早于 LongToolOutputMiddleware

如果 formatter 放在 middleware 之后，会有根本问题：

- middleware 看到的还是 raw dict
- middleware 的截断依据仍是 `result` / `content` 等原始字段，而不是最终的 LLM 文本
- formatter 之后还有机会重新引入包装结构，导致 LLM 仍看到“截断后的 dict 壳子”

因此正确顺序应是：

```text
raw tool output
  → formatter
  → llm-facing output
  → LongToolOutputMiddleware
  → final llm-facing output
  → ToolResultBlock
```

### 为什么采用 YAML formatter

NexAU 已经有清晰的 YAML tool 定义入口（`ToolYamlSchema` / `Tool.from_yaml()`）。formatter 通过 YAML 暴露的好处是：

- 符合当前工具声明模型
- 可以不修改 tool implementation 就切换 LLM 输出样式
- builtin tool、外部工具、MCP 包装工具都能统一接入
- 支持按工具定制，同时保持框架默认行为

## 设计

### 概述

引入一个**双通道工具输出模型**：

```text
Tool implementation
    ↓
raw tool output
    ↓
formatter resolution
    ├─ YAML 指定 custom formatter
    └─ 未指定 → 默认 XML formatter
    ↓
llm_tool_output
    ↓
after_tool middleware
    ├─ tool_output      (raw channel)
    └─ llm_tool_output  (LLM channel)
    ↓
final llm_tool_output
    ↓
ToolResultBlock.content
```

核心原则：

1. **`tool_output` 保留原始结构**，供事件、调试、frontend、程序逻辑使用
2. **`llm_tool_output` 是唯一 LLM 输入源**，由 formatter 生成，并由 middleware 截断
3. **默认总有 formatter**，即使 YAML 未配置，也使用内置 XML formatter
4. **LongToolOutputMiddleware 针对 `llm_tool_output` 工作**，而不是只看 raw dict

### 详细设计

#### 1. Tool YAML 增加 `formatter` 字段

在 `ToolYamlSchema` 中新增可选字段：

```python
class ToolYamlSchema(BaseModel):
    type: Literal["tool"] | None = Field(default=None)
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    ...
    binding: str | None = None
    formatter: str | None = None
```

语义：

- 若配置，则表示 custom formatter
- 若未配置，则框架自动回退到内置 `xml` formatter

本 RFC 约束：

- builtin formatter alias **首期只支持 `xml`**
- 未来如需 `markdown` / `plain`，另起后续增量设计，不在本 RFC 首批范围内

建议支持两种写法：

```yaml
formatter: xml
```

```yaml
formatter: nexau.archs.tool.formatters.agent:format_agent_tool_output
```

#### 1.1 非 YAML Tool 也支持 formatter

除 `Tool.from_yaml()` 外，纯 Python 构造的 `Tool(...)` 也应暴露 `formatter` 参数，并保持与 YAML 一致的默认行为：

```python
Tool(
    name="my_tool",
    description="...",
    input_schema={...},
    implementation=my_impl,
    formatter="xml",  # optional
)
```

规则：

- 未显式传入时，同样默认回退到内置 `xml` formatter
- 允许传 builtin alias（当前仅 `xml`）
- 允许传 import path 或直接传 callable（如框架已有此能力）

这样可以保证 YAML Tool 与 Python Tool 的行为一致，不引入两套规则。

#### 2. Formatter 接口

formatter 是一个纯函数，输入为工具执行上下文，输出为 LLM-facing 内容：

```python
@dataclass(frozen=True)
class ToolFormatterContext:
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: object
    tool_call_id: str | None
    is_error: bool


ToolFormatter = Callable[[ToolFormatterContext], object]
```

返回值约定：

- `str`：最常见，直接作为 LLM 文本
- `dict` / `list`：允许 formatter 返回结构，供后续 `coerce_tool_result_content()` 继续处理
- image-compatible 结构：保留现有 multimodal 管道

#### 3. 默认 XML formatter

当 tool YAML 未配置 formatter 时，统一使用 NexAU 内置 XML formatter。

默认 XML formatter 的目标不是保留原始 JSON，而是生成**稳定、扁平、可截断**的文本边界。例如：

```xml
<tool_result>
  <meta>
    <status>success</status>
    <sub_agent_name>explore</sub_agent_name>
    <sub_agent_id>d6a23025ed0c</sub_agent_id>
  </meta>
  <body field="result"><![CDATA[
## Answer

...长文本正文...
  ]]></body>
</tool_result>
```

默认 XML formatter 规则：

1. **字符串输出**：可直接返回原字符串，避免无意义包裹
2. **图片输出 bypass XML formatter**：对 MCP tool 与 Python tool，只要 formatter 输入中检测到 image-like / multimodal image 结果，就直接 bypass XML formatter，原样交给后续 `coerce_tool_result_content()` 处理
3. **Dict / List 输出**：渲染为 XML 文本
4. **忽略 display-only 字段**：如 `returnDisplay`
5. **单字段正文直通**：若在剥离 `returnDisplay` 等 display-only 字段后，Dict 只剩 `content` 或 `result` 一个键，则直接返回该值，不再包一层 XML
6. **识别主体字段**：优先 `result`、`content`、`stdout`、`stderr`、`message`，再回退到最长 multiline string
7. **短字段进入 `<meta>`**
8. **长正文进入 `<body>`**

#### 4. Custom formatter 的 YAML 配置方式

例如 AgentTool 可以配置专用 formatter：

```yaml
name: Agent
binding: nexau.archs.tool.builtin.agent_tool:call_sub_agent
formatter: nexau.archs.tool.formatters.agent:format_agent_tool_output
```

实现后补充的 builtin 示例：`run_shell_command` 也可以配置专用 formatter，以输出更接近 Claude Code BashTool 的 shell transcript 文本，而不是通用 XML：

```yaml
name: run_shell_command
binding: nexau.archs.tool.builtin.shell_tools:run_shell_command
formatter: nexau.archs.tool.formatters.shell:format_run_shell_command_output
```

其目标输出形态示意：

```text
hello world
warn: something happened

Command running in background with ID: 123. Output is being written to: /tmp/.../stdout.txt
```

custom formatter 可以输出更贴近工具语义的文本：

```text
Sub-agent finished (sub_agent_name: explore, sub_agent_id: d6a23025ed0c).

## Answer

...正文...
```

也可以继续输出工具特定 XML：

```xml
<sub_agent_result name="explore" id="d6a23025ed0c" status="success">
  <body><![CDATA[
## Answer
...正文...
  ]]></body>
</sub_agent_result>
```

#### 5. After-tool 双通道：`tool_output` + `llm_tool_output`

当前 `AfterToolHookInput` 只有一个 `tool_output` 字段，不足以区分：

- 原始 runtime 结果
- 发给 LLM 的格式化结果

本 RFC 提议扩展为双通道：

```python
@dataclass
class AfterToolHookInput(BeforeToolHookInput):
    tool_output: Any = None
    llm_tool_output: Any = None


@dataclass
class HookResult:
    ...
    tool_output: Any | None = None
    llm_tool_output: Any | None = None
```

职责划分：

| 字段 | 含义 | 主要消费者 |
|------|------|------------|
| `tool_output` | 原始工具结果 | frontend、事件、日志、调试、程序逻辑 |
| `llm_tool_output` | formatter 后结果 | LongToolOutputMiddleware、ToolResultBlock、LLM |

兼容策略：

- 老 middleware 继续读 `tool_output`，行为不变
- 新的 LLM-oriented middleware 读 `llm_tool_output`
- `LongToolOutputMiddleware` 优先读写 `llm_tool_output`

建议进一步明确迁移分层：

1. **raw-oriented middleware**：继续消费 `tool_output`
   - 例如事件分发、调试日志、frontend display
2. **LLM-oriented middleware**：消费 `llm_tool_output`
   - 例如 LongToolOutputMiddleware
3. **混合型 middleware**：可同时读取两个通道
   - 例如需要把 raw metadata 与 llm preview 同时记录的日志逻辑

#### 6. 挂载点：formatter 在 middleware 之前执行

建议链路：

```python
raw_output = output if output is not None else content

llm_output = tool.format_output_for_llm(
    tool_input=tool_parameters,
    tool_output=raw_output,
    tool_call_id=str(call_id),
    is_error=bool(feedback.get("is_error")),
)

hook_input = AfterToolHookInput(
    agent_state=agent_state,
    sandbox=sandbox,
    tool_name=tool_name,
    tool_input=tool_parameters,
    tool_output=raw_output,
    llm_tool_output=llm_output,
    tool_call_id=tool_call_id,
)

after_result = middleware_manager.run_after_tool(hook_input, llm_output)
final_llm_output = after_result.llm_tool_output if provided else llm_output

tool_result_block = ToolResultBlock(
    tool_use_id=str(call_id),
    content=coerce_tool_result_content(final_llm_output, fallback_text=None),
    is_error=bool(feedback.get("is_error")),
)
```

其中 `tool.format_output_for_llm(...)` 负责：

1. 解析 YAML 中的 `formatter`
2. 若缺失则回退到内置 XML formatter
3. 构造 `ToolFormatterContext`
4. 调用 formatter 返回 `llm_tool_output`

#### 7. 修改 `LongToolOutputMiddleware`

`LongToolOutputMiddleware` 需要升级为：**优先处理 formatter 后的 `llm_tool_output`**。

新的行为：

1. 若存在 `llm_tool_output`，对其进行长度测量、截断、落盘、提示注入
2. 若不存在 `llm_tool_output`，再回退到旧行为处理 `tool_output`
3. middleware 修改结果时，优先回写 `llm_tool_output`
4. `tool_output` 原则上保持原始结构，不因 LLM 截断被污染

伪代码：

```python
def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
    llm_output = hook_input.llm_tool_output
    if llm_output is None:
        llm_output = hook_input.tool_output

    serialized = self._serialize_for_measurement(llm_output)
    if len(serialized) <= self.max_output_chars:
        return HookResult.no_changes()

    truncated = self._truncate(serialized)
    saved_path = self._save_to_temp_file(full_text=serialized, ...)
    hinted = truncated + self._build_hint(saved_path)

    return HookResult.with_modifications(llm_tool_output=hinted)
```

这样保证：**被 middleware 截断的内容，就是最终发给 LLM 的内容。**

#### 7.1 其他 middleware 的迁移策略

除 `LongToolOutputMiddleware` 外，还需要明确其他 after-tool middleware 对双通道的使用边界：

- **AgentEventsMiddleware**：默认继续使用 `tool_output` 生成事件内容，因为前端/事件系统目前更偏向消费原始结构化结果，而不是 LLM-facing 文本
- **LoggingMiddleware.after_tool**：建议保留 `tool_output` 预览，同时可选增加 `llm_tool_output` 预览，便于排查“raw 输出”和“模型实际看到的输出”不一致问题
- **未来新增 middleware**：如果其目标是影响 LLM 上下文，应默认优先使用 `llm_tool_output`

这意味着双通道不是临时兼容层，而是明确的职责分离。

#### 8. `coerce_tool_result_content()` 的调整

一旦 formatter 成为标准入口，就不应继续依赖：

```python
fallback_text=str(content)
```

否则即使 formatter 和 middleware 已正确处理，最终仍可能退化成 Python repr。

建议：

- executor 侧传入 `fallback_text=None`
- `coerce_tool_result_content()` 继续负责图片 / multimodal 兼容
- 对 formatter 产出的字符串直接原样进入 `ToolResultBlock.content`

## 示例

### 示例 1：默认 XML formatter + LongToolOutputMiddleware

原始输出：

```python
{
    "status": "success",
    "sub_agent_name": "explore",
    "sub_agent_id": "d6a23025ed0c",
    "message": "Find the RuntimeBundle struct...",
    "result": "## Answer\n\nNexAU handles tool results through..."
}
```

formatter 生成：

```xml
<tool_result>
  <meta>
    <status>success</status>
    <sub_agent_name>explore</sub_agent_name>
    <sub_agent_id>d6a23025ed0c</sub_agent_id>
  </meta>
  <body field="result"><![CDATA[
## Answer

NexAU handles tool results through...
  ]]></body>
</tool_result>
```

若超长，`LongToolOutputMiddleware` 截断的是上面的 XML 文本，而不是原始 dict。

### 示例 2：Agent tool 使用 custom formatter

YAML：

```yaml
name: Agent
formatter: nexau.archs.tool.formatters.agent:format_agent_tool_output
```

LLM 最终收到：

```text
Sub-agent finished (sub_agent_name: explore, sub_agent_id: d6a23025ed0c).

## Answer

NexAU handles tool results through...
```

若超长，middleware 截断的是这段文本本身。

### 示例 3：简单 Dict 走默认 XML formatter

原始输出：

```python
{"status": "ok", "message": "File created", "path": "/tmp/test.py"}
```

格式化后：

```xml
<tool_result>
  <meta>
    <status>ok</status>
    <message>File created</message>
    <path>/tmp/test.py</path>
  </meta>
</tool_result>
```

## 权衡取舍

### 考虑过的替代方案

| 方案 | 优点 | 缺点 | 决定 |
|------|------|------|------|
| A: 仅把 `str(dict)` 改成 `json.dumps` | 改动小 | 仍然是结构包裹，长正文仍埋在字段里 | 否 |
| B: formatter 放在 middleware 之后 | 侵入较小 | middleware 截断的不是最终 LLM 输入对象 | 否 |
| C: 所有工具强制固定 formatter，不允许配置 | 简单 | 无法针对 Agent / MCP / Bash 等工具做定制优化 | 否 |
| **D: tool YAML 配置 custom formatter，默认 XML formatter，且 formatter 先于 middleware 执行** | 统一入口、截断对象正确、可扩展 | 需要扩展 hook 双通道 | **采用** |

### 缺点

1. **XML 会引入少量额外 token**，但比 `str(dict)` 噪声更可控
2. **hook 双通道增加了执行链路复杂度**
3. **默认 XML formatter 仍有启发式**，例如主体字段识别不可能覆盖所有极端情况

## 实现计划

### Phase 1: formatter 基础设施

- [x] 在 `ToolYamlSchema` 增加 `formatter` 字段
- [x] 在 `Tool` / `Tool.from_yaml()` 中解析 formatter 配置，并让纯 Python `Tool(...)` 也支持 formatter 参数
- [x] 建立 formatter resolver，支持 builtin alias 与 import path（首期 builtin alias 仅 `xml`）
- [x] 实现内置默认 XML formatter
- [x] 为 image-like / multimodal image 输出建立 XML formatter bypass 规则（覆盖 MCP tool 与 Python tool）

### Phase 2: after-tool 双通道接入

- [x] 扩展 `AfterToolHookInput` / `HookResult`，支持 `llm_tool_output`
- [x] formatter 在 after-tool middleware 之前执行，生成 `llm_tool_output`
- [x] 修改 `LongToolOutputMiddleware`，优先对 `llm_tool_output` 做截断与持久化
- [x] 保留 `tool_output` 原始结构，避免 LLM-oriented 截断污染 raw output
- [x] 明确 AgentEventsMiddleware / LoggingMiddleware 的双通道消费策略

### Phase 3: ToolResultBlock 链路与工具适配

- [x] `ToolResultBlock` 构建只消费最终 `llm_tool_output`
- [x] 移除 `fallback_text=str(content)` 这类 repr 回退路径
- [x] 为 AgentTool 增加专用 custom formatter
- [x] 评估 Bash / MCP / 其他内置工具是否需要自定义 formatter（本次已为 `run_shell_command` 落地 Claude Code 风格 formatter）

### 相关文件

| 文件 | 说明 |
|------|------|
| `nexau/archs/tool/tool.py` | Tool YAML schema、formatter 解析、Tool 封装 |
| `nexau/archs/main_sub/execution/hooks.py` | after-tool hook 输入/输出增加 `llm_tool_output` |
| `nexau/archs/main_sub/execution/tool_executor.py` | 组装 raw output / llm output 双通道并执行 middleware |
| `nexau/archs/main_sub/execution/middleware/long_tool_output.py` | 改为基于 formatter 后输出做截断 |
| `nexau/archs/main_sub/execution/executor.py` | `ToolResultBlock` 仅消费最终 `llm_tool_output` |
| `nexau/core/messages.py` | `coerce_tool_result_content()` 去掉 repr 兜底依赖 |
| `nexau/archs/tool/formatters/` | 新增 formatter 实现目录 |

## 测试方案

### 单元测试

- `ToolYamlSchema` 能正确解析 `formatter` 字段
- 纯 Python `Tool(...)` 未配置 formatter 时也能默认回退到 `xml`
- formatter resolver 能正确处理：未配置 → `xml`、builtin alias（仅 `xml`）、import path
- 默认 XML formatter 对简单 Dict、长 `result` Dict、嵌套对象、纯字符串行为正确
- 默认 XML formatter 对剥离 display-only 字段后仅剩 `content` / `result` 的 Dict，直接返回正文值
- MCP tool / Python tool 的 image-like output 能正确 bypass XML formatter
- `LongToolOutputMiddleware` 在存在 `llm_tool_output` 时优先截断它，而不是 `tool_output`

### 集成测试

- Sub-agent tool：返回长 `result` 时，LLM 接收的是 formatter 文本而不是 Dict repr
- `LongToolOutputMiddleware` + formatter：formatter 先生成 LLM 文本，middleware 对该文本截断，最终 LLM 看到截断后的 formatter 文本
- 自定义 formatter：YAML 中指定 custom formatter 后，LLM 看到自定义格式而不是默认 XML
- `run_shell_command`：LLM 接收 Claude Code 风格 shell 结果文本（stdout / stderr / background info），而不是 metadata-heavy dict 或通用 XML
- AgentEventsMiddleware 继续基于 raw `tool_output` 发事件，不受 `llm_tool_output` 截断影响

### 手动验证

- 运行一次包含 Sub-agent 的真实对话，确认 tool_result 不再出现 `{'status': ...}` 这种 repr
- 人工检查长输出被截断后，LLM 看到的仍是 XML / custom formatter 文本，而不是 dict 壳子

## 已确认的设计决策

1. builtin formatter alias 首期**只支持 `xml`**
2. 纯 Python 构造的 `Tool(...)` 也支持 `formatter` 参数，且默认同样回退到 XML formatter
3. 对于 MCP tool 与 Python tool，只要输出是图片 / image-like multimodal 结果，就 **bypass XML formatter**
4. after-tool 阶段采用 **`tool_output` + `llm_tool_output` 双通道**，并按职责拆分 middleware 消费路径
5. display-only 字段剥离后，如果工具输出只剩单个 `content` 或 `result` 键，则直接把该值作为 LLM-facing output
6. `run_shell_command` 采用专用 shell formatter，按 Claude Code BashTool 的风格向 LLM暴露 stdout / stderr / background 信息

## 未解决的问题

1. 如果未来需要 `markdown` / `plain` formatter，builtin alias 命名与语义如何设计？
2. AgentEventsMiddleware 是否需要在未来事件协议中同时携带 raw output 与 llm output？
3. `llm_tool_output` 是否需要在调试日志和 tracing 中默认落盘，以便排查 formatter 问题？

## 参考资料

- Claude Code (`~/claude_code_2188`) 的 per-tool tool-result formatting 思路
- NexAU `nexau/archs/tool/tool.py` — tool YAML 定义与加载逻辑
- NexAU `nexau/archs/main_sub/execution/hooks.py` — middleware hook 数据结构
- NexAU `nexau/archs/main_sub/execution/tool_executor.py` — after-tool middleware 链路
- NexAU `nexau/archs/main_sub/execution/executor.py` — `ToolResultBlock` 构建点
- NexAU `nexau/core/messages.py` — `coerce_tool_result_content()`
