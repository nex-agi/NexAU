# RFC-0014: 提示词工程与模板化

- **状态**: implemented
- **优先级**: P2
- **标签**: `prompt`, `templating`, `jinja2`, `dx`, `architecture`
- **影响服务**: `nexau/archs/main_sub/prompt_builder.py`, `nexau/archs/main_sub/prompt_handler.py`, `nexau/archs/main_sub/prompts/*.j2`, `nexau/archs/main_sub/config/base.py`, `nexau/archs/main_sub/config/config.py`, `nexau/archs/main_sub/agent.py`, `examples/*/systemprompt*.md`, `examples/*/*.yaml`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

NexAU 把"运行时给 LLM 的 system prompt"抽象成一条管线：用户在 `AgentConfig.system_prompt` 里以 **string / file / jinja** 三种类型之一指定 base prompt，框架渲染后再依次拼接 tools 文档、sub-agents 文档、tool execution instructions、可选的 `NEXAU.md` 项目说明，得到最终发给 LLM 的字符串。本 RFC 描述这条管线的两个核心类（`PromptBuilder` 与 `PromptHandler`）、5 份内置 Jinja2 模板、`system_prompt_suffix` 与 `NEXAU.md` 的注入位置、相对路径解析时机、以及 `examples/` 下"多 system prompt 变体"（codex / swe / zh / apply_patch / v1）的命名约定。

## 动机

prompt 是 agent 的灵魂；NexAU 必须解决的设计问题：

1. **prompt 来源多样**：用户可能写一段内联 string、引用一个 `.md` 文件、或一个 Jinja2 `.j2` 文件。框架不能让用户为这三种来源分别学一套 API。
2. **prompt 内必须能引用运行时事实**：当前 agent 名、可用工具列表、sub-agent 列表、tool 参数 schema、运行时 context（如 working_directory）。模板引擎不能选 `str.format`（不支持 `{% for %}`），必须选支持 control flow 的 Jinja2。
3. **多个变体文件常态化**：`examples/code_agent/` 下有 6 份 `systemprompt*.md`（默认 / codex / swe / zh / apply_patch / v1），分别给不同 LLM provider 或 ablation 实验用。命名约定必须可读、易选。
4. **structured-tool-call 模式不需要拼工具说明**：当 LLM 走 OpenAI/Anthropic 原生 function-calling 时，工具 schema 由 SDK 直接传，不能再在 prompt 里重复一份否则浪费 token。
5. **项目级补充上下文**：用户希望在 sandbox work_dir 放一份 `NEXAU.md`（类似 Claude Code 的 `CLAUDE.md`）自动注入，而不是每次都改 system prompt。

这些诉求若被分散在 ad-hoc 代码里，会导致两个 agent 的 prompt 拼接顺序不一致、模板变量名漂移、`NEXAU.md` 在某些路径下被忽略。本 RFC 把它们冻结为一条**唯一的 build path**。

## 设计

### 概述

```
                 ┌──────────────────────────────────────────────┐
                 │ AgentConfig (base.py:41-43)                  │
                 │  - system_prompt: str | None                 │
                 │  - system_prompt_type: "string"|"file"|jinja │
                 │  - system_prompt_suffix: str | None          │
                 └────────────────┬─────────────────────────────┘
                                  │ Agent.run()
                                  ▼
                ┌─────────────────────────────────────────────┐
                │ PromptBuilder.build_system_prompt           │
                │ (prompt_builder.py:87)                      │
                └───────┬─────────────────┬───────────────────┘
                        │                 │
                        ▼                 ▼
        ┌────────────────────────┐ ┌────────────────────────────┐
        │ _get_base_system_prompt│ │ _build_capabilities_docs   │
        │ (prompt_builder.py:131)│ │ (prompt_builder.py:220)    │
        │  → PromptHandler.       │ │  → tools_template.j2       │
        │    create_dynamic_prompt│ │  → sub_agents_template.j2  │
        │    (prompt_handler.py:193)│ │                          │
        └──────────┬─────────────┘ └──────────────┬─────────────┘
                   │ + suffix                     │
                   │ + NEXAU.md (if present)      │
                   └──────────────┬───────────────┘
                                  │ + tool_execution_instructions.j2
                                  │   (skip if structured tool calls)
                                  ▼
                  ┌────────────────────────────────┐
                  │ final system prompt string     │
                  │ (passed to LLM)                │
                  └────────────────────────────────┘
```

最终拼接顺序（prompt_builder.py:108-125）：

```
[base_prompt][system_prompt_suffix][NEXAU.md block][tools docs][sub-agents docs][tool_execution_instructions]
```

### 详细设计

#### 1. 三种 prompt 类型

`AgentConfigBase`（base.py:41-43）声明：

```python
system_prompt: str | None = None
system_prompt_type: Literal["string", "file", "jinja"] = "string"
system_prompt_suffix: str | None = None
```

`PromptHandler.process_prompt`（prompt_handler.py:47-73）按 `prompt_type` 分发：

| `prompt_type` | 处理 | 模板能力 |
|---------------|------|----------|
| `string` | `str.format(**context)`（prompt_handler.py:75-92） | 仅 `{var}` 替换；`KeyError` 时原样保留 |
| `file` | 读文件后 `str.format`（prompt_handler.py:94-125） | 同 string；额外做路径回退到 cwd |
| `jinja` | 读文件后 `Jinja2.Template.render(**context)`（prompt_handler.py:127-160） | 完整 Jinja2 控制流（`{% for %}`, `{% if %}`） |

**核心约束**：`string` 类型用 `str.format` 而不是 Jinja，是因为大多数用户写内联字符串时只需要简单替换；引入 Jinja 会迫使他们 escape `{`/`}`。但当用户在 `string` 类型里需要循环时，框架仍会用 Jinja 渲染——`PromptHandler.create_dynamic_prompt`（prompt_handler.py:193-220）对 `string` 类型走 `jinja_env.from_string(...).render(...)`（prompt_handler.py:215-216），并不直接调用 `_process_string_prompt`。这是**两条 string 处理路径**：

- `process_prompt(prompt_type="string", ...)` → `str.format` 路径（向后兼容）
- `create_dynamic_prompt(template_type="string", ...)` → Jinja 渲染路径（PromptBuilder 实际使用）

`PromptBuilder` 调用的是后者（prompt_builder.py:148），因此**最终运行时 string 类型也走 Jinja**。`process_prompt` 是公开但当前未被 PromptBuilder 直接使用的二级 API，用于第三方扩展。

#### 2. 路径解析时机

`system_prompt_type ∈ {"file", "jinja"}` 时，`AgentConfig.system_prompt` 是路径字符串。**路径解析发生在 config build 阶段**，不在运行时（config.py:828-849）：

```python
def build_system_prompt_path(self) -> AgentConfigBuilder:
    system_prompt = self.agent_params.get("system_prompt")
    system_prompt_type = self.agent_params.get("system_prompt_type", "string")

    if system_prompt and system_prompt_type in ["file", "jinja"] and not Path(system_prompt).is_absolute():
        system_prompt = self.base_path / system_prompt
        if not Path(system_prompt).exists():
            raise ConfigError(f"System prompt file not found: {system_prompt}")
        self.agent_params["system_prompt"] = str(system_prompt)
    return self
```

**约束**：

- 路径相对于 YAML 配置文件所在目录（`self.base_path`），不是 cwd。这让 `system_prompt: ./systemprompt.md` 在不同 cwd 下行为一致。
- 文件**必须**在 build 阶段存在，否则 `ConfigError`。运行时 `PromptHandler._process_jinja_prompt` 还有第二道 cwd 回退（prompt_handler.py:135-139）作为 defense in depth，正常路径用不到。
- 已经是绝对路径时跳过解析。

#### 3. PromptBuilder 主流程

`PromptBuilder.build_system_prompt`（prompt_builder.py:87-129）的执行顺序固定：

```python
base_prompt = self._get_base_system_prompt(agent_config, runtime_context)
if include_tool_instructions:
    capabilities_docs = self._build_capabilities_docs(tools, sub_agents, runtime_context)
    execution_instructions = self._get_tool_execution_instructions() or ""
    return f"{base_prompt}{capabilities_docs}{execution_instructions}"
else:
    return base_prompt
```

**`include_tool_instructions=False` 的触发条件**：在 `Agent.run`（agent.py:814-820）把 `not use_structured_tool_calls` 传进来——也就是说，当 `tool_call_mode` 是 `openai` / `anthropic` 等原生 function-calling 模式时，工具 schema 由 SDK 直接处理，**整段 capabilities + execution instructions 被跳过**。仅当 `tool_call_mode="xml"` 时这段才进入 prompt。

#### 4. `_get_base_system_prompt`：base prompt 渲染 + suffix + NEXAU.md

代码（prompt_builder.py:131-167）：

```python
if not agent_config.system_prompt:
    return self._get_default_system_prompt(agent_config.name)

context = self._build_template_context(runtime_context)
rendered = self.prompt_handler.create_dynamic_prompt(
    agent_config.system_prompt,
    agent_config,
    additional_context=context,
    template_type=agent_config.system_prompt_type,
)
if agent_config.system_prompt_suffix:
    rendered += agent_config.system_prompt_suffix

nexau_md = self._load_nexau_md(agent_config, runtime_context)
if nexau_md:
    rendered += f"\n\n# Project Instructions (NEXAU.md)\n\n{nexau_md}"

return rendered
```

**两个 append 点都在 base prompt 之后、capabilities 之前**：

| 段落 | 作用 | 来源 |
|------|------|------|
| `system_prompt_suffix` | 团队模式注入 team-context（RFC-0002 用） | `AgentConfig.system_prompt_suffix`（base.py:43） |
| `NEXAU.md` block | 项目级补充指令（类似 Claude Code 的 CLAUDE.md） | sandbox work_dir / `NEXAU.md`（prompt_builder.py:182-218） |

`_load_nexau_md` 的 work_dir 解析顺序（prompt_builder.py:197-199）：

1. `runtime_context["working_directory"]`（运行时显式给）
2. `agent_config.sandbox_config.work_dir`（YAML 配置给）
3. 都没有 → 跳过注入

`NEXAU.md` 不存在或读取失败 → 静默跳过（仅 logger.warning），不抛异常。这种"best-effort"设计避免在没有 sandbox 的场景下强制要求 `NEXAU.md`。

#### 5. capabilities 文档：tools + sub-agents

由 `_build_capabilities_docs`（prompt_builder.py:220-241）拼接两段，顺序固定为 **tools 在前、sub-agents 在后**：

- `_build_tools_documentation`（prompt_builder.py:243-276）→ 用 `tools_template.j2` 渲染
- `_build_subagents_documentation`（prompt_builder.py:278-303）→ 用 `sub_agents_template.j2` 渲染

每个 tool 在渲染前被规范化为 `ToolInfo` TypedDict（prompt_builder.py:42-50）：

```python
{
    "name": str,
    "description": str,
    "template_override": str | None,   # tool 自定义文档片段
    "parameters": list[ToolParameter],  # JSON Schema → Python 类型字符串
    "as_skill": bool,
    "skill_description": str | None,
}
```

JSON Schema → Python 类型的映射在 `_get_python_type_from_json_schema`（prompt_builder.py:53-70）：`string→str / integer→int / number→float / boolean→bool / array→list / object→dict`。这层归一化让 prompt 里的参数类型描述对人类可读（不是 `"integer"` 而是 `"int"`）。

#### 6. 5 份内置 Jinja2 模板

位于 `nexau/archs/main_sub/prompts/`：

| 文件 | 用途 | 由谁加载 |
|------|------|----------|
| `default_system_prompt.j2` | `agent_config.system_prompt` 为空时的 fallback | `_get_default_system_prompt`（prompt_builder.py:169-180） |
| `tools_template.j2` | 工具文档段（XML 模式下嵌入 prompt） | `_build_tools_documentation`（prompt_builder.py:250） |
| `sub_agents_template.j2` | sub-agent 文档段 | `_build_subagents_documentation`（prompt_builder.py:284） |
| `tool_execution_instructions.j2` | XML tool call 的格式与"一次只发一种 tool call"约束 | `_get_tool_execution_instructions`（prompt_builder.py:335-345） |
| `tools_template_for_skill_detail.j2` | tool 在以 skill 形式展示时的扩展模板 | 由 tool template_override / skill_description 间接触发 |

**Jinja2 环境配置**（prompt_builder.py:80-84）：

```python
self.jinja_env = Environment(
    loader=FileSystemLoader(self.prompts_dir),
    trim_blocks=True,
    lstrip_blocks=True,
)
```

`trim_blocks=True / lstrip_blocks=True` 是关键：让 Jinja 控制流（`{% for %}{% endfor %}`）不在输出里留多余空行——否则 system prompt 会被 Jinja 自动换行污染。

#### 7. `examples/` 下的多 prompt 变体约定

`examples/code_agent/` 下并存多份 `systemprompt*.md`：

| 文件 | 用途 |
|------|------|
| `systemprompt.md` | 默认（中性、CLI 风格） |
| `systemprompt_codex.md` | OpenAI Codex 调优版 |
| `systemprompt_swe.md` | SWE-bench 调优版（强调 patch 输出） |
| `systemprompt_apply_patch.md` | 强制 `apply_patch` workflow 的版本 |
| `systemprompt_zh.md` | 中文版 |
| `systemprompt_v1.md` | 历史版本（保留作 ablation 基线） |

对应的 YAML 配置（如 `code_agent_codex.yaml` / `code_agent_gpt_swe_prompt.yaml`）通过 `system_prompt: ./systemprompt_xxx.md` + `system_prompt_type: jinja` 切换。**命名约定**：`systemprompt_<purpose>.md`；purpose 段是单一短词（codex / swe / zh / v1 / apply_patch）。

`examples/nexau_building_team/` 沿用同样模式但语义不同：`systemprompt_<role>.md`（leader / builder / rfc_writer），role 是团队角色名而非 model 调优维度。两套约定在不同目录里不冲突，但读者要从目录上下文区分。

### 示例

最小 YAML 例（`examples/code_agent/code_agent.yaml`）：

```yaml
type: agent
name: nexau_code_agent
system_prompt: ./systemprompt.md
system_prompt_type: jinja
tool_call_mode: openai      # 触发 include_tool_instructions=False
max_iterations: 300
llm_config:
  model: ${env.LLM_MODEL}
  ...
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
  ...
```

build 阶段：

1. `AgentConfigBuilder.build_system_prompt_path` 把 `./systemprompt.md` 解析成 `examples/code_agent/systemprompt.md` 的绝对路径。
2. 文件存在性检查通过；`agent_params["system_prompt"]` 改写为绝对路径字符串。

run 阶段，`Agent.run` → `PromptBuilder.build_system_prompt`：

1. `_get_base_system_prompt`：因 `system_prompt_type="jinja"`，走 `PromptHandler.create_dynamic_prompt`（prompt_handler.py:211-213）→ `_process_jinja_prompt`（prompt_handler.py:127-160）读文件 + Jinja 渲染，可用变量含 `agent_name / agent_id / system_prompt_type / timestamp` 与 `runtime_context` 合并出来的 `working_directory` 等。
2. 没配 `system_prompt_suffix` → 跳过。
3. 若 `examples/code_agent/NEXAU.md` 存在 → 末尾追加 `# Project Instructions (NEXAU.md)\n\n<content>`。
4. 因 `tool_call_mode="openai"` → `use_structured_tool_calls=True`（agent.py:819），不追加 capabilities + execution instructions（让 OpenAI SDK 自己处理 function calling）。

最终 prompt 仅含 base + 可选 NEXAU.md。

## 权衡取舍

### 考虑过的替代方案

1. **完全用 Jinja2 替代 `str.format`**：`process_prompt` string 分支也走 Jinja。拒绝原因：用户内联简单字符串写 `"Hello {name}"` 时，Jinja 把 `{name}` 当作字面量，必须改写为 `"Hello {{ name }}"`——破坏直觉。当前的"两条 string 路径"是有意保留兼容。
2. **Mustache / Handlebars 替代 Jinja2**：跨语言更标准。拒绝原因：Python 生态里 Jinja2 是事实标准，且支持自定义 filter / global，比 Mustache 灵活；NexAU 不是跨语言项目。
3. **不内置 capabilities templates，让用户在 system_prompt 里手写**：拒绝原因：(a) 工具列表运行时变化；(b) 多 agent 之间会出现格式漂移；(c) sub-agent 的 `<tool_use>agent:X</tool_use>` 调用格式必须由框架统一。
4. **把 `NEXAU.md` 作为 first-class config 字段**：拒绝原因：约定优于配置——用户在 work_dir 放 `NEXAU.md` 就生效，零配置；YAML 里加字段反而增加心智负担。
5. **suffix 改成 prefix（在 base prompt 之前注入）**：拒绝原因：当前 suffix 用于 RFC-0002 团队模式注入"你是 X 团队的成员"。放在 base 之后让 base prompt 的人格设定先成立，team-context 作为补充更自然；放在前面会被后面的 base prompt 覆盖语气。

### 缺点

1. **拼接顺序硬编码**：`base + suffix + NEXAU.md + tools + sub-agents + execution_instructions` 的顺序在 `build_system_prompt` 里写死，用户无法改。需要新顺序时只能改框架。
2. **Jinja 模板放在 package data 里，用户无法 override**：`tools_template.j2` 等 5 份模板在 `nexau/archs/main_sub/prompts/` 下打包发布。用户想自定义工具文档格式，只能 fork repo。Tool 自身的 `template_override` 字段（tool 级别覆盖文档片段）部分缓解这个问题，但模板大框架不可改。
3. **`include_tool_instructions` 与 `tool_call_mode` 的耦合不够显式**：判定写在 `Agent.run`（agent.py:819）的一行 `not use_structured_tool_calls`；用户读 `PromptBuilder` 单独看不出来"什么时候 capabilities 被跳过"。
4. **`NEXAU.md` 注入对 prompt cache 不友好**：每次 run 都重新读文件 + 字符串拼接；若 `NEXAU.md` 较大，每次都会让 prompt 变化（实际不变但 LLM provider 可能视为新前缀）——长期看应缓存哈希。
5. **string 类型有"两条路径"是隐藏复杂度**：`process_prompt` vs `create_dynamic_prompt` 对 `"string"` 行为不同；只有读源码才能发现 `PromptBuilder` 实际走 Jinja 而非 `str.format`。

## 实现计划

### 阶段划分

- [x] Phase 1: `PromptBuilder` + `PromptHandler` 双层（prompt_builder.py / prompt_handler.py 全部）
- [x] Phase 2: 5 份 Jinja2 模板与 trim_blocks 配置（nexau/archs/main_sub/prompts/）
- [x] Phase 3: `AgentConfigBuilder.build_system_prompt_path` 路径解析（config.py:828-849）
- [x] Phase 4: `system_prompt_suffix`（base.py:43）与 RFC-0002 团队模式集成
- [x] Phase 5: `NEXAU.md` 自动注入（prompt_builder.py:182-218）
- [x] Phase 6: `include_tool_instructions` 在 structured tool call 模式下短路（agent.py:819）
- [ ] Phase 7（未来）：模板可 override 机制——允许用户在 work_dir 提供同名 `.j2` 覆盖内置模板
- [ ] Phase 8（未来）：prompt DSL——结构化描述"persona / capabilities / constraints / examples"四段，框架渲染成 prompt
- [ ] Phase 9（未来）：`NEXAU.md` 内容哈希 + 缓存，避免冗余渲染影响 prompt cache

### 相关文件

- `nexau/archs/main_sub/prompt_builder.py` - 主管线（base + suffix + NEXAU + capabilities + execution）
- `nexau/archs/main_sub/prompt_handler.py` - 三类型分发（string / file / jinja）
- `nexau/archs/main_sub/prompts/default_system_prompt.j2` - 无配置时的 fallback
- `nexau/archs/main_sub/prompts/tools_template.j2` - 工具文档段模板
- `nexau/archs/main_sub/prompts/sub_agents_template.j2` - sub-agent 文档段模板
- `nexau/archs/main_sub/prompts/tool_execution_instructions.j2` - XML 模式下的执行规则
- `nexau/archs/main_sub/prompts/tools_template_for_skill_detail.j2` - skill 形式工具的扩展模板
- `nexau/archs/main_sub/config/base.py` - `system_prompt / system_prompt_type / system_prompt_suffix` 定义
- `nexau/archs/main_sub/config/config.py` - `build_system_prompt_path` 路径解析与校验
- `nexau/archs/main_sub/agent.py` - `PromptBuilder` 实例化与 `build_system_prompt` 调用点
- `examples/code_agent/systemprompt*.md` - 多变体 prompt 命名约定示例
- `examples/nexau_building_team/systemprompt_*.md` - 角色化 prompt 命名约定示例

## 测试方案

### 单元测试

- **PromptHandler 三类型分发**：
  - `process_prompt(prompt="Hello {name}", prompt_type="string", context={"name":"X"})` → `"Hello X"`。
  - `process_prompt(prompt="<path-to-file>", prompt_type="file", ...)` 读文件并 format。
  - `process_prompt(prompt="<path-to-j2>", prompt_type="jinja", context={"items":[1,2]})` → 含 `{% for %}` 循环正确展开。
  - `process_prompt(prompt_type="invalid")` → `ValueError`。
- **PromptHandler.create_dynamic_prompt** 对 `"string"` 类型走 Jinja：`base_template="{% if x %}Y{% endif %}"` + `additional_context={"x":True}` → `"Y"`（验证不是走 `str.format`）。
- **`_load_nexau_md` 优先级**：mock 文件系统，验证 `runtime_context["working_directory"]` 优先于 `agent_config.sandbox_config.work_dir`，二者皆无则返回 `None`。
- **`_load_nexau_md` 失败静默**：文件存在但权限错误 → 返回 `None`，仅 logger.warning，不抛异常。
- **拼接顺序**：mock 各子函数返回固定字符串，验证最终输出严格按 `base + suffix + NEXAU + tools + sub-agents + execution_instructions` 拼接。
- **`include_tool_instructions=False` 跳过 capabilities**：返回值 == base prompt（无 tools / sub-agents / execution_instructions）。
- **`build_system_prompt_path` 路径解析**：相对路径被解析为 `base_path / 相对路径` 的绝对路径；不存在 → `ConfigError`；绝对路径不变。
- **JSON Schema 类型映射**：`_get_python_type_from_json_schema("integer")` → `"int"`；未知类型回落到 `"str"`。

### 集成测试

- **端到端跑 example**：用 `examples/code_agent/code_agent.yaml`（`tool_call_mode="openai"`）启动一个 agent，捕获最终 system prompt 字符串；断言：(a) 不含 "## Available Tools" 段（被 short-circuit）；(b) 不含 `tool_execution_instructions.j2` 内容；(c) 仅含 base prompt + 可选 NEXAU.md。
- **切换 `tool_call_mode="xml"` 重跑**：相同 YAML 改一行，捕获 prompt；断言：(a) 含 "## Available Tools" + 每个工具 XML 用法；(b) 含 "## Available Sub-Agents" 当 sub_agents 非空；(c) 末尾含 "CRITICAL TOOL EXECUTION INSTRUCTIONS"。
- **多变体切换**：把 `system_prompt: ./systemprompt.md` 改成 `./systemprompt_codex.md`，断言 prompt 主体改变但 capabilities + execution 段保持一致。
- **`NEXAU.md` 自动注入**：在 sandbox work_dir 写一份 `NEXAU.md`，断言 prompt 末尾出现 `# Project Instructions (NEXAU.md)\n\n<content>`。

### 手动验证

1. 在 `examples/code_agent/` 下临时增删 `NEXAU.md`，重启 agent 看 prompt 是否包含/不包含项目说明段。
2. 修改 `tools_template.j2` 中的 `{% for tool %}` 循环格式，重跑 agent 看 prompt 中工具段是否同步变化（验证模板热加载——实际是每次 build 都重读，因此立即生效）。
3. 在 YAML 中把 `system_prompt_suffix` 设为 `"\n\n你是团队 alpha 的研究员。"`，确认它出现在 base 之后、capabilities 之前的位置。

## 未解决的问题

1. **模板用户可 override 的机制**：当前 `tools_template.j2` 等内置模板是 package data，用户无法不 fork 就替换。是否引入"work_dir / `.nexau/prompts/*.j2` 优先于 package data"的查找顺序？
2. **prompt DSL**：是否值得提供更高层的 prompt-as-code 抽象（如 BAML / DSPy 风格），让用户描述意图而不是写模板？这会引入新依赖。
3. **prompt cache 与 `NEXAU.md` 的交互**：`NEXAU.md` 内容变更会让整个前缀变化，破坏 LLM provider 的 prompt cache。是否在拼接时显式分段 + 让 LLM 客户端按段缓存？
4. **多变体的 metadata 缺失**：`systemprompt_codex.md` 与 `systemprompt_swe.md` 之间的差异只能通过 diff 看出。是否在文件 frontmatter 加 `purpose: codex tuning` 字段，让 README 自动列举？
5. **`system_prompt_type="string"` 的两条路径**：`process_prompt` 用 `str.format`、`create_dynamic_prompt` 用 Jinja，对外名义相同。是否统一为单一路径并 deprecate `process_prompt`？

## 参考资料

- `nexau/archs/main_sub/prompt_builder.py:21` — `from jinja2 import Environment, FileSystemLoader`
- `nexau/archs/main_sub/prompt_builder.py:73` — `class PromptBuilder`
- `nexau/archs/main_sub/prompt_builder.py:79` — `prompts_dir`
- `nexau/archs/main_sub/prompt_builder.py:80-84` — Jinja2 `Environment` with `trim_blocks` / `lstrip_blocks`
- `nexau/archs/main_sub/prompt_builder.py:85` — `PromptHandler` 实例化
- `nexau/archs/main_sub/prompt_builder.py:87` — `build_system_prompt`
- `nexau/archs/main_sub/prompt_builder.py:108` — `base_prompt` 取得
- `nexau/archs/main_sub/prompt_builder.py:110-125` — `include_tool_instructions` 短路与最终拼接
- `nexau/archs/main_sub/prompt_builder.py:131` — `_get_base_system_prompt`
- `nexau/archs/main_sub/prompt_builder.py:137-139` — fallback 到默认 prompt
- `nexau/archs/main_sub/prompt_builder.py:148` — `PromptHandler.create_dynamic_prompt` 调用
- `nexau/archs/main_sub/prompt_builder.py:156-157` — `system_prompt_suffix` 追加
- `nexau/archs/main_sub/prompt_builder.py:160-162` — `NEXAU.md` block 追加
- `nexau/archs/main_sub/prompt_builder.py:169` — `_get_default_system_prompt`
- `nexau/archs/main_sub/prompt_builder.py:182` — `_load_nexau_md`
- `nexau/archs/main_sub/prompt_builder.py:197-199` — work_dir 解析顺序（runtime_context > sandbox_config）
- `nexau/archs/main_sub/prompt_builder.py:205` — `<work_dir>/NEXAU.md` 查找
- `nexau/archs/main_sub/prompt_builder.py:220` — `_build_capabilities_docs`
- `nexau/archs/main_sub/prompt_builder.py:243` — `_build_tools_documentation`
- `nexau/archs/main_sub/prompt_builder.py:250` — `tools_template` 加载
- `nexau/archs/main_sub/prompt_builder.py:278` — `_build_subagents_documentation`
- `nexau/archs/main_sub/prompt_builder.py:284` — `sub_agents_template` 加载
- `nexau/archs/main_sub/prompt_builder.py:53-70` — `_get_python_type_from_json_schema`
- `nexau/archs/main_sub/prompt_builder.py:42-50` — `ToolInfo` TypedDict
- `nexau/archs/main_sub/prompt_builder.py:335` — `_get_tool_execution_instructions`
- `nexau/archs/main_sub/prompt_builder.py:363` — `_load_prompt_template`
- `nexau/archs/main_sub/prompt_handler.py:31` — `class PromptHandler`
- `nexau/archs/main_sub/prompt_handler.py:38-45` — `_setup_jinja`
- `nexau/archs/main_sub/prompt_handler.py:47` — `process_prompt`
- `nexau/archs/main_sub/prompt_handler.py:67-72` — 类型 dispatch
- `nexau/archs/main_sub/prompt_handler.py:75` — `_process_string_prompt`（`str.format`）
- `nexau/archs/main_sub/prompt_handler.py:94` — `_process_file_prompt`
- `nexau/archs/main_sub/prompt_handler.py:127` — `_process_jinja_prompt`
- `nexau/archs/main_sub/prompt_handler.py:135-139` — cwd fallback for jinja file
- `nexau/archs/main_sub/prompt_handler.py:162` — `validate_prompt_type`
- `nexau/archs/main_sub/prompt_handler.py:172` — `get_default_context`
- `nexau/archs/main_sub/prompt_handler.py:193` — `create_dynamic_prompt`
- `nexau/archs/main_sub/prompt_handler.py:211-216` — `template_type` dispatch in `create_dynamic_prompt`
- `nexau/archs/main_sub/prompts/default_system_prompt.j2` — 默认 fallback 模板
- `nexau/archs/main_sub/prompts/tools_template.j2` — 工具文档段
- `nexau/archs/main_sub/prompts/sub_agents_template.j2` — sub-agent 文档段
- `nexau/archs/main_sub/prompts/tool_execution_instructions.j2` — XML 模式执行规则
- `nexau/archs/main_sub/prompts/tools_template_for_skill_detail.j2` — skill 形式工具扩展模板
- `nexau/archs/main_sub/config/base.py:41` — `system_prompt`
- `nexau/archs/main_sub/config/base.py:42` — `system_prompt_type`
- `nexau/archs/main_sub/config/base.py:43` — `system_prompt_suffix`
- `nexau/archs/main_sub/config/config.py:180` — builder chain 中调用 `build_system_prompt_path`
- `nexau/archs/main_sub/config/config.py:828` — `build_system_prompt_path`
- `nexau/archs/main_sub/config/config.py:841-842` — 相对路径分支判定与 `base_path / 相对路径` 拼接
- `nexau/archs/main_sub/agent.py:48` — 导入 `PromptBuilder`
- `nexau/archs/main_sub/agent.py:188` — `self.prompt_builder = PromptBuilder()`
- `nexau/archs/main_sub/agent.py:814-820` — `Agent.run` 中 `build_system_prompt` 调用
- `nexau/archs/main_sub/agent.py:819` — `include_tool_instructions=not use_structured_tool_calls`

### 相关 RFC

- `rfcs/0002-agent-team.md` — 团队模式通过 `system_prompt_suffix` 注入 team-context
- `rfcs/0006-rfc-catalog-completion-master-plan.md` — 本 RFC 的任务 T8
- `rfcs/0007-tool-system-architecture-and-binding.md` — 工具元数据来源（tool capabilities 文档段依赖）
- `rfcs/0009-sandbox-isolation-and-lifecycle.md` — sandbox work_dir 是 `NEXAU.md` 的查找根
- `rfcs/0018-skill-system-and-embeddings.md` — skill 形式工具的扩展模板（`tools_template_for_skill_detail.j2`）将由本 RFC 与之共同覆盖
