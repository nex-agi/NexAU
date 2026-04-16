# RFC-0018: 技能系统与嵌入模型

- **状态**: implemented
- **优先级**: P2
- **标签**: `skill`, `tool`, `dx`, `architecture`, `prompt`
- **影响服务**: `nexau/archs/main_sub/skill.py`、`nexau/archs/tool/builtin/skill_tool.py`、`nexau/archs/tool/builtin/description/skill_tool.yaml`、`nexau/archs/main_sub/agent.py`、`nexau/archs/main_sub/config/config.py`、`nexau/archs/tool/tool.py`、`examples/*/skills/*/SKILL.md`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

本 RFC 描述 NexAU 的"技能（Skill）"子系统：以 SKILL.md 文件夹形式承载领域知识、以 LoadSkill 工具按需注入详细内容到上下文，以及把带 `as_skill=True` 的工具自动物化为技能项。同时记录"嵌入模型"目标维度——当前实现走精确名匹配，向量召回属于未实现部分，本 RFC 在"未解决的问题"中明确划界。技能系统在 `nexau/archs/main_sub/skill.py` 集中实现，供 `AgentConfigBuilder._finalize` 与 `Agent.__init__` 在装配阶段统一接入。

## 动机

ReAct agent 的系统提示词预算紧张：把所有领域细节都压进 system prompt 会浪费 token、且对很多任务无关。同时 nexau 同时支持人工编写的 SKILL.md（带 frontmatter 的 markdown 文件夹）和把工具按"高描述成本工具"标记为 skill 两种形态，需要统一的注册、查询与渲染管道。技能系统的核心动机：

- **按需加载**：只把每个技能的简短描述放进 LoadSkill 工具描述中，详细内容（XML usage、参数表、长 markdown）由模型主动调用 LoadSkill 拉取（`skill.py:183-197`、`skill.py:200-224`）。
- **统一形态**：人工 SKILL.md 与 tool-as-skill 走同一个 `skill_registry`，避免两套查找逻辑（`agent.py:172-185`、`config.py:691-693`）。
- **零侵入注入**：当且仅当存在技能或 `as_skill=True` 的工具时，才在工具列表里追加 LoadSkill 工具，避免给无技能 agent 增加噪声（`skill.py:227-241`）。

嵌入模型方向（语义检索式技能召回）属于未来扩展：当前仓库的 `grep -r "embedding\|vector\|RAG" nexau/` 在生产代码中无任何匹配，说明该能力尚未落地。本 RFC 把现状写实，避免后续读者误判。

## 设计

### 概述

技能子系统由四个层次组成：

1. **数据模型**：`Skill` 类承载 `name / description / detail / folder`（`skill.py:26-31`）。
2. **加载器**：`Skill.from_folder` 从 SKILL.md 解析 YAML frontmatter + body（`skill.py:33-34`、`skill.py:47-48`）。
3. **物化器**：`build_tool_skill` 把 `Tool` 转成 `Skill`，用 `build_tool_skill_detail` 渲染详细内容（`skill.py:118-170`、`skill.py:173-180`）。
4. **运行时入口**：`load_skill` 在 `skill_registry` 中按 name 精确查找；`build_load_skill_tool` 决定是否注入 LoadSkill 工具（`skill.py:183-197`、`skill.py:227-241`）。

`AgentConfigBuilder._finalize` 在配置阶段调用 `build_load_skill_tool` 并把结果追加进 `self.tools`（`config.py:260-263`）。`Agent.__init__` 在初始化阶段把 `config.skills` 与 tool-derived skills 合并写入 `skill_registry` 和 `global_storage`（`agent.py:172-185`）。

### 详细设计

#### Skill 数据模型与 SKILL.md 解析

`Skill` 是 plain Python class（非 Pydantic），构造签名 `name / description / detail / folder`（`skill.py:26-31`）。`from_folder` 接受文件夹路径，要求文件夹内必须存在 `SKILL.md`，否则抛 `FileNotFoundError`（`skill.py:33-45`）。`_load_yaml_formatted` 是个手写的 frontmatter 解析器：

- 首行必须是 `---`，否则 `ValueError`（`skill.py:66-72`）。
- 从第二行起扫描，直到再次出现 `---` 行才认为 frontmatter 结束；扫不到也 `ValueError`（`skill.py:74-82`）。
- frontmatter 区块用 `yaml.safe_load` 解析为 metadata dict，body 区块 `.strip()` 后作为 `detail`（`skill.py:84-95`）。

例如 `examples/code_agent/skills/skill-creator/SKILL.md` 起始就是 `---\nname: skill-creator\ndescription: ...\n---`，body 才是给模型看的"如何创建技能"说明。

#### 工具到技能的物化

`Tool` 数据类提供两个相关字段：`skill_description: str | None`（技能简介，进 LoadSkill 描述里）与 `as_skill: bool = False`（是否要被物化为技能）（`tool.py:46`、`tool.py:80`）。

`build_tool_skill(tool, tool_call_mode)` 直接构造 `Skill(name=tool.name, description=tool.skill_description, detail=build_tool_skill_detail(...), folder="")`（`skill.py:173-180`）。`folder=""` 表明 tool-skill 没有真实文件夹来源。

`build_tool_skill_detail` 根据 `tool_call_mode` 走两个分支（`skill.py:118-170`）：

- **结构化模式**（`STRUCTURED_TOOL_CALL_MODES`，对应 OpenAI 函数调用、Anthropic tool_use 等）：只输出 `# Tool Skill: <name>` + `## Detailed Description` + 可选 `## Additional Usage Guidance`（`skill.py:123-138`）。
- **XML 模式**：除上述描述外，再追加 `## XML Usage` 区块，从 `tool.input_schema` 渲染 `<tool_use><tool_name>…</tool_name><parameter>…</parameter></tool_use>` 模板（`skill.py:140-170`）。

参数行由 `_render_xml_parameter_lines` 拼装，遍历 `properties` 与 `required` 集合，每条形如 `<param_name>desc (required, type: string)</param_name>`（`skill.py:98-115`）。`description` 缺省回退到 "Parameter value"，`type` 缺省回退到 `string`。

#### LoadSkill 工具与运行时查询

`load_skill(skill_name, agent_state)` 是绑定到 LoadSkill 的执行函数（`skill.py:183-197`）：

1. 从 `agent_state.get_global_value("skill_registry", {})` 取出当前 agent 的技能字典。
2. 不存在直接 `raise ValueError(f"Skill {skill_name} not found")`。
3. 命中后返回固定 XML 包裹的字符串：`<SkillDetails><SkillName>…</SkillName><SkillFolder>…</SkillFolder><SkillDescription>…</SkillDescription><SkillDetail>…</SkillDetail></SkillDetails>`，并在前面加一句"路径相对 skill 文件夹"的提示。

`generate_skill_tool_description(skills, tools)` 生成 LoadSkill 工具的描述补丁（`skill.py:200-224`）：

- 先遍历显式 skills，每个写一个 `<SkillBrief>` 块（`skill.py:205-211`）。
- 再遍历 tools，对所有 `as_skill=True` 且名字未在 skills 中重复的，再写一个 `<SkillBrief>`；若标记为 skill 但没填 `skill_description` 直接 `ValueError`（`skill.py:213-221`）。
- 整体由 `<Skills>…</Skills>` 包裹（`skill.py:202`、`skill.py:223`）。

`build_load_skill_tool(tools, skills)` 是注入决策点（`skill.py:227-241`）：

- 若 tools 列表中已存在 LoadSkill，返回 `None`（避免重复）（`skill.py:228-229`）。
- 否则计算 `has_skilled_tools = any(t.as_skill for t in tools)`；若 `has_skilled_tools` 或 `skills` 非空，从 `archs/tool/builtin/description/skill_tool.yaml` 加载工具定义、binding 到 `load_skill`、并把 `generate_skill_tool_description(...)` 拼接到 `description` 末尾（`skill.py:232-240`）。
- 否则返回 `None`，agent 不感知 LoadSkill。

LoadSkill 的 YAML 定义见 `nexau/archs/tool/builtin/description/skill_tool.yaml`：单个必填参数 `skill_name: string`，`additionalProperties: false`，schema 走 JSON Schema draft-07（`skill_tool.yaml:1-14`）。

#### 装配链路

`AgentConfigBuilder._finalize` 在 model_validator 阶段调用 `build_load_skill_tool(self.tools, self.skills)`，命中则 `self.tools.append(load_skill_tool)`（`config.py:260-263`）。

`AgentConfigBuilder.build_skills`（`config.py:686-696`）从 `agent_params["tools"]` 中收集 `tool.as_skill==True` 的项，调用 `build_tool_skill(tool, tool_call_mode=...)` 加进 `skills` 列表，再写回 `self.agent_params["skills"]`。

`Agent.__init__` 阶段（`agent.py:170-185`）：

1. 取 `tools = self.config.tools`（已包含 LoadSkill）。
2. `runtime_skills = list(self.config.skills)`，并按 `as_skill` + name 去重把工具补充进来（`agent.py:172-177`）。
3. 构建 `tool_registry` 与 `serial_tool_name`（`agent.py:180-181`）。
4. `self.skill_registry = {skill.name: skill for skill in runtime_skills}` 并 `self.global_storage.set("skill_registry", ...)`（`agent.py:184-185`）。

这样 `load_skill` 在被 LLM 调用时，可以走 `agent_state.get_global_value("skill_registry", {})` 拿到当前 agent 的字典；不同 agent 有自己的 global_storage，互不污染。

#### Tool 字段对外可配置

`ToolYamlSchema`（`tool.py:46`）允许在 YAML 定义中声明 `skill_description: str | None = None`。`Tool.__init__` 接收 `as_skill: bool = False`、`skill_description: str | None = None` 两个字段（`tool.py:69-70`、`tool.py:79-80`）。`Tool.__str__` 在调试输出里追加 `Skill description:` 行（`tool.py:302-303`）。

### 示例

#### 示例 1：人工 SKILL.md（`examples/code_agent/skills/skill-creator/SKILL.md`）

```markdown
---
name: skill-creator
description: Guide for creating effective skills. ...
license: Complete terms in LICENSE.txt
---

# Skill Creator

This skill provides guidance for creating effective skills.
```

`Skill.from_folder("examples/code_agent/skills/skill-creator")` 解析后得到：

- `name="skill-creator"`
- `description="Guide for creating effective skills. ..."`
- `detail="# Skill Creator\n\nThis skill provides guidance ..."`（`---` 后的全部 body）
- `folder="<absolute path to skill-creator>"`

#### 示例 2：tool-as-skill 自动物化

某 YAML 工具声明了 `skill_description: "Apply file edits in patch syntax"` 且 `as_skill: true`。`AgentConfigBuilder.build_skills` 会把它喂给 `build_tool_skill`，得到一个 `Skill(name="...", description="Apply file edits in patch syntax", detail=<XML/markdown 渲染结果>, folder="")`，然后进入 `skill_registry`。LoadSkill 描述里就会多出一段 `<SkillBrief>Skill Name: ...</SkillBrief>` 引导模型识别。

#### 示例 3：LoadSkill 调用响应

LLM 输出 `LoadSkill(skill_name="skill-creator")` →`load_skill("skill-creator", agent_state)` →返回：

```text
Found the skill details of `skill-creator`.
Note that the paths mentioned in skill description are relative to the skill folder.
<SkillDetails>
<SkillName>skill-creator</SkillName>
<SkillFolder>/abs/path/to/skill-creator</SkillFolder>
<SkillDescription>Guide for creating effective skills. ...</SkillDescription>
<SkillDetail># Skill Creator ...</SkillDetail>
</SkillDetails>
```

模型据此按需把 `SkillDetail` 当作上下文继续推理。

## 权衡取舍

### 考虑过的替代方案

- **把全部技能详细内容直接拼进 system prompt**：实现最简单，但浪费 token 且 agent 任务多半只用其中一两条。本设计把详细内容延迟到 LoadSkill 调用，节省启动期上下文。
- **Skill 用 Pydantic 类**：可在加载时强校验。但 `Skill` 几乎只在 nexau 内部传递，且 frontmatter 已由 `_load_yaml_formatted` 校验，转 Pydantic 收益有限，因此保持 plain class（`skill.py:26-31`）。
- **frontmatter 用第三方库（python-frontmatter）解析**：会引入额外依赖。当前手写解析器仅 50 行（`skill.py:47-95`），覆盖 NexAU 用到的最小子集即可。
- **基于嵌入向量的语义召回**：`load_skill` 走精确名匹配（`skill.py:185-187`），意味着模型必须知道准确技能名才能加载。理论上向量召回（embedding + cosine top-k）可以让"我想做 X"自动找到合适技能。**当前未实现**——见"未解决的问题"。
- **LoadSkill 工具默认始终注入**：可减少决策分支。但对完全不依赖技能的 agent 会增加无意义的 system prompt 体积，因此保留 `has_skilled_tools or skills` 门控（`skill.py:232-233`）。

### 缺点

- **精确名匹配脆弱**：模型若拼错技能名（大小写、连字符），`load_skill` 直接抛错（`skill.py:186-187`），需要靠 `<SkillBrief>` 列表里的 `Skill Name` 字段引导。
- **Skill 详细内容无大小限制**：`skill.detail` 可能是数千行的 markdown，加载后会迅速吞噬上下文窗口。约束在写 SKILL.md 的人那一侧，框架不做截断。
- **`folder=""` 是 tool-skill 的特殊值**：`load_skill` 把 `folder` 直接拼进响应里（`skill.py:193`），对 tool-skill 会出现空字符串字段，需要消费方注意（人类 SKILL.md 有真实路径，tool-skill 没有）。
- **`build_tool_skill_detail` 的 `STRUCTURED_TOOL_CALL_MODES` 判定隐含工具调用语义**：意味着 OpenAI/Anthropic 模式下不会渲染 `## XML Usage`，但 XML 模式下会强制写 XML 模板，跨模式切换要重新生成 detail。
- **嵌入维度缺失**：题目所述"嵌入模型"完全未落地（见下文）。

## 实现计划

### 阶段划分

- [x] Phase 1: `Skill` 类与 `from_folder` 加载器（`skill.py:26-45`）
- [x] Phase 2: SKILL.md frontmatter 解析（`skill.py:47-95`）
- [x] Phase 3: tool-as-skill 物化（`build_tool_skill` / `build_tool_skill_detail` / `_render_xml_parameter_lines`，`skill.py:98-180`）
- [x] Phase 4: 运行时 `load_skill` + `skill_registry` 全局存储（`skill.py:183-197`、`agent.py:184-185`）
- [x] Phase 5: LoadSkill 自动注入与描述生成（`skill.py:200-241`、`config.py:260-263`）
- [x] Phase 6: `Tool.skill_description` / `Tool.as_skill` 字段对外暴露（`tool.py:46/69-70/79-80`）
- [ ] Phase 7（未来）: 基于嵌入向量的技能召回与排序（参见"未解决的问题"）

### 相关文件

- `nexau/archs/main_sub/skill.py` - Skill 类、frontmatter 解析、tool 物化、LoadSkill 注入与运行时查询全在此文件
- `nexau/archs/tool/builtin/skill_tool.py` - 公开 re-export（`load_skill`、`generate_skill_tool_description`）
- `nexau/archs/tool/builtin/description/skill_tool.yaml` - LoadSkill 工具的 YAML 定义（`skill_name` 必填）
- `nexau/archs/main_sub/agent.py` - `Agent.__init__` 在 `agent.py:170-185` 构建 `skill_registry` 并写入 global_storage
- `nexau/archs/main_sub/config/config.py` - `AgentConfigBuilder._finalize`（`config.py:260-263`）注入 LoadSkill；`build_skills`（`config.py:686-696`）把 tool-as-skill 转 Skill
- `nexau/archs/tool/tool.py` - `ToolYamlSchema.skill_description`（`tool.py:46`）、`Tool.__init__` 接受 `as_skill` / `skill_description`（`tool.py:69-70`、`tool.py:79-80`）
- `examples/code_agent/skills/skill-creator/SKILL.md`、`examples/nexau_building_team/skills/rfc/SKILL.md` 等 7 个示例 - 真实 SKILL.md 模板

## 测试方案

### 单元测试

- `Skill.from_folder` 在缺失 SKILL.md 时抛 `FileNotFoundError`（`skill.py:44-45`）。
- `_load_yaml_formatted` 在缺失起始 `---`、缺失闭合 `---`、frontmatter YAML 非法时分别抛 `ValueError`（`skill.py:66-72`、`skill.py:74-82`、`skill.py:90-95`）。
- `_render_xml_parameter_lines` 对空 schema、缺 `description`、缺 `type`、含 `default` 四种情况均输出符合预期的 `<param>...</param>` 行（`skill.py:98-115`）。
- `build_tool_skill_detail` 在 `STRUCTURED_TOOL_CALL_MODES` 与 XML 模式下分别产出对应模板，`template_override` 存在时追加 `## Additional Usage Guidance`（`skill.py:118-170`）。
- `load_skill` 命中返回正确 XML、未命中抛 `ValueError`（`skill.py:183-197`）。
- `generate_skill_tool_description` 对 `as_skill=True` 但缺 `skill_description` 的工具抛 `ValueError`，对 skills/tools 重名只输出一次 `<SkillBrief>`（`skill.py:213-221`）。
- `build_load_skill_tool` 三分支：已存在 LoadSkill 返回 None；无 skill/无 as_skill 工具返回 None；其他情况返回正确绑定的 LoadSkill Tool（`skill.py:227-241`）。

### 集成测试

- 装载真实 SKILL.md 文件夹（如 `examples/nexau_building_team/skills/rfc/`）创建 Agent，验证 `agent.skill_registry` 包含该技能、`agent.config.tools` 包含 LoadSkill。
- 配置一个 `as_skill: true` 的 YAML 工具，跑完 `AgentConfigBuilder` 全链，验证 `skill_registry` 含同名 Skill、`build_tool_skill_detail` 输出符合 `tool_call_mode`。
- 跑一段对话：模型调用 LoadSkill 时收到 `<SkillDetails>`；调用 LoadSkill 时给错误名得到错误信息可继续。
- 验证无技能 agent 完全不见 LoadSkill 工具（`build_load_skill_tool` 返回 None 路径）。

### 手动验证

- `examples/nexau_building_team/main.py` 一类示例脚本启动后，打印 `agent.skill_registry.keys()` 应包含 SKILL.md 文件夹中的全部技能名。
- 跑 `nexau chat`，模型若被引导调用 LoadSkill 应能收到带 `<SkillDetail>` 的回复。

## 未解决的问题

- **嵌入模型缺位**：`load_skill` 走 `skill_registry` 字典精确查找（`skill.py:185-187`），仓库内 `nexau/` 目录无任何 embedding/vector/RAG 实现。RFC 题目中的"嵌入模型"维度尚未落地。后续若引入，需要确定：
  1. 嵌入存储位置（独立向量索引文件？挂在 SKILL.md 同级？）；
  2. 召回入口形态（新增 `SearchSkill` 工具？让 LoadSkill 接收自然语言？）；
  3. 与现有精确名匹配的兼容（同名优先 vs. 语义召回 top-k）。
- **`folder=""` 在 tool-skill 上的语义泄漏**：当前 `load_skill` 响应里 `<SkillFolder></SkillFolder>` 会出现空标签（`skill.py:193`），是否需要在 tool-skill 上略过该字段？
- **`Skill.detail` 大小预算**：缺少长度阈值或 truncation 策略，长 SKILL.md 可能瞬间灌满上下文。
- **多 agent skill_registry 隔离边界**：当前每个 Agent 自己持有 `skill_registry` 并写入各自 `global_storage`，sub-agent 是否需要继承父 agent 的技能仍未明确（参见 RFC-0013 委派语义）。
- **YAML frontmatter `name` 与文件夹名不一致时的策略**：`from_folder` 不会校验 `metadata["name"]` 与 `folder.name` 是否一致，可能造成调试困惑。

## 参考资料

- `rfcs/0006-rfc-catalog-completion-master-plan.md` - T12 子任务来源
- `rfcs/0007-tool-system-architecture-and-binding.md` - Tool 字段定义与绑定模型（依赖项）
- `rfcs/0014-prompt-engineering-and-templating.md` - LoadSkill 描述与系统提示词协作
- `rfcs/0017-cli-and-yaml-config-loading.md` - SKILL.md 文件夹通过 YAML 配置接入 agent
- `nexau/archs/main_sub/skill.py` - 技能子系统全部实现
- `nexau/archs/tool/builtin/description/skill_tool.yaml` - LoadSkill 工具 schema
- `examples/code_agent/skills/skill-creator/SKILL.md`、`examples/nexau_building_team/skills/rfc/SKILL.md` - 真实 SKILL.md 模板
