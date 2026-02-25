# RFC-0000: [Agent 名称]

- **状态**: draft | accepted | implemented | superseded | rejected
- **优先级**: P0 | P1 | P2 | P3
- **标签**: `agent`, `tool`, `skill`, 等
- **Agent 角色**: Agent 角色名如 `multi-modal-extractor`, `ads-judger` 等
- **创建日期**: YYYY-MM-DD
- **更新日期**: YYYY-MM-DD

## 摘要

一段话描述这个 Agent 的定位：它是谁、为谁服务、解决什么问题。

## 动机

为什么需要构建这个 Agent？当前存在什么痛点或空白？

## Agent 设计

### 定位与职责边界

描述 Agent 的核心职责和能力边界：

- **角色定义**: 这个 Agent 扮演什么角色？（如：代码审查专家、研究助手、运维工程师）
- **目标用户**: 谁会使用这个 Agent？使用场景是什么？
- **职责边界**: 这个 Agent 负责什么、不负责什么？明确划定能力范围。

> ⚠️ 只描述"是什么"和"为什么"，不要写具体的 YAML 配置或代码。

### System Prompt 设计原则

描述 system prompt 的设计思路，而非 prompt 本文：

- **人设与语气**: Agent 应以什么身份和风格与用户交互？
- **核心工作流**: Agent 的主要工作步骤是什么？（用自然语言描述流程逻辑，不写 prompt 原文）
- **关键约束**: Agent 必须遵守哪些规则或限制？（如：必须先读文件再修改、不允许执行危险命令等）
- **输出规范**: Agent 的输出应遵循什么格式或标准？

> ⚠️ 这里描述的是 prompt 的设计意图和原则，不是 prompt 的具体内容。实现阶段再根据这些原则编写实际 prompt。

### 工具设计

列出 Agent 需要的工具能力，以及选型理由：

#### Builtin 工具

列出计划使用的内置工具及选用原因：

| 工具 | 选用理由 |
|------|----------|
| 例：`read_file` | Agent 需要读取用户指定的文件进行分析 |
| 例：`run_shell_command` | Agent 需要执行构建命令验证结果 |


### Builtin Tools Catalog

#### File Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| read_file | `nexau.archs.tool.builtin.file_tools:read_file` | `builtin_tools/tools/read_file.tool.yaml` |
| write_file | `nexau.archs.tool.builtin.file_tools:write_file` | `builtin_tools/tools/write_file.tool.yaml` |
| replace | `nexau.archs.tool.builtin.file_tools:replace` | `builtin_tools/tools/replace.tool.yaml` |
| search_file_content | `nexau.archs.tool.builtin.file_tools:search_file_content` | `builtin_tools/tools/search_file_content.tool.yaml` |
| glob | `nexau.archs.tool.builtin.file_tools:glob` | `builtin_tools/tools/Glob.tool.yaml` |
| list_directory | `nexau.archs.tool.builtin.file_tools:list_directory` | `builtin_tools/tools/list_directory.tool.yaml` |
| read_many_files | `nexau.archs.tool.builtin.file_tools:read_many_files` | `builtin_tools/tools/read_many_files.tool.yaml` |

#### Web Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| google_web_search | `nexau.archs.tool.builtin.web_tools:google_web_search` | `builtin_tools/tools/WebSearch.tool.yaml` |
| web_fetch | `nexau.archs.tool.builtin.web_tools:web_fetch` | `builtin_tools/tools/WebFetch.tool.yaml` |

#### Shell Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| run_shell_command (sync) | `nexau.archs.tool.builtin.shell_tools:run_shell_command` | `builtin_tools/tools/run_shell_command_sync.tool.yaml` |

#### Session Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| write_todos | `nexau.archs.tool.builtin.session_tools:write_todos` | `builtin_tools/tools/write_todos.tool.yaml` |
| complete_task | `nexau.archs.tool.builtin.session_tools:complete_task` | `builtin_tools/tools/complete_task.tool.yaml` |
| ask_user | `nexau.archs.tool.builtin.session_tools:ask_user` | `builtin_tools/tools/ask_user.tool.yaml` |


#### 自定义工具（如需要）

对于内置工具无法覆盖的能力，描述需要创建的自定义工具：

| 工具名称 | 能力描述 | 为什么不能用 builtin |
|----------|----------|---------------------|
| 例：`query_database` | 查询项目数据库获取统计信息 | 内置工具没有数据库查询能力 |

> ⚠️ 只描述工具的能力和选型理由，不要写 tool YAML schema 或 Python 实现。

### Skills 设计（如需要）

如果 Agent 需要 Skills，列出每个 Skill 的目的：

| Skill 名称 | 目的 | 何时激活 |
|------------|------|----------|
| 例：`code-review` | 提供代码审查的最佳实践和检查清单 | 用户请求代码审查时 |
| 例：`rfc` | 提供 RFC 模板和写作规范 | 用户需要撰写设计文档时 |

> ⚠️ 只描述 Skill 的用途和触发条件，不要写 SKILL.md 的具体内容。

## 权衡取舍

### 考虑过的替代方案

列出在设计过程中考虑过但未采用的方案，以及原因。例如：

- 为什么某个能力选择自定义工具而不是 builtin？
- 为什么 prompt 采用某种工作流而不是另一种？

### 已知局限

当前设计的已知缺点或限制。

## 验证标准

描述如何判断这个 Agent 构建成功：

- **功能验证**: 列出 Agent 应能完成的典型任务场景
- **边界验证**: 列出 Agent 应正确拒绝或降级处理的场景

> ⚠️ 描述验证场景，不要写测试代码。

## 未解决的问题

需要进一步讨论的设计问题。

## 参考资料

相关链接、文档、已有 Agent 的参考。
