# NexAU Cloud RFCs

本目录包含 NexAU Cloud 项目的 RFC（Request for Comments）文档。

## RFC 是什么

RFC 是一种用于记录技术设计决策的文档格式。每个 RFC 描述一个特定的功能、架构变更或技术决策，包括：

- **问题背景**：为什么需要这个变更
- **设计方案**：如何解决问题
- **权衡取舍**：考虑过的替代方案
- **实现状态**：当前进度

## RFC 状态


| 状态             | 说明          |
| -------------- | ----------- |
| `draft`        | 草稿，正在讨论     |
| `accepted`     | 已接受，待实现     |
| `implementing` | 实现中         |
| `implemented`  | 已实现         |
| `superseded`   | 被更新的 RFC 取代 |
| `rejected`     | 已拒绝         |


## RFC 列表

### 基础设施


| RFC                                             | 标题              | 状态           | 优先级 |
| ----------------------------------------------- | --------------- | ------------ | --- |
| [RFC-0001](./0001-state-persistence-on-stop.md) | Agent 中断时状态持久化   | draft        | P1  |


### Agent 协作（RFC-0002）


| RFC                                            | 标题                      | 状态    | 优先级 |
| ---------------------------------------------- | ----------------------- | ----- | --- |
| [RFC-0002](./0002-agent-team.md)               | AgentTeam 多 Agent 协作框架 | draft | P0  |


### 可靠性（RFC-0003）


| RFC                                            | 标题                      | 状态    | 优先级 |
| ---------------------------------------------- | ----------------------- | ----- | --- |
| [RFC-0003](./0003-llm-failover-middleware.md)   | LLM 自动降级中间件             | implemented | P1  |


## 架构图

### 整体架构

TBA

## 如何提交 RFC

> **详细规范请参考 [RFC 撰写指南**](./WRITING_GUIDE.md)

1. 阅读 [RFC 撰写指南](./WRITING_GUIDE.md) 了解格式规范
2. 复制 `0000-template.md` 为 `NNNN-title.md`
3. 按规范填写 RFC 内容（注意使用 [Mermaid 图表样式规范](./mermaid-style-guide.md)）
4. 更新本文件的 RFC 列表
5. 提交 PR 进行讨论
6. 获得批准后合并

## RFC 编号规则

- 使用 4 位数字编号，如 `0001`
- 编号顺序分配，不跳号
- 被 superseded 的 RFC 保留原编号
- 相关功能的 RFC 使用连续编号（如 Runtime 组 0008-0009，Sandbox 组 0013-0014）

## 相关文档

- [RFC 撰写指南](./WRITING_GUIDE.md) - RFC 格式、内容、图表规范
- [RFC 模板](./0000-template.md) - RFC 基础模板
- [Mermaid 图表样式规范](./mermaid-style-guide.md) - 统一图表颜色规范
