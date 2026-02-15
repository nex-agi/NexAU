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
| [RFC-0003](./0003-observability.md)             | 统一日志与监控         | draft        | P2  |
| [RFC-0004](./0004-api-docs.md)                  | API 文档完善        | draft        | P3  |
| [RFC-0005](./0005-service-discovery.md)         | 服务发现与染色泳道       | draft        | P2  |
| [RFC-0006](./0006-request-level-token.md)       | 请求级 Token 认证    | implemented  | P1  |
| [RFC-0007](./0007-openapi-client-generation.md) | OpenAPI 客户端统一生成 | implemented  | P2  |
| [RFC-0012](./0012-configuration-system.md)      | 统一配置系统          | implementing | P1  |


### 服务架构


| RFC                                              | 标题                                    | 状态          | 优先级 |
| ------------------------------------------------ | ------------------------------------- | ----------- | --- |
| [RFC-0002](./0002-agent-gateway.md)              | Agent Gateway 实现                      | implemented | P1  |
| [RFC-0010](./0010-service-architecture.md)       | 服务架构简化重构                              | implemented | P0  |
| [RFC-0011](./0011-project-version-management.md) | Project Version & Artifact Management | implemented | P0  |


### Agent Runtime 组（RFC-0008 ~ 0009）


| RFC                                         | 标题                          | 服务                       | 状态          | 优先级 |
| ------------------------------------------- | --------------------------- | ------------------------ | ----------- | --- |
| [RFC-0008](./0008-agent-runtime-sidecar.md) | Agent Runtime Sidecar Proxy | `agent-runtime-sidecar`  | implemented | P1  |
| [RFC-0009](./0009-agent-runtime-service.md) | Agent Runtime Service       | `agent-runtime` (Python) | implemented | P1  |


### Sandbox 组（RFC-0013 ~ 0014）


| RFC                                         | 标题                    | 服务                      | 状态          | 优先级 |
| ------------------------------------------- | --------------------- | ----------------------- | ----------- | --- |
| [RFC-0013](./0013-agent-sandbox-manager.md) | Agent Sandbox Manager | `agent-sandbox-manager` | implemented | P1  |
| [RFC-0014](./0014-sandboxd.md)              | Sandbox Daemon (gRPC) | `sandboxd`              | implemented | P1  |


### 数据模型（RFC-0015）


| RFC                                     | 标题                     | 状态    | 优先级 |
| --------------------------------------- | ---------------------- | ----- | --- |
| [RFC-0015](./0015-unified-artifacts.md) | Unified Artifacts 统一制品 | draft | P0  |


### CI/CD（RFC-0016, RFC-0037）


| RFC                                        | 标题              | 状态    | 优先级 |
| ------------------------------------------ | --------------- | ----- | --- |
| [RFC-0016](./0016-ci-coverage-strategy.md) | CI/CD 覆盖率统一收集方案 | draft | P1  |
| [RFC-0037](./0037-k8s-integration-test-ci.md) | Kind-based K8s Integration Test CI | draft | P1  |


### Bugfix（RFC-0017, RFC-0019）


| RFC                                               | 标题                 | 状态          | 优先级 |
| ------------------------------------------------- | ------------------ | ----------- | --- |
| [RFC-0017](./0017-artifact-upload-confirm-fix.md) | 修复 Artifact 上传确认流程 | draft       | P1  |
| [RFC-0019](./0019-project-description-field.md)   | 项目描述字段支持           | implemented | P1  |


### 前端功能（RFC-0018, RFC-0020）


| RFC                                           | 标题             | 状态    | 优先级 |
| --------------------------------------------- | -------------- | ----- | --- |
| [RFC-0018](./0018-artifact-content-viewer.md) | Artifact 内容浏览器 | draft | P1  |
| [RFC-0020](./0020-breadcrumb-navigation.md)   | 前端面包屑导航        | draft | P2  |
| [RFC-0030](./0030-playground-sandbox-snapshot-preview.md) | Playground 沙箱快照预览 | draft | P1  |
| [RFC-0031](./0031-frontend-otel-enhancement.md) | 前端 OpenTelemetry 增强 | implemented | P1  |


### 安全（RFC-0021, RFC-0031）


| RFC                                              | 标题                         | 状态    | 优先级 |
| ------------------------------------------------ | -------------------------- | ----- | --- |
| [RFC-0021](./0021-backend-api-authentication.md) | Backend API Authentication | draft | P0  |
| [RFC-0031](./0031-network-policy.md)             | Helm NetworkPolicy 网络隔离  | draft | P1  |


### Sandbox 网络（RFC-0022）


| RFC                                      | 标题                   | 状态           | 优先级 |
| ---------------------------------------- | -------------------- | ------------ | --- |
| [RFC-0022](./0022-sandbox-networking.md) | Sandbox 网络与 E2B 协议支持 | implementing | P0  |


### Sandbox 集成（RFC-0023, RFC-0025）


| RFC                                                | 标题                     | 状态          | 优先级 |
| -------------------------------------------------- | ---------------------- | ----------- | --- |
| [RFC-0023](./0023-sandbox-template-abstraction.md) | Sandbox Profile 抽象     | implemented | P1  |
| [RFC-0025](./0025-nexau-sandbox-integration.md)    | NexAU Agent Sandbox 集成 | draft       | P1  |


### Sandbox 性能（RFC-0033）


| RFC                                            | 标题                      | 状态    | 优先级 |
| ---------------------------------------------- | ----------------------- | ----- | --- |
| [RFC-0033](./0033-sandbox-warm-pool.md)        | K8s Sandbox Warm Pool   | draft | P0  |


## 架构图

### 整体架构 (RFC-0010)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              NexAU Cloud 架构                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    ┌─────────────┐                                                              │
│    │   Nginx     │                                                              │
│    │   :8080     │                                                              │
│    └──────┬──────┘                                                              │
│           │                                                                     │
│    ┌──────┴──────────────────────────────────────────────────────────┐          │
│    │                                                                 │          │
│    ▼                                                                 ▼          │
│  ┌─────────────────┐                                    ┌─────────────────┐     │
│  │    Backend      │                                    │  Agent Gateway  │     │
│  │    :8001        │                                    │    :8002        │     │
│  │  (控制平面)      │                                    │  (数据平面)      │     │
│  │  RFC-0010/0011  │                                    │  RFC-0002/0010  │     │
│  └────────┬────────┘                                    └────────┬────────┘     │
│           │                                                      │              │
│           │                    ┌─────────────────┐               │              │
│           │                    │ Session Manager │◄──────────────┤              │
│           │                    │    :8005        │               │              │
│           │                    │  RFC-0006       │               │              │
│           │                    └────────┬────────┘               │              │
│           │                             │                        │              │
│           ▼                             ▼                        ▼              │
│    ┌──────────────────────────────────────────────────────────────────┐         │
│    │                        PostgreSQL / Redis                        │         │
│    └──────────────────────────────────────────────────────────────────┘         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Agent Runtime 与 Sandbox 架构 (RFC-0008/0009/0013/0014)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Agent Runtime Pod                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌───────────────────────┐                              │
│  │  Sidecar Proxy   │    │   Agent Runtime       │                              │
│  │  (RFC-0008)      │    │   (RFC-0009)          │                              │
│  │  :3001 LLM       │◄───│   Python FastAPI      │                              │
│  │  :3002 Langfuse  │    │   内置 Tools          │                              │
│  └──────────────────┘    └───────────┬───────────┘                              │
└─────────────────────────────────────┼───────────────────────────────────────────┘
                                      │ REST API
┌─────────────────────────────────────▼───────────────────────────────────────────┐
│              Agent Sandbox Manager (RFC-0013) :8008                              │
│              Rust 服务，管理沙盒生命周期                                          │
│              支持 Docker+gVisor / Bubblewrap 后端                                │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │ Docker/Bubblewrap
┌─────────────────────────────────────▼───────────────────────────────────────────┐
│                            gVisor Sandbox                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  sandboxd (RFC-0014) :49983                                               │  │
│  │  Connect RPC + gRPC Daemon                                                │  │
│  │  - Process Service: 进程执行 (bash)                                       │  │
│  │  - Filesystem Service: 文件系统 (read/write/list/stat)                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│  /home/user (Session 专属)    /tmp (Session 专属)    /agent (只读，项目代码)     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 如何提交 RFC

> **详细规范请参考 [RFC 撰写指南**](./WRITING_GUIDE.md)

1. 阅读 [RFC 撰写指南](./WRITING_GUIDE.md) 了解格式规范
2. 复制 `0000-template.md` 为 `NNNN-title.md`
3. 按规范填写 RFC 内容（注意使用 [Mermaid 图表样式规范](../mermaid-style-guide.md)）
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
- [Mermaid 图表样式规范](../mermaid-style-guide.md) - 统一图表颜色规范
