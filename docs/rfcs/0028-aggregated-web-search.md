# RFC-0028: 聚合 Web 搜索内置工具

- **状态**: draft
- **优先级**: P1
- **标签**: `builtin-tools`, `dx`, `信创`
- **影响服务**: NexAU runtime（`nexau/archs/tool/builtin/web_tools/`、`config.py` 内置工具注入）
- **创建日期**: 2026-07-30
- **更新日期**: 2026-07-30

## 摘要

把内置联网搜索从**硬绑 Serper** 改为**可切换服务商的聚合搜索**：一个 `web_search`
工具接口，后端支持 Serper / 豆包搜索 / 百度 AI 搜索 / 北坡聚合搜索，通过环境变量
`SEARCH_PROVIDER` 切换，并按 RFC-0197 注册为 runtime 内置工具。

**本 RFC 不删除也不修改 `google_web_search`**，现有 examples / docs / 用户 Agent 全部
不受影响，`SERPER_API_KEY` 继续可用。

## 动机

当前 `nexau/archs/tool/builtin/web_tools/web_tool.py` 的 `SerperSearch` 是唯一实现：

```python
class SerperSearch:
    def __init__(self, ...):
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("Serper API key is required")
        self.base_url = "https://google.serper.dev/"
```

这带来三个现实问题：

1. **信创 / 内网环境不可用**。Serper 背后是 Google，私有化与政务内网场景下不通；
   而这正是 NexAU 的主要落地场景之一。
2. **中文检索质量与成本不占优**。国内服务商（豆包搜索、百度 AI 搜索）在中文内容、
   政务站点覆盖、权威度分级上有明显优势，且延迟更低（实测豆包 < 1s、Serper 1~2s）。
3. **能力面被最小公倍数锁死**。Serper 没有「只要权威来源」「行业垂直检索」这类字段，
   而这些在政务场景是刚需；反过来，Serper 的 `gl` / `hl` / `location` 也是别家没有的。
   硬绑一家意味着**只能暴露交集**。

换服务商目前需要改代码、重新打包、重新部署。合理的做法是让它成为**部署期配置**。

## 设计

### 概述

三层结构：

```text
                      ┌──────────────────────────────────┐
  Agent 调用 ─────────►│ web_search(query, ...19 个参数)   │  统一接口
                      └───────────────┬──────────────────┘
                                      │  SEARCH_PROVIDER 决定实例化谁
                      ┌───────────────▼──────────────────┐
                      │ SearchProviderBase               │  重试 / 退避 /
                      │   重试·退避·错误归一·参数告警     │  错误归一 / 告警
                      └───┬────────┬────────┬────────┬───┘
                          │        │        │        │
                     Serper     Seed     Baidu   XiaoBei   各自只实现
                    (Google) (豆包搜索) (百度AI) (聚合)     "发一次请求+归一"
```

**Provider（服务商）与 Engine（底层搜索引擎）是两层**，不能合并：多数服务商的引擎
固定（Serper 背后即 Google），但聚合型服务商自己会并发打多个引擎，这时才需要
`SEARCH_ENGINE` 选 `google` / `bing` / `baidu`。

### 详细设计

#### 1. 新增模块

`nexau/archs/tool/builtin/web_tools/aggregated_search.py`，导出 `web_search()`，
返回结构与现有 `google_web_search` 完全一致（gemini-cli 风格）：

```python
{"content": ..., "returnDisplay": ..., "sources": [...], "provider": "Seed"}
# 失败时：{"content": ..., "returnDisplay": ..., "error": {"message": ..., "type": ...}}
```

因此**前端与 system prompt 无需任何改动**即可从 `google_web_search` 迁移过来。

#### 2. 环境变量

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `SEARCH_PROVIDER` | 否 | `Serper` | `Serper` / `Seed` / `Baidu` / `XiaoBei` |
| `SEARCH_API_KEY` | 是 | — | 当前服务商密钥；缺省时回落 `SERPER_API_KEY`（向后兼容） |
| `SEARCH_ENGINE` | 否 | 空 | 底层引擎，仅聚合型服务商生效 |
| `SEARCH_BASE_URL` | 否 | 各家内置 | 覆盖上游地址（私有化 / 内网代理） |
| `SEARCH_TIMEOUT` | 否 | `30` | 单次请求超时秒数 |
| `SEARCH_MAX_RETRIES` | 否 | `3` | **总**尝试次数（含首次），下限 1 |

此外**每个工具参数都能用环境变量设部署级默认值**，命名规则
`SEARCH_` + 参数名大写（自带 `search_` 前缀的去重），如
`content_format` → `SEARCH_CONTENT_FORMAT`。

取值优先级：**调用方显式传参 > 环境变量 > 内置默认值**。

> ⚠️ 为区分「没传」与「显式传了与默认值相同的值」，工具函数签名的默认值一律是
> `None` 哨兵。实测模型会把 schema 里的默认值原样回传（如
> `content_format="text"`），若按值判等，部署方设的默认会时灵时不灵。

#### 3. 注册为内置工具（RFC-0197）

**条件注入**：与 `read_file` / `run_shell_command` 这些「零配置即可用」的内置工具不同，
搜索必须有服务商密钥才能工作。若无条件注入，每个 Agent 的工具列表里都会多一个
「一调就报缺 Key」的工具——既白占上下文，又诱导模型误用。因此单列一张条件表：

```python
# nexau/archs/main_sub/config/config.py
_CONDITIONAL_BUILTIN_TOOL_BINDINGS = (
    (
        "web_search",
        "nexau.archs.tool.builtin.web_tools:web_search",
        ("SEARCH_API_KEY", "SERPER_API_KEY"),   # 任一非空即注入
    ),
)
```

即：**配了密钥就自动拥有 `web_search`，没配就当它不存在**。

> 工具名用 snake_case（与 `read_file` / `write_file` 等 runtime 内置工具一致），
> 也刻意避开 examples 里既有的 `WebSearch.tool.yaml`——那 4 份描述的是绑定
> `google_web_search` 的旧工具，RFC-0197 的 schema 单一事实源校验要求同名
> `.tool.yaml` 全仓逐字节一致，不能混为一谈。

schema 落在 RFC-0197 收口的单一事实源目录：
`nexau/archs/tool/builtin/schemas/web_search.tool.yaml`。

按 RFC-0197 的既有语义，**agent 自声明或插件贡献的同名工具优先**，runtime 只补齐缺失项——
已经显式声明 `WebSearch` 的 Agent 不会被覆盖。

#### 4. 能力不齐的处理原则

四家服务商能力差异很大，处理原则三条**缺一不可**：

1. **不报错**：不支持的参数不影响调用成功；
2. **能降级就降级**：Serper / XiaoBei 没有结构化站点过滤，但 query 透传给 Google 系
   引擎，因此把 `sites` 编译成 `site:` 算子。实测 XiaoBei 上 `arxiv.org` 命中率从
   纯本地过滤的 1/28 提升到 20/20，耗时 33s → 6s；
3. **必须留痕**：每个被忽略的参数打一条 `warning`。静默忽略是最难排查的一类问题
   （"我明明配了 `authority_only`"）。

每个 Provider 声明 `IGNORED_PARAMS`，基类统一比对告警。

### 示例

```yaml
# agent.yaml —— 什么都不用写；只要配了搜索密钥，
# WebSearch 就由 runtime 自动注入（RFC-0197 注入机制 + 本 RFC 的条件判定）
```

```bash
# 部署时切换服务商，代码与制品不变
SEARCH_PROVIDER=Seed
SEARCH_API_KEY=<豆包搜索 Key>

# 政务部署：默认只要官方权威来源
SEARCH_AUTHORITY_ONLY=true
SEARCH_INDUSTRY=gov
```

## 权衡取舍

### 考虑过的替代方案

**A. 直接改 `SerperSearch` 支持多服务商。** 否决：`google_web_search` 被 10+ 处
examples / docs 引用，现有单测还 patch 了 `google_web_search._web_search` 这个内部
接缝，改动会连环破坏；且 `web_search()` 现有返回结构（`{"status": ..., "results": ...}`）
与新接口不同。

**B. 每家服务商各出一个内置工具**（`serper_search` / `doubao_search` …）。否决：模型要
在多个语义重叠的工具间选择，`description` 极难写；且换服务商仍要改 `agent.yaml`。

**C. 用 MCP 外挂搜索服务。** 否决：联网搜索是基础能力，不应引入额外部署组件；
且 MCP 无法解决"能力面取交集"的问题。

### 缺点

1. **代码量增加**：多服务商适配器约 1500 行（含大量上游契约注释）。缓解：基类吃掉
   重试/退避/告警/参数解析，每个 Provider 子类只有 60~120 行。
2. **参数面变大**（19 个）：可能增加模型选择负担。缓解：除 `query` 外全部可省略，
   全省略时行为与现有 `google_web_search` 等价。
3. **四家的行为差异无法完全抹平**：如 `full_content` 在百度上是 no-op（其
   `snippet` 与 `content` 返回同一段文本）。缓解：支持度矩阵写进模块 docstring 与
   tool schema，运行时告警补齐。

## 实现计划

### 阶段划分

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1 | `aggregated_search.py` + 四个 Provider + 单测 | 本 PR |
| P2 | `schemas/web_search.tool.yaml` + 条件注入注册 | 本 PR |
| P3 | examples / docs 迁移到内置 `WebSearch` | 后续 PR |
| P4 | `google_web_search` 标记 deprecated（不删除） | 后续 PR |

### 相关文件

- `nexau/archs/tool/builtin/web_tools/aggregated_search.py`（新增）
- `nexau/archs/tool/builtin/web_tools/__init__.py`（导出 `web_search`）
- `nexau/archs/tool/builtin/schemas/web_search.tool.yaml`（新增）
- `nexau/archs/main_sub/config/config.py`（新增 `_CONDITIONAL_BUILTIN_TOOL_BINDINGS` 与条件判定）
- `tests/unit/test_config.py`、`tests/unit/test_plugin_adapter_quickstart.py`（内置工具集合断言随之更新）
- `tests/unit/test_builtin_tools/test_aggregated_search.py`（新增）
- `tests/integration/test_config_integration.py`（内置工具集合断言随之更新）

## 测试方案

| # | 触发条件 | 期望结果 | 验证方式 |
|---|---|---|---|
| 1 | `query` 为空 / 纯空白 | `error.type == "INVALID_QUERY"`，不发请求 | 单测 |
| 2 | 未设 `SEARCH_API_KEY` 与 `SERPER_API_KEY` | `error.type == "WEB_SEARCH_CONFIG_ERROR"`，报错文案含变量名 | 单测 |
| 3 | `SEARCH_PROVIDER` 取非法值 | `WEB_SEARCH_CONFIG_ERROR`，列出合法取值 | 单测 |
| 4 | 只设 `SERPER_API_KEY`（旧用户场景） | 正常工作，provider 为 `Serper` | 单测（向后兼容） |
| 5 | 四家分别注入 mock transport | 请求 URL / 鉴权头 / 请求体字段符合各自文档 | 单测（白盒断言请求体） |
| 6 | 给不支持某参数的服务商传该参数 | 调用**成功**，且 `caplog` 中有对应 warning | 单测 |
| 7 | 环境变量设参数默认值 | 生效；调用方显式传参时被覆盖 | 单测（三级优先级） |
| 8 | 环境变量类型写错（`SEARCH_NUM_RESULTS=abc`） | 回退内置默认 + warning，不抛异常 | 单测 |
| 9 | 上游 5xx / 超时 | 指数退避，共尝试 `SEARCH_MAX_RETRIES` 次后返回 `WEB_SEARCH_FAILED` | 单测（计数 mock 调用次数） |
| 9b | `SEARCH_MAX_RETRIES` 配成 `0` / 负数 | 钳到 1，**仍会发出一次请求**而非恒失败 | 单测 |
| 10 | 上游 4xx（非 429） | **不重试**，直接返回 | 单测（断言只调用 1 次） |
| 11 | 豆包 HTTP 200 但 `ResponseMetadata.Error` 非空 | 按错误码区分可重试/不可重试 | 单测 |
| 12 | 结果为 0 条 | 返回**成功**且 `sources == []`，不是 error | 单测 |
| 13 | 配了 `SEARCH_API_KEY`（或 `SERPER_API_KEY`）| 未声明 `web_search` 的 agent 自动获得该工具 | 单测（参数化两个变量名）|
| 14 | 两个密钥变量都未配 | **不注入** `web_search` | 单测 |
| 15 | agent 已自声明 `web_search` | runtime **不覆盖**其声明（复用 RFC-0197 dedup 语义）| 既有单测 |

所有单测**不打真实上游**（注入 mock transport / patch），CI 无需任何 API Key。

真实上游连通性已在 NAC beta 环境人工验证：Serper / 豆包 / 百度 / 北坡四家均跑通
成功路径（含站点过滤、时效过滤、权威度过滤、自定义日期区间）。

## 未解决的问题

1. **是否要把 `google_web_search` 直接改为委托新实现？** 本 RFC 选择不改（零风险），
   代价是短期内两套实现并存。P4 阶段再评估。
2. ~~`XiaoBei` 是否适合进仓库~~ —— **已确认保留**（公开服务，申请 Key 需联系服务管理员）。
3. **是否需要 provider 级健康检查 / 自动 failover**：当前一次调用只打一家，某家挂了
   需要人工切换。自动 failover 涉及配额与计费归属，留待后续 RFC。

## 参考资料

- RFC-0197（NAC 仓库）— runtime 内置基础工具与 `.tool.yaml` schema 单一事实源
- [Serper API](https://serper.dev/playground)
- [豆包搜索 Custom 版](https://docs.volcengine.com/docs/87772/2272953)
- [百度 AI 搜索](https://cloud.baidu.com/doc/qianfan-api/s/Wmbq4z7e5)
- [北坡聚合搜索](https://search.xiaobei.top/docs)
