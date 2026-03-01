# RFC-0003: LLM 自动降级中间件 (Failover Middleware)

- **状态**: implemented
- **优先级**: P1
- **标签**: `architecture`, `reliability`
- **影响服务**: Agent 执行层、LLM 调用链
- **创建日期**: 2026-02-28
- **更新日期**: 2026-02-28

## 摘要

当主 LLM provider 因服务端错误（500/502/503）或限流（429）不可用时，自动按配置顺序切换到备用 provider，保障 Agent 执行不中断。通过 Middleware 机制实现，零侵入核心 LLMCaller 代码。

## 动机

生产环境中 LLM provider 可能因以下原因不可用：
- 服务端内部错误（HTTP 500/502/503）
- 限流（HTTP 429 / RateLimitError）
- 网关超时
- 区域性故障

当前 NexAU 的 `LLMCaller._call_with_retry()` 只支持对同一 provider 的指数退避重试，无法在 provider 级别做降级。NexTask 分支实现了一个 failover 方案，但以 ~240 行侵入性代码直接嵌入 `LLMCaller` 核心类，存在以下问题：

1. 侵入性强，所有方法带 `_nextask_` 前缀
2. 直接突变共享 `llm_config` 对象
3. 大量 `Any` / `getattr`，违反类型安全规范
4. 只支持单一 fallback，不可逆
5. 配置绕道存放在 `global_storage`

NexAU 已有完善的 `Middleware.wrap_model_call()` 机制，是实现 failover 的理想载体。

## 设计

### 概述

新增 `LLMFailoverMiddleware`，继承 `Middleware` 基类，通过 `wrap_model_call(params, call_next)` 拦截 LLM 调用。主 provider 失败且匹配触发条件时，创建新的 `LLMConfig` + SDK client，以新 `ModelCallParams` 调用 `call_next`，不修改原始对象。

```
Agent YAML
  → MiddlewareManager 加载 LLMFailoverMiddleware
    → wrap_model_call(params, call_next)
      → try: call_next(params)           # 主 provider
      → except: 匹配触发条件?
        → _apply_fallback(params, provider)  # 创建新 params
        → call_next(new_params)              # 备用 provider
```

### 详细设计

#### 触发条件 (FailoverTrigger)

匹配逻辑为 OR：
- `status_codes`: HTTP 状态码列表，通过 `isinstance(exc, openai.APIStatusError)` 类型安全提取
- `exception_types`: 异常类名列表，如 `"ConnectionError"`

#### 降级链 (FallbackProvider)

支持多个备用 provider，按配置顺序依次尝试。每个 fallback 的 `llm_config` 字段继承主 provider 未覆盖的配置（如 model、temperature）。

#### 熔断器 (CircuitBreaker, 可选)

状态机: CLOSED → OPEN → HALF_OPEN → CLOSED
- 连续失败超过 `failure_threshold` 次后进入 OPEN，直接跳过主 provider
- `recovery_timeout_seconds` 后进入 HALF_OPEN，允许一次探测

#### 不可变性

`_apply_fallback()` 通过 `LLMConfig.copy()` + `copy.copy(params)` 创建新对象，不修改原始 `params` 或 `llm_config`。

### 示例

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.llm_failover:LLMFailoverMiddleware
    params:
      trigger:
        status_codes: [500, 502, 503, 529]
        exception_types: ["RateLimitError", "InternalServerError"]
      fallback_providers:
        - name: "backup-gateway"
          llm_config:
            base_url: ${env.FALLBACK_LLM_BASE_URL}
            api_key: ${env.FALLBACK_LLM_API_KEY}
        - name: "emergency"
          llm_config:
            model: "gpt-4o"
            base_url: ${env.EMERGENCY_LLM_BASE_URL}
            api_key: ${env.EMERGENCY_LLM_API_KEY}
            api_type: "openai_chat_completion"
      circuit_breaker:
        failure_threshold: 3
        recovery_timeout_seconds: 60
```

## 权衡取舍

### 考虑过的替代方案

1. **侵入式修改 LLMCaller**（NexTask 方案）：直接在 `_call_with_retry()` 的 except 块中嵌入 failover 逻辑。优点是实现直接；缺点是侵入性强、突变共享状态、不可测试、违反类型安全规范。
2. **Multi-Provider LLMConfig**：扩展 `LLMConfig` 支持多个 provider 配置。缺点是破坏单一职责，复杂化配置模型。

### 缺点

- 每次 failover 都会重建 SDK client（`openai.OpenAI()` / `anthropic.Anthropic()`），有少量开销
- 不同 provider 的 tool calling 格式可能不兼容（如从 Anthropic 降级到 OpenAI 时 tool schema 格式不同）

## 实现计划

### 阶段划分

- [x] Phase 1: 核心 middleware 实现 + 单元测试
- [x] Phase 2: 集成测试
- [ ] Phase 3: 审计日志（可选，通过 logging 已覆盖基本需求）

### 相关文件

- `nexau/archs/main_sub/execution/middleware/llm_failover.py` — 核心实现
- `tests/unit/test_llm_failover_middleware.py` — 单元测试（20 cases）
- `tests/integration/test_llm_failover_integration.py` — 集成测试（3 cases）
- `nexau/archs/main_sub/execution/hooks.py` — Middleware 基类和 ModelCallParams

## 测试方案

### 单元测试

已实现 20 个测试用例，覆盖：
- `_extract_status_code`: OpenAI/Anthropic SDK 异常状态码提取
- `_CircuitBreaker`: 状态机转换（CLOSED/OPEN/HALF_OPEN/reset）
- `LLMFailoverMiddleware` 构造：默认/自定义 trigger、circuit breaker
- `wrap_model_call`: 主 provider 成功、failover 成功、非匹配错误透传、所有 provider 失败、异常类名匹配、配置继承、不可变性、多级降级、熔断器跳过主 provider

### 手动验证

```bash
# 运行测试
uv run pytest tests/unit/test_llm_failover_middleware.py -v --no-cov

# 类型检查
uv run pyright nexau/archs/main_sub/execution/middleware/llm_failover.py
uv run mypy nexau/archs/main_sub/execution/middleware/llm_failover.py
```

## 未解决的问题

1. 跨 provider tool calling 兼容性：从 Anthropic 降级到 OpenAI 时，tool schema 格式需要转换，当前未处理
2. 是否需要支持 failover 后自动切回（当前每次调用独立判断，circuit breaker 的 HALF_OPEN 已部分覆盖）

## 参考资料

- NexTask 分支 `llm_caller.py` failover 实现
- NexAU Middleware 体系：`nexau/archs/main_sub/execution/hooks.py`
