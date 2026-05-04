# RFC-0023 Merge Runbook（飞书友好版）

> **目的**：4 个 PR 串成的 stack，讲清楚每个 PR 干啥、依赖关系、CI 怎么跑、怎么按顺序合
>
> **创建日期**：2026-05-04 · **状态**：ready-to-merge

---

## 一句话讲清这几个 PR

把现有「两套独立的 LLM 流式解析器」（Set A 推事件 + Set B 拉 ModelResponse）合并成「一套 Set A」，删掉 ~8400 行重复代码，同时把测试从「等价性 parity」升级到「真实 LLM 端到端 e2e」覆盖。

---

## Stack 拓扑

```
main
 └── PR-A #508  feat: parity 测试基建（3-axis）           ← 准备合
     └── PR-B #510  feat: ModelCallFinishedEvent 等 sidecar event
         └── PR-C.1 #513  feat: Set A build() 返回 vendor 类型 + axis-4 parity
             └── PR-C.2 #514  feat: Set B 退役 + 41 个新测试
```

**合并顺序**：`#508 → #510 → #513 → #514`，按顺序点 merge button 即可。每个 PR 合掉后下一个 PR 的 base 会自动指向 main。

---

## 每个 PR 一句话

| PR | 行数 | 一句话 |
|---|---|---|
| **[#508](https://github.com/china-qijizhifeng/nexau/pull/508)** | +约 4k | 录 SSE fixture + parity test 基建（Set A vs Set B 等价断言） |
| **[#510](https://github.com/china-qijizhifeng/nexau/pull/510)** | +约 1.5k | 加 `ModelCallFinishedEvent` 让 AG-UI 端能承载 stop_reason / model_name |
| **[#513](https://github.com/china-qijizhifeng/nexau/pull/513)** | +约 2k / -约 1k | Set A 加 `build()` 方法返回 vendor 类型；axis-4 parity 验证 |
| **[#514](https://github.com/china-qijizhifeng/nexau/pull/514)** | +约 800 / -约 8.4k | 删 Set B；llm_caller 走 Set A；19 live e2e + 22 mock unit + 4 traced e2e |

---

## CI 怎么跑

### 7 个 job（都要绿）
| Job | 跑啥 | 大概耗时 |
|---|---|---|
| `lint` | ruff format + ruff check | ~30s |
| `typecheck` | mypy + pyright | ~1min |
| `windows-quality` | Windows 平台 quality 检查 | ~3min |
| `windows-target-tests` | Windows target 集成 | ~3min |
| `windows-entrypoint-smoke` | Windows entrypoint smoke | ~1min |
| **`test-saas`** | E2B SaaS + **本 PR 的 19 live LLM e2e** | ~10min |
| `test-selfhost` | E2B 自托管 sandbox | ~5min |
| `codecov/patch` | 改动行覆盖率 ≥80% | 自动 |

### test-saas 需要的 GitHub Secrets / Variables

```
secrets:
  LLM_API_KEY                          # 主 LLM key (14.103 网关)
  E2B_API_KEY                          # E2B SaaS
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY  # 已有
  NORTHGATE_API_KEY                    # 新增 (PR-C.2): northgate.xiaobei.top 网关
  TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY  # 新增 (PR-C.2): 14.103 网关上 Gemini 路由 (paid tier)

variables:
  LLM_BASE_URL = http://14.103.60.158:3001/v1
  LLM_MODEL = nex-agi/deepseek-v3.1-nex-1
  LANGFUSE_HOST = https://langfuse.xiaobei.top/
```

PR-C.2 在 ci.yml 里加了这几行：
```yaml
NEXAU_RUN_LIVE_LLM_TESTS: "1"
GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
NORTHGATE_API_KEY: ${{ secrets.NORTHGATE_API_KEY }}
TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY: ${{ secrets.TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY }}
```

### 本地跑 e2e（验证用）

```bash
TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY=<key>  \
NORTHGATE_API_KEY=<key>                    \
NEXAU_RUN_LIVE_LLM_TESTS=1                 \
LLM_API_KEY=ignored                        \
uv run pytest tests/integration/test_aggregator_live_e2e.py -v

# 期望: 23 passed in ~95s
```

---

## 测试覆盖矩阵

| 维度 | OpenAI Chat | OpenAI Responses | Anthropic | Gemini |
|---|---|---|---|---|
| Happy path (sync) | ✅ | ✅ | ✅ | ✅ |
| Tool calling | ✅ | ✅ | ✅ | ✅ |
| Reasoning/thinking | ✅ (deepseek-v4-pro) | ✅ (relaxed) | ✅ (thinking + sig) | ✅ (3.1-pro) |
| Async variant | ✅ | ✅ | ✅ | ✅ |
| **Tracing branch** | ✅ | ✅ | ✅ | ✅ |
| Multi-chunk (long) | ✅ | — | ✅ | — |
| shutdown_event break | ✅ | — | — | — |

**Mock unit 补的边界 case**（22 个）：
- `_maybe_wrap_stream_idle_timeout` × 6（httpx/requests/builtin/duck-typed timeout + 2 negative）
- `_get_event_emitter` × 3（chain walk + no-op fallback）
- `_resolve_run_id` × 3
- `_chat_completion_to_model_response` × 2（正向 + ValueError on empty）
- `_process_stream_chunk` × 4（drop / mutate / no-op）
- Anthropic aggregator edge cases × 4（truncated / 重复 block_start / fragmented JSON / invalid JSON fallback）

---

## ⚠️ 已知 Pre-existing Flake（**不**阻塞合并）

### 1. `test_thinking_cross_validation_langfuse::test_task_b_thinking_logic_computation`
- **症状**：让 LLM 算 `120 + 180 + 360`，模型偶尔答 600（应该 660），导致 ` < 1.0` 容差挂掉
- **原因**：deepseek-v3.1-nex-1 推理对、最后心算翻车（典型 LLM 算术 flake）
- **跟 PR-C.2 无关**：跟踪 issue [#516](https://github.com/china-qijizhifeng/nexau/issues/516)

### 2. `test_interrupt_persistence::test_finally_flush_on_cancelled_error`
- **症状**：asyncio cancellation timing flake
- **跟 PR-C.2 无关**：建议单独 issue 跟踪

### 3. test-selfhost E2B sandbox 偶发连接超时（>120s）
- **症状**：socket connect 到 prod-e2b.xiaobei.top 超时
- **原因**：infra 网络偶发
- **跟踪 issue**：[#517](https://github.com/china-qijizhifeng/nexau/issues/517)

### CI 重试策略
test-saas 失败时 → 看是不是 #516 / #517 / interrupt_persistence flake → 是的话直接 re-run；不是的话才查代码。

---

## Follow-up issues（**不阻塞合并**，PR-C.2 之外的清扫工作）

| Issue | 内容 | 优先级 |
|---|---|---|
| [#515](https://github.com/china-qijizhifeng/nexau/issues/515) | refactor: Gemini base_url 单一规则（避免 substring-match 特判） | P2 |
| [#516](https://github.com/china-qijizhifeng/nexau/issues/516) | flake: thinking_cross_validation 数学算错容忍度 | P3 |
| [#517](https://github.com/china-qijizhifeng/nexau/issues/517) | ci: E2B sandbox 偶发超时 | P2 |

---

## RFC-0023 解锁的下游工作

RFC-0023 §阶段 ③ 完成后，**RFC-0022 Phase 2**（iter-level 持久化）不再阻塞。

- [PR #503](https://github.com/china-qijizhifeng/nexau/pull/503)：RFC-0022 Phase 1（协议原语）—— 跟 RFC-0023 stack **正交**，可独立合并，建议先 rebase + 重跑 CI
- RFC-0022 Phase 2 PR：待写，基于 main，会同时用到 #503 的 `event_id` / `RUN_START` 和 RFC-0023 的单源解析

---

## 需要 reviewer 关注什么

### #508 (PR-A)
- parity test 设计：3-axis（强等价 / 弱差距 / vendor truth）
- 录 fixture 的 SSE 文件大小：是否能接受

### #510 (PR-B)
- `ModelCallFinishedEvent` schema 是否符合 AG-UI 协议
- 「Occam pass」删了 `usage` / `provider_extras` 字段，确认这些信息已经在别的 event 里有了

### #513 (PR-C.1)
- `build()` 返回 vendor 类型而不是 dict 的设计是否合理
- axis-4 parity（byte-equal）测试值不值

### #514 (PR-C.2)
- **重点 review 的代码改动**：`llm_caller.py` 8 个 call site 的 swap，每个走 sync + async 两条路径
- **重点验证**：`_chat_completion_to_model_response` —— 这是 review 中发现的 OpenAI Chat shape mismatch 修复
- **测试覆盖**：23 e2e + 22 unit，是否还有该补的 case
- **删了什么**：`tests/aggregator_parity/` axis 1/2/4，`tests/unit/` 8 个 Set B 相关测试，看删的对不对

---

## 一行命令本地复刻 PR-C.2 全套测试

```bash
TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY=<key> \
NORTHGATE_API_KEY=<key> \
NEXAU_RUN_LIVE_LLM_TESTS=1 \
LLM_API_KEY=ignored \
uv run pytest tests/integration/test_aggregator_live_e2e.py \
              tests/unit/test_llm_caller_helpers.py \
              -v --tb=short

# 期望: 45 passed in ~100s
```
