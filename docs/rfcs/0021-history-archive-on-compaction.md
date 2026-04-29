# RFC-0021: 上下文压缩时归档历史消息到 sandbox

- **状态**: draft
- **优先级**: P2
- **标签**: `architecture`, `dx`
- **影响服务**: NexAU (`ContextCompactionMiddleware`, sandbox)
- **创建日期**: 2026-04-27
- **更新日期**: 2026-04-27

## 摘要

`ContextCompactionMiddleware` 在压缩时会丢弃或折叠旧消息——压缩后原始消息从工作上下文中彻底消失。本 RFC 在压缩发生时把"被移除的原始消息" + 一行 boundary 元数据 **append 到 sandbox 内的单文件 `transcript.jsonl`**，并在 summary 中注入路径提示。Agent 通过已有的 `Read`/`Grep` 工具自助召回，无需新增召回工具。多次压缩通过纯 append-only 模式自然支持，无需差集索引。

## 动机

现状：

- `SlidingWindowCompaction` 把旧消息折叠成 summary，原文丢失；
- `ToolResultCompaction` 把旧 tool result 替换为占位字符串，结果体丢失；
- `UserModelFullTraceAdaptiveCompaction` 紧急路径切两段都做摘要，全部原文丢失。

模型在压缩之后若发现需要某条早期 tool result 或某轮对话的细节，无法找回——只能依赖 summary 中的少量陈述（往往不够）。

NexAU 已有的 `AgentRunActionModel` 事件溯源虽然在 DB 里保留了 APPEND 历史，但：

1. 需要 DB 查询 + 跨表扫描；
2. 不是 agent 的"自助"路径，需要新工具或定制查询；
3. 设计目标是 session 持久化，不是面向召回的快速读路径。

需要一条**轻量旁路**：压缩时直接落盘到 sandbox，agent 用现有文件工具自取。

## 非目标

1. 不替换或修改任何现有压缩策略（`SlidingWindowCompaction` / `ToolResultCompaction` / `UserModelFullTraceAdaptiveCompaction`）；
2. 不引入新的内置工具（agent 通过 `Read`/`Grep` 召回）；
3. 不做跨 agent 共享 / 跨 session 持久化（sandbox 本是 per-agent，归档随 sandbox 生命周期）；
4. 不替换 `AgentRunActionModel` 的事件溯源持久化；
5. 不实现 FTS / 倒排索引等高级搜索（首版纯文件，agent 用 `Grep`）。

## 设计

### 概述

```
压缩前消息列表 messages_before
        │
        │ ContextCompactionMiddleware._compact_messages()
        │
        ├─→ compaction_strategy.compact() → messages_after
        │
        ├─→ removed = messages_before − messages_after  (按 message_id 差集，保留原序)
        │
        ├─→ HistoryArchiveWriter.write_round(removed, ...)
        │     └─ append 到 {sandbox}/.nexau_history_archive/transcript.jsonl
        │           ├─ N 行: 每行一条原始 Message JSON
        │           └─ 1 行: {"_boundary": {round, compacted_at, ...}}
        │
        └─→ inject_archive_hint(messages_after) → 把路径提示并入最后一条 summary 消息
```

### 详细设计

#### 存储位置

写入当前 agent 的 sandbox 根目录下的 `.nexau_history_archive/transcript.jsonl`。Sandbox 路径通过 `nexau.archs.sandbox.get_sandbox(agent_state)` 获得（与 `save_memory.py` 一致）。

```
{sandbox_root}/.nexau_history_archive/
  transcript.jsonl     # append-only, 每行一条记录
```

#### 文件格式：单文件 + 内联 boundary

`transcript.jsonl` 的每行是两种形态之一：

**A. 序列化 Message** —— `Message.model_dump_json()` 输出：

```json
{"id": "uuid-...", "role": "user", "content": [...], "metadata": {...}, "created_at": "..."}
```

**B. Boundary 元数据** —— 用 `_boundary` 顶层 key 区分（`Message` 不可能有这个键）：

```json
{"_boundary": {
  "round": 1,
  "compacted_at": "2026-04-27T10:30:00Z",
  "agent_id": "agent_xxx",
  "run_id": "run_xxx",
  "trigger_reason": "token_threshold_75pct",
  "strategy": "SlidingWindowCompaction",
  "tokens_before": 150000,
  "tokens_after": 30000,
  "removed_message_count": 47,
  "first_message_id": "uuid",
  "last_message_id": "uuid",
  "summary_message_id": "uuid",
  "preview": "first 300 chars of first removed user/assistant message..."
}}
```

读取侧用一行 `if "_boundary" in obj:` 即可分辨两类。`preview` 字段保留是为了让 agent `Grep _boundary` 一次就能扫到每轮主题。

#### ImageBlock: base64 外置到 `images/` 子目录

普通 text/tool message 序列化到 transcript.jsonl 一行就完了。但 `ImageBlock` 如果带了 `base64` data，原始 dump 会让 transcript 体积爆炸（一张 1MB 的图 → base64 ~1.4MB → 单行 1.4MB）。

处理：

- **URL ImageBlock**：原样保留，序列化进 transcript.jsonl 一行（开销几十字节，可忽略）
- **Base64 ImageBlock**：归档时
  1. 解码 base64 字节，写入 `{archive_dir}/images/{message_id}-{block_idx}.{ext}`（`ext` 由 `mime_type` 推：jpg/png/gif/webp/...，未知 → `bin`）
  2. 把内存 message 的 ImageBlock **拷贝一份**（`model_copy(deep=True)`，原 message 不动），拷贝里 `base64` 清空、`url = "file:images/{...}.{ext}"`
  3. 序列化拷贝进 transcript.jsonl

最终归档目录长这样：

```
{sandbox}/.nexau_history_archive/
  transcript.jsonl              # message 行 + boundary 行
  images/                       # 仅当本会话出现 base64 ImageBlock 时才创建
    {msg_id_1}-{idx}.png
    {msg_id_2}-{idx}.jpg
```

`BoundaryRecord` 多一个字段 `extracted_images: int` 记录本轮外置了多少张图（boundary 行里也透出，便于审计）。

实现细节：
- 失败回退：写盘失败/decode 出错时该 block 保持原样（fallback inline），转而记 `logger.warning`，不中断压缩
- 不变性：`removed` 入参里的原 `Message` 引用永远不变（agent 后续 active context 可能还要拿这些消息，不能 mutate）
- 零开销：本轮没有任何 base64 图片时不会创建 `images/` 目录、不复制消息

#### 多轮压缩：纯 append

第 N 轮压缩：

1. `before_ids = {m.id for m in messages_before}`
2. `after_ids = {m.id for m in compaction_strategy.compact(messages_before)}`
3. `removed = [m for m in messages_before if m.id not in after_ids]`
4. **append** N 行 Message JSON + 1 行 `_boundary` 到 `transcript.jsonl`

**性质**：

| 性质 | 实现 |
|---|---|
| **append-only 不可变** | 老内容永不重写，写入失败也只影响新行 |
| **不重复存储** | removed 是差集，前一轮的 summary 被下一轮再次压缩时自然进入 transcript（其 id 进入差集） |
| **崩溃安全** | 单文件 append，不存在 manifest/round 不一致；最坏情况是文件末尾一行半截，跳过即可 |
| **重启幂等** | 启动时扫所有 `_boundary` 行取 max round，下一轮 = max+1 |
| **agent 召回** | `Grep <keyword> .nexau_history_archive/transcript.jsonl` 一次到位 |

> **写入实现**：当前用 `sandbox.write_file()` 做 read+rewrite append（兼容所有 sandbox 后端）。这意味着每轮 IO ∝ 累计大小，单 session 几 MB 量级毫秒级，无需优化。如果未来 sandbox 抽象层加 `append_file()` API，可零成本切过去。

#### Summary 提示注入

启用 `save_history` 时，每轮压缩后**自动**在 summary 末尾追加路径提示，让 agent 知道归档存在并能用工具召回：

```
📁 [Archive] {N} earlier message(s) archived across {M} compaction round(s) (latest: round {R}).
To recall earlier conversation, use your file tools on `.nexau_history_archive/transcript.jsonl`:
  • `search_file_content` to grep for keywords (each matched line is a serialized Message)
  • `read_file` for full chronological view
  • Boundary lines `{"_boundary": ...}` mark each compaction round
```

`ToolResultCompaction` 不产生 summary 消息——则注入一条独立 `Role.FRAMEWORK` 消息携带同样提示。

不再单独提供 hint opt-out 开关——没有"归档但不告诉 agent"的真实场景；归档的整个意义就是让 agent 能召回。

#### 配置（`CompactionConfig`）

```python
save_history: bool = True
"""开启历史消息文件归档（默认 Opt-out: 启用压缩即归档）。

归档目录位置固定为 ``{sandbox.work_dir}/.nexau_history_archive/``,
不暴露成 config —— 避免用户输入路径穿越, 默认值就是设计上的唯一选择。

启用归档时, summary 末尾会自动注入"如何召回"的提示文本。
"""
```

> 注：归档目录名在源码里是常量 `ARCHIVE_SUBDIR = ".nexau_history_archive"`，**不允许通过 config 改写**。理由：(1) 没有真实场景需要改名；(2) 如果暴露成字符串字段，恶意 / 误传 `"../foo"` 或 `"/tmp/leak"` 之类会让归档逸出 sandbox。

#### 实现挂载点

唯一侵入点：`ContextCompactionMiddleware._compact_messages()`。三处压缩入口（`before_model` / `after_model` / `wrap_model_call` 紧急路径）共用一个辅助方法 `_maybe_archive_compaction()`：

1. 检查 `save_history` flag 和 `agent_state` 可用性
2. 计算 removed 差集
3. 按需 lazy 创建 `HistoryArchiveWriter`
4. 调用 `write_round()`
5. 按需注入 hint

`HistoryArchiveWriter`（`history_archive.py`）：

```python
class HistoryArchiveWriter:
    @classmethod
    def from_sandbox(cls, *, agent_state: Any, subdir: str) -> "HistoryArchiveWriter | None":
        """从 agent_state 提取 sandbox; 失败返回 None。"""

    def write_round(
        self,
        *,
        removed: list[Message],
        tokens_before: int | None,
        tokens_after: int | None,
        trigger_reason: str,
        strategy_name: str,
        run_id: str | None,
        agent_id: str | None,
    ) -> BoundaryRecord | None:
        """append removed messages + 1 boundary 行到 transcript.jsonl。"""
```

### 示例

```yaml
# agent.yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.context_compaction:ContextCompactionMiddleware
    params:
      max_context_tokens: 200000
      auto_compact: true
      threshold: 0.75
      compaction_strategy: "llm_summary"
      keep_iterations: 3
      # —— 本 RFC 新增 (默认值即如下, 启用归档时 hint 自动注入)
      save_history: true
```

触发 3 轮压缩后：

```
{sandbox}/.nexau_history_archive/transcript.jsonl
  # 第 1 轮被移除的 5 条 Message JSON
  # 1 行 _boundary (round=1)
  # 第 2 轮被移除的 4 条 Message JSON (含 round 1 produced summary)
  # 1 行 _boundary (round=2)
  # 第 3 轮被移除的 6 条 Message JSON (含 round 2 produced summary)
  # 1 行 _boundary (round=3)
```

Agent 后续 turn 中接收到 summary 内嵌 hint，调 `Grep "agent_events_middleware" transcript.jsonl` 一次召回。

## 权衡取舍

### 考虑过的替代方案

**A. 多文件: `round_NNNN.jsonl` + `manifest.jsonl`** （初版实现，已废弃）：
- 每轮一个 round 文件 + 一个 manifest 索引行；
- 优点：每轮文件不可变；manifest 给"按轮"统计提供索引；
- 缺点：（1）召回需要"先 Read manifest 再 Read round 文件"两步；（2）manifest 增量写实际上也是 read+rewrite，并未省 IO；（3）round 编号是给人看的，对 agent 没用；（4）实现代码量是单文件版的两倍。
- 决定：**改用单文件**。append-only 单文件本身就是不可变 prefix，所有声称的优势单文件都有；agent 召回一步到位。

**B. 新增 `recall_history` 内置工具**：
- 优点：模式化召回（list_rounds / get_round / search_text / get_messages），可加 token cap、分页；
- 缺点：（1）需要给所有 agent 自动注册或显式配置；（2）查询逻辑与 agent 已有的 `Read`/`Grep` 重复；（3）维护成本更高。
- 决定：**不采用**。Agent 已经熟练使用 `Read`/`Grep`，归档落到文件就行。

**C. 扩展 `AgentRunActionModel` 加新字段或新表存归档**：
- 优点：所有持久化收敛在 session DB；
- 缺点：（1）DB 不便 grep；（2）和现有 APPEND/REPLACE 语义重叠；（3）额外多后端兼容工作（SQL/JSONL/Memory/Remote）。
- 决定：**不采用**。归档读路径要的是文本可搜，DB 不合适。

**D. NEXAU_HOME 全局路径而非 sandbox**：
- 优点：跨 agent 共享归档；
- 缺点：（1）目标场景就是"本 agent 自救"，跨 agent 不必要；（2）违反 sandbox 隔离原则；（3）需要额外清理策略。
- 决定：**不采用**。

### 缺点

1. 每轮压缩多一次磁盘 IO。当前用 `sandbox.write_file()` read+rewrite append，IO ∝ 累计大小。对常见 session 几十轮、几 MB 内毫秒级，无需优化；超大 session 可未来加 `append_file()` API。
2. Sandbox 不存在或不可写时静默跳过归档（degrade gracefully），通过 `logger.warning` 记录。
3. 第三方压缩策略若返回的 messages 不带稳定 `id`，差集计算会失效——目前所有现有策略都保留 message id（含新生成的 summary message），合规。

## 实现计划

### 阶段划分

- [x] Phase 1: 配置扩展（`CompactionConfig` 三个新字段）
- [x] Phase 2: `HistoryArchiveWriter` 单文件实现 + 单元测试
- [x] Phase 3: `ContextCompactionMiddleware` 接入 + hint 注入
- [x] Phase 4: 端到端验证（多轮压缩 + 召回 + 不变量）

### 相关文件

- `nexau/archs/main_sub/execution/middleware/context_compaction/config.py` — 配置字段
- `nexau/archs/main_sub/execution/middleware/context_compaction/middleware.py` — 接入归档 + hint 注入
- `nexau/archs/main_sub/execution/middleware/context_compaction/history_archive.py` — 归档读写器
- `tests/unit/test_history_archive.py` — 测试
- `examples/test_history_archive_e2e.py` — 真实 LLM e2e 验证脚本

## 测试方案

### 单元测试

- `test_writes_messages_then_boundary_line`：单轮写入后 `transcript.jsonl` 含 N 行 Message + 1 行 `_boundary`
- `test_round_numbers_increment`：多轮压缩后 boundary 行的 round 字段递增
- `test_transcript_is_append_only`：第 2 轮写入不修改第 1 轮已写入的 prefix（字节级）
- `test_resume_after_existing_transcript`：预置 transcript 仅有 round 1 的 boundary，writer 实例化后下一轮 = 2
- `test_diff_excludes_kept_messages`：返回 messages 中保留的 id 不进入归档；新生成的 summary id 不在 before_ids，不会进归档
- `test_creates_archive_dir` / `test_returns_none_when_*`：`get_sandbox` 抛错或返回 None 时 graceful degrade
- `test_summary_id_recorded` / `test_no_summary_id_when_absent`：boundary 的 `summary_message_id` 字段
- `test_preview_skips_system_role` / `test_preview_truncated_to_300`：preview 生成
- `test_hint_mentions_path_and_search_tools`：hint 文本包含路径 + 真实工具名 (`search_file_content` / `read_file`)

### 集成测试

`tests/unit/test_history_archive.py::TestMiddlewareArchiveIntegration` 4 个用例覆盖 middleware 与 writer 的耦合：

- `test_maybe_archive_writes_diff_and_injects_hint`：单文件 + summary hint 注入
- `test_maybe_archive_skipped_when_flag_false`：flag 关时不创建归档
- `test_no_summary_falls_back_to_framework_message`：无 summary 时注入 framework 消息
- `test_multiple_rounds_append_to_same_file`：多轮始终写入同一 transcript 文件

### 手动验证

```bash
NEXAU_LOG_LEVEL=DEBUG \
  uv run python examples/test_history_archive_e2e.py
ls $SANDBOX_ROOT/.nexau_history_archive/   # 应只有 transcript.jsonl
grep -c '"_boundary"' $SANDBOX_ROOT/.nexau_history_archive/transcript.jsonl   # 行数 == 压缩轮数
```

召回验证：

1. 让 agent 跑多轮对话直到触发 2-3 轮压缩
2. 在新 turn 询问"刚才第一轮我们讨论的 X 是什么"
3. 观察 agent 是否调用 `Grep` / `Read` 读取 `.nexau_history_archive/transcript.jsonl`

## 未解决的问题

1. 单文件涨到几十 MB 后的 read+rewrite 成本：目前用 `sandbox.write_file()` 全量重写，IO ∝ 累计大小。对常规 session 没问题，但极端长会话需要给 `BaseSandbox` 加 `append_file()` API。等出现需求再做。
2. 跨 sandbox 持久化（保留旧 session transcript 供新 session 召回）的需求暂未明确——如有再考虑导出/导入流程。
3. 是否暴露 `HistoryArchivedEvent` 给 UI 流？倾向暂不加，需求出现再补。

## 参考资料

- RFC-0004: 上下文溢出紧急压缩
- RFC-0016: Micro-compact（微压缩）
- `nexau/archs/main_sub/execution/middleware/context_compaction/` 现有实现
- `nexau/archs/tool/builtin/session_tools/save_memory.py`（sandbox 写文件参考实现）
- Claude Code 的 transcript 单文件方案（`~/.claude/projects/{proj}/{sessionId}.jsonl` + `SystemCompactBoundaryMessage`）—— 启发了从多文件改单文件的决定
