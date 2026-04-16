# RFC-0019: 图像令牌化与缓存

- **状态**: implemented
- **优先级**: P3
- **标签**: `multimodal`, `image`, `cache`, `prompt-cache`, `architecture`
- **影响服务**: `nexau/core/messages.py`、`nexau/core/adapters/legacy.py`、`nexau/core/adapters/anthropic_messages.py`、`nexau/archs/tool/builtin/file_tools/read_visual_file.py`、`nexau/archs/main_sub/agent.py`、`nexau/archs/main_sub/execution/llm_caller.py`、`nexau/archs/main_sub/execution/model_response.py`、`nexau/archs/main_sub/execution/middleware/context_compaction/middleware.py`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

本 RFC 描述 NexAU 中图像数据的"令牌化"路径（从 tool 输出 / 用户输入中识别图像、规整成 UMP `ImageBlock`、再转成各厂商 vendor 格式），以及 prompt 缓存机制（OpenAI Responses API 的 `prompt_cache_key`、Anthropic 的 `cache_control: ephemeral`、以及 `cache_creation_input_tokens` / `cache_read_input_tokens` 在 token 计费上的归一化）。两条主线在 nexau 内部交叉：image 进入 prompt 是缓存命中的高价值候选项，而缓存又决定了多模态 prompt 的真实计费方式。本 RFC 同时把"图像令牌化（按视觉 token 数计费）"中目前 nexau 没有实现的部分（如 OpenAI vision tile 计算）归入未解决的问题。

## 动机

NexAU 已经支持工具返回图像（`ToolOutputImage` / `read_visual_file`）和 user 多模态消息，但实现散落在多个文件、且 vendor 间差异很大：

- **OpenAI Chat Completions** 不允许 `role=tool` 消息携带 image parts；需要把图像折叠到一条额外的 user 消息里（`legacy.py:288-351`）。
- **OpenAI Responses API** 既要 `prompt_cache_key` 走 `extra_body`，又要保留图像在 tool_result 上下文里。
- **Anthropic Messages API** 走原生 `image` block，并支持 `cache_control: ephemeral` 标记（`llm_caller.py:520-536`）。
- 多视觉模型按"视觉 token"计费（如 GPT-4o 把图像切 tile 后计 token），但 nexau 当前无该计算，所有 vision token 由 vendor usage 字段反算。

prompt cache 同样跨 vendor：

- OpenAI 通过 `prompt_cache_key`（路由 hint）让相同 key 的请求命中同一个 cache 集群；nexau 在 `Agent.__init__` 自动注入唯一 key（`agent.py:149-155`）。
- Anthropic 必须在 prompt content block 上显式打 `cache_control` 才能进缓存；nexau 在 LLM 调用前把 `cache_control: ephemeral` 注入到最后一条 user message 的首个 content block（`llm_caller.py:528-532`、`llm_caller.py:560-564`）。
- 缓存命中后 vendor 用 `cache_creation_input_tokens` / `cache_read_input_tokens` 报账，nexau 必须把这些字段折算回"标准 input_tokens"（`model_response.py:140-175`），否则上下文压实中间件会算错总 token（`context_compaction/middleware.py:230-249`）。

把以上散落的实现合并到一篇 RFC，便于后续扩展（如 vision token 估算、按 model 类型精细化缓存 TTL）。

## 设计

### 概述

数据流分两条主线：

1. **图像令牌化路径**：tool 返回 → `coerce_tool_result_content` → `ImageBlock` → vendor 适配（OpenAI Chat / Anthropic / Responses）→ vendor 内部 image tokenization。
2. **缓存路径**：`Agent.__init__` 注入 `prompt_cache_key`（OpenAI Responses 专用）→ LLM caller 在 messages 首块上打 `cache_control`（Anthropic 专用）→ vendor 返回 usage with cache fields → `model_response._normalize_usage` 折算回标准 input_tokens → context compaction middleware 用归一化后的 token 数判断窗口压力。

两条路径的接合点是 `_as_legacy_openai_chat_dict`（`messages.py:307-396`）与 `messages_to_legacy_openai_chat`（`legacy.py:288-351`）：image 进入 vendor payload 时正好处在 cache_control 注入位置之前，因此一个带图的 user message 可以同时作为 cache anchor 和 multimodal prompt。

### 详细设计

#### UMP 图像数据模型

`ImageBlock` 是 Pydantic 模型（`messages.py:45-60`）：

- `type: Literal["image"] = "image"`（discriminator 标签）。
- `url: str | None`、`base64: str | None`，二选一。
- `mime_type: str = "image/jpeg"`。
- `detail: Literal["low", "high", "auto"] = "auto"`（多模态模型的精度提示，OpenAI 风格）。
- `@model_validator(mode="after") _validate_source` 保证 `url` 与 `base64` 恰好二选一，否则 `ValueError`（`messages.py:54-60`）。

`ToolOutputImage` 与 `ToolOutputImageDict` 是给开发者写工具时的"友好返回类型"（`messages.py:63-79`、`messages.py:82-95`）：

- `ToolOutputImage(image_url="https://..." | "data:...")`，可选 `detail`。
- `_validate` 阻止空字符串 `image_url`。
- TypedDict 形式提供等价 dict shape（`type: "input_image"`、`image_url`、`detail`）。

`ToolResultContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]`（`messages.py:264`）正式声明 tool result 的 content 是 text + image 的判别联合。`ToolResultBlock.content` 类型是 `str | list[ToolResultContentBlock]`，向后兼容历史的 plain string（`messages.py:267-273`）。

#### data URL 解析与图像归一

`_parse_base64_data_url` 是个鲁棒的 `data:<mime>;base64,<data>` 解析器（`messages.py:98-119`）：

- 不以 `data:` 开头返回 `None`。
- `,` 切分；缺失 `;base64` header 返回 `None`。
- mime 缺省 `image/jpeg`。
- 容忍 `b'...'` / `b"..."` 包裹（防 `str(bytes)` 误用）和单/双引号包裹。
- 用 `"".join(data.split())` 删除所有空白以兼容厂商对 base64 的严格要求。

`parse_base64_data_url`（`messages.py:122-129`）是公开包装，注释强调 callers 经常会粘贴带空白的 base64。

`_image_block_from_image_url`（`messages.py:132-137`）在解析成功时构造 `ImageBlock(base64=..., mime_type=...)`、否则当作 URL `ImageBlock(url=...)`。

#### `coerce_tool_result_content` 递归 coercion

`coerce_tool_result_content(value, *, fallback_text=None)`（`messages.py:140-243`）是 image 进入消息体系的核心入口。它接受任意工具返回值，输出 `str | list[TextBlock | ImageBlock]`。

`from_mapping` 处理单个 dict 项（`messages.py:157-185`）：

- `type ∈ {text, output_text, input_text}` → `TextBlock`。
- `type ∈ {input_image, image}` → 优先 `image_url`/`url`，否则 `base64`+`media_type`/`mime_type`。
- 未识别类型尝试退回 `text`/`content` 字段成 `TextBlock`，避免丢失 user-visible payload。

`coerce_any` 递归处理任意输入（`messages.py:187-218`）：

- `None` → `[]`；`str` → 单 `TextBlock`（空字符串丢弃）。
- `dict` 没 `type` 字段时探查 `content`/`result` 包裹（兼容 ToolExecutor 输出形态）。
- `list` 递归。
- `ToolOutputImage` 直接转 `ImageBlock`。
- 兜底 `str(item)`。

主体逻辑（`messages.py:220-243`）：

- 如果输入是字符串，先尝试 `json.loads`，解析成功后再走 `coerce_any`，**仅当含图像时才返回 blocks**；否则保持历史行为返回 `fallback_text` 或原字符串。
- 非字符串输入直接 `coerce_any`；含图像才返回 list；否则把 text blocks 拼成单一字符串以保持旧行为。

这样工具返回 `{"type": "image", "image_url": "..."}` 自然变成 `ImageBlock`，而老式 `{"result": "..."}` 仍走字符串路径。

#### `Message._as_legacy_openai_chat_dict` 多模态分支

`messages.py:307-388` 实现了把 UMP `Message` 渲染成 legacy OpenAI Chat dict 的 best-effort 转换：

- `Role.TOOL`：取首个 `ToolResultBlock`，list content 时用 `"<image>"` 占位（OpenAI Chat tool-role 不接受 image parts）（`messages.py:315-324`）。
- 其它 role：先扫一遍 `has_images = any(isinstance(b, ImageBlock) for b in self.content)`；若有图，文本块改用 `content_parts=[{"type":"text","text":...}]` 拼装，否则走平面 `text_parts`（`messages.py:334-345`）。
- `ToolUseBlock` 转 `tool_calls[*].function.arguments`，优先用 `raw_input` 否则 `json.dumps`（`messages.py:348-358`）。
- `ImageBlock` 渲染为 `{"type":"image_url", "image_url":{"url":..., "detail":...}}`，`detail=="auto"` 时省略 detail 字段（`messages.py:371-379`）。

#### `messages_to_legacy_openai_chat` 的 tool_image_policy

`legacy.py:288-351` 是 OpenAI Chat 适配的核心函数。`tool_image_policy` 默认 `"inject_user_message"`，因为 OpenAI Chat tool-role 消息禁止 image parts：

- `_image_part_to_image_url_obj`（`legacy.py:309-314`）拼装 `image_url` 对象，`detail=="auto"` 时省略 detail。
- `_emit_tool_result_as_messages`（`legacy.py:316-351`）把 tool_result 拆成两条消息：先一条 `role=tool` 文本（图像位置用 `<image>` 字面量），再一条 `role=user` 多模态消息携带真实 image parts，前缀文本 `f"Images returned by tool call {tool_call_id}:"`。
- 如果策略是 `"embed_in_tool_message"`（用于 Responses API 的 input 重构），则一条 tool message 同时含 text+image parts，跳过额外 user message（`legacy.py:337-339`）。

#### `messages_from_legacy_openai_chat` 反向：image 与 inject 还原

`legacy.py:96-138` 处理 OpenAI Chat 反向转 UMP 时的图像识别：

- `part_type ∈ {image, image_url, input_image}` 都视为图像（`legacy.py:96`）。
- `url` 优先从顶层 `url`、其次 `image_url.url`、再次 `image_url`（input_image 风格）（`legacy.py:97-101`）。
- `b64` 直接读 `base64`，`mime` 缺省 `image/jpeg`（`legacy.py:102-103`）。
- `detail` 限定到 `{low, high, auto}`，否则强制 `auto`（`legacy.py:104-107`）。
- 若 `url` / `b64` 是 `data:...` 形式，调用 `parse_base64_data_url` 拆出 mime + 纯 base64 后再构造 `ImageBlock`，避免重复保留 `data:` 前缀（`legacy.py:108-126`）。

`_INJECTED_TOOL_IMAGES_RE`（`legacy.py:29`）匹配前面 `_emit_tool_result_as_messages` 注入的"Images returned by tool call X:"前缀，配合 `legacy.py:182-220` 的回滚逻辑：扫到这种 user message 时，把里面的 image parts 合并回前面对应 tool message 的 `ToolResultBlock.content`、并丢弃这条 user message。这样 `to_vendor_format` → `from_legacy` 是幂等的。

#### Anthropic 适配与 cache_control

`anthropic_payload_from_legacy_openai_chat`（`anthropic_messages.py:140-168`）是 NexAU 内现存最直接的"legacy OpenAI dict → UMP → Anthropic 内容块"通路：

- 调用 `messages_from_legacy_openai_chat` 拿 UMP。
- 用 `AnthropicMessagesAdapter().to_vendor_format` 转 Anthropic system + messages（含图像 block，按 Anthropic 原生格式）。
- 当传入 `system_cache_control_ttl` 时，把 `cache_control: {"type": "ephemeral", "ttl": <ttl>}` 打在最后一条 user message 的首个 text content 上（`anthropic_messages.py:156-166`）。

实际线上 `llm_caller._call_anthropic` 不走 `system_cache_control_ttl`，而是直接在请求侧手动注入：

- 非流式（`llm_caller.py:520-536`）：`openai_to_anthropic_message` 之后，给 last user message 的首个 content 打 `cache_control: ephemeral`（注释说明 ttl 因 litellm 兼容问题被注释掉）。
- 流式（`llm_caller.py:555-568`）同样逻辑。
- `kwargs.pop("anthropic_cache_control_ttl", None)` 把代理参数从 vendor kwargs 中剥离。

#### OpenAI Responses API 与 prompt_cache_key

`Agent.__init__` 在 `agent.py:149-155` 自动注入：

- `if api_type == "openai_responses":` 先看 `llm_config.get_param("prompt_cache_key")`，缺失则 `uuid.uuid4()` 写回。
- 注释明确"每个 agent 生命周期使用固定的 key（跨轮次不变），不同 agent 使用不同 key"。

`llm_caller._call_openai_responses` 处理 vendor SDK 限制（`llm_caller.py:797-804`）：

- `prompt_cache_key` 不能直接作为 SDK kwarg（OpenAI SDK v2+ 拒绝非标准字段），必须移入 `extra_body`。
- 同时函数把 `extra_body` pop / 合并 / 回填，避免污染主 payload。

#### usage 归一化（cache token accounting）

`model_response._normalize_usage`（`model_response.py:140-175`）的 cache 处理：

- `direct_input = input_tokens or prompt_tokens`（`model_response.py:149`）。
- `cache_creation = usage.get("cache_creation_input_tokens", 0)`（`model_response.py:152`）。
- `cache_read = usage.get("cache_read_input_tokens", 0)`（`model_response.py:153`）。
- 若 usage 含任一 cache 字段，记录 `input_tokens_uncached = direct_input` 后把 `direct_input = cache_creation + cache_read + direct_input`（`model_response.py:155-158`）。
- 标准化输出 `input_tokens / completion_tokens / reasoning_tokens / total_tokens`，`total_tokens` 不存在时用三者求和（`model_response.py:160-173`）。

`reasoning_tokens` 兜底从 `completion_tokens_details.reasoning_tokens` 或 `output_tokens_details.reasoning_tokens` 读取（`model_response.py:162-167`），覆盖 OpenAI ChatCompletions 报账格式。

#### context compaction 中间件的 cache 防御

`ContextCompactionMiddleware._get_current_tokens`（`middleware.py:230-249`）做了一道防御：

- 若 usage 已含 `input_tokens_uncached`（说明 nexau 已经归一化过），直接用 `input_val + output_val`。
- 否则视为原始 vendor usage，主动加上 `cache_creation + cache_read`（`middleware.py:238-241`）。
- 解析失败 fallback 到日志告警 + `None`（`middleware.py:244-249`）。

这避免了"vendor 把 cache 算在 cache_creation/cache_read，nexau 仅看 input_tokens 而误判窗口未满"的 bug。

#### `read_visual_file` 工具：图像与视频的统一入口

`nexau/archs/tool/builtin/file_tools/read_visual_file.py` 是默认携带的视觉文件读取工具：

- `MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024`（`read_visual_file.py:28`）：图像 10MB 上限。
- `IMAGE_EXTENSIONS = {.png, .jpg, .jpeg, .gif, .bmp, .webp, .tiff, .tif, .svg}`（`read_visual_file.py:31-41`）。
- `VIDEO_EXTENSIONS = {.mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .m4v}`（`read_visual_file.py:42`）。
- `VIDEO_FRAME_INTERVAL_SEC = 5` / `VIDEO_MAX_FRAMES = 10`（`read_visual_file.py:45-46`）。
- `_read_video_frames`（`read_visual_file.py:59-167`）：在沙箱内用 `ffmpeg -vf fps=1/<interval>` 抽帧，尝试 `scale=W:-2` 保持比例；ffmpeg 缺失抛"ffmpeg not found in sandbox"；超出 `max_frames` 时按均匀采样裁剪；每帧编码成 base64 data URL，附 `label: "Frame N / ~Ts"`。
- `_resize_image_in_sandbox`（`read_visual_file.py:170-201`）：用 `ffmpeg scale='min(W,iw)':-2`，仅在原图宽度超阈值时下采样；输出固定 JPEG。
- `_read_image_file`（`read_visual_file.py:204-249`）：base64 编码原图，可选 `image_max_size` 触发降采样、降采样后强制 mime=`image/jpeg`。
- `read_visual_file`（`read_visual_file.py:252-441`）：调度 image/video 分支，超大图返回 `FILE_TOO_LARGE` 结构化错误，权限错误返回 `PERMISSION_DENIED`，类型不匹配建议改用 `read_file`。

工具返回的 `{"type": "image", "image_url": "data:...;base64,..."}` 形态正好与 `coerce_tool_result_content` 的入参契约对齐，无需中间转换层。

### 示例

#### 示例 1：tool 返回单张图像

```python
def my_tool() -> dict:
    return {"type": "image", "image_url": "data:image/png;base64,iVBORw0KGgo..."}
```

`coerce_tool_result_content(my_tool())` 走 `from_mapping`（`messages.py:165-178`）→ `_image_block_from_image_url`（`messages.py:132-137`）→ `_parse_base64_data_url` 解析 → `ImageBlock(base64="iVBOR...", mime_type="image/png", detail="auto")`。

`ToolResultBlock(content=[<ImageBlock>])` 然后随消息体：

- 走 OpenAI Chat 适配：`messages_to_legacy_openai_chat` 把它拆成 `role=tool` text + `role=user` multimodal（`legacy.py:316-351`）。
- 走 Anthropic 适配：image 直接成 Anthropic 原生 `image` content block，且最后一条 user message 首块带 `cache_control: ephemeral`。

#### 示例 2：`read_visual_file("photo.png")`

返回结构：

```python
{
    "content": {"type": "image", "image_url": "data:image/png;base64,...", "detail": "auto"},
    "returnDisplay": "Read image file: photo.png",
}
```

`ToolExecutor` 包装后由 `coerce_tool_result_content` 识别 `content` 字段（`messages.py:200-203`），变成 `ImageBlock`，下游与示例 1 一致。

#### 示例 3：OpenAI Responses 多轮对话的 prompt cache

Agent 启动时 `agent.py:149-155` 注入 `prompt_cache_key="abcd-1234"`（一次性 UUID）。每轮调用时 `llm_caller.py:800-802` 把它放进 `extra_body`。OpenAI Responses 服务器看到相同 key 的请求路由到同一缓存集群，前缀稳定的 system prompt + 工具描述部分命中 cache_read，usage 返回 `{"cache_read_input_tokens": 12000, "input_tokens": 200, ...}`。`_normalize_usage` 把 input 折算为 12200，context compaction 据此正确估算窗口压力。

#### 示例 4：Anthropic 缓存 + 视频

调用 Anthropic 流式：`llm_caller.py:555-568` 在 `openai_to_anthropic_message` 之后，把 `cache_control: ephemeral` 打到 last user message 首个 content。如果这条 user message 含视频帧（多个 image block），首个 content 是首帧图像，缓存 anchor 即"首个视觉块之前的全部内容"。这意味着只要 system prompt + 工具列表 + 历史不变，下次相同前缀的请求就能命中 cache_read。

## 权衡取舍

### 考虑过的替代方案

- **统一 vendor-agnostic 图像类型，让适配器只做格式转换**：当前已经接近这个方向（UMP `ImageBlock`），但 OpenAI Chat 的 tool-role 限制（不支持 image parts）必须由 `tool_image_policy` 在转换层处理（`legacy.py:288-351`），无法纯靠 UMP 抽象。
- **总是把图像编码成 base64 data URL**：实现简单但放大 payload 体积；保留 `url` 选项让 user 端可以传 hosted URL，省带宽（`messages.py:47-48`）。
- **在 `ImageBlock` 内嵌 vision token 估算字段**：能让 context compaction 提前判断；但准确的 vision tokenization 因 model 而异（如 GPT-4o tile size 与图像分辨率耦合），目前保留为未解决。
- **prompt cache key 用 agent.name 而非 UUID**：让重启后 cache 仍可命中。但 agent.name 跨进程可能冲突，UUID 更安全；命中靠 OpenAI Responses 自身 prefix matching，并不强依赖 key 复用（`agent.py:149-155`）。
- **Anthropic cache_control 打到 system message**：理论上 cache prefix 会更长。当前实现把 cache anchor 放在最后一条 user 的首块（`llm_caller.py:528-532`），这样多轮对话每轮都能命中"system + 历史前缀"。这是 NexAU 历史选择，TTL 字段因 litellm 不接受被注释掉。

### 缺点

- **`tool_image_policy="inject_user_message"` 注入的额外 user message 可能影响 round counting / 中间件的"用户回合"语义**：`_INJECTED_TOOL_IMAGES_RE` 在反向转换时会把它合并掉（`legacy.py:182-223`），但任何中途观察消息序列的中间件可能误算。
- **图像没有 token 预算保护**：`MAX_FILE_SIZE_BYTES = 10MB`（`read_visual_file.py:28`）只是文件字节上限，不是 vision token 上限；用户传 10MB 高分辨率 PNG 可能在 GPT-4o 下产生数千 vision token，无任何告警。
- **Anthropic `cache_control` 的 TTL 因 litellm 兼容问题硬编码缺省**：`llm_caller.py:530-531` / `llm_caller.py:562-563` 注释掉了 ttl 字段，意味着缓存生命周期由 vendor 默认控制。
- **`prompt_cache_key` 仅对 `openai_responses` 注入**（`agent.py:151`），ChatCompletions 不在此优化范围内。
- **`coerce_tool_result_content` 的字符串分支**：JSON 解码失败的字符串直接返回 `fallback_text` 或原 value（`messages.py:228-233`），如果工具开发者把 image data URL 直接塞进 `str` 而非 dict，会被当成纯文本（无图像）。文档需要明确"工具返回 dict / list 才能识别为图像"。

## 实现计划

### 阶段划分

- [x] Phase 1: UMP `ImageBlock` / `ToolOutputImage` 数据模型 + data URL 解析（`messages.py:45-137`）
- [x] Phase 2: `coerce_tool_result_content` 多模态 coercion（`messages.py:140-243`）
- [x] Phase 3: `ToolResultContentBlock` discriminator union（`messages.py:264-273`）
- [x] Phase 4: OpenAI Chat 适配 + `tool_image_policy` 双向逻辑（`legacy.py:96-223`、`legacy.py:288-351`）
- [x] Phase 5: Anthropic 适配 + system_cache_control_ttl 兼容包装（`anthropic_messages.py:140-168`）
- [x] Phase 6: `read_visual_file` 工具（图像 + 视频抽帧 + ffmpeg 缩放）（`read_visual_file.py:28-441`，含 `read_visual_file.py:31-41` IMAGE_EXTENSIONS、`read_visual_file.py:42` VIDEO_EXTENSIONS、`read_visual_file.py:59-167` 抽帧、`read_visual_file.py:170-201` 缩放）
- [x] Phase 7: Agent 注入 `prompt_cache_key`（`agent.py:149-155`）+ Anthropic `cache_control` 注入（`llm_caller.py:520-568`）+ OpenAI Responses `extra_body` 路由（`llm_caller.py:797-804`）
- [x] Phase 8: usage cache 归一化（`model_response.py:140-175`）+ context compaction 防御（`middleware.py:230-249`）
- [ ] Phase 9（未来）: vision token 预估（按 model 类型 + 图像分辨率）
- [ ] Phase 10（未来）: Anthropic cache TTL 重新启用（litellm 兼容修复后）

### 相关文件

- `nexau/core/messages.py` - UMP 图像 block + `coerce_tool_result_content` + legacy chat 渲染
- `nexau/core/adapters/legacy.py` - OpenAI Chat 双向适配 + `tool_image_policy` + image inject 还原
- `nexau/core/adapters/anthropic_messages.py` - Anthropic 适配 + `system_cache_control_ttl` 兼容
- `nexau/archs/tool/builtin/file_tools/read_visual_file.py` - 视觉文件读取（图像 / 视频抽帧）
- `nexau/archs/main_sub/agent.py` - `Agent.__init__` 注入 `prompt_cache_key`（`agent.py:149-155`）
- `nexau/archs/main_sub/execution/llm_caller.py` - Anthropic `cache_control: ephemeral` 注入（`llm_caller.py:520-568`），OpenAI Responses `prompt_cache_key → extra_body`（`llm_caller.py:797-804`）
- `nexau/archs/main_sub/execution/model_response.py` - `_normalize_usage` cache 折算（`model_response.py:140-175`）
- `nexau/archs/main_sub/execution/middleware/context_compaction/middleware.py` - cache token 防御计算（`middleware.py:230-249`）

## 测试方案

### 单元测试

- `ImageBlock._validate_source` 在缺 url+base64、同时 url+base64 两种非法状态各抛 `ValueError`（`messages.py:54-60`）。
- `_parse_base64_data_url` 在非 `data:` 前缀、缺 `,`、缺 `;base64`、`b'...'`/`'...'`/`"..."` 包裹各分支返回符合预期（`messages.py:98-119`）。
- `_image_block_from_image_url` 对 `data:` URL 命中 base64 路径，对 https URL 命中 url 路径（`messages.py:132-137`）。
- `coerce_tool_result_content`：dict image 形态、`ToolOutputImage` 实例、混合 list、`{"content": [...]}` 包裹、JSON 字符串带 image、纯文本 fallback，6 种输入分别走预期分支并返回正确 block 类型（`messages.py:140-243`）。
- `Message._as_legacy_openai_chat_dict`：`has_images=True/False` 两条路径分别产出 `content_parts` 与 `text_parts` 形态（`messages.py:334-388`）。
- `messages_to_legacy_openai_chat` 在 `inject_user_message` 与 `embed_in_tool_message` 两种 policy 下，含图像 tool result 分别产出对应数量与形状的消息（`legacy.py:288-351`）。
- `messages_from_legacy_openai_chat` 反向：注入的 "Images returned by tool call X:" + image_url parts 能合并回 ToolResultBlock 并丢弃 user message（`legacy.py:182-223`）。
- `anthropic_payload_from_legacy_openai_chat`：`system_cache_control_ttl` 给定时，最后一条 user message 首个 text content 含 `cache_control: ephemeral`（`anthropic_messages.py:156-166`）。
- `_normalize_usage`：含 `cache_creation_input_tokens` / `cache_read_input_tokens` 时返回 `input_tokens` 等于三者之和、并保留 `input_tokens_uncached`；不含时不做折算（`model_response.py:140-175`）。
- `_get_current_tokens` 对已归一化（含 `input_tokens_uncached`）与原始 vendor usage 两种形态各自正确计算（`middleware.py:230-249`）。

### 集成测试

- 配置一个返回 `ToolOutputImage(image_url="data:...")` 的工具，跑完整 LLM 调用链（OpenAI Chat / Anthropic / OpenAI Responses 三选一），验证 vendor payload 含正确 image part、且 Anthropic 路径 last user content 首块带 `cache_control: ephemeral`。
- 用 `read_visual_file` 读真实小图（< 10MB），断言下游 `ImageBlock` 含正确 mime 与 base64。
- OpenAI Responses 多轮对话验证 `extra_body.prompt_cache_key` 一致性，且第二轮 usage 含 `cache_read_input_tokens`。
- 跑包含 `cache_creation_input_tokens` 的人工 usage，验证 `ContextCompactionMiddleware` 不会把缓存命中的部分误算成 0。

### 手动验证

- `read_visual_file` 跑一段视频，确认 ffmpeg 抽帧成功、`Frame N / ~Ts` 标签正确。
- `agent.py:149-155` 启动一个 OpenAI Responses agent，日志中应有"Injected prompt_cache_key=..."条目。
- 真机跑 Anthropic claude-sonnet：在第二轮请求前后比较 usage 报账，应能看到 `cache_read_input_tokens` 增长。

## 未解决的问题

- **Vision token 预估**：当前不计算图像在具体模型下的 token 数（如 GPT-4o tile-based），仅靠 vendor usage 反算。后续若做"提交前预算检查"，需要 model-aware 的 vision tokenizer 表。
- **Anthropic `cache_control.ttl` 重启**：`llm_caller.py:530-531` / `llm_caller.py:562-563` 因 litellm 不接受 ttl 字段而注释掉，需追踪 litellm 升级或绕过 litellm 直接走 anthropic SDK。
- **`prompt_cache_key` 仅 openai_responses**：ChatCompletions 是否需要类似 hint？目前没有明确需求驱动。
- **`detail` 字段缺乏自动选择策略**：`auto` / `low` / `high` 全部由 caller 决定，未来可基于图像分辨率自动建议（小图自动 `low`）。
- **视频抽帧默认 `5s/帧 × 10 帧上限`** 是经验值，缺自适应（短视频可能丢中间）。
- **`MAX_FILE_SIZE_BYTES` 与 vision token 限额脱钩**：10MB 文件 ≠ 安全 token 数；需要新限额维度。
- **`ToolOutputImage` 与 dict 形态并存**：开发者偶有混淆，文档需明示推荐 typed 形式。

## 参考资料

- `rfcs/0006-rfc-catalog-completion-master-plan.md` - T13 子任务来源
- `rfcs/0010-llm-aggregator-event-stream.md` - LLM 聚合器事件流（依赖项 T4）
- `rfcs/0007-tool-system-architecture-and-binding.md` - 工具返回值规范（`coerce_tool_result_content` 的契约）
- `rfcs/0011-middleware-hook-composition.md` - context compaction 中间件接入点
- `nexau/core/messages.py` - UMP 图像 block + 多模态 coercion
- `nexau/core/adapters/legacy.py` - OpenAI Chat 适配
- `nexau/core/adapters/anthropic_messages.py` - Anthropic 适配
- `nexau/archs/tool/builtin/file_tools/read_visual_file.py` - 视觉文件读取工具
- `nexau/archs/main_sub/execution/llm_caller.py` - cache_control / prompt_cache_key 注入
- `nexau/archs/main_sub/execution/model_response.py` - usage cache 归一化
