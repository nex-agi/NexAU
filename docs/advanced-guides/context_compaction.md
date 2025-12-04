## Context Compaction

Context Compaction is a middleware that automatically manages conversation history when token limits are approached. Once configured, it monitors token usage after each model call and performs compaction automatically when trigger conditions are met.

**How it works:**
1. **Automatic Monitoring**: Tracks token usage after every model call
2. **Smart Triggering**: Automatically compacts when token usage exceeds the threshold (default: 75%)
3. **Safe Execution**: Only compacts when appropriate (e.g., skips if the last assistant message has no tool calls)

### Compaction Strategies

#### 1. Tool Result Compaction (Recommended)

Preserves all conversation structure while compacting old tool results. Fast and cost-free.

**What it preserves:**
- System prompt
- All user messages
- All assistant messages
- Recent tool results (after the last assistant message)

**What it compacts:**
- Old tool result content is replaced with: `"Tool call result has been compacted"`

**Configuration:**

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.context_compaction:ContextCompactionMiddleware
    params:
      max_context_tokens: 100000
      auto_compact: true
      threshold: 0.75
      compaction_strategy: "tool_result_compaction"
```

**Python Configuration:**
```python
ContextCompactionMiddleware(
    max_context_tokens=100000,
    auto_compact=True,
    threshold=0.75,
    compaction_strategy="tool_result_compaction",
)
```

#### 2. Sliding Window Compaction

Summarizes old conversation rounds using an LLM while keeping recent iterations unchanged.

**How it works:**
- Groups messages into iterations (user → assistant → tools)
- Keeps the most recent N iterations in full
- Summarizes older iterations using a separate LLM call
- Injects the summary into the conversation context

**Configuration:**

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.context_compaction:ContextCompactionMiddleware
    params:
      max_context_tokens: 100000
      auto_compact: true
      threshold: 0.75
      compaction_strategy: "sliding_window"
      window_size: 2  # Keep last 2 iterations

      # Required: LLM for summarization
      summary_model: ${env.SUMMARY_MODEL}
      summary_base_url: ${env.SUMMARY_BASE_URL}
      summary_api_key: ${env.SUMMARY_API_KEY}

      # Optional: Custom summary prompt
      compact_prompt_path: "./prompts/custom_compact_prompt.md"
```

**Python Configuration:**

```python
ContextCompactionMiddleware(
    max_context_tokens=100000,
    auto_compact=True,
    threshold=0.75,
    compaction_strategy="sliding_window",
    window_size=2,
    summary_model="nex-n1",
    summary_base_url="base_url",
    summary_api_key="sk-...",
)
```

### Trigger Strategy

The middleware automatically monitors token usage after each model call. When usage exceeds the configured threshold percentage, compaction is triggered automatically.

**YAML Configuration:**
```yaml
threshold: 0.75  # Trigger at 75% of max_context_tokens
```

**Python Configuration:**
```python
ContextCompactionMiddleware(
    max_context_tokens=100000,
    threshold=0.75,
)
```

**Safety checks:**
- Compaction is automatically skipped if the last assistant message has no tool calls