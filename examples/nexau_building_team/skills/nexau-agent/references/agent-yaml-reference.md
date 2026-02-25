# Agent YAML Configuration Reference

Complete field reference for NexAU agent YAML configuration files.

## Minimal Config

```yaml
type: agent
name: my_agent
system_prompt: ./systemprompt.md
llm_config:
  model: ${env.LLM_MODEL}
  base_url: ${env.LLM_BASE_URL}
  api_key: ${env.LLM_API_KEY}
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
```

## Full Config

```yaml
type: agent                          # Required. Must be "agent"
name: agent_name                     # Required. Unique identifier
description: What the agent does     # Optional. Human-readable description

# Context and iteration limits
max_context_tokens: 200000           # Max context window size (default: 200000)
max_iterations: 80                   # Max agent loop iterations (default: 50)

# System prompt
system_prompt: ./systemprompt.md     # Path to system prompt file (relative to YAML)
system_prompt_type: jinja            # "jinja" for template variables, "string" for plain text

# Tool calling format
tool_call_mode: openai               # "openai" | "xml" | "anthropic"

# LLM configuration
llm_config:
  model: ${env.LLM_MODEL}           # Model name/ID
  base_url: ${env.LLM_BASE_URL}     # API base URL
  api_key: ${env.LLM_API_KEY}       # API key
  max_tokens: 16000                  # Max output tokens per response
  temperature: 0.7                   # Sampling temperature (0.0-1.0)
  stream: False                      # Enable streaming (True/False)
  api_type: openai_chat_completion   # "openai_chat_completion" | "openai_responses" | "anthropic_chat_completion"

# Sandbox configuration
sandbox_config:
  type: local                        # Sandbox type
  _work_dir: ${env.SANDBOX_WORK_DIR} # Working directory for file/shell tools

# Tools
tools:
  - name: tool_name                  # Tool identifier
    yaml_path: ./tools/tool.tool.yaml  # Path to tool YAML (relative to agent YAML)
    binding: module.path:function     # Python import path to binding function

# Skills (folder-based)
skills:
  - ./skills/skill-name              # Path to skill folder (relative to agent YAML)

# Stop tools — tools that halt the agent loop when called
stop_tools: [complete_task]          # List of tool names

# Middlewares
middlewares: []                      # List of middleware instances

# Tracers
tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
  # Or with params:
  - import: nexau.archs.tracer.adapters.langfuse:LangfuseTracer
    params:
      public_key: ${env.LANGFUSE_PUBLIC_KEY}
      secret_key: ${env.LANGFUSE_SECRET_KEY}
      host: https://us.cloud.langfuse.com
```

## Field Details

### `system_prompt_type`

| Value | Use When | Template Variables |
|-------|----------|-------------------|
| `string` | Plain text prompt, no variables | None |
| `jinja` | Need `{{ date }}`, `{{ username }}`, `{{ working_directory }}` | Passed via `context` dict in `agent.run()` |

### `tool_call_mode`

| Value | Use With |
|-------|----------|
| `openai` | OpenAI-compatible APIs (GPT, Claude via proxy, local models) |
| `xml` | XML-based tool calling |
| `anthropic` | Anthropic native API |

### `api_type`

| Value | Use With |
|-------|----------|
| `openai_chat_completion` | Standard OpenAI Chat Completions API (default) |
| `openai_responses` | OpenAI Responses API (for gpt-5-codex) |
| `anthropic_chat_completion` | Anthropic Messages API |

### `temperature` Guidelines

| Range | Use For |
|-------|---------|
| 0.0-0.3 | Precise tasks (code generation, structured output) |
| 0.3-0.5 | Balanced tasks (analysis, implementation) |
| 0.5-0.7 | Creative tasks (writing, brainstorming) |
| 0.7-1.0 | Highly creative tasks |

### `max_iterations` Guidelines

| Range | Use For |
|-------|---------|
| 20-50 | Simple single-purpose agents |
| 50-100 | Complex agents with multiple tools |
| 100-200 | Team leaders coordinating sub-agents |
| 200-300 | Long-running interactive agents |

### Environment Variable Substitution

Use `${env.VAR_NAME}` syntax in YAML values. The variable is resolved at config load time from the process environment.

```yaml
llm_config:
  model: ${env.LLM_MODEL}           # Reads os.environ["LLM_MODEL"]
  api_key: ${env.LLM_API_KEY}       # Reads os.environ["LLM_API_KEY"]
```
