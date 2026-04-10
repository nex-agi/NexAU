# Tool Output Formatters

Tool output formatters control **what the model actually receives** after a tool finishes.

This matters because a tool usually has at least two audiences:

- the **frontend / runtime**, which may want structured metadata such as `returnDisplay`, file paths, or status fields
- the **LLM**, which usually benefits from a flatter, token-efficient representation

Since RFC-0017, NexAU separates these two concerns explicitly.

## Mental Model

When a tool returns, NexAU now keeps two channels:

- `tool_output`: the normalized **raw runtime output**
- `llm_tool_output`: the **model-facing output** produced by the formatter

The execution flow is:

1. Tool implementation returns a Python value.
2. NexAU normalizes it into `tool_output`.
3. The tool's formatter converts that raw output into `llm_tool_output`.
4. `after_tool` middlewares run and can inspect or modify **both** channels.
5. The final `llm_tool_output` is serialized back into the conversation and sent to the model.

So the formatter is the boundary between:

- **raw tool semantics**
- **prompt-facing tool text / multimodal content**

## What the Formatter Changes

Formatters affect:

- the text or multimodal payload appended to the conversation as the tool result
- what `LongToolOutputMiddleware` usually truncates
- what later `after_tool` middlewares should treat as the model-facing payload

Formatters do **not** replace the raw output channel. Middleware and other runtime components can still inspect the original structured result via `hook_input.tool_output`.

## Default Behavior: XML Formatter

If a tool does not configure a formatter, NexAU uses the built-in `xml` formatter.

Its behavior is intentionally simple:

- **string output** → passed through directly
- **image / multimodal output** → passed through without XML wrapping
- **dict / list / scalar output** → rendered into stable XML text
- display-only fields like `returnDisplay` → stripped from the model-facing output

### Single-field unwrap fast path

After display-only fields are stripped, these values are unwrapped directly:

```json
{"content": "hello"}
{"result": "hello"}
```

Instead of wrapping them in XML, NexAU sends the bare value to the model. This avoids noisy wrappers around the common case where the tool really has only one body field.

### Example

Raw tool output:

```python
{
    "status": "success",
    "result": "Found 3 matches",
    "returnDisplay": "3 matches",
}
```

Model-facing output after the default formatter:

```xml
<tool_result>
  <meta>
    <status>success</status>
  </meta>
  <body field="result"><![CDATA[
Found 3 matches
]]></body>
</tool_result>
```

If the raw output were only:

```python
{"result": "Found 3 matches", "returnDisplay": "3 matches"}
```

then the model would receive just:

```text
Found 3 matches
```

## Why This Matters for Middleware Authors

If you write `after_tool` middleware, you must decide **which channel you want to modify**.

### Raw channel vs model-facing channel

Inside `after_tool`, you receive:

- `hook_input.tool_output`: raw structured output
- `hook_input.llm_tool_output`: already-formatted output for the model

Use these rules:

- modify **`tool_output`** if you want to change the raw runtime result
- modify **`llm_tool_output`** if you want to change what the model sees
- if you only change `tool_output`, the model-facing payload may stay unchanged

This is the biggest behavioral change introduced by formatter support.

### Example: redact only the model-facing output

```python
from nexau.archs.main_sub.execution.hooks import HookResult, Middleware


class RedactSecretsMiddleware(Middleware):
    def after_tool(self, hook_input):
        llm_output = hook_input.llm_tool_output
        if isinstance(llm_output, str):
            redacted = llm_output.replace("sk-live-", "sk-***-")
            return HookResult.with_modifications(llm_tool_output=redacted)
        return HookResult.no_changes()
```

### Example: normalize raw output before later middleware reads it

```python
from nexau.archs.main_sub.execution.hooks import HookResult, Middleware


class AddStatusFieldMiddleware(Middleware):
    def after_tool(self, hook_input):
        if not isinstance(hook_input.tool_output, dict):
            return HookResult.no_changes()

        updated = dict(hook_input.tool_output)
        updated.setdefault("status", "success")
        return HookResult.with_modifications(tool_output=updated)
```

If you want the model to also see that new field, you should also update `llm_tool_output` explicitly.

## Formatter Impact on LongToolOutputMiddleware

`LongToolOutputMiddleware` runs in `after_tool`, which means it sees formatter output.

In practice it prefers:

1. `hook_input.llm_tool_output` if present
2. otherwise `hook_input.tool_output`

So for normal tool execution, truncation usually happens against the **already formatted, model-facing payload**, not the original Python object.

This has two important consequences:

- if your formatter returns a flatter string, truncation acts on that flatter string
- if your formatter returns a dict with `content` or `result`, the middleware truncates that field

When designing a formatter, think about whether the output shape is friendly to later truncation and redaction.

## Built-in Example: `run_shell_command`

`run_shell_command` uses a custom formatter instead of the default XML formatter.

Why?

Because raw shell results often contain runtime metadata like:

- `stdout`
- `stderr`
- `exit_code`
- background task IDs
- file paths for redirected logs

That structure is useful for the runtime, but the model usually wants a flatter result, such as:

- combined command output
- a clear background-task status sentence
- a concise interruption / timeout message

This is exactly what a custom formatter is for: preserve rich raw output, but present a cleaner prompt-facing result.

## How to Configure a Formatter

You can configure a formatter in tool YAML with the `formatter` field.

### Option 1: built-in alias

```yaml
type: tool
name: my_tool
description: Example tool
formatter: xml
input_schema:
  type: object
  properties: {}
```

### Option 2: import path

```yaml
type: tool
name: run_shell_command
description: Execute shell commands
formatter: my_project.tool_formatters:format_shell_output
input_schema:
  type: object
  properties:
    command:
      type: string
  required: [command]
```

You can also pass a callable directly when creating a `Tool` in Python.

## Writing a Custom Formatter

A formatter is a callable that receives `ToolFormatterContext` and returns the model-facing output.

```python
from nexau.archs.tool.formatters import ToolFormatterContext


def format_search_output(context: ToolFormatterContext) -> object:
    raw = context.tool_output

    if not isinstance(raw, dict):
        return raw

    hits = raw.get("hits")
    if isinstance(hits, list):
        lines = [f"Found {len(hits)} hits:"]
        for item in hits[:10]:
            lines.append(f"- {item}")
        return "\n".join(lines)

    return raw
```

### Available context

`ToolFormatterContext` includes:

- `tool_name`
- `tool_input`
- `tool_output`
- `tool_call_id`
- `is_error`

`is_error` is especially useful when you want different formatting for success and failure paths.

## Formatter Design Guidelines

Prefer formatters when you need to:

- flatten noisy structured results for the model
- preserve runtime metadata without dumping all of it into the prompt
- produce a tool-specific text format that is clearer than generic XML
- keep truncation and downstream middleware operating on a cleaner payload

Avoid custom formatters when the default XML rendering is already good enough.

As a rule of thumb:

- **use default `xml`** for ordinary structured tools
- **use a custom formatter** for tools with highly specialized runtime output shapes

## Failure Behavior

Formatter failures are defensive:

1. NexAU first tries the configured formatter.
2. If it fails, NexAU falls back to the built-in `xml` formatter.
3. If that also fails, NexAU falls back to the raw tool output.

This keeps tool execution robust even when a custom formatter is buggy.

## Recommended Pattern for Custom Tools

For most tools, this is a good pattern:

1. return a rich raw dict from the tool implementation
2. keep `returnDisplay` for frontend-only summaries
3. use a formatter to shape the LLM-facing result
4. use `after_tool` middleware only for cross-cutting concerns like truncation, redaction, logging, or auditing

That separation keeps tool code, formatter logic, and middleware behavior easier to reason about.
