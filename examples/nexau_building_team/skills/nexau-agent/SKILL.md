---
name: nexau-agent
description: Guide for building NexAU agents from scratch. This skill should be used when implementing a new NexAU agent — including YAML configuration, system prompt, tool definitions, tool bindings, entry point scripts.
---

# NexAU Agent Builder

This skill provides the procedural knowledge and reference material needed to implement NexAU agents correctly and efficiently.

## When to Use

- Implementing a new standalone agent (YAML config + system prompt + tools + entry point)
- Adding custom tools to an existing agent
- Writing or refining system prompts for NexAU agents
- Setting up skills for an agent
- Looking up NexAU framework concepts (agents, tools, LLMs, transports, sessions, etc.)

> **NexAU Documentation**: The `docs/` directory contains the full user-facing documentation for NexAU. Start with `docs/index.md` for an overview, `docs/getting-started.md` for setup, `docs/core-concepts/` for fundamentals (agents, tools, LLMs), and `docs/advanced-guides/` for topics like skills, hooks, MCP, sandbox, transports, and session management.

## Agent Implementation Workflow

### 1. Understand the Requirements

Before writing any code, read the RFC or requirements document thoroughly. Identify:

- Agent purpose and capabilities
- Required tools (builtin vs custom)
- LLM configuration needs (model, temperature, token limits)
- Any skills the agent should have

### 2. Create the Directory Structure

A standalone agent follows this layout:

```
agent_name/
├── agent_name.yaml          # Agent configuration
├── systemprompt.md           # System prompt (Jinja2 or plain string)
├── start.py                  # Entry point script
├── tools/                    # Tool definitions
│   ├── read_file.tool.yaml
│   ├── write_file.tool.yaml
│   └── custom_tool.tool.yaml
├── skills/                   # Optional skills
│   └── skill-name/
│       └── SKILL.md
└── custom_tools/             # Custom tool implementations (if any)
    └── custom_tool.py
```


### 3. Write the Agent YAML Config

Refer to `references/agent-yaml-reference.md` for the complete field reference. Key points:

- Use `${env.VAR_NAME}` for environment variable substitution (LLM keys, sandbox paths)
- Set `system_prompt_type: jinja` when using template variables like `{{ date }}`
- Set `system_prompt_type: string` for plain text prompts
- Choose `tool_call_mode: openai` for OpenAI-compatible APIs
- Reference tool YAMLs with relative paths: `./tools/tool.tool.yaml`
- Reference system prompt with relative path: `./systemprompt.md`

Template available at: `assets/templates/agent.yaml`

### 4. Define Tools

For each tool the agent needs:

1. Check if a builtin tool exists — see `references/builtin-tools-reference.md`
2. If builtin, copy the tool YAML from an existing example and set the binding path
3. If custom, create both a `.tool.yaml` definition and a Python implementation

Refer to `references/tool-yaml-reference.md` for the tool YAML schema.

Template available at: `assets/templates/tool.tool.yaml`

### 5. Write the System Prompt

Follow the conventions in `references/system-prompt-guide.md`:

- Start with a clear role description
- Define a numbered step-by-step workflow
- Include guidelines and constraints
- Add template variables at the bottom: `{{ date }}`, `{{ username }}`, `{{ working_directory }}`

Template available at: `assets/templates/systemprompt.md`

### 6. Create the Entry Point

For standalone agents, use `Agent.from_yaml()`.

Template available at: `assets/templates/start.py` (standalone)

### 7. Verify the Implementation

- **Validate agent config**:
  ```bash
  python {THIS_SKILL_FOLDER}/scripts/validate_agent.py path/to/agent.yaml
  ```
  - Add `--recursive` to also validate sub-agent configs
  - Add `--json` for machine-readable output
- Check Python syntax: `python -c "import ast; ast.parse(open('start.py').read())"`
- Check imports: `python -c "from nexau import Agent"`
- Run the agent with a simple test query

## Common Patterns

### Environment Variable Substitution in YAML

```yaml
llm_config:
  model: ${env.LLM_MODEL}
  base_url: ${env.LLM_BASE_URL}
  api_key: ${env.LLM_API_KEY}
```

### Adding Builtin Tools

```yaml
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
```

### Adding Skills

```yaml
skills:
  - ./skills/skill-name
```

### Stop Tools

Use `stop_tools` to define tools that halt the agent loop when called:

```yaml
stop_tools: [complete_task]    # For task-based agents
stop_tools: [ask_user]         # For interactive agents
```

### Tracers

```yaml
tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

## Builtin Tools Catalog

All builtin tool YAML configs are located under `builtin_tools/tools/` in this skill folder. When adding a builtin tool to an agent, copy the YAML to your agent's `tools/` directory and set the binding path accordingly.

### File Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| read_file | `nexau.archs.tool.builtin.file_tools:read_file` | `builtin_tools/tools/read_file.tool.yaml` |
| write_file | `nexau.archs.tool.builtin.file_tools:write_file` | `builtin_tools/tools/write_file.tool.yaml` |
| replace | `nexau.archs.tool.builtin.file_tools:replace` | `builtin_tools/tools/replace.tool.yaml` |
| search_file_content | `nexau.archs.tool.builtin.file_tools:search_file_content` | `builtin_tools/tools/search_file_content.tool.yaml` |
| glob | `nexau.archs.tool.builtin.file_tools:glob` | `builtin_tools/tools/Glob.tool.yaml` |
| list_directory | `nexau.archs.tool.builtin.file_tools:list_directory` | `builtin_tools/tools/list_directory.tool.yaml` |


### Web Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| google_web_search | `nexau.archs.tool.builtin.web_tools:google_web_search` | `builtin_tools/tools/WebSearch.tool.yaml` |
| web_fetch | `nexau.archs.tool.builtin.web_tools:web_fetch` | `builtin_tools/tools/WebFetch.tool.yaml` |

### Shell Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| run_shell_command (sync) | `nexau.archs.tool.builtin.shell_tools:run_shell_command` | `builtin_tools/tools/run_shell_command_sync.tool.yaml` |

### Session Tools

| Tool | Binding | Config YAML |
|------|---------|-------------|
| write_todos | `nexau.archs.tool.builtin.session_tools:write_todos` | `builtin_tools/tools/write_todos.tool.yaml` |
| complete_task | `nexau.archs.tool.builtin.session_tools:complete_task` | `builtin_tools/tools/complete_task.tool.yaml` |
| ask_user | `nexau.archs.tool.builtin.session_tools:ask_user` | `builtin_tools/tools/ask_user.tool.yaml` |


> For detailed parameter descriptions of each tool, see `references/builtin-tools-reference.md`.

## Bundled Resources

| Path | Purpose |
|------|---------|
| `references/agent-yaml-reference.md` | Complete agent YAML config field reference |
| `references/tool-yaml-reference.md` | Tool YAML definition schema and examples |
| `references/builtin-tools-reference.md` | Catalog of all builtin tools with binding paths |
| `references/system-prompt-guide.md` | System prompt writing conventions and patterns |
| `assets/templates/agent.yaml` | Starter agent YAML config template |
| `assets/templates/tool.tool.yaml` | Starter tool YAML definition template |
| `assets/templates/systemprompt.md` | Starter system prompt template |
| `assets/templates/start.py` | Standalone agent entry point template |
| `scripts/validate_agent.py` | Agent YAML validator (schema + file reference checks) |
| `docs/index.md` | NexAU documentation index — entry point for all user-facing docs |
| `docs/getting-started.md` | Installation, setup, and first agent walkthrough |
| `docs/core-concepts/agents.md` | Agent architecture, lifecycle, and configuration |
| `docs/core-concepts/tools.md` | Tool system — definition, binding, execution |
| `docs/core-concepts/llms.md` | LLM provider configuration and aggregator usage |
| `docs/advanced-guides/skills.md` | Skill system — folder-based and tool-based skills |
| `docs/advanced-guides/hooks.md` | Hook system for lifecycle event handling |
| `docs/advanced-guides/mcp.md` | Model Context Protocol integration |
| `docs/advanced-guides/sandbox.md` | Sandbox configuration and isolation |
| `docs/advanced-guides/session-management.md` | Session persistence and history management |
| `docs/advanced-guides/streaming-events.md` | Streaming event types and handling |
| `docs/advanced-guides/transports.md` | Transport layer — HTTP, stdio, WebSocket, gRPC |
| `docs/advanced-guides/templating.md` | Jinja2 templating in system prompts and configs |
| `docs/advanced-guides/context_compaction.md` | Context compaction middleware for long conversations |
| `docs/advanced-guides/global-storage.md` | Global storage for cross-turn agent state |
| `docs/advanced-guides/tracer.md` | Tracing and observability setup |
| `docs/advanced-guides/image.md` | Image input handling |

## Setting Up Skills for an Agent

Skills are reusable capabilities that extend an agent's knowledge and workflows. NexAU supports two types: **folder-based skills** and **tool-based skills**. Both are automatically registered and discoverable via the `LoadSkill` tool at runtime.

### Folder-Based Skills

A folder-based skill is a self-contained directory with a `SKILL.md` file. To add one to an agent:

1. Create the skill directory under the agent's `skills/` folder:

```
skills/
└── my-skill/
    ├── SKILL.md              # Required — metadata + instructions
    ├── scripts/              # Optional — executable code
    ├── references/           # Optional — context docs loaded on demand
    └── assets/               # Optional — templates, images, output files
```

2. Reference the skill in the agent YAML:

```yaml
skills:
  - ./skills/my-skill
```

The agent will see a brief description of the skill in its system prompt and can call `LoadSkill` to read the full `SKILL.md` content when needed.

### Tool-Based Skills

A tool-based skill is a regular tool marked with `as_skill: true` in its YAML definition. This is useful for simple, well-defined capabilities that do not need extensive documentation.

```yaml
# tools/generate_code.tool.yaml
type: tool
name: generate_code
description: Generates code based on specifications
as_skill: true
skill_description: Code generation skill for multiple programming languages
input_schema:
  type: object
  properties:
    language:
      type: string
    specification:
      type: string
  required: [language, specification]
```

The `skill_description` field is required when `as_skill: true`. It provides the brief text shown in the skill registry.

### Combining Both Types

Both skill types can coexist in the same agent:

```yaml
# Agent YAML
skills:
  - ./skills/data-analysis       # Folder-based skill
  - ./skills/report-writing      # Folder-based skill

tools:
  - name: web_search
    yaml_path: ./tools/web_search.tool.yaml   # Tool-based skill (as_skill: true)
    binding: nexau.archs.tool.builtin.web_tools:google_web_search
```

### How Skills Work at Runtime

1. All skills are registered in the agent's `skill_registry` during initialization
2. A `LoadSkill` tool is automatically added to the agent
3. The agent sees brief descriptions of all available skills in its system prompt
4. When the agent needs detailed information, it calls `LoadSkill` with the skill name
5. The full `SKILL.md` content (and skill folder path) is returned to the agent

## Writing High-Quality Skills

A well-crafted skill transforms a general-purpose agent into a specialized one. Follow these principles to maximize effectiveness.

### SKILL.md Structure

Every `SKILL.md` must start with YAML frontmatter containing `name` and `description`:

```markdown
---
name: my-skill
description: Brief description of what this skill does. This skill should be used when [trigger conditions].
---

# Skill Title

[Purpose — what this skill provides, in 1-2 sentences]

## When to Use

[Concrete trigger conditions — when should the agent load this skill]

## Workflow / Instructions

[Step-by-step procedural knowledge]

## Bundled Resources

[Table of scripts, references, and assets with their purposes]
```

### Metadata Quality

The `name` and `description` in the frontmatter determine when the agent will load the skill. Write them carefully:

- **name**: Use lowercase kebab-case (`my-skill`, not `MySkill` or `my_skill`)
- **description**: Be specific about what the skill does and when to use it. Use third-person form (e.g., "This skill should be used when..." instead of "Use this skill when...")

### Progressive Disclosure

Skills use a three-level loading system to manage context efficiently:

1. **Metadata** (name + description) — Always in the agent's system prompt (~100 words)
2. **SKILL.md body** — Loaded when the agent calls `LoadSkill` (target: <5k words)
3. **Bundled resources** — Loaded on demand as the agent reads specific files (unlimited)

Keep `SKILL.md` lean by moving detailed reference material to `references/` files. Only include essential procedural instructions and workflow guidance in the main body.

### Writing Style

- Use **imperative/infinitive form** (verb-first instructions), not second person
- Write "To accomplish X, do Y" rather than "You should do X"
- Keep instructions actionable and specific — vague steps produce vague behavior
- Reference bundled resources explicitly so the agent knows they exist and when to use them

### Bundled Resources Guidelines

| Type | Directory | When to Include | Example |
|------|-----------|-----------------|---------|
| Scripts | `scripts/` | Same code is rewritten repeatedly or deterministic reliability is needed | `scripts/validate_config.py` |
| References | `references/` | Documentation the agent should consult while working | `references/api_schema.md` |
| Assets | `assets/` | Files used in the output (templates, images, boilerplate) | `assets/templates/starter.yaml` |

**Avoid duplication**: Information should live in either `SKILL.md` or reference files, not both. For large reference files (>10k words), include grep search patterns in `SKILL.md` to help the agent locate relevant sections.

### Choosing Between Skill Types

| Criteria | Folder-Based | Tool-Based |
|----------|-------------|------------|
| Needs extensive documentation | ✓ | |
| Includes multiple files (scripts, templates) | ✓ | |
| Single, well-defined capability | | ✓ |
| Tool description is sufficient documentation | | ✓ |
| Will be shared across projects | ✓ | |

### Common Pitfalls

- Putting too much detail in `SKILL.md` instead of `references/` — bloats context on every load
- Vague frontmatter description — the agent cannot determine when to use the skill
- Missing resource references — the agent does not know bundled files exist unless told
- Absolute paths in documentation — use relative paths within the skill folder
- Duplicating information between `SKILL.md` and reference files
