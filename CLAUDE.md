# Claude Code Development Guide

This is the entry point for understanding NexAU's implementation. For user documentation on how to use NexAU, see [docs/](./docs/).

## Quick Reference

### Module Implementation Guides

For detailed implementation guides for each module, see:

- **[Agent System](./nexau/archs/main_sub/CLAUDE.md)** - Agent container, executor, middleware
- **[Tool System](./nexau/archs/tool/CLAUDE.md)** - Tool definition, execution, binding
- **[Session Management](./nexau/archs/session/CLAUDE.md)** - ORM, repositories, history persistence
- **[Transport System](./nexau/archs/transports/CLAUDE.md)** - HTTP, stdio, WebSocket, gRPC
- **[LLM Aggregators](./nexau/archs/llm/llm_aggregators/CLAUDE.md)** - Event streaming, providers

### User Documentation

- **[Documentation Index](./docs/index.md)** - Main user-facing documentation
- **[Getting Started](./docs/getting-started.md)** - Installation, setup, first agent
- **[Core Concepts](./docs/core-concepts/)** - Agents, Tools, LLMs
- **[Advanced Guides](./docs/advanced-guides/)** - Skills, Hooks, Tracing, MCP

## Development Commands

This project uses **uv** for package management and **Make** for workflow orchestration. Python 3.12+ is required.

```bash
# Install dependencies and pre-commit hooks (first time setup)
make install

# Format code (auto-fix)
make format

# Run linter (auto-fix mode)
make lint

# Run all type checking (mypy + pyright)
make typecheck

# Run tests with coverage
make test

# Full CI pipeline locally
make ci
```

### Running Individual Tests

```bash
# Run a single test file
uv run pytest path/to/test_file.py

# Run a specific test function
uv run pytest path/to/test_file.py::test_function_name

# Run with verbose output
uv run pytest path/to/test_file.py -v

# Run tests without coverage (faster)
uv run pytest path/to/test_file.py --no-cov
```

### Type Checking

```bash
# Run mypy only
make mypy

# Run pyright only
uv run pyright

# Generate type coverage reports
make mypy-coverage
# Reports saved to: mypy_reports/type_html/index.html
```

## Type Safety Guidelines

This section documents the **mandatory type safety coding standards** that apply across the codebase, especially in `llm_aggregators` module. These rules establish a strict type safety culture.

### Core Principles

1. **Type Checkers Are Your Friends**
   - Type checker errors = actual code problems
   - Suppressing errors only delays runtime failures
   - Refactor instead of ignore

2. **Zero Tolerance Policy**
   - ❌ No `# type: ignore` comments
   - ❌ No `Any` type usage
   - ❌ No dynamic attribute access (`getattr`/`hasattr`)

3. **Documentation Through Types**
   - Type annotations are the best documentation
   - Clear types make code understandable
   - Type checkers catch common errors

### Forbidden Patterns

#### 1. Never Use `# type: ignore`

```python
# ❌ Disables type checking entirely
self._on_event(ThinkingTextMessageStartEvent())  # type: ignore

# ✅ Use TYPE_CHECKING for import cycles
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from some_module import EventType

event: EventType = get_typed_event()
```

#### 2. Never Use `Any`

```python
# ❌ Loses all type safety
def process_data(data: Any) -> Any:
    return data.process()

# ✅ Use TypeVar for generics
from typing import TypeVar
T = TypeVar('T')
def process_data(data: T) -> T:
    return data

# ✅ Use Union for limited types
from typing import Union
def process_event(event: Union[TextEvent, AudioEvent]) -> None:
    match event:
        case TextEvent(): handle_text(event)
        case AudioEvent(): handle_audio(event)
```

#### 3. No Dynamic Attribute Access

```python
# ❌ getattr and hasattr are not type-safe
value = getattr(obj, "name", default)
if hasattr(item, "name") and item.name:
    process(item.name)

# ✅ Use type guards + match statement
from typing import TypeGuard

def is_event_with_name(item: ResponseStreamEvent) -> TypeGuard[EventWithName]:
    return hasattr(item, "name") and isinstance(item.name, str)

if is_event_with_name(item) and item.name:
    process(item.name)

# ✅ Use pattern matching
match item:
    case EventWithName(name=name) if name:
        process(name)
```

#### 4. No String-Based Type Narrowing

```python
# ❌ Bypassing type system
item_type_str = str(item.type)
if item_type_str == "response.reasoning_summary_text.delta":
    # ...

# ✅ Direct literal comparison (type checker can infer)
if item.type == "response.reasoning_summary_text.delta":
    # ...

# ✅ Use pattern matching (most elegant and safe)
match item.type:
    case "response.reasoning_summary_text.delta":
        # ...
```

### Type Safety Checklist

Before committing code, verify:

- [ ] No `# type: ignore` comments
- [ ] No `# type: ignore[...]` specific error suppression
- [ ] No `Any` type usage
- [ ] All functions have complete type signatures
- [ ] No `getattr`/`setattr` for attribute access
- [ ] All properties have explicit type declarations
- [ ] IDE type checker shows no errors (warnings OK)
- [ ] Using type guards instead of dynamic checks
- [ ] Using pattern matching for type handling
- [ ] Run `make typecheck` before committing

## Debugging Tips

### Enable Logging

The framework uses Python logging throughout:

```python
import logging
logging.basicConfig(level=logging.INFO)
# Or DEBUG for more verbosity
```

### Common Issues

- **Import errors**: Ensure `uv sync` ran successfully
- **Type errors**: Run `make typecheck` to find issues
- **Tool call failures**: Check tool schema matches implementation signature
- **Token limit exceeded**: Configure context compaction middleware
- **Sub-agent not found**: Verify agent name matches config key
- **Session storage issues**: Check database permissions and path
- **Transport connection errors**: Verify port availability and network configuration
