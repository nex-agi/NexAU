# LLM Aggregators Implementation Guide

This module contains event-driven streaming architecture for unified LLM provider integration.

## Architecture Overview

Modern event-driven architecture for unified streaming:

```
Raw API Chunks (Provider-Specific)
    ↓
Aggregator (Provider-Specific Parser)
    ↓
Unified Event Objects (Provider-Agnostic)
    ↓
Event Handlers (UI/Framework)
```

**Why this pattern?**

1. Same event interface for all providers
2. Handles partial JSON, incremental tool call parsing
3. UI-agnostic (works with any frontend)

## Key Components

### Aggregator ABC (`events.py`)

Abstract base class for accumulating input items into a built output.

**Generic Pattern**:

```python
class Aggregator[AggregatorInputT, AggregatorOutputT](ABC):
    """Base class for aggregators that accumulate input items."""

    @abstractmethod
    def aggregate(self, item: AggregatorInputT) -> None:
        """Aggregate a single input item.

        Raises:
            RuntimeError: If called after build() or on completed aggregator
        """
        raise NotImplementedError

    @abstractmethod
    def build(self) -> AggregatorOutputT:
        """Build the final result from aggregated items.

        Returns:
            The complete aggregated output

        Raises:
            RuntimeError: If called before any items were aggregated
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Reset the aggregator state for reuse.

        This allows the aggregator to be reused for processing a new
        sequence without creating a new instance.
        """
        raise NotImplementedError
```

**Usage Pattern**:

```python
class MyAggregator(Aggregator[Chunk, Result]):
    def __init__(self):
        self._chunks: list[Chunk] = []

    def aggregate(self, item: Chunk) -> None:
        self._chunks.append(item)

    def build(self) -> Result:
        return Result(chunks=self._chunks)

    def clear(self) -> None:
        self._chunks = []

# Usage
aggregator = MyAggregator()
for chunk in stream:
    aggregator.aggregate(chunk)

result = aggregator.build()
aggregator.clear()  # Reuse
```

### Event Types (`events.py`)

Unified event types for all LLM providers.

#### Text Message Events

```python
class TextMessageStartEvent(BaseEvent):
    """Text message start event with run_id for multi-agent support."""
    run_id: str  # ID of the agent run that produced this event

class TextMessageContentEvent(BaseEvent):
    """Text message content event with delta."""
    delta: str  # Incremental text content

class TextMessageEndEvent(BaseEvent):
    """Text message end event."""
    message_id: str  # Links to StartEvent
```

#### Thinking Message Events

```python
class ThinkingTextMessageStartEvent(BaseEvent):
    """Thinking message start event."""
    parent_message_id: str  # ID of the parent message (for correlation to parent)
    thinking_message_id: str  # Unique ID for the thinking message
    run_id: str  # ID of the agent run that produced this event

class ThinkingTextMessageContentEvent(BaseEvent):
    """Thinking message content event."""
    thinking_message_id: str  # Unique identifier linking to the start event
    delta: str  # Incremental thinking content

class ThinkingTextMessageEndEvent(BaseEvent):
    """Thinking message end event."""
    thinking_message_id: str  # Unique identifier linking to the start event
```

#### Tool Call Events

```python
class ToolCallStartEvent(BaseEvent):
    """Tool call start event."""
    tool_call_id: str  # Unique identifier for this tool call
    parent_message_id: str  # Links to message
    run_id: str  # ID of the agent run
    name: str  # Tool name

class ToolCallArgsEvent(BaseEvent):
    """Tool call args event with delta."""
    tool_call_id: str  # Links to StartEvent
    delta: dict[str, Any]  # Incremental arguments

class ToolCallEndEvent(BaseEvent):
    """Tool call end event."""
    tool_call_id: str  # Links to StartEvent

class ToolCallResultEvent(BaseEvent):
    """Event for sending tool execution result back to LLM."""
    tool_call_id: str  # Unique identifier for this tool call
    content: str  # Tool execution result (JSON string or plain text)
    role: Literal["tool"] | None = "tool"
```

#### Image Events

```python
class ImageMessageStartEvent(BaseEvent):
    """Event indicating the start of an image message."""
    message_id: str  # Unique identifier for the message
    mime_type: str = "image/jpeg"  # MIME type of the image
    run_id: str  # ID of the agent run that produced this event

class ImageMessageContentEvent(BaseEvent):
    """Event containing base64-encoded image data."""
    message_id: str  # Links to StartEvent
    delta: str  # Base64-encoded image data

class ImageMessageEndEvent(BaseEvent):
    """Event indicating the end of an image message."""
    message_id: str  # Links to StartEvent
```

#### Run Lifecycle Events

```python
class RunStartedEvent(BaseEvent):
    """Run started event with full tracing IDs."""
    agent_id: str  # ID of the agent
    run_id: str  # ID of the run
    root_run_id: str  # ID of the root run
    timestamp: int  # Unix timestamp in milliseconds

class RunFinishedEvent(BaseEvent):
    """Run finished event."""
    thread_id: str  # Session ID
    run_id: str  # ID of the run
    result: str  # Agent response
    timestamp: int  # Unix timestamp in milliseconds

class RunErrorEvent(BaseEvent):
    """Run error event."""
    run_id: str  # ID of the agent run
    message: str  # Error message
    timestamp: int  # Unix timestamp in milliseconds

class TransportErrorEvent(BaseEvent):
    """Event indicating a transport-level error."""
    message: str  # Error description
    timestamp: int | None = None  # Unix timestamp in milliseconds
```

### OpenAI Chat Completion Aggregator (`openai_chat_completion/openai_chat_completion_aggregator.py`)

Parser for OpenAI Chat Completions API streaming responses.

**Initialization**:

```python
from nexau.archs.llm.llm_aggregators import (
    OpenAIChatCompletionAggregator,
    Event,
)

def handle_event(event: Event):
    """Handle unified streaming events"""
    if event.type == "TEXT_MESSAGE_CONTENT":
        print(event.delta, end="")

aggregator = OpenAIChatCompletionAggregator(
    on_event=handle_event,
    run_id="run_123",
)
```

**Supported Chunk Types**:

- `ChatCompletionChunk` from `openai.types.chat`
- `response.created` / `response.done` from `openai.types.responses`

**Event Mapping**:

| Raw Chunk Type | Emitted Event |
| -------------- | -------------- |
| `delta.content` | `TextMessageContentEvent` |
| `delta.tool_calls[]` | `ToolCallStartEvent`, `ToolCallArgsEvent` |
| `finish_reason` | `TextMessageEndEvent` |

### OpenAI Responses Aggregator (`openai_responses/openai_responses_aggregator.py`)

Parser for OpenAI Responses API (GPT-5 Codex style).

**Initialization**:

```python
from nexau.archs.llm.llm_aggregators import (
    OpenAIResponsesAggregator,
    Event,
)

def handle_event(event: Event):
    """Handle unified streaming events"""
    if event.type == "TEXT_MESSAGE_CONTENT":
        print(event.delta, end="")

aggregator = OpenAIResponsesAggregator(
    on_event=handle_event,
    run_id="run_123",
)
```

**Supported Event Types**:

- `ResponseStreamEvent` from `openai.types.responses`
- Supports reasoning tokens, thinking content, images, and tool calls

**Event Mapping**:

| Raw Event Type | Emitted Event |
| -------------- | -------------- |
| `response.created` | `RunStartedEvent` |
| `response.output.message.content.delta` | `TextMessageContentEvent` |
| `response.output.message.reasoning.delta` | `ThinkingTextMessageContentEvent` |
| `response.output.tool_call.delta` | `ToolCallArgsEvent` |
| `response.done` | `RunFinishedEvent` |

## Key Patterns

### Aggregator Implementation Pattern

Implement a custom aggregator for a new LLM provider:

```python
from nexau.archs.llm.llm_aggregators.events import Aggregator, Event

class CustomLLMAggregator(Aggregator[CustomChunk, CustomResult]):
    """Aggregator for Custom LLM provider."""

    def __init__(self, on_event: Callable[[Event], None], run_id: str):
        self._on_event = on_event
        self._run_id = run_id
        self._current_message_id = None
        self._chunks: list[CustomChunk] = []

    def aggregate(self, item: CustomChunk) -> None:
        """Process a chunk from LLM stream."""
        self._chunks.append(item)

        # Parse chunk and emit unified events
        if item.type == "text_delta":
            self._on_event(
                TextMessageContentEvent(
                    message_id=self._current_message_id,
                    delta=item.content,
                    run_id=self._run_id,
                )
            )
        elif item.type == "tool_call":
            # Emit tool call events
            pass

    def build(self) -> CustomResult:
        """Build final result from accumulated chunks."""
        return CustomResult(chunks=self._chunks)

    def clear(self) -> None:
        """Reset for reuse."""
        self._chunks = []
        self._current_message_id = None
```

### Type-Safe Event Emission Pattern

Emit events without type errors using wrapper functions:

```python
from typing import TYPE_CHECKING
from nexau.archs.llm.llm_aggregators import Event

if TYPE_CHECKING:
    from .events import TextMessageContentEvent

def _emit_text_delta(self, delta: str) -> None:
    """Wrapper function ensures type checking passes"""
    event = TextMessageContentEvent(
        message_id=self._current_message_id,
        delta=delta,
        run_id=self._run_id,
    )
    self._on_event(event)  # ✅ Type checker passes

# Usage
if item.type == "text_delta":
    self._emit_text_delta(item.delta)
```

**Key insight**: Break problems into small wrapper functions that handle specific types.

### Event Handler Pattern

Handle different event types:

```python
from nexau.archs.llm.llm_aggregators import Event

def handle_event(event: Event):
    """Handle all event types."""
    match event.type:
        case "TEXT_MESSAGE_CONTENT":
            print(event.delta, end="")
        case "TOOL_CALL_ARGS_EVENT":
            print(f"Tool: {event.name}", end=": ")
            print(event.delta)
        case "RUN_FINISHED_EVENT":
            print(f"\nDone. Response: {event.result}")
```

## Type Safety Requirements

This module enforces strict type safety:

### Core Principles

1. **No `# type: ignore` comments**
2. **No `Any` type usage**
3. **No dynamic attribute access** (`getattr`/`hasattr`)
4. **Full type signatures on all functions**

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
def process_event(event: Any) -> Any:
    return event.process()

# ✅ Use pattern matching
from nexau.archs.llm.llm_aggregators import Event

def process_event(event: Event) -> str:
    match event.type:
        case "TEXT_MESSAGE_CONTENT":
            return event.delta
        case "TOOL_CALL_ARGS_EVENT":
            return str(event.delta)
```

#### 3. No Dynamic Attribute Access

```python
# ❌ getattr and hasattr are not type-safe
event_type = getattr(event, "type", None)

# ✅ Use direct attribute access
event_type = event.type

# ✅ Use pattern matching
match event:
    case TextMessageContentEvent():
        pass
```

#### 4. No String-Based Type Narrowing

```python
# ❌ Bypassing type system
if str(event.type) == "text_message_content":
    # ...

# ✅ Direct literal comparison (type checker can infer)
if event.type == "TEXT_MESSAGE_CONTENT":
    # ...

# ✅ Use pattern matching (most elegant and safe)
match event.type:
    case "TEXT_MESSAGE_CONTENT":
        # ...
```

## Common Issues

### Event Not Emitted

**Error**: Events not being emitted by aggregator

**Solution**: Ensure `on_event` callback is passed correctly:

```python
# Incorrect
aggregator = OpenAIChatCompletionAggregator(run_id="run_123")  # Missing on_event

# Correct
aggregator = OpenAIChatCompletionAggregator(
    on_event=handle_event,  # ← Required
    run_id="run_123",
)
```

### Type Error on Event Emission

**Error**: `Type error: Argument 1 has incompatible type`

**Solution**: Use wrapper functions for type-safe emission:

```python
# Incorrect
self._on_event(TextMessageContentEvent(delta=item.delta))  # Missing required fields

# Correct
def _emit_text_delta(self, delta: str) -> None:
    event = TextMessageContentEvent(
        message_id=self._current_message_id,
        delta=delta,
        run_id=self._run_id,
    )
    self._on_event(event)
```

### Aggregator State Not Cleared

**Error**: Previous run's state leaking into new run

**Solution**: Always call `aggregator.clear()` before reuse:

```python
# First run
aggregator = OpenAIChatCompletionAggregator(...)
for chunk in stream1:
    aggregator.aggregate(chunk)
result1 = aggregator.build()

# Second run - clear state first
aggregator.clear()
for chunk in stream2:
    aggregator.aggregate(chunk)
result2 = aggregator.build()
```

### Run ID Not Propagated

**Error**: Events don't have correct `run_id`

**Solution**: Always pass `run_id` to aggregator initialization:

```python
# Incorrect
aggregator = OpenAIChatCompletionAggregator(on_event=handle_event)

# Correct
aggregator = OpenAIChatCompletionAggregator(
    on_event=handle_event,
    run_id=agent_state.run_id,  # ← Important
)
```
