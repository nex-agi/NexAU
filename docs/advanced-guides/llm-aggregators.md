# LLM Aggregators

> **⚠️ Experimental**: This feature is under active development. APIs may change.

LLM Aggregators turn **vendor-specific streaming chunks** into **vendor-specific messages (responses)**. You feed chunks from an LLM stream into an aggregator, then call `build()` to get the complete response (e.g. OpenAI `ChatCompletion` or `Response`).

## Core idea: chunks → message

```
Vendor stream (e.g. OpenAI)     Aggregator              Vendor message
─────────────────────────      ──────────               ──────────────
 chunk 1  ──┐
 chunk 2  ──┼──  aggregate()  ──►  build()  ──►  ChatCompletion / Response
 chunk N  ──┘
```

- **Input**: Provider-specific chunks (`ChatCompletionChunk`, `ResponseStreamEvent`, etc.).
- **Output**: Provider-specific message (`ChatCompletion`, `Response`, etc.).
- **Flow**: Call `aggregate(chunk)` for each chunk, then `build()` to get the final message. Use `clear()` before reusing the same aggregator.

Aggregators optionally emit **streaming events** during `aggregate()` (e.g. text deltas, tool calls) for live UI updates. If you only need the final message, you can pass a no-op `on_event`. See [Streaming Events](./streaming-events.md) for event types and handlers.

## Aggregator ABC

The base pattern is generic over input (chunks) and output (message):

```python
from nexau.archs.llm.llm_aggregators import Aggregator

class MyAggregator(Aggregator[Chunk, Message]):
    def aggregate(self, item: Chunk) -> None:
        """Process one chunk; accumulate into internal state."""
        ...

    def build(self) -> Message:
        """Produce the final vendor message from accumulated chunks."""
        ...

    def clear(self) -> None:
        """Reset state for reuse."""
        ...
```

## Available aggregators

### OpenAI Chat Completion

**Input**: `ChatCompletionChunk` (OpenAI Chat Completions API stream).
**Output**: `ChatCompletion`.

```python
from nexau.archs.llm.llm_aggregators import OpenAIChatCompletionAggregator

def noop(_): pass

agg = OpenAIChatCompletionAggregator(on_event=noop, run_id="run_1")

for chunk in openai_client.beta.chat.completions.stream(...):
    agg.aggregate(chunk)

response: ChatCompletion = agg.build()
agg.clear()
```

### OpenAI Responses

**Input**: `ResponseStreamEvent` (OpenAI Responses API, e.g. GPT-5 style).
**Output**: `Response`. Supports reasoning/thinking, tool calls, images.

```python
from nexau.archs.llm.llm_aggregators import OpenAIResponsesAggregator

def noop(_): pass

agg = OpenAIResponsesAggregator(on_event=noop, run_id="run_1")

async for event in openai_client.responses.create(..., stream=True):
    agg.aggregate(event)

response: Response = agg.build()
agg.clear()
```

## Usage patterns

### Chunks → message only (no streaming UI)

Use a no-op `on_event` if you only care about the final message:

```python
def noop(_): pass

agg = OpenAIChatCompletionAggregator(on_event=noop, run_id="run_1")
for chunk in stream:
    agg.aggregate(chunk)
msg = agg.build()
agg.clear()
```

### Chunks → message + streaming events

Implement `on_event` to handle incremental updates (e.g. print deltas, update UI). See [Streaming Events](./streaming-events.md) for event types and handler patterns.

```python
def on_event(event):
    if event.type == "TEXT_MESSAGE_CONTENT":
        print(event.delta, end="", flush=True)

agg = OpenAIChatCompletionAggregator(on_event=on_event, run_id="run_1")
for chunk in stream:
    agg.aggregate(chunk)
msg = agg.build()
agg.clear()
```

## Custom aggregator

To support another vendor, implement `Aggregator[ChunkT, MessageT]`:

```python
from nexau.archs.llm.llm_aggregators import Aggregator

class CustomAggregator(Aggregator[CustomChunk, CustomMessage]):
    def __init__(self):
        self._chunks: list[CustomChunk] = []

    def aggregate(self, item: CustomChunk) -> None:
        self._chunks.append(item)

    def build(self) -> CustomMessage:
        return self._parse_chunks_to_message(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
```

You can add `on_event` and emit unified events during `aggregate()`; see [Streaming Events](./streaming-events.md).

## Best practices

1. **Always `clear()` before reuse**
   Avoid leaking state from a previous run.

2. **Pass `run_id`**
   Required by the built-in aggregators; used for tracing and multi-agent support.

3. **Use a no-op `on_event`** when you only need the final message:
   `on_event=lambda _: None`.

## Common issues

- **Events not firing**
  Ensure you pass `on_event`. Use a no-op if you don’t need events.

- **Stale state between runs**
  Call `agg.clear()` before processing a new stream.

- **Missing `run_id`**
  Built-in aggregators require `run_id` in their constructor.
