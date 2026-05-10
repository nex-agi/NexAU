# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RFC-0024 invariant — ``ToolResultBlock.raw_output`` MUST NOT leak to the LLM.

The field is persisted into ``nexau_agent_run_actions.append_messages`` so
the playground Compare panel + future UI consumers can render typed
fields (returnDisplay, duration_ms, custom meta). It deliberately
duplicates content the LLM already sees via ``content`` / ``llm_tool_output``
(stripped form) and is meant to remain server-side only — sending it
back to the model would (a) waste tokens, (b) leak internal fields
(stdout file paths, debug counters, etc.), and (c) confuse the model
with two near-duplicate views of the same tool result.

Every UMP → provider serializer (OpenAI Chat, OpenAI Responses,
Anthropic Messages, Gemini) must drop this field. This test plants a
unique sentinel value into ``raw_output`` and asserts none of the
serialized payloads contains it — strong enough to catch a future
regression where someone wires ``**block.model_dump()`` or dumps a
JSON-serialized ``ToolResultBlock`` whole-cloth.
"""

from __future__ import annotations

import json
from typing import Any

from nexau.core.messages import Message, Role, ToolResultBlock
from nexau.core.serializers.anthropic_messages import (
    serialize_ump_to_anthropic_messages_payload,
)
from nexau.core.serializers.gemini_messages import serialize_ump_to_gemini_messages_payload
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload

# Unique strings that should never appear in payload sent to LLM.
_SENTINEL_TOKEN = "DO_NOT_LEAK_TO_LLM_5d2f1a9c"
_SENTINEL_RETURN_DISPLAY = "internal_return_display_label_e7b3"
_SENTINEL_DURATION_MS = 999777
_SENTINEL_INTERNAL_PATH = "/var/lib/nexau-internal/stdout-buffer.bin"


def _make_message_with_raw_output() -> Message:
    """Plant sentinels into raw_output that have no business reaching the LLM."""
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call_invariant_1",
                content="OK: 42 lines processed",  # this string IS allowed to reach LLM
                is_error=False,
                raw_output={
                    "content": "OK: 42 lines processed",
                    "returnDisplay": _SENTINEL_RETURN_DISPLAY,
                    "duration_ms": _SENTINEL_DURATION_MS,
                    "stdout_file": _SENTINEL_INTERNAL_PATH,
                    "_internal_debug": _SENTINEL_TOKEN,
                },
            )
        ],
        metadata={"tool_name": "demo_tool"},
    )


def _flatten_to_json_string(payload: Any) -> str:
    """JSON-serialize so any nested dict/list value gets searched."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _assert_no_raw_output_leak(payload: Any, *, label: str) -> None:
    rendered = _flatten_to_json_string(payload)
    for sentinel in (
        _SENTINEL_TOKEN,
        _SENTINEL_RETURN_DISPLAY,
        _SENTINEL_INTERNAL_PATH,
        '"raw_output"',
        # duration_ms shows up on success-path UI but the integer marker is unique
        # enough to not clash with arbitrary tool output:
        str(_SENTINEL_DURATION_MS),
    ):
        assert sentinel not in rendered, f"[{label}] raw_output sentinel {sentinel!r} leaked into LLM-bound payload:\n{rendered[:600]}..."


def test_completion_provider_serializer_drops_raw_output() -> None:
    """RFC-0024: OpenAI-Chat-style tool messages must not carry raw_output.

    Test name avoids the substrings ``openai`` / ``chat`` / ``llm`` because
    ``tests/conftest.py::pytest_collection_modifyitems`` auto-attaches
    ``@pytest.mark.llm`` to any test whose name contains those — which then
    skips the test under PR-loop CI (no LLM_API_KEY env). This is a pure-
    serialization test that doesn't touch a real LLM and must stay in PR
    loop.
    """
    msgs = [_make_message_with_raw_output()]
    payload = serialize_ump_to_openai_chat_payload(msgs)
    _assert_no_raw_output_leak(payload, label="openai_chat")
    # Sanity: the LLM-visible content survives.
    assert any(isinstance(item, dict) and "OK: 42 lines processed" in str(item.get("content")) for item in payload), (
        f"expected llm-visible content to reach payload: {payload}"
    )


def test_anthropic_messages_serializer_drops_raw_output() -> None:
    """RFC-0024: Anthropic ``tool_result`` blocks must not carry raw_output."""
    msgs = [_make_message_with_raw_output()]
    payload = serialize_ump_to_anthropic_messages_payload(msgs)
    _assert_no_raw_output_leak(payload, label="anthropic_messages")


def test_gemini_serializer_drops_raw_output() -> None:
    """RFC-0024: Gemini ``functionResponse`` parts must not carry raw_output.

    Gemini wants a prior tool_use to anchor func_name, so we precede the
    tool result with the matching assistant message that requested it.
    """
    from nexau.core.messages import ToolUseBlock

    msgs = [
        Message.user("trigger demo_tool"),
        Message(
            role=Role.ASSISTANT,
            content=[ToolUseBlock(id="call_invariant_1", name="demo_tool", input={})],
        ),
        _make_message_with_raw_output(),
    ]
    contents, _ = serialize_ump_to_gemini_messages_payload(msgs)
    _assert_no_raw_output_leak(contents, label="gemini_messages")


# ---------------------------------------------------------------------------
# RFC-0024: ``_coerce_raw_output`` helper — direct unit coverage
# ---------------------------------------------------------------------------
#
# The executor's ``_coerce_raw_output(output)`` narrows tool feedback's
# loosely-typed ``output`` (``object``) into the dict/list shape that
# ``ToolResultBlock.raw_output`` accepts, falling through to None for
# scalars. Direct unit test below exercises all 3 branches (the integration
# path through the LLM serializers above only hits the dict branch).


def test_coerce_raw_output_passes_dict_through() -> None:
    from nexau.archs.main_sub.execution.executor import _coerce_raw_output

    result = _coerce_raw_output({"foo": 1, "bar": [2, 3]})
    assert result == {"foo": 1, "bar": [2, 3]}


def test_coerce_raw_output_passes_list_through() -> None:
    from nexau.archs.main_sub.execution.executor import _coerce_raw_output

    result = _coerce_raw_output([{"a": 1}, {"b": 2}])
    assert result == [{"a": 1}, {"b": 2}]


def test_coerce_raw_output_returns_none_for_scalar_or_none() -> None:
    """Scalars / strings / None must round-trip to None (cannot persist
    structured raw_output for them; ToolResultBlock.content carries the text)."""
    from nexau.archs.main_sub.execution.executor import _coerce_raw_output

    assert _coerce_raw_output(None) is None
    assert _coerce_raw_output("plain text output") is None
    assert _coerce_raw_output(42) is None
    assert _coerce_raw_output(True) is None


def test_coerce_raw_output_keeps_trivial_result_wrapping_verbatim() -> None:
    """``raw_output`` is post-``ToolExecutor.finalize_tool_execution`` raw —
    framework does NOT strip ``{"result": <scalar>}`` wrappings even though
    they look trivial. UI consumers wanting to hide them filter themselves.

    Pinned because an earlier iteration added strip-trivial-wrapping logic
    based on a review suggestion; reverted to keep field semantics aligned
    with the name "raw_output". See PR #533 review thread.
    """
    from nexau.archs.main_sub.execution.executor import _coerce_raw_output

    assert _coerce_raw_output({"result": "OK"}) == {"result": "OK"}
    assert _coerce_raw_output({"result": 42}) == {"result": 42}
    assert _coerce_raw_output({"result": True}) == {"result": True}
    assert _coerce_raw_output({"result": None}) == {"result": None}
    assert _coerce_raw_output({"result": {"a": 1}}) == {"result": {"a": 1}}
    assert _coerce_raw_output({"result": [1, 2, 3]}) == {"result": [1, 2, 3]}
    assert _coerce_raw_output({"result": "ok", "duration_ms": 42}) == {"result": "ok", "duration_ms": 42}
