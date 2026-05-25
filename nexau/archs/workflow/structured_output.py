"""Structured output helpers for Agent workflow nodes.

RFC-0027: Agent structured output compatibility path

The runtime supports a dynamic ``complete_task`` stop tool and a fenced JSON
fallback. Provider-native structured output remains an optimization hook; the
portable path is implemented here so workflow Agent nodes can always validate a
machine-readable result.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import cast

import jsonschema
from json_repair import repair_json

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import SessionManager
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.core import BaseTracer
from nexau.archs.workflow.config import WorkflowOutputMode
from nexau.archs.workflow.types import JsonObject, json_object, json_value

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(ValueError):
    """Raised when structured output cannot be extracted or validated."""


def validate_json_schema_output(output: JsonObject, schema: JsonObject) -> None:
    """Validate a workflow/agent structured output against JSON Schema."""

    try:
        jsonschema.validate(instance=output, schema=cast(dict[str, object], schema))
    except jsonschema.ValidationError as exc:
        raise StructuredOutputError(f"Structured output failed schema validation: {exc.message}") from exc


def parse_json_block(text: str) -> JsonObject:
    """Extract exactly one fenced JSON object from model text."""

    matches = _JSON_BLOCK_PATTERN.findall(text)
    if len(matches) != 1:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            matches = [stripped]
        else:
            raise StructuredOutputError("Expected exactly one fenced ```json``` block")
    repaired = repair_json(matches[0])
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("Structured output JSON could not be parsed") from exc
    return json_object(parsed, label="structured output")


def parse_structured_response(text: str, schema: JsonObject) -> JsonObject:
    """Parse a JSON object from a response and validate it."""

    stripped = text.strip()
    parsed: JsonObject
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json_object(json.loads(repair_json(stripped)), label="structured response")
        except json.JSONDecodeError as exc:
            raise StructuredOutputError("Structured response JSON could not be parsed") from exc
    else:
        parsed = parse_json_block(stripped)
    validate_json_schema_output(parsed, schema)
    return parsed


def build_complete_task_tool(schema: JsonObject, *, output_name: str | None = None) -> Tool:
    """Build a dynamic ``complete_task`` stop tool for a structured schema."""

    name = output_name or "structured_output"
    description = f"Submit the final {name} object and finish this agent node."
    input_schema = deepcopy(schema)

    def complete_task(**kwargs: object) -> JsonObject:
        output = {key: json_value(value) for key, value in kwargs.items()}
        validate_json_schema_output(output, schema)
        return output

    return Tool(
        name="complete_task",
        description=description,
        input_schema=input_schema,
        implementation=complete_task,
        disable_parallel=True,
    )


def structured_prompt_suffix(schema: JsonObject, mode: WorkflowOutputMode) -> str:
    """Return instruction text that asks the model to satisfy the schema."""

    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    if mode == "json_block":
        return (
            "You must finish by returning exactly one fenced ```json``` code block. "
            "The JSON object must satisfy this schema:\n"
            f"{schema_text}"
        )
    return (
        "You must call the `complete_task` tool exactly once when the task is complete. "
        "The tool arguments must satisfy this JSON Schema:\n"
        f"{schema_text}"
    )


async def run_agent_structured_async(
    *,
    config: AgentConfig,
    message: str,
    output_schema: JsonObject,
    output_mode: WorkflowOutputMode = "auto",
    output_retries: int = 3,
    output_name: str | None = None,
    session_manager: SessionManager | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    tracer: BaseTracer | None = None,
) -> JsonObject:
    """Run an Agent and return schema-valid structured output."""

    from nexau.archs.main_sub.agent import Agent

    effective_mode: WorkflowOutputMode = "complete_task" if output_mode in {"auto", "native", "complete_task"} else "json_block"
    errors: list[str] = []
    current_message = message

    for attempt in range(output_retries + 1):
        structured_config = _build_structured_agent_config(
            config=config,
            output_schema=output_schema,
            output_mode=effective_mode,
            output_name=output_name,
            tracer=tracer,
        )
        agent = await Agent.create(
            config=structured_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        response = await agent.run_async(message=current_message, run_id=run_id)
        response_text = response[0] if isinstance(response, tuple) else response
        try:
            return parse_structured_response(response_text, output_schema)
        except StructuredOutputError as exc:
            errors.append(str(exc))
            if attempt >= output_retries:
                break
            current_message = (
                f"{message}\n\nThe previous response did not satisfy the required output contract: {exc}. "
                "Retry and provide only the required final structured output."
            )

    raise StructuredOutputError("; ".join(errors) if errors else "Structured output was not produced")


def run_agent_structured(
    *,
    config: AgentConfig,
    message: str,
    output_schema: JsonObject,
    output_mode: WorkflowOutputMode = "auto",
    output_retries: int = 3,
    output_name: str | None = None,
    session_manager: SessionManager | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    tracer: BaseTracer | None = None,
) -> JsonObject:
    """Synchronous wrapper for ``run_agent_structured_async``."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_agent_structured_async(
                config=config,
                message=message,
                output_schema=output_schema,
                output_mode=output_mode,
                output_retries=output_retries,
                output_name=output_name,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                run_id=run_id,
                tracer=tracer,
            )
        )
    raise RuntimeError("run_agent_structured() cannot be called from a running event loop; use run_agent_structured_async()")


def _build_structured_agent_config(
    *,
    config: AgentConfig,
    output_schema: JsonObject,
    output_mode: WorkflowOutputMode,
    output_name: str | None,
    tracer: BaseTracer | None,
) -> AgentConfig:
    # Tracers often own live SDK clients (Langfuse/OpenTelemetry) that are not
    # deepcopy-safe. Preserve those runtime objects by reference while cloning
    # ordinary agent config.
    original_tracers = list(config.tracers)
    original_resolved_tracer = config.resolved_tracer
    memo: dict[int, object] = {id(runtime_tracer): runtime_tracer for runtime_tracer in original_tracers}
    if original_resolved_tracer is not None:
        memo[id(original_resolved_tracer)] = original_resolved_tracer
    structured_config = deepcopy(config, memo)
    if tracer is not None:
        structured_config.tracers = [tracer]
        structured_config.resolved_tracer = tracer
    else:
        structured_config.tracers = list(original_tracers)
        structured_config.resolved_tracer = original_resolved_tracer
    suffix = structured_prompt_suffix(output_schema, output_mode)
    existing_suffix = structured_config.system_prompt_suffix or ""
    structured_config.system_prompt_suffix = f"{existing_suffix}\n\n{suffix}".strip()
    structured_config.output_schema = output_schema
    structured_config.output_mode = output_mode
    structured_config.output_name = output_name
    if output_mode != "json_block":
        complete_task_tool = build_complete_task_tool(output_schema, output_name=output_name)
        structured_config.tools = [*structured_config.tools, complete_task_tool]
        structured_config.stop_tools = set(structured_config.stop_tools or set())
        structured_config.stop_tools.add("complete_task")
        structured_config.max_iterations = max(structured_config.max_iterations, 2)
    return structured_config


def schema_from_mapping(schema: Mapping[str, object]) -> JsonObject:
    """Convert a generic mapping into the local JSON object alias."""

    return json_object(dict(schema), label="schema")
