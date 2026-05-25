"""RFC-0027 live LLM workflow end-to-end test."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.workflow import WorkflowConfig, WorkflowExecutor, WorkflowStore
from nexau.archs.workflow.types import json_array, json_object


@pytest.mark.llm
def test_workflow_agent_node_real_llm_structured_output() -> None:
    asyncio.run(_run_workflow_agent_node_real_llm_structured_output())


async def _run_workflow_agent_node_real_llm_structured_output() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "real_llm_structured_workflow",
            "nodes": {
                "start": {"type": "start", "output": {"instruction": "{{ inputs.instruction }}"}},
                "answer": {
                    "type": "agent",
                    "agent": "structured_answerer",
                    "input": {"instruction": "{{ nodes.start.output.instruction }}"},
                    "output_mode": "complete_task",
                    "output_retries": 2,
                    "output_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string", "enum": ["ok"]}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
                "finish": {"type": "end", "output": {"answer": "{{ nodes.answer.output.answer }}"}},
            },
            "edges": {"start": "answer", "answer": "finish"},
        }
    )
    agent_config = AgentConfig(
        name="structured_answerer",
        system_prompt=(
            "You are inside an automated workflow test. Follow the workflow input exactly. "
            "When asked for the final answer, submit the literal string ok."
        ),
        llm_config=LLMConfig(),
        tool_call_mode="structured",
        max_iterations=4,
    )
    engine = InMemoryDatabaseEngine()
    session_manager = SessionManager(engine=engine)
    store = WorkflowStore(engine)
    executor = WorkflowExecutor(workflow=workflow, store=store, agents={"structured_answerer": agent_config})

    result = await executor.run_async(
        inputs={"instruction": "Return the final structured answer with answer equal to ok."},
        run_id="wf_real_llm_structured",
        user_id="test_user",
        session_id="workflow_real_llm_session",
        session_manager=session_manager,
    )

    assert result.status.value == "completed"
    assert result.output == {"answer": "ok"}
    folded = await store.fold("wf_real_llm_structured")
    assert folded.status.value == "completed"
    assert folded.node_outputs["answer"] == {"answer": "ok"}


@pytest.mark.llm
def test_workflow_subgraph_real_llm_human_resume(tmp_path: Path) -> None:
    asyncio.run(_run_workflow_subgraph_real_llm_human_resume(tmp_path))


async def _run_workflow_subgraph_real_llm_human_resume(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    review_path = graph_dir / "review_answer.workflow.yaml"
    review_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: review_answer",
                "inputs:",
                "  answer:",
                "    type: string",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      answer: "{{ inputs.answer }}"',
                "  review:",
                "    type: human",
                "    prompt: Review the structured answer.",
                "    input:",
                '      answer: "{{ nodes.start.output.answer }}"',
                "    output_schema:",
                "      type: object",
                "      properties:",
                "        approved:",
                "          type: boolean",
                "      required: [approved]",
                "      additionalProperties: false",
                "  finish:",
                "    type: end",
                "    output:",
                '      approved: "{{ nodes.review.output.approved }}"',
                '      answer: "{{ nodes.start.output.answer }}"',
                "edges:",
                "  start: review",
                "  review: finish",
            ]
        ),
        encoding="utf-8",
    )
    workflow_path = tmp_path / "subgraph_real_llm.workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: real_llm_subgraph_workflow",
                "includes:",
                "  graphs:",
                "    review_answer: ./graphs/review_answer.workflow.yaml",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      instruction: "{{ inputs.instruction }}"',
                "  answer:",
                "    type: agent",
                "    agent: structured_answerer",
                "    input:",
                '      instruction: "{{ nodes.start.output.instruction }}"',
                "    output_mode: complete_task",
                "    output_retries: 2",
                "    output_schema:",
                "      type: object",
                "      properties:",
                "        answer:",
                "          type: string",
                "          enum: [ok]",
                "      required: [answer]",
                "      additionalProperties: false",
                "  review_answer:",
                "    type: subgraph",
                "    graph: review_answer",
                "    input:",
                '      answer: "{{ nodes.answer.output.answer }}"',
                "    output_schema:",
                "      type: object",
                "      properties:",
                "        approved:",
                "          type: boolean",
                "        answer:",
                "          type: string",
                "      required: [approved, answer]",
                "      additionalProperties: false",
                "  finish:",
                "    type: end",
                "    output:",
                '      approved: "{{ nodes.review_answer.output.approved }}"',
                '      answer: "{{ nodes.review_answer.output.answer }}"',
                "edges:",
                "  start: answer",
                "  answer: review_answer",
                "  review_answer: finish",
            ]
        ),
        encoding="utf-8",
    )
    agent_config = AgentConfig(
        name="structured_answerer",
        system_prompt=(
            "You are inside an automated workflow subgraph test. Follow the workflow input exactly. "
            "When asked for the final answer, submit the literal string ok."
        ),
        llm_config=LLMConfig(),
        tool_call_mode="structured",
        max_iterations=4,
    )
    engine = InMemoryDatabaseEngine()
    session_manager = SessionManager(engine=engine)
    store = WorkflowStore(engine)
    executor = WorkflowExecutor(
        workflow=WorkflowConfig.from_yaml(workflow_path),
        store=store,
        agents={"structured_answerer": agent_config},
    )

    waiting = await executor.run_async(
        inputs={"instruction": "Return the final structured answer with answer equal to ok."},
        run_id="wf_real_llm_subgraph",
        user_id="test_user",
        session_id="workflow_subgraph_real_llm_session",
        session_manager=session_manager,
    )

    assert waiting.status.value == "waiting"
    assert waiting.checkpoint_id is not None
    checkpoint = await store.get_checkpoint(waiting.checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.scope_path == "review_answer/review"
    assert checkpoint.input == {"answer": "ok"}

    completed = await executor.resume_async(
        run_id="wf_real_llm_subgraph",
        checkpoint_id=waiting.checkpoint_id,
        output={"approved": True},
        user_id="test_user",
        session_id="workflow_subgraph_real_llm_session",
        session_manager=session_manager,
    )

    assert completed.status.value == "completed"
    assert completed.output == {"approved": True, "answer": "ok"}
    events = await store.list_events("wf_real_llm_subgraph")
    assert any(event.event_type == "subgraph_completed" for event in events)


@pytest.mark.llm
def test_workflow_parallel_map_real_llm_structured_output() -> None:
    asyncio.run(_run_workflow_parallel_map_real_llm_structured_output())


async def _run_workflow_parallel_map_real_llm_structured_output() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "real_llm_parallel_map_workflow",
            "durable": {"mode": "node_boundary", "default_parallelism": 2, "max_parallelism": 2},
            "nodes": {
                "start": {"type": "start", "output": {"items": "{{ inputs.items }}"}},
                "map_items": {
                    "type": "parallel_map",
                    "items": "{{ nodes.start.output.items }}",
                    "item_name": "work_item",
                    "item_key": "{{ work_item.id }}",
                    "max_concurrency": 2,
                    "body": "answer_item",
                    "failure_policy": "collect_errors",
                    "collect": {"output": {"results": "{{ results }}", "errors": "{{ errors }}", "stats": "{{ stats }}"}},
                    "output_schema": {
                        "type": "object",
                        "properties": {"results": {"type": "array"}, "errors": {"type": "array"}, "stats": {"type": "object"}},
                        "required": ["results", "errors", "stats"],
                        "additionalProperties": False,
                    },
                },
                "answer_item": {
                    "type": "agent",
                    "agent": "parallel_answerer",
                    "input": {"id": "{{ work_item.id }}", "instruction": "{{ work_item.instruction }}"},
                    "output_mode": "complete_task",
                    "output_retries": 2,
                    "output_schema": {
                        "type": "object",
                        "properties": {"item_id": {"type": "string"}, "answer": {"type": "string", "enum": ["ok"]}},
                        "required": ["item_id", "answer"],
                        "additionalProperties": False,
                    },
                },
                "finish": {
                    "type": "end",
                    "output": {
                        "results": "{{ nodes.map_items.output.results }}",
                        "stats": "{{ nodes.map_items.output.stats }}",
                    },
                },
            },
            "edges": {"start": "map_items", "map_items": "finish"},
        }
    )
    agent_config = AgentConfig(
        name="parallel_answerer",
        system_prompt=(
            "You are inside an automated workflow parallel-map test. "
            "For each workflow item, copy the provided id into item_id and submit answer exactly as ok."
        ),
        llm_config=LLMConfig(),
        tool_call_mode="structured",
        max_iterations=4,
    )
    engine = InMemoryDatabaseEngine()
    session_manager = SessionManager(engine=engine)
    store = WorkflowStore(engine)
    executor = WorkflowExecutor(workflow=workflow, store=store, agents={"parallel_answerer": agent_config})

    result = await executor.run_async(
        inputs={
            "items": [
                {"id": "A", "instruction": "Return answer ok for item A."},
                {"id": "B", "instruction": "Return answer ok for item B."},
            ]
        },
        run_id="wf_real_llm_parallel_map",
        user_id="test_user",
        session_id="workflow_parallel_real_llm_session",
        session_manager=session_manager,
    )

    assert result.status.value == "completed"
    assert result.output is not None
    assert result.output["stats"] == {"total": 2, "completed": 2, "failed": 0, "waiting": 0, "uncertain": 0}
    results = json_array(result.output["results"], label="results")
    entries = [json_object(entry, label="parallel result") for entry in results]
    assert [entry["key"] for entry in entries] == ["A", "B"]
    answers: list[str] = []
    for entry in entries:
        output = json_object(entry["output"], label="parallel output")
        answer = output["answer"]
        assert isinstance(answer, str)
        answers.append(answer)
    assert answers == ["ok", "ok"]

    folded = await store.fold("wf_real_llm_parallel_map")
    assert folded.status.value == "completed"
    parallel_map = next(iter(folded.parallel_maps.values()))
    assert [item.status for item in parallel_map.items] == ["completed", "completed"]
    events = await store.list_events("wf_real_llm_parallel_map")
    assert any(event.event_type == "parallel_map_started" for event in events)
    assert sum(1 for event in events if event.event_type == "parallel_item_completed") == 2
