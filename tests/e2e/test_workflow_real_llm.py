"""RFC-0027 live LLM workflow end-to-end test."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.workflow import WorkflowConfig, WorkflowExecutor, WorkflowStore


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
