"""RFC-0027 live LLM workflow end-to-end test."""

from __future__ import annotations

import asyncio

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
