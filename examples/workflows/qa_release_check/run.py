from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import BaseTracer, SpanType
from nexau.archs.workflow import JsonObject, WorkflowConfig, WorkflowExecutor, WorkflowStore
from nexau.archs.workflow.executor import AgentNodeRunner
from nexau.archs.workflow.types import json_object, json_value

EXAMPLE_DIR = Path(__file__).parent
WORKFLOW_PATH = EXAMPLE_DIR / "qa_release.workflow.yaml"
REQUIRED_ENV_KEYS = ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_API_TYPE")
REQUIRED_LANGFUSE_ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
DEFAULT_SESSION_ID = "qa_release_check_example"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QA release check workflow example.")
    parser.add_argument(
        "--requirement",
        default="Checkout retry should show a clear error after three failed payment attempts.",
        help="Product requirement or release change to verify.",
    )
    parser.add_argument("--max-cases", type=int, default=2, choices=(1, 2, 3), help="Maximum generated QA cases.")
    parser.add_argument("--run-id", default=None, help="Optional stable workflow run id.")
    parser.add_argument("--reject", action="store_true", help="Resume the human checkpoint with approved=false.")
    parser.add_argument("--mock-agents", action="store_true", help="Run full workflow with deterministic local agent outputs.")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse tracing for workflow and node spans.")
    parser.add_argument("--langfuse-debug", action="store_true", help="Enable Langfuse SDK debug logging.")
    parser.add_argument("--langfuse-session-id", default=DEFAULT_SESSION_ID, help="Langfuse session id for this example run.")
    parser.add_argument("--langfuse-trace-id", default=None, help="Optional Langfuse trace id to attach this run to.")
    parser.add_argument("--validate-only", action="store_true", help="Load and validate YAML without calling the LLM.")
    return parser.parse_args()


def require_env() -> None:
    missing = [key for key in REQUIRED_ENV_KEYS if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} or put them in a repo-level .env before running this live example.")


def require_langfuse_env() -> None:
    missing = [key for key in REQUIRED_LANGFUSE_ENV_KEYS if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} or put them in a repo-level .env before enabling --langfuse.")


def build_langfuse_tracer(args: argparse.Namespace) -> BaseTracer | None:
    if not args.langfuse:
        return None

    require_langfuse_env()

    from nexau.archs.tracer.adapters import LangfuseTracer

    host = os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    print(f"langfuse tracing enabled: host={host}, session_id={args.langfuse_session_id}")
    return LangfuseTracer(
        host=host,
        debug=args.langfuse_debug,
        session_id=args.langfuse_session_id,
        trace_id=args.langfuse_trace_id,
        tags=["example", "workflow", "qa_release_check"],
        metadata={"example": "qa_release_check", "runtime": "nexau_workflow"},
    )


def close_tracer(tracer: BaseTracer | None) -> None:
    if tracer is None:
        return
    print("flushing langfuse tracer...")
    tracer.flush()
    tracer.shutdown()


def dump_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def get_cases(planner_output: JsonObject) -> list[JsonObject]:
    raw_cases = planner_output.get("cases")
    if not isinstance(raw_cases, list):
        raise RuntimeError("Planner output did not contain a cases array.")

    cases: list[JsonObject] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        cases.append(json_object(raw_case, label=f"case[{index}]"))
    return cases


def mock_agent_configs() -> dict[str, AgentConfig]:
    llm_config = LLMConfig(
        model="mock-workflow-model",
        base_url="http://localhost/mock/v1",
        api_key="mock-api-key",
        api_type="openai_chat_completion",
    )
    return {
        "qa_planner": AgentConfig(name="qa_planner", llm_config=llm_config),
        "qa_runner": AgentConfig(name="qa_runner", llm_config=llm_config),
    }


def build_mock_agent_runner(tracer: BaseTracer | None) -> AgentNodeRunner:
    async def runner(
        *,
        agent_name: str,
        agent_config: AgentConfig | None,
        input_data: JsonObject,
        output_schema: JsonObject | None,
        run_id: str,
        node_id: str,
        scope_path: str,
    ) -> JsonObject:
        del agent_config, output_schema, run_id, node_id

        if tracer is None:
            return mock_agent_output(agent_name=agent_name, input_data=input_data, scope_path=scope_path)

        trace_ctx = TraceContext(
            tracer,
            f"Agent: {agent_name}",
            SpanType.AGENT,
            inputs={"input": input_data},
            attributes={
                "agent_name": agent_name,
                "workflow.mock_agent": True,
                "workflow.scope_path": scope_path,
            },
        )
        with trace_ctx:
            output = mock_agent_output(agent_name=agent_name, input_data=input_data, scope_path=scope_path)
            trace_ctx.set_outputs({"response": output})
            return output

    return runner


def mock_agent_output(
    *,
    agent_name: str,
    input_data: JsonObject,
    scope_path: str,
) -> JsonObject:
    if agent_name == "qa_planner":
        max_cases_value = input_data.get("max_cases", 2)
        max_cases = max_cases_value if isinstance(max_cases_value, int) else 2
        cases: list[JsonObject] = [
            {
                "id": "C001",
                "title": "Payment retry error is visible",
                "steps": ["Submit a payment with a failing test card three times.", "Observe the checkout error state."],
                "expected": "Checkout shows a clear retry-limit error.",
            },
            {
                "id": "C002",
                "title": "Retry state does not create duplicate order",
                "steps": ["Trigger three failed payment attempts.", "Inspect the order confirmation state."],
                "expected": "No duplicate confirmed order is created.",
            },
            {
                "id": "C003",
                "title": "Recovery path remains available",
                "steps": ["Trigger three failed attempts.", "Switch to a valid payment method."],
                "expected": "The customer can recover without restarting checkout.",
            },
        ]
        return {"cases": json_value(cases[:max_cases])}

    raw_case = input_data.get("case")
    case = json_object(raw_case, label="runner.case")
    return {
        "case_id": str(case.get("id", "unknown")),
        "status": "passed",
        "evidence": f"Mock execution satisfied expected result in {scope_path}: {case.get('expected', '')}",
    }


async def run_workflow(args: argparse.Namespace) -> None:
    load_dotenv()

    workflow = WorkflowConfig.from_yaml(WORKFLOW_PATH)
    workflow.vars["max_cases"] = args.max_cases

    if args.validate_only:
        print(f"validated workflow: {workflow.name}")
        print("nodes:")
        for node_id, node in workflow.nodes.items():
            print(f"  - {node_id}: {node.type}")
        print("included agents:")
        for agent_name, agent_path in workflow.includes.agents.items():
            print(f"  - {agent_name}: {agent_path}")
        print("included graphs:")
        for graph_name, graph_path in workflow.includes.graphs.items():
            graph = workflow.included_graphs[graph_name]
            print(f"  - {graph_name}: {graph_path} ({len(graph.nodes)} nodes)")
        return

    tracer = build_langfuse_tracer(args)
    try:
        if not args.mock_agents:
            require_env()

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)
        store = WorkflowStore(engine)
        executor = WorkflowExecutor(
            workflow=workflow,
            store=store,
            agents=mock_agent_configs() if args.mock_agents else None,
            agent_runner=build_mock_agent_runner(tracer) if args.mock_agents else None,
            tracer=tracer,
        )
        run_id = args.run_id or f"wf_qa_release_{uuid.uuid4().hex[:8]}"

        print(f"starting run: {run_id}")
        waiting = await executor.run_async(
            inputs={"requirement": args.requirement},
            run_id=run_id,
            user_id="workflow_example_user",
            session_id=DEFAULT_SESSION_ID,
            session_manager=session_manager,
        )

        print(f"status after planner: {waiting.status.value}")
        if waiting.checkpoint_id is None:
            print("workflow finished without a human checkpoint:")
            print(dump_json(waiting.output))
            return

        checkpoint = await store.get_checkpoint(waiting.checkpoint_id)
        folded = await store.fold(run_id)
        planner_output = folded.node_outputs.get("generate_cases", {})
        cases = get_cases(planner_output)

        if checkpoint is not None:
            print(f"checkpoint: {checkpoint.checkpoint_id}")
            print(f"checkpoint scope: {checkpoint.scope_path}")
            print(f"prompt: {checkpoint.prompt}")
            print("review input:")
            print(dump_json(checkpoint.input))

        review_output: JsonObject = {
            "approved": not args.reject,
            "cases": json_value(cases),
            "review_note": "Auto-approved by the example runner." if not args.reject else "Rejected by the example runner.",
        }

        print("resuming checkpoint with:")
        print(dump_json(review_output))

        completed = await executor.resume_async(
            run_id=run_id,
            checkpoint_id=waiting.checkpoint_id,
            output=review_output,
            user_id="workflow_example_user",
            session_id=DEFAULT_SESSION_ID,
            session_manager=session_manager,
        )

        events = await store.list_events(run_id)
        subgraph_events = [event for event in events if event.event_type.startswith("subgraph_")]
        print(f"final status: {completed.status.value}")
        print("final output:")
        print(dump_json(completed.output))
        print(f"event count: {len(events)}")
        print("subgraph events:")
        for event in subgraph_events:
            print(f"  - {event.event_type}: node={event.node_id}, scope={event.scope_path}, graph={event.payload.get('graph_id')}")
    finally:
        close_tracer(tracer)


def main() -> None:
    asyncio.run(run_workflow(parse_args()))


if __name__ == "__main__":
    main()
