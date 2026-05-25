"""RFC-0029 workflow store fold tests."""

from __future__ import annotations

import asyncio

from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.workflow import WorkflowStore
from nexau.archs.workflow.store import event_payload


def test_store_fold_reconstructs_parallel_map_items_and_checkpoints() -> None:
    asyncio.run(_run_store_fold_reconstructs_parallel_map_items_and_checkpoints())


async def _run_store_fold_reconstructs_parallel_map_items_and_checkpoints() -> None:
    store = WorkflowStore(InMemoryDatabaseEngine())
    await store.create_run(run_id="wf_parallel_fold", workflow_name="parallel_fold", inputs={}, definition_snapshot={})
    await store.append_event(run_id="wf_parallel_fold", event_type="workflow_run_started", payload=event_payload(inputs={}))
    await store.append_event(
        run_id="wf_parallel_fold",
        event_type="parallel_map_started",
        node_id="map_items",
        payload=event_payload(
            body_node_id="run_item",
            max_concurrency=2,
            failure_policy="collect_errors",
            result_order="input",
            items=[
                {"index": 0, "key": "A", "item": {"id": "A"}, "scope_path": "map_items[A]", "body_node_id": "run_item"},
                {"index": 1, "key": "B", "item": {"id": "B"}, "scope_path": "map_items[B]", "body_node_id": "run_item"},
            ],
        ),
    )
    await store.append_event(
        run_id="wf_parallel_fold",
        event_type="parallel_item_completed",
        node_id="map_items",
        payload=event_payload(
            item_index=0,
            item_key="A",
            item_scope_path="map_items[A]",
            body_node_id="run_item",
            output={"value": "A"},
        ),
    )
    await store.append_event(
        run_id="wf_parallel_fold",
        event_type="parallel_item_failed",
        node_id="map_items",
        payload=event_payload(
            item_index=1,
            item_key="B",
            item_scope_path="map_items[B]",
            body_node_id="run_item",
            error={"message": "failed B"},
        ),
    )
    await store.append_event(
        run_id="wf_parallel_fold",
        event_type="checkpoint_created",
        node_id="review",
        scope_path="map_items[A]/run_item/review",
        payload=event_payload(checkpoint_id="ckpt_A"),
    )
    await store.append_event(
        run_id="wf_parallel_fold",
        event_type="checkpoint_created",
        node_id="review",
        scope_path="map_items[B]/run_item/review",
        payload=event_payload(checkpoint_id="ckpt_B"),
    )
    await store.append_event(
        run_id="wf_parallel_fold",
        event_type="checkpoint_resumed",
        node_id="review",
        scope_path="map_items[A]/run_item/review",
        payload=event_payload(checkpoint_id="ckpt_A"),
    )

    folded = await store.fold("wf_parallel_fold")

    parallel_map = next(iter(folded.parallel_maps.values()))
    assert parallel_map.body_node_id == "run_item"
    assert parallel_map.max_concurrency == 2
    assert parallel_map.items[0].status == "completed"
    assert parallel_map.items[0].output == {"value": "A"}
    assert parallel_map.items[1].status == "failed"
    assert parallel_map.items[1].error == {"message": "failed B"}
    assert parallel_map.completed_item_order == ["map_items[A]"]
    assert folded.waiting_checkpoint_ids == ["ckpt_B"]
    assert folded.waiting_checkpoint_id == "ckpt_B"
    assert folded.waiting_scope_path == "map_items[B]/run_item/review"


def test_store_fold_keeps_parallel_node_failure_from_failing_run() -> None:
    asyncio.run(_run_store_fold_keeps_parallel_node_failure_from_failing_run())


async def _run_store_fold_keeps_parallel_node_failure_from_failing_run() -> None:
    store = WorkflowStore(InMemoryDatabaseEngine())
    await store.create_run(run_id="wf_parallel_failure", workflow_name="parallel_failure", inputs={}, definition_snapshot={})
    await store.append_event(run_id="wf_parallel_failure", event_type="workflow_run_started", payload=event_payload(inputs={}))
    await store.append_event(
        run_id="wf_parallel_failure",
        event_type="node_failed",
        node_id="run_item",
        scope_path="map_items[A]/run_item",
        payload=event_payload(parallel_node_id="map_items", item_scope_path="map_items[A]", error="item failed"),
    )

    folded = await store.fold("wf_parallel_failure")

    assert folded.status.value == "running"
    assert "wf_parallel_failure:run_item:map_items[A]/run_item" in folded.failed_node_runs
