"""Workflow YAML configuration models.

RFC-0027: Workflow DSL 与配置解析
RFC-0028: Workflow 子图支持
RFC-0029: Workflow 并行 Map 与结果收集

The parser validates the YAML-facing workflow contract before execution:
node shapes, edge references, simple control graph cycles, durable side-effect
requirements, and JSON Schema snippets used for structured outputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self, cast

import jsonschema
import yaml
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexau.archs.workflow.types import JsonObject, JsonValue, json_object

WorkflowNodeType = Literal[
    "start",
    "agent",
    "tool",
    "mcp",
    "if_else",
    "while",
    "human",
    "transform",
    "set_state",
    "end",
    "note",
    "subgraph",
    "parallel_map",
]
SideEffectPolicy = Literal["read_only", "idempotent_write", "external_write", "local_write"]
WorkflowOutputMode = Literal["auto", "native", "complete_task", "json_block"]
SubgraphExecutionMode = Literal["inline"]
ParallelMapResultOrder = Literal["input", "completion"]
ParallelMapFailurePolicy = Literal["fail_fast", "fail_after_all", "collect_errors"]

PARALLEL_MAP_BODY_NODE_TYPES: set[WorkflowNodeType] = {
    "agent",
    "tool",
    "mcp",
    "transform",
    "set_state",
    "subgraph",
}
PARALLEL_MAP_RESERVED_CONTEXT_NAMES = {"inputs", "vars", "state", "nodes", "run", "node", "item_key"}


def _empty_json_object() -> JsonObject:
    return {}


def _empty_nodes() -> dict[str, WorkflowNode]:
    return {}


def _empty_edges() -> dict[str, str | list[str]]:
    return {}


def _empty_branches() -> list[WorkflowBranch]:
    return []


def _empty_includes() -> WorkflowIncludes:
    return WorkflowIncludes()


def _empty_included_graphs() -> dict[str, WorkflowConfig]:
    return {}


class WorkflowConfigError(ValueError):
    """Raised when a workflow definition is invalid."""


class RetryPolicy(BaseModel):
    """Retry policy for durable workflow nodes."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=1, ge=1)
    backoff: Literal["none", "linear", "exponential"] = "none"
    on_uncertain: Literal["human_review", "retry", "fail"] | None = None


class DurableConfig(BaseModel):
    """Durable execution configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["node_boundary"] = "node_boundary"
    default_retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    lease_timeout_seconds: float = Field(default=30.0, gt=0)
    max_subgraph_depth: int = Field(default=5, ge=0)
    default_parallelism: int = Field(default=1, ge=1)
    max_parallelism: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _validate_parallelism(self) -> Self:
        if self.default_parallelism > self.max_parallelism:
            raise WorkflowConfigError("durable.default_parallelism must not exceed durable.max_parallelism")
        return self


class WorkflowInputSpec(BaseModel):
    """Workflow input field specification."""

    model_config = ConfigDict(extra="allow")

    type: str
    description: str | None = None


class WorkflowIncludes(BaseModel):
    """Relative resources referenced by workflow nodes."""

    model_config = ConfigDict(extra="forbid")

    agents: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, str] = Field(default_factory=dict)
    graphs: dict[str, str] = Field(default_factory=dict)


class GraphRef(BaseModel):
    """Resolved external graph reference for RFC-0028 snapshots."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str


class WorkflowBranch(BaseModel):
    """Conditional branch for an ``if_else`` node."""

    model_config = ConfigDict(extra="forbid")

    if_: str = Field(alias="if")
    next: str


class WorkflowNode(BaseModel):
    """Generic workflow node definition.

    RFC-0027 defines a discriminated set of node types. The first
    implementation keeps the YAML model compact and validates type-specific
    requirements in ``WorkflowConfig`` so all node definitions share a stable
    field vocabulary.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: WorkflowNodeType
    description: str | None = None

    output: JsonValue | None = None
    input: JsonValue | None = None
    update: JsonValue | None = None
    prompt: str | None = None

    agent: str | None = None
    tool: str | None = None
    server: str | None = None
    graph: str | None = None
    execution: SubgraphExecutionMode = "inline"

    output_schema: JsonObject | None = None
    output_mode: WorkflowOutputMode | None = None
    output_retries: int | None = Field(default=None, ge=0)
    output_name: str | None = None

    branches: list[WorkflowBranch] = Field(default_factory=_empty_branches)
    else_: str | None = Field(default=None, alias="else")

    condition: str | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    scope_key: str | None = None
    body: str | None = None
    items: JsonValue | None = None
    item_name: str = "item"
    index_name: str = "index"
    item_key: str | None = None
    max_concurrency: int | None = Field(default=None, ge=1)
    result_order: ParallelMapResultOrder = "input"
    failure_policy: ParallelMapFailurePolicy = "fail_fast"
    collect: JsonObject | None = None
    init: JsonValue | None = None
    state_in: JsonValue | None = None
    state_out: JsonValue | None = None

    side_effect: SideEffectPolicy = "read_only"
    idempotency_key: str | None = None
    retry_policy: RetryPolicy | None = None

    ui: JsonObject | None = None


class WorkflowConfig(BaseModel):
    """Top-level workflow configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["workflow"] = "workflow"
    version: str
    name: str
    description: str | None = None
    inputs: dict[str, WorkflowInputSpec] = Field(default_factory=dict)
    vars: JsonObject = Field(default_factory=_empty_json_object)
    durable: DurableConfig = Field(default_factory=DurableConfig)
    includes: WorkflowIncludes = Field(default_factory=_empty_includes)
    output: JsonValue | None = None
    output_schema: JsonObject | None = None
    nodes: dict[str, WorkflowNode] = Field(default_factory=_empty_nodes)
    edges: dict[str, str | list[str]] = Field(default_factory=_empty_edges)

    source_path: str | None = Field(default=None, exclude=True)
    included_graphs: dict[str, WorkflowConfig] = Field(default_factory=_empty_included_graphs, exclude=True, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowConfig:
        """Load and validate a workflow YAML file."""

        workflow_path = Path(path)
        return cls._from_yaml_path(workflow_path, root_max_depth=None, include_stack=())

    @classmethod
    def from_snapshot(cls, snapshot: JsonObject) -> WorkflowConfig:
        """Restore a workflow graph from a durable definition snapshot."""

        raw = dict(snapshot)
        raw_source_path = raw.pop("_source_path", None)
        raw_graphs = raw.pop("_included_graphs", {})
        config = cls.model_validate(raw)
        config.source_path = str(raw_source_path) if isinstance(raw_source_path, str) and raw_source_path else None
        if isinstance(raw_graphs, dict):
            for graph_name, graph_snapshot in raw_graphs.items():
                if isinstance(graph_snapshot, dict):
                    config.included_graphs[graph_name] = cls.from_snapshot(json_object(graph_snapshot, label=f"{graph_name} snapshot"))
        return config

    @classmethod
    def _from_yaml_path(
        cls,
        workflow_path: Path,
        *,
        root_max_depth: int | None,
        include_stack: tuple[Path, ...],
    ) -> WorkflowConfig:
        resolved_path = workflow_path.expanduser().resolve()
        if root_max_depth is not None and len(include_stack) > root_max_depth:
            raise WorkflowConfigError(f"Workflow subgraph include depth exceeds max_subgraph_depth={root_max_depth}: {resolved_path}")
        if resolved_path in include_stack:
            cycle = " -> ".join(str(item) for item in (*include_stack, resolved_path))
            raise WorkflowConfigError(f"Workflow subgraph includes contain recursion: {cycle}")
        if not resolved_path.exists():
            raise WorkflowConfigError(f"Workflow YAML file not found: {workflow_path}")
        with open(resolved_path, encoding="utf-8") as file_obj:
            raw = yaml.safe_load(file_obj)
        config = cls.model_validate(json_object(raw, label="workflow YAML"))
        config.source_path = str(resolved_path)
        effective_max_depth = config.durable.max_subgraph_depth if root_max_depth is None else root_max_depth
        config._resolve_included_graphs(root_max_depth=effective_max_depth, include_stack=(*include_stack, resolved_path))
        return config

    def definition_snapshot(self) -> JsonObject:
        """Return a JSON-safe workflow definition snapshot with expanded graph includes."""

        snapshot = json_object(self.model_dump(mode="json", by_alias=True, exclude={"source_path", "included_graphs"}))
        if self.source_path is not None:
            snapshot["_source_path"] = self.source_path
        if self.included_graphs:
            snapshot["_included_graphs"] = {
                name: graph.definition_snapshot() for name, graph in sorted(self.included_graphs.items(), key=lambda item: item[0])
            }
        return snapshot

    @property
    def start_node_id(self) -> str:
        """Return the unique start node id."""

        starts = [node_id for node_id, node in self.nodes.items() if node.type == "start"]
        if len(starts) != 1:
            raise WorkflowConfigError("Workflow must contain exactly one start node")
        return starts[0]

    @property
    def base_path(self) -> Path:
        """Directory used to resolve workflow includes."""

        if self.source_path is None:
            return Path.cwd()
        return Path(self.source_path).parent

    def graph_ref(self, graph_name: str) -> GraphRef:
        """Return the declared include reference for a subgraph."""

        graph_path = self.includes.graphs.get(graph_name)
        if graph_path is None:
            raise WorkflowConfigError(f"Subgraph reference not found in includes.graphs: {graph_name}")
        return GraphRef(name=graph_name, path=str((self.base_path / graph_path).expanduser().resolve()))

    @model_validator(mode="after")
    def _validate_workflow(self) -> Self:
        if not self.nodes:
            raise WorkflowConfigError("Workflow must define at least one node")

        start_count = sum(1 for node in self.nodes.values() if node.type == "start")
        if start_count != 1:
            raise WorkflowConfigError("Workflow must contain exactly one start node")

        self._validate_edges()
        self._validate_node_requirements()
        self._validate_output_schemas()
        self._validate_edge_cycles()
        return self

    def _resolve_included_graphs(self, *, root_max_depth: int, include_stack: tuple[Path, ...]) -> None:
        # RFC-0028: 解析外部 YAML 子图，确保启动 run 前图定义已经展开
        for graph_name, graph_path in self.includes.graphs.items():
            resolved_graph_path = (self.base_path / graph_path).expanduser().resolve()
            graph_config = self._from_yaml_path(
                resolved_graph_path,
                root_max_depth=root_max_depth,
                include_stack=include_stack,
            )
            self.included_graphs[graph_name] = graph_config

    def _validate_edges(self) -> None:
        for source, targets in self.edges.items():
            if source not in self.nodes:
                raise WorkflowConfigError(f"Edge source references unknown node: {source}")
            for target in self._iter_edge_targets(targets):
                if target not in self.nodes:
                    raise WorkflowConfigError(f"Edge target references unknown node: {target}")

        for node_id, node in self.nodes.items():
            if node.type == "if_else":
                for branch in node.branches:
                    if branch.next not in self.nodes:
                        raise WorkflowConfigError(f"if_else node {node_id!r} references unknown branch target: {branch.next}")
                if node.else_ is not None and node.else_ not in self.nodes:
                    raise WorkflowConfigError(f"if_else node {node_id!r} references unknown else target: {node.else_}")
            if node.type == "while" and node.body is not None and node.body not in self.nodes:
                raise WorkflowConfigError(f"while node {node_id!r} references unknown body node: {node.body}")
            if node.type == "parallel_map" and node.body is not None and node.body not in self.nodes:
                raise WorkflowConfigError(f"parallel_map node {node_id!r} references unknown body node: {node.body}")

    def _validate_node_requirements(self) -> None:
        default_uncertain = self.durable.default_retry_policy.on_uncertain
        for node_id, node in self.nodes.items():
            if node.type == "agent" and not node.agent:
                raise WorkflowConfigError(f"agent node {node_id!r} requires 'agent'")
            if node.type == "tool" and not node.tool:
                raise WorkflowConfigError(f"tool node {node_id!r} requires 'tool'")
            if node.type == "mcp" and (not node.server or not node.tool):
                raise WorkflowConfigError(f"mcp node {node_id!r} requires 'server' and 'tool'")
            if node.type == "human" and not node.prompt:
                raise WorkflowConfigError(f"human node {node_id!r} requires 'prompt'")
            if node.type == "subgraph":
                if not node.graph:
                    raise WorkflowConfigError(f"subgraph node {node_id!r} requires 'graph'")
                if node.graph not in self.includes.graphs:
                    raise WorkflowConfigError(f"subgraph node {node_id!r} references unknown includes.graphs key: {node.graph}")
            if node.type == "if_else" and not node.branches and node.else_ is None:
                raise WorkflowConfigError(f"if_else node {node_id!r} requires branches or else")
            if node.type == "while":
                if node.condition is None:
                    raise WorkflowConfigError(f"while node {node_id!r} requires 'condition'")
                if node.max_iterations is None:
                    raise WorkflowConfigError(f"while node {node_id!r} requires 'max_iterations'")
                if node.body is None:
                    raise WorkflowConfigError(f"while node {node_id!r} requires 'body'")
            if node.type == "parallel_map":
                self._validate_parallel_map_node(node_id, node)
            if node.side_effect in {"idempotent_write", "external_write"}:
                node_uncertain = node.retry_policy.on_uncertain if node.retry_policy is not None else None
                if not node.idempotency_key and not node_uncertain and not default_uncertain:
                    raise WorkflowConfigError(f"{node.side_effect} node {node_id!r} requires 'idempotency_key' or an on_uncertain policy")

    def _validate_parallel_map_node(self, node_id: str, node: WorkflowNode) -> None:
        if node.items is None:
            raise WorkflowConfigError(f"parallel_map node {node_id!r} requires 'items'")
        if node.body is None:
            raise WorkflowConfigError(f"parallel_map node {node_id!r} requires 'body'")
        body = self.nodes.get(node.body)
        if body is None:
            raise WorkflowConfigError(f"parallel_map node {node_id!r} references unknown body node: {node.body}")
        if body.type not in PARALLEL_MAP_BODY_NODE_TYPES:
            raise WorkflowConfigError(f"parallel_map node {node_id!r} body {node.body!r} has unsupported type: {body.type}")
        if self._node_has_incoming_edge(node.body):
            raise WorkflowConfigError(f"parallel_map node {node_id!r} body {node.body!r} must not be referenced by normal edges")
        if node.max_concurrency is not None and node.max_concurrency > self.durable.max_parallelism:
            raise WorkflowConfigError(f"parallel_map node {node_id!r} max_concurrency exceeds durable.max_parallelism")
        if node.item_name in PARALLEL_MAP_RESERVED_CONTEXT_NAMES:
            raise WorkflowConfigError(f"parallel_map node {node_id!r} item_name uses a reserved context name: {node.item_name}")
        if node.index_name in PARALLEL_MAP_RESERVED_CONTEXT_NAMES or node.index_name == node.item_name:
            raise WorkflowConfigError(
                f"parallel_map node {node_id!r} index_name uses an invalid or reserved context name: {node.index_name}"
            )
        if body.side_effect in {"idempotent_write", "external_write"} and node.item_key is None:
            raise WorkflowConfigError(f"parallel_map node {node_id!r} requires explicit item_key for side-effecting body {node.body!r}")

    def _node_has_incoming_edge(self, node_id: str) -> bool:
        for targets in self.edges.values():
            if node_id in self._iter_edge_targets(targets):
                return True
        return False

    def _validate_output_schemas(self) -> None:
        for node_id, node in self.nodes.items():
            if node.output_schema is None:
                continue
            schema = cast(dict[str, object], node.output_schema)
            try:
                validator_for(schema).check_schema(schema)
            except jsonschema.SchemaError as exc:
                raise WorkflowConfigError(f"Node {node_id!r} has invalid output_schema: {exc.message}") from exc
        if self.output_schema is not None:
            schema = cast(dict[str, object], self.output_schema)
            try:
                validator_for(schema).check_schema(schema)
            except jsonschema.SchemaError as exc:
                raise WorkflowConfigError(f"Workflow {self.name!r} has invalid output_schema: {exc.message}") from exc

    def _validate_edge_cycles(self) -> None:
        adjacency: dict[str, list[str]] = {node_id: list(self._iter_edge_targets(self.edges.get(node_id, []))) for node_id in self.nodes}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str, path: tuple[str, ...]) -> None:
            if node_id in visiting:
                cycle_path = " -> ".join((*path, node_id))
                raise WorkflowConfigError(f"Workflow edges contain a cycle: {cycle_path}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for target in adjacency[node_id]:
                visit(target, (*path, node_id))
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id, ())

    @staticmethod
    def _iter_edge_targets(targets: str | list[str] | Iterable[str]) -> Iterable[str]:
        if isinstance(targets, str):
            return (targets,)
        return tuple(targets)


type WorkflowGraphConfig = WorkflowConfig
