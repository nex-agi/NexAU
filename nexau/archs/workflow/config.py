"""Workflow YAML configuration models.

RFC-0027: Workflow DSL 与配置解析

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
]
SideEffectPolicy = Literal["read_only", "idempotent_write", "external_write", "local_write"]
WorkflowOutputMode = Literal["auto", "native", "complete_task", "json_block"]


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
    init: JsonValue | None = None

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
    nodes: dict[str, WorkflowNode] = Field(default_factory=_empty_nodes)
    edges: dict[str, str | list[str]] = Field(default_factory=_empty_edges)

    source_path: str | None = Field(default=None, exclude=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowConfig:
        """Load and validate a workflow YAML file."""

        workflow_path = Path(path)
        if not workflow_path.exists():
            raise WorkflowConfigError(f"Workflow YAML file not found: {workflow_path}")
        with open(workflow_path, encoding="utf-8") as file_obj:
            raw = yaml.safe_load(file_obj)
        config = cls.model_validate(json_object(raw, label="workflow YAML"))
        config.source_path = str(workflow_path)
        return config

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
            if node.type == "if_else" and not node.branches and node.else_ is None:
                raise WorkflowConfigError(f"if_else node {node_id!r} requires branches or else")
            if node.type == "while":
                if node.condition is None:
                    raise WorkflowConfigError(f"while node {node_id!r} requires 'condition'")
                if node.max_iterations is None:
                    raise WorkflowConfigError(f"while node {node_id!r} requires 'max_iterations'")
                if node.body is None:
                    raise WorkflowConfigError(f"while node {node_id!r} requires 'body'")
            if node.side_effect in {"idempotent_write", "external_write"}:
                node_uncertain = node.retry_policy.on_uncertain if node.retry_policy is not None else None
                if not node.idempotency_key and not node_uncertain and not default_uncertain:
                    raise WorkflowConfigError(f"{node.side_effect} node {node_id!r} requires 'idempotency_key' or an on_uncertain policy")

    def _validate_output_schemas(self) -> None:
        for node_id, node in self.nodes.items():
            if node.output_schema is None:
                continue
            schema = cast(dict[str, object], node.output_schema)
            try:
                validator_for(schema).check_schema(schema)
            except jsonschema.SchemaError as exc:
                raise WorkflowConfigError(f"Node {node_id!r} has invalid output_schema: {exc.message}") from exc

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
