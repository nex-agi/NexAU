"""Tool implementation for the Northau framework."""

import functools
import inspect
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import jsonschema
import yaml
from diskcache import Cache

from northau.archs.main_sub.agent_state import AgentState

cache = Cache("./.tool_cache")


def cache_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(func, "__self__"):
            self = func.__self__
            method = f"{self.__class__.__name__}."
        else:
            method = ""
        method += func.__name__

        args = [arg for arg in args if not isinstance(arg, AgentState)]

        kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, AgentState)}

        key = json.dumps(
            {"method": method, "args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str,
            ensure_ascii=False,
        )

        result = cache.get(key)
        if result is None:
            # 只有缓存中没有时才执行函数
            result = func(*args, **kwargs)
            cache.set(key, result)

        return result

    return wrapper


class Tool:
    """Tool class that represents a callable function with schema validation."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        implementation: Callable,
        use_cache: bool = False,
        disable_parallel: bool = False,
        template_override: str | None = None,
        timeout: int | None = None,
    ):
        """Initialize a tool with schema and implementation."""
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.implementation = implementation
        self.template_override = template_override
        self.timeout = timeout
        self.disable_parallel = disable_parallel

        if use_cache:
            self.implementation = cache_result(self.implementation)

        # Validate schema
        self._validate_schema()

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        binding: Callable,
        **kwargs,
    ) -> "Tool":
        """Load tool definition from YAML file and bind to implementation."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Tool YAML file not found: {yaml_path}")

        with open(path) as f:
            tool_def = yaml.safe_load(f)

        # Extract required fields
        name = tool_def.get("name")
        description = tool_def.get("description", "")
        input_schema = tool_def.get("input_schema", {})
        use_cache = tool_def.get("use_cache", False)
        disable_parallel = tool_def.get("disable_parallel", False)

        if "global_storage" in input_schema:
            raise ValueError(
                f"Tool definition of `{name}` contains 'global_storage' field in {yaml_path}, "
                "which will be injected by the framework, please remove it from the tool definition.",
            )

        if "agent_state" in input_schema:
            raise ValueError(
                f"Tool definition of `{name}` contains 'agent_state' field in {yaml_path}, "
                "which will be injected by the framework, please remove it from the tool definition."
            )

        template_override = tool_def.get("template_override", None)

        if not name:
            raise ValueError(
                f"Tool definition missing 'name' field in {yaml_path}",
            )

        # Create tool instance
        return cls(
            name=name,
            description=description,
            input_schema=input_schema,
            implementation=binding,
            use_cache=use_cache,
            disable_parallel=disable_parallel,
            template_override=template_override,
            **kwargs,
        )

    def execute(self, **params) -> dict:
        """Execute the tool with given parameters."""
        # Handle agent_state parameter
        filtered_params = params.copy()
        if "agent_state" in params:
            # Check if the function signature accepts agent_state
            sig = inspect.signature(self.implementation)
            if "agent_state" not in sig.parameters:
                # Remove agent_state if function doesn't accept it
                filtered_params.pop("agent_state", None)

                # For backwards compatibility, check if function accepts global_storage
                if "global_storage" in sig.parameters:
                    agent_state = params["agent_state"]
                    filtered_params["global_storage"] = agent_state.global_storage

        # Validate parameters (excluding agent_state and global_storage for schema validation)
        validation_params = {k: v for k, v in filtered_params.items() if k not in ("agent_state", "global_storage")}
        if not self.validate_params(validation_params):
            raise ValueError(
                f"Invalid parameters for tool '{self.name}': {validation_params}",
            )

        try:
            # Execute the implementation
            result = self.implementation(**filtered_params)

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"result": result}

            return result

        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "tool_name": self.name,
            }

    def validate_params(self, params: dict) -> bool:
        """Validate parameters against schema.

        Only validates parameters that are defined in the schema.
        Extra parameters (injected by hooks or with default values) are ignored.
        """
        # Extract only the parameters that are defined in the schema
        schema_properties = self.input_schema.get("properties", {})
        schema_params = {k: v for k, v in params.items() if k in schema_properties}

        try:
            # Validate only the schema-defined parameters
            jsonschema.validate(schema_params, self.input_schema)
            return True
        except jsonschema.ValidationError as e:
            print(
                f"Invalid parameters for tool '{self.name}': {schema_params}, error: {e}",
            )
            return False

    def _validate_schema(self):
        """Validate that the input schema is valid JSON Schema."""
        try:
            # Check if it's a valid JSON Schema
            jsonschema.validators.validator_for(
                self.input_schema,
            ).check_schema(self.input_schema)
        except jsonschema.SchemaError as e:
            raise ValueError(
                f"Invalid JSON Schema for tool '{self.name}': {e}",
            )

    def get_schema(self) -> dict:
        """Get the tool's input schema."""
        return self.input_schema.copy()

    def get_info(self) -> dict:
        """Get tool information."""
        return {
            "name": self.name,
            "template_override": self.template_override,
            "description": self.description,
            "input_schema": self.input_schema,
            "timeout": self.timeout,
        }

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', implementation={self.implementation.__name__})"

    def __str__(self) -> str:
        return f"Tool '{self.name}': {self.description[:50]}{'...' if len(self.description) > 50 else ''}"
