# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool implementation for the NexAU framework."""

import functools
import inspect
import json
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from diskcache import Cache
from pydantic import BaseModel, ConfigDict, Field

from nexau.archs.main_sub.agent_state import AgentState

logger = logging.getLogger(__name__)

cache = Cache("./.tool_cache")


class ToolYamlSchema(BaseModel):
    """Schema describing a tool YAML definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    skill_description: str | None = None
    use_cache: bool = False
    disable_parallel: bool = False
    template_override: str | None = None
    timeout: int | None = Field(default=None, gt=0)
    builtin: str | None = None


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


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class Tool:
    """Tool class that represents a callable function with schema validation."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        implementation: Callable | str | None,
        skill_description: str | None = None,
        as_skill: bool = False,
        use_cache: bool = False,
        disable_parallel: bool = False,
        template_override: str | None = None,
        timeout: int | None = None,
    ):
        """Initialize a tool with schema and implementation."""
        self.name = name
        self.description = description
        self.skill_description = skill_description
        self.as_skill = as_skill
        self.input_schema = input_schema
        self.implementation = None
        self.implementation_import_path = None
        if isinstance(implementation, str):
            self.implementation_import_path = implementation
            self.implementation = None  # lazy import and bind at runtime
        else:
            self.implementation = implementation
        self.template_override = template_override
        self.timeout = timeout
        self.disable_parallel = disable_parallel
        self.use_cache = use_cache

        # Validate schema
        self._validate_schema()

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        binding: Callable | str | None,
        as_skill: bool = False,
        **kwargs,
    ) -> "Tool":
        """Load tool definition from YAML file and bind to implementation."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Tool YAML file not found: {yaml_path}")

        with open(path) as f:
            tool_def_model = ToolYamlSchema.model_validate(yaml.safe_load(f))
        tool_def = tool_def_model.model_dump()

        # Extract required fields
        name = tool_def["name"]
        description = tool_def["description"]
        skill_description = tool_def.get("skill_description", "")
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

        template_override = tool_def.get("template_override")
        timeout = tool_def.get("timeout")

        # Create tool instance
        return cls(
            name=name,
            description=description,
            skill_description=skill_description,
            input_schema=input_schema,
            implementation=binding,
            as_skill=as_skill,
            use_cache=use_cache,
            disable_parallel=disable_parallel,
            template_override=template_override,
            timeout=timeout,
            **kwargs,
        )

    def execute(self, **params) -> dict:
        """Execute the tool with given parameters."""

        if self.implementation is None:
            if self.implementation_import_path:
                logger.info(f"Dynamic importing tool implementation '{self.name}': {self.implementation_import_path}")

                from nexau.archs.config.config_loader import import_from_string

                func = import_from_string(str(self.implementation_import_path))

                if self.use_cache:
                    func = cache_result(func)
                self.implementation = func
            else:
                raise ValueError(f"Tool '{self.name}' has no implementation")

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
            "skill_description": self.skill_description,
            "input_schema": self.input_schema,
            "timeout": self.timeout,
        }

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', implementation={self.implementation.__name__})"

    def __str__(self) -> str:
        tool_str = f"Tool '{self.name}': {self.description[:50]}{'...' if len(self.description) > 50 else ''}"
        if self.skill_description:
            tool_str += f"\nSkill description: {self.skill_description}"
        return tool_str
