"""Tool implementation for the Northau framework."""

import yaml
import jsonschema
import inspect
import traceback
from typing import Dict, Callable, Optional
from pathlib import Path


class Tool:
    """Tool class that represents a callable function with schema validation."""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict,
        implementation: Callable,
        template_override: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """Initialize a tool with schema and implementation."""
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.implementation = implementation
        self.template_override = template_override
        self.timeout = timeout
        
        # Validate schema
        self._validate_schema()
    
    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        binding: Callable,
        **kwargs
    ) -> 'Tool':
        """Load tool definition from YAML file and bind to implementation."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Tool YAML file not found: {yaml_path}")
        
        with open(path, 'r') as f:
            tool_def = yaml.safe_load(f)
        
        # Extract required fields
        name = tool_def.get('name')
        description = tool_def.get('description', '')
        input_schema = tool_def.get('input_schema', {})
        
        if "global_storage" in input_schema:
            raise ValueError(f"Tool definition of `{name}` contains 'global_storage' field in {yaml_path}, which will be injected by the framework, please remove it from the tool definition.")
        
        template_override = tool_def.get('template_override', None)
        
        if not name:
            raise ValueError(f"Tool definition missing 'name' field in {yaml_path}")
        
        # Create tool instance
        return cls(
            name=name,
            description=description,
            input_schema=input_schema,
            implementation=binding,
            template_override=template_override,
            **kwargs
        )
    
    def execute(self, **params) -> Dict:
        """Execute the tool with given parameters."""
        # Handle global_storage parameter
        filtered_params = params.copy()
        if 'global_storage' in params:
            # Check if the function signature accepts global_storage
            sig = inspect.signature(self.implementation)
            if 'global_storage' not in sig.parameters:
                # Remove global_storage if function doesn't accept it
                filtered_params.pop('global_storage', None)
        
        # Validate parameters (excluding global_storage for schema validation)
        validation_params = {k: v for k, v in filtered_params.items() if k != 'global_storage'}
        if not self.validate_params(validation_params):
            raise ValueError(f"Invalid parameters for tool '{self.name}': {validation_params}")
        
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
                "tool_name": self.name
            }
    
    def validate_params(self, params: Dict) -> bool:
        """Validate parameters against schema.
        
        Only validates parameters that are defined in the schema.
        Extra parameters (injected by hooks or with default values) are ignored.
        """
        # Extract only the parameters that are defined in the schema
        schema_properties = self.input_schema.get('properties', {})
        schema_params = {k: v for k, v in params.items() if k in schema_properties}
        
        try:
            # Validate only the schema-defined parameters
            jsonschema.validate(schema_params, self.input_schema)
            return True
        except jsonschema.ValidationError as e:
            print(f"Invalid parameters for tool '{self.name}': {schema_params}, error: {e}")
            return False
    
    def _validate_schema(self):
        """Validate that the input schema is valid JSON Schema."""
        try:
            # Check if it's a valid JSON Schema
            jsonschema.validators.validator_for(self.input_schema).check_schema(self.input_schema)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid JSON Schema for tool '{self.name}': {e}")
    
    
    def get_schema(self) -> Dict:
        """Get the tool's input schema."""
        return self.input_schema.copy()
    
    def get_info(self) -> Dict:
        """Get tool information."""
        return {
            "name": self.name,
            "template_override": self.template_override,
            "description": self.description,
            "input_schema": self.input_schema,
            "timeout": self.timeout
        }
    
    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', implementation={self.implementation.__name__})"
    
    def __str__(self) -> str:
        return f"Tool '{self.name}': {self.description[:50]}{'...' if len(self.description) > 50 else ''}"