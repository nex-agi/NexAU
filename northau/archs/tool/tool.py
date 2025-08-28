"""Tool implementation for the Northau framework."""

import yaml
import json
import jsonschema
from typing import Dict, Callable, Optional, Any
from pathlib import Path


class Tool:
    """Tool class that represents a callable function with schema validation."""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict,
        implementation: Callable,
        content: Optional[str] = None,
        cache_results: bool = False,
        timeout: Optional[int] = None
    ):
        """Initialize a tool with schema and implementation."""
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.implementation = implementation
        self.content = content
        self.cache_results = cache_results
        self.timeout = timeout
        
        # Cache for results if enabled
        self._cache = {} if cache_results else None
        
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
        content = tool_def.get('content', None)
        
        if not name:
            raise ValueError(f"Tool definition missing 'name' field in {yaml_path}")
        
        # Create tool instance
        return cls(
            name=name,
            description=description,
            input_schema=input_schema,
            implementation=binding,
            content=content,
            **kwargs
        )
    
    def execute(self, **params) -> Dict:
        """Execute the tool with given parameters."""
        # Validate parameters
        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for tool '{self.name}': {params}")
        
        # Check cache if enabled
        if self.cache_results and self._cache is not None:
            cache_key = self._generate_cache_key(params)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        try:
            # Execute the implementation
            result = self.implementation(**params)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"result": result}
            
            # Cache result if enabled
            if self.cache_results and self._cache is not None:
                cache_key = self._generate_cache_key(params)
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "tool_name": self.name
            }
    
    def validate_params(self, params: Dict) -> bool:
        """Validate parameters against schema."""
        try:
            jsonschema.validate(params, self.input_schema)
            return True
        except jsonschema.ValidationError as e:
            print(f"Invalid parameters for tool '{self.name}': {params}, error: {e}")
            return False
    
    def _validate_schema(self):
        """Validate that the input schema is valid JSON Schema."""
        try:
            # Check if it's a valid JSON Schema
            jsonschema.validators.validator_for(self.input_schema).check_schema(self.input_schema)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid JSON Schema for tool '{self.name}': {e}")
    
    def _generate_cache_key(self, params: Dict) -> str:
        """Generate a cache key from parameters."""
        # Sort parameters to ensure consistent keys
        sorted_params = json.dumps(params, sort_keys=True)
        return f"{self.name}:{sorted_params}"
    
    def clear_cache(self):
        """Clear the tool's cache."""
        if self._cache is not None:
            self._cache.clear()
    
    def get_schema(self) -> Dict:
        """Get the tool's input schema."""
        return self.input_schema.copy()
    
    def get_info(self) -> Dict:
        """Get tool information."""
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "input_schema": self.input_schema,
            "cache_enabled": self.cache_results,
            "timeout": self.timeout
        }
    
    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', implementation={self.implementation.__name__})"
    
    def __str__(self) -> str:
        return f"Tool '{self.name}': {self.description[:50]}{'...' if len(self.description) > 50 else ''}"