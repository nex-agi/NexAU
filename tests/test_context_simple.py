#!/usr/bin/env python3
"""Simple test for agent context functionality."""

from northau.archs.main_sub.agent_context import (
    AgentContext, 
    get_state, 
    get_config, 
    update_state, 
    update_config,
    get_state_value,
    set_state_value
)


def tool_function(**params):
    """Example tool function that uses context."""
    print("=== Inside tool_function ===")
    
    try:
        # Get current context
        state = get_state()
        config = get_config()
        
        print(f"State: {state}")
        print(f"Config: {config}")
        
        # Update state
        query = params.get('query', 'No query')
        update_state(
            query_count=state.get('query_count', 0) + 1,
            last_query=query
        )
        
        # Use config values
        max_length = config.get('max_length', 100)
        debug_mode = config.get('debug_mode', False)
        
        # Get specific state value
        user_id = get_state_value('user_id', 'unknown')
        
        # Set a new state value
        set_state_value('last_tool_call', 'tool_function')
        
        return {
            "processed_query": query,
            "user_id": user_id,
            "query_count": get_state_value('query_count'),
            "max_length": max_length,
            "debug_mode": debug_mode,
            "final_state": get_state()
        }
        
    except Exception as e:
        return {"error": str(e)}


def main():
    """Test the context system directly."""
    print("Testing Agent Context System (Direct)")
    print("=" * 50)
    
    # Test 1: Basic context usage
    print("\n1. Testing basic context usage:")
    
    initial_state = {
        "session_id": "session_123",
        "user_id": "user_456",
        "query_count": 0
    }
    
    initial_config = {
        "max_length": 500,
        "debug_mode": True,
        "api_version": "v1.0"
    }
    
    with AgentContext(state=initial_state, config=initial_config):
        result1 = tool_function(query="What is AI?")
        print(f"Result 1: {result1}")
    
    print("\n" + "=" * 50)
    
    # Test 2: Nested contexts
    print("\n2. Testing nested contexts:")
    
    outer_state = {"level": "outer", "counter": 1}
    outer_config = {"mode": "outer_mode"}
    
    with AgentContext(state=outer_state, config=outer_config):
        print(f"Outer context - State: {get_state()}")
        
        inner_state = {"level": "inner", "counter": 2}
        inner_config = {"mode": "inner_mode"}
        
        with AgentContext(state=inner_state, config=inner_config):
            print(f"Inner context - State: {get_state()}")
            result2 = tool_function(query="Nested test")
            print(f"Result 2: {result2}")
        
        print(f"Back to outer context - State: {get_state()}")
    
    print("\n" + "=" * 50)
    
    # Test 3: Context outside of with statement
    print("\n3. Testing context access outside with statement:")
    
    try:
        state = get_state()
        print("This should not work!")
    except RuntimeError as e:
        print(f"Expected error: {e}")
    
    print("\n✅ Direct context test completed!")


if __name__ == "__main__":
    main()