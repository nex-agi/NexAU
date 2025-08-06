#!/usr/bin/env python3
"""Test script for agent context functionality."""

from northau.archs.main_sub.agent import create_agent
from northau.archs.tool.tool import Tool
from northau.archs.main_sub.agent_context import get_state, get_config, update_state, update_config


def test_tool_implementation(**params):
    """Test tool that uses the agent context."""
    print("=== Inside test_tool_implementation ===")
    
    # Get current state and config from context
    try:
        current_state = get_state()
        print(f"Current state: {current_state}")
        
        current_config = get_config()
        print(f"Current config: {current_config}")
        
        # Update state with query information
        update_state(
            query_count=current_state.get('query_count', 0) + 1,
            last_query=params.get('query', 'No query provided')
        )
        
        # Read updated state
        updated_state = get_state()
        print(f"Updated state: {updated_state}")
        
        # Use config values
        response_format = current_config.get('response_format', 'default')
        max_length = current_config.get('max_length', 100)
        
        return {
            "result": f"Processed query: {params.get('query', 'No query')}",
            "response_format": response_format,
            "max_length": max_length,
            "query_count": updated_state.get('query_count', 0),
            "state_keys": list(updated_state.keys()),
            "config_keys": list(current_config.keys())
        }
        
    except Exception as e:
        return {"error": f"Context access failed: {e}"}


def main():
    """Test the agent context system."""
    print("Testing Agent Context System")
    print("=" * 50)
    
    # Create a test tool
    test_tool = Tool(
        name="test_context_tool",
        description="A tool that demonstrates context usage",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to process"
                }
            },
            "required": ["query"]
        },
        implementation=test_tool_implementation
    )
    
    # Create agent with initial state and config
    agent = create_agent(
        name="test_agent",
        tools=[test_tool],
        initial_state={
            "session_id": "test_session_123",
            "user_id": "user_456",
            "query_count": 0
        },
        initial_config={
            "response_format": "json",
            "max_length": 500,
            "debug_mode": True,
            "api_version": "v1.0"
        },
        system_prompt="You are a test agent that demonstrates state and config management. Use the test_context_tool to process user queries."
    )
    
    print("\n1. Testing agent with initial context:")
    
    # Test basic functionality
    response1 = agent.run(
        message="Please use the test_context_tool with query: 'What is the weather today?'",
        state={"additional_info": "Runtime state addition"},
        config={"temp_setting": "enabled"}
    )
    
    print(f"Response 1: {response1}")
    
    print("\n" + "=" * 50)
    print("\n2. Testing agent with updated context:")
    
    # Test with different state/config
    response2 = agent.run(
        message="Use test_context_tool again with query: 'How are you doing?'",
        state={"session_id": "updated_session_789", "additional_info": "Second call"},
        config={"response_format": "xml", "max_length": 200}
    )
    
    print(f"Response 2: {response2}")
    
    print("\n" + "=" * 50)
    print("✅ Agent context test completed!")


if __name__ == "__main__":
    main()