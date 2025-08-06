#!/usr/bin/env python3
"""Test agent context with actual tool execution."""

from northau.archs.main_sub.agent import create_agent
from northau.archs.tool.tool import Tool
from northau.archs.main_sub.agent_context import (
    AgentContext, 
    get_state, 
    get_config, 
    update_state, 
    get_state_value,
    set_state_value
)


def query_processor_tool(**params):
    """Tool that processes queries using context."""
    query = params.get('query', '')
    
    # Get context
    state = get_state()
    config = get_config()
    
    # Update query count
    current_count = state.get('query_count', 0)
    update_state(
        query_count=current_count + 1,
        last_query=query,
        last_processed_at=f"step_{current_count + 1}"
    )
    
    # Use config for processing
    max_length = config.get('max_length', 100)
    debug_mode = config.get('debug_mode', False)
    
    processed_query = query[:max_length] if len(query) > max_length else query
    
    result = {
        "original_query": query,
        "processed_query": processed_query,
        "query_count": get_state_value('query_count'),
        "user_id": get_state_value('user_id', 'anonymous'),
        "session_id": get_state_value('session_id', 'no_session'),
        "truncated": len(query) > max_length,
        "debug_info": get_state() if debug_mode else "debug_disabled"
    }
    
    return result


def user_tracker_tool(**params):
    """Tool that tracks user information using context."""
    action = params.get('action', 'view')
    
    state = get_state()
    config = get_config()
    
    # Update user activity
    activities = state.get('user_activities', [])
    activities.append(action)
    
    update_state(
        user_activities=activities,
        last_activity=action,
        total_actions=len(activities)
    )
    
    # Use config for response formatting
    response_format = config.get('response_format', 'json')
    
    result = {
        "action_recorded": action,
        "total_actions": len(activities),
        "recent_activities": activities[-3:],  # Last 3 activities
        "user_id": get_state_value('user_id'),
        "format": response_format
    }
    
    return result


def main():
    """Test context with tools."""
    print("Testing Agent Context with Tools")
    print("=" * 50)
    
    # Create tools
    query_tool = Tool(
        name="query_processor",
        description="Processes user queries with context awareness",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to process"}
            },
            "required": ["query"]
        },
        implementation=query_processor_tool
    )
    
    tracker_tool = Tool(
        name="user_tracker", 
        description="Tracks user activities with context",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "The action to track"}
            },
            "required": ["action"]
        },
        implementation=user_tracker_tool
    )
    
    # Create agent
    agent = create_agent(
        name="context_aware_agent",
        tools=[query_tool, tracker_tool],
        initial_state={
            "user_id": "user_123",
            "session_id": "session_456",
            "query_count": 0,
            "user_activities": []
        },
        initial_config={
            "max_length": 50,
            "debug_mode": True,
            "response_format": "json",
            "log_level": "info"
        }
    )
    
    # Test 1: Direct tool execution with context
    print("\n1. Testing direct tool execution with context:")
    
    with AgentContext(state=agent.initial_state, config=agent.initial_config):
        print("Current state:", get_state())
        print("Current config:", get_config())
        
        # Test query processor tool
        result1 = agent.tool_registry["query_processor"].execute(
            query="This is a long query that should be truncated based on config settings"
        )
        print(f"\nQuery processor result: {result1}")
        
        # Test user tracker tool
        result2 = agent.tool_registry["user_tracker"].execute(action="search")
        print(f"\nUser tracker result: {result2}")
        
        # Test another query
        result3 = agent.tool_registry["query_processor"].execute(
            query="Short query"
        )
        print(f"\nSecond query result: {result3}")
        
        print(f"\nFinal state: {get_state()}")
    
    print("\n" + "=" * 50)
    
    # Test 2: Running agent with runtime state/config
    print("\n2. Testing agent.run with runtime state/config:")
    
    # The agent should use context internally during tool execution
    # This would work with LLM integration, but in fallback mode we can't see tool execution
    
    try:
        response = agent.run(
            message="Process this query and track user activity",
            state={"additional_context": "runtime_data", "priority": "high"},
            config={"max_length": 25, "debug_mode": False}
        )
        print(f"Agent response: {response}")
    except Exception as e:
        print(f"Agent run error (expected in fallback mode): {e}")
    
    print("\n✅ Context with tools test completed!")


if __name__ == "__main__":
    main()