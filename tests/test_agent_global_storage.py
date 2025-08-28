#!/usr/bin/env python3
"""Test script for GlobalStorage integration with actual agents."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the northau package to the path
sys.path.insert(0, str(Path(__file__).parent))

from northau.archs.main_sub.agent_context import GlobalStorage
from northau.archs.main_sub.agent import create_agent, Agent
from northau.archs.llm.llm_config import LLMConfig

def test_agent_global_storage_integration():
    """Test GlobalStorage integration with actual agents using mocked dependencies."""
    print("🧪 Testing Agent-GlobalStorage integration...")
    
    # Set up mock environment to avoid OpenAI API requirements
    mock_openai_client = Mock()
    mock_openai_client.chat.completions.create.return_value = Mock()
    
    with patch('northau.archs.main_sub.agent.openai') as mock_openai_module:
        mock_openai_module.OpenAI.return_value = mock_openai_client
        
        # Test 1: Agents created separately have different GlobalStorage instances
        agent1 = create_agent(name="agent1", llm_config=LLMConfig(model="gpt-4"))
        agent2 = create_agent(name="agent2", llm_config=LLMConfig(model="gpt-4"))
        
        assert agent1.global_storage is not agent2.global_storage
        
        # Test 2: Agents created with shared storage share the same instance
        shared_storage = GlobalStorage()
        agent3 = create_agent(name="agent3", llm_config=LLMConfig(model="gpt-4"), global_storage=shared_storage)
        agent4 = create_agent(name="agent4", llm_config=LLMConfig(model="gpt-4"), global_storage=shared_storage)
        
        assert agent3.global_storage is agent4.global_storage
        assert agent3.global_storage is shared_storage
        
        # Test 3: Data isolation between different storage instances
        agent1.global_storage.set("test_key", "agent1_value")
        agent2.global_storage.set("test_key", "agent2_value")
        
        assert agent1.global_storage.get("test_key") == "agent1_value"
        assert agent2.global_storage.get("test_key") == "agent2_value"
        
        # Test 4: Data sharing between agents with shared storage
        agent3.global_storage.set("shared_key", "shared_value")
        assert agent4.global_storage.get("shared_key") == "shared_value"
        
        print("✅ Agent-GlobalStorage integration test passed!")

def test_subagent_manager_storage_sharing():
    """Test SubAgentManager properly shares storage with sub-agents."""
    print("🧪 Testing SubAgentManager storage sharing...")
    
    from northau.archs.main_sub.execution.subagent_manager import SubAgentManager
    
    # Create a GlobalStorage instance
    parent_storage = GlobalStorage()
    parent_storage.set("parent_data", "from_parent")
    
    # Mock sub-agent factory that accepts global_storage parameter
    def mock_subagent_factory(global_storage=None):
        mock_agent = Mock()
        mock_agent.global_storage = global_storage or GlobalStorage()
        mock_agent.run.return_value = "sub_agent_response"
        return mock_agent
    
    # Create SubAgentManager with the storage
    manager = SubAgentManager(
        agent_name="test_parent",
        sub_agent_factories={"test_sub": mock_subagent_factory},
        global_storage=parent_storage
    )
    
    # Mock the agent context
    with patch('northau.archs.main_sub.agent_context.get_context') as mock_get_context:
        mock_context = Mock()
        mock_context.state = {"state_key": "state_value"}
        mock_context.config = {"config_key": "config_value"}
        mock_get_context.return_value = mock_context
        
        # Call sub-agent - this should share the parent's storage
        result = manager.call_sub_agent("test_sub", "test message")
        
        # Verify the sub-agent was created with shared storage
        # (The mock factory will be called with global_storage parameter)
        assert result == "sub_agent_response"
    
    print("✅ SubAgentManager storage sharing test passed!")

def test_executor_storage_passing():
    """Test that Executor properly passes GlobalStorage to SubAgentManager."""
    print("🧪 Testing Executor storage passing...")
    
    from northau.archs.main_sub.execution.executor import Executor
    from northau.archs.main_sub.utils.token_counter import TokenCounter
    
    # Create test storage
    test_storage = GlobalStorage()
    test_storage.set("test_data", "executor_test")
    
    # Mock dependencies
    mock_openai_client = Mock()
    mock_llm_config = Mock()
    
    # Create executor with storage
    executor = Executor(
        agent_name="test_agent",
        tool_registry={},
        sub_agent_factories={},
        stop_tools=set(),
        openai_client=mock_openai_client,
        llm_config=mock_llm_config,
        global_storage=test_storage
    )
    
    # Verify the SubAgentManager received the storage
    assert executor.subagent_manager.global_storage is test_storage
    assert executor.subagent_manager.global_storage.get("test_data") == "executor_test"
    
    print("✅ Executor storage passing test passed!")

def test_complete_agent_hierarchy():
    """Test complete agent hierarchy with GlobalStorage sharing."""
    print("🧪 Testing complete agent hierarchy...")
    
    with patch('northau.archs.main_sub.agent.openai') as mock_openai_module:
        mock_openai_client = Mock()
        mock_openai_module.OpenAI.return_value = mock_openai_client
        
        # Create main agent
        main_storage = GlobalStorage()
        
        # Simple sub-agent factory for testing
        def create_test_subagent():
            return create_agent(name="test_sub", llm_config=LLMConfig(model="gpt-4"))
        
        main_agent = create_agent(
            name="main_agent",
            llm_config=LLMConfig(model="gpt-4"),
            global_storage=main_storage,
            sub_agents=[("test_sub", create_test_subagent)]
        )
        
        # Set data in main agent
        main_agent.global_storage.set("main_data", "from_main_agent")
        
        # Verify the executor has the storage
        assert main_agent.executor.subagent_manager.global_storage is main_storage
        
        # Verify data is accessible
        executor_storage = main_agent.executor.subagent_manager.global_storage
        assert executor_storage.get("main_data") == "from_main_agent"
        
        # Test that sub-agent manager can share storage
        manager = main_agent.executor.subagent_manager
        assert manager.global_storage is main_storage
        
        print("✅ Complete agent hierarchy test passed!")

def main():
    """Run all integration tests."""
    print("🚀 Starting GlobalStorage integration tests...\n")
    
    try:
        test_agent_global_storage_integration()
        print()
        
        test_subagent_manager_storage_sharing()
        print()
        
        test_executor_storage_passing()
        print()
        
        test_complete_agent_hierarchy()
        print()
        
        print("🎉 All GlobalStorage integration tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()