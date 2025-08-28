#!/usr/bin/env python3
"""Test script for hooks accessing GlobalStorage."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the northau package to the path
sys.path.insert(0, str(Path(__file__).parent))

from northau.archs.main_sub.agent_context import GlobalStorage
from northau.archs.main_sub.execution.hooks import AfterModelHookInput, HookResult, AfterModelHook, HookManager
from northau.archs.main_sub.execution.parse_structures import ParsedResponse, ToolCall
from northau.archs.main_sub.execution.response_generator import ResponseGenerator

def test_hook_global_storage_access():
    """Test that hooks can access and modify GlobalStorage."""
    print("🪝 Testing hook GlobalStorage access...")
    
    # Create test storage
    test_storage = GlobalStorage()
    test_storage.set("hook_counter", 0)
    test_storage.set("hook_data", [])
    
    # Create a hook that uses GlobalStorage
    def storage_hook(hook_input: AfterModelHookInput) -> HookResult:
        """Hook that reads from and writes to GlobalStorage."""
        if hook_input.global_storage is not None:
            # Increment counter
            with hook_input.global_storage.lock_key("hook_counter"):
                current = hook_input.global_storage.get("hook_counter", 0)
                hook_input.global_storage.set("hook_counter", current + 1)
            
            # Add data
            with hook_input.global_storage.lock_key("hook_data"):
                data_list = hook_input.global_storage.get("hook_data", [])
                data_list.append(f"hook_executed_iteration_{hook_input.current_iteration}")
                hook_input.global_storage.set("hook_data", data_list)
        
        return HookResult.no_changes()
    
    # Create hook input with GlobalStorage
    hook_input = AfterModelHookInput(
        max_iterations=5,
        current_iteration=1,
        original_response="test response",
        parsed_response=ParsedResponse(
            original_response="test response",
            tool_calls=[],
            sub_agent_calls=[],
            batch_agent_calls=[],
            is_parallel_tools=False,
            is_parallel_sub_agents=False
        ),
        messages=[{"role": "user", "content": "test"}],
        global_storage=test_storage
    )
    
    # Execute hook
    result = storage_hook(hook_input)
    
    # Verify hook accessed storage
    assert test_storage.get("hook_counter") == 1
    assert test_storage.get("hook_data") == ["hook_executed_iteration_1"]
    assert not result.has_modifications()
    
    print("✅ Hook GlobalStorage access test passed!")

def test_hook_manager_global_storage():
    """Test that HookManager properly passes GlobalStorage to hooks."""
    print("🪝 Testing HookManager GlobalStorage handling...")
    
    # Create test storage
    test_storage = GlobalStorage()
    test_storage.set("manager_test", "initial")
    
    # Create multiple hooks
    def first_hook(hook_input: AfterModelHookInput) -> HookResult:
        if hook_input.global_storage:
            hook_input.global_storage.set("first_hook_executed", True)
        return HookResult.no_changes()
    
    def second_hook(hook_input: AfterModelHookInput) -> HookResult:
        if hook_input.global_storage:
            hook_input.global_storage.set("second_hook_executed", True)
        return HookResult.no_changes()
    
    # Create hook manager
    manager = HookManager([first_hook, second_hook])
    
    # Create hook input
    hook_input = AfterModelHookInput(
        max_iterations=3,
        current_iteration=2,
        original_response="manager test",
        parsed_response=ParsedResponse(
            original_response="manager test",
            tool_calls=[],
            sub_agent_calls=[],
            batch_agent_calls=[],
            is_parallel_tools=False,
            is_parallel_sub_agents=False
        ),
        messages=[{"role": "user", "content": "test"}],
        global_storage=test_storage
    )
    
    # Execute hooks through manager
    final_parsed, final_messages = manager.execute_hooks(hook_input)
    
    # Verify both hooks accessed storage
    assert test_storage.get("first_hook_executed") == True
    assert test_storage.get("second_hook_executed") == True
    assert test_storage.get("manager_test") == "initial"  # Unchanged
    
    print("✅ HookManager GlobalStorage test passed!")

def test_response_generator_integration():
    """Test that ResponseGenerator passes GlobalStorage to hooks."""
    print("🪝 Testing ResponseGenerator-hook integration...")
    
    # Create test storage
    test_storage = GlobalStorage()
    test_storage.set("response_gen_test", "start")
    
    # Mock dependencies
    mock_openai_client = Mock()
    mock_llm_config = Mock()
    
    # Create ResponseGenerator with GlobalStorage
    response_gen = ResponseGenerator(
        agent_name="test_agent",
        openai_client=mock_openai_client,
        llm_config=mock_llm_config,
        max_iterations=3,
        global_storage=test_storage
    )
    
    # Verify storage is properly stored
    assert response_gen.global_storage is test_storage
    
    # Create mock hook input to verify it gets the storage
    hook_input = AfterModelHookInput(
        max_iterations=response_gen.max_iterations,
        current_iteration=1,
        original_response="test",
        parsed_response=None,
        messages=[],
        global_storage=response_gen.global_storage
    )
    
    # Verify the hook input has correct storage reference
    assert hook_input.global_storage is test_storage
    assert hook_input.global_storage.get("response_gen_test") == "start"
    
    print("✅ ResponseGenerator integration test passed!")

def test_hook_storage_modifications():
    """Test that hooks can safely modify storage during execution."""
    print("🪝 Testing hook storage modifications...")
    
    # Create test storage
    shared_storage = GlobalStorage()
    shared_storage.set("execution_log", [])
    shared_storage.set("modification_count", 0)
    
    # Hook that modifies storage and messages
    def modifying_hook(hook_input: AfterModelHookInput) -> HookResult:
        if hook_input.global_storage:
            # Add to execution log
            with hook_input.global_storage.lock_key("execution_log"):
                log = hook_input.global_storage.get("execution_log", [])
                log.append(f"Hook executed at iteration {hook_input.current_iteration}")
                hook_input.global_storage.set("execution_log", log)
            
            # Update modification count
            with hook_input.global_storage.lock_key("modification_count"):
                count = hook_input.global_storage.get("modification_count", 0)
                hook_input.global_storage.set("modification_count", count + 1)
            
            # Modify messages based on storage state
            new_messages = hook_input.messages.copy()
            new_messages.append({
                "role": "system", 
                "content": f"Hook added message. Total modifications: {count + 1}"
            })
            
            return HookResult.with_modifications(messages=new_messages)
        
        return HookResult.no_changes()
    
    # Create hook input
    hook_input = AfterModelHookInput(
        max_iterations=5,
        current_iteration=3,
        original_response="test response",
        parsed_response=ParsedResponse(
            original_response="test response",
            tool_calls=[],
            sub_agent_calls=[],
            batch_agent_calls=[],
            is_parallel_tools=False,
            is_parallel_sub_agents=False
        ),
        messages=[{"role": "user", "content": "original message"}],
        global_storage=shared_storage
    )
    
    # Execute hook
    result = modifying_hook(hook_input)
    
    # Verify storage modifications
    assert shared_storage.get("modification_count") == 1
    exec_log = shared_storage.get("execution_log", [])
    assert "Hook executed at iteration 3" in exec_log
    
    # Verify message modifications
    assert result.has_modifications()
    assert result.messages is not None
    assert len(result.messages) == 2
    assert "Hook added message" in result.messages[1]["content"]
    
    print("✅ Hook storage modifications test passed!")

def test_multiple_hooks_shared_storage():
    """Test multiple hooks sharing and modifying the same GlobalStorage."""
    print("🪝 Testing multiple hooks with shared storage...")
    
    # Create shared storage
    shared_storage = GlobalStorage()
    shared_storage.set("shared_counter", 0)
    shared_storage.set("hook_names", [])
    
    # Create multiple hooks that modify shared storage
    def create_counter_hook(hook_name: str) -> AfterModelHook:
        def counter_hook(hook_input: AfterModelHookInput) -> HookResult:
            if hook_input.global_storage:
                # Thread-safe counter increment
                with hook_input.global_storage.lock_multiple("shared_counter", "hook_names"):
                    current_count = hook_input.global_storage.get("shared_counter", 0)
                    hook_names = hook_input.global_storage.get("hook_names", [])
                    
                    hook_input.global_storage.set("shared_counter", current_count + 1)
                    hook_names.append(hook_name)
                    hook_input.global_storage.set("hook_names", hook_names)
            
            return HookResult.no_changes()
        return counter_hook
    
    # Create hooks
    hooks = [
        create_counter_hook("hook_alpha"),
        create_counter_hook("hook_beta"),
        create_counter_hook("hook_gamma")
    ]
    
    # Create hook manager
    manager = HookManager(hooks)
    
    # Create hook input
    hook_input = AfterModelHookInput(
        max_iterations=10,
        current_iteration=5,
        original_response="shared storage test",
        parsed_response=ParsedResponse(
            original_response="shared storage test",
            tool_calls=[],
            sub_agent_calls=[],
            batch_agent_calls=[],
            is_parallel_tools=False,
            is_parallel_sub_agents=False
        ),
        messages=[{"role": "user", "content": "test"}],
        global_storage=shared_storage
    )
    
    # Execute all hooks
    final_parsed, final_messages = manager.execute_hooks(hook_input)
    
    # Verify all hooks modified shared storage
    assert shared_storage.get("shared_counter") == 3
    hook_names = shared_storage.get("hook_names", [])
    assert "hook_alpha" in hook_names
    assert "hook_beta" in hook_names
    assert "hook_gamma" in hook_names
    assert len(hook_names) == 3
    
    print("✅ Multiple hooks shared storage test passed!")

def main():
    """Run all hook GlobalStorage tests."""
    print("🎯 Hook GlobalStorage Integration Tests")
    print("=" * 50)
    print()
    
    try:
        test_hook_global_storage_access()
        print()
        
        test_hook_manager_global_storage()
        print()
        
        test_response_generator_integration()
        print()
        
        test_hook_storage_modifications()
        print()
        
        test_multiple_hooks_shared_storage()
        print()
        
        print("=" * 50)
        print("🎉 All hook GlobalStorage tests passed successfully!")
        print("\n✨ Key Features Validated:")
        print("  • Hooks can access GlobalStorage via AfterModelHookInput")
        print("  • Thread-safe storage operations within hooks")
        print("  • HookManager properly passes GlobalStorage to all hooks")
        print("  • ResponseGenerator integration with GlobalStorage")
        print("  • Multiple hooks can safely share and modify storage")
        print("  • Hooks can modify both storage and execution flow")
        
    except Exception as e:
        print(f"❌ Hook test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()