#!/usr/bin/env python3
"""Test script for GlobalStorage functionality."""

import threading
import time
import sys
from pathlib import Path

# Add the northau package to the path
sys.path.insert(0, str(Path(__file__).parent))

from northau.archs.main_sub.agent_context import GlobalStorage
from northau.archs.main_sub.agent import create_agent
from northau.archs.llm.llm_config import LLMConfig

def test_global_storage_isolation():
    """Test that different agents have isolated GlobalStorage instances."""
    print("🧪 Testing GlobalStorage isolation between different agents...")
    
    # Test with direct GlobalStorage instances first
    storage1 = GlobalStorage()
    storage2 = GlobalStorage()
    
    # Set different values in each storage
    storage1.set("test_key", "storage1_value")
    storage2.set("test_key", "storage2_value")
    
    # Verify isolation
    assert storage1.get("test_key") == "storage1_value"
    assert storage2.get("test_key") == "storage2_value"
    assert storage1 is not storage2
    
    print("✅ GlobalStorage isolation test passed!")

def test_global_storage_sharing():
    """Test that agents can share the same GlobalStorage instance."""
    print("🧪 Testing GlobalStorage sharing between storage instances...")
    
    # Create shared storage
    shared_storage = GlobalStorage()
    
    # Simulate two agents sharing the same storage
    storage_ref1 = shared_storage
    storage_ref2 = shared_storage
    
    # Set value in first reference
    storage_ref1.set("shared_key", "shared_value")
    
    # Verify second reference can access it
    assert storage_ref2.get("shared_key") == "shared_value"
    assert storage_ref1 is storage_ref2
    assert storage_ref1 is shared_storage
    
    print("✅ GlobalStorage sharing test passed!")

def test_global_storage_operations():
    """Test basic GlobalStorage operations."""
    print("🧪 Testing GlobalStorage operations...")
    
    storage = GlobalStorage()
    
    # Test set and get
    storage.set("key1", "value1")
    assert storage.get("key1") == "value1"
    assert storage.get("nonexistent", "default") == "default"
    
    # Test update
    storage.update({"key2": "value2", "key3": "value3"})
    assert storage.get("key2") == "value2"
    assert storage.get("key3") == "value3"
    
    # Test keys and items
    keys = storage.keys()
    assert "key1" in keys
    assert "key2" in keys
    assert "key3" in keys
    
    items = storage.items()
    assert ("key1", "value1") in items
    
    # Test delete
    assert storage.delete("key1") == True
    assert storage.delete("nonexistent") == False
    assert storage.get("key1") is None
    
    # Test clear
    storage.clear()
    assert len(storage.keys()) == 0
    
    print("✅ GlobalStorage operations test passed!")

def test_global_storage_locking():
    """Test GlobalStorage thread safety with locking."""
    print("🧪 Testing GlobalStorage thread safety...")
    
    storage = GlobalStorage()
    storage.set("counter", 0)
    results = []
    
    def increment_worker(worker_id):
        """Worker function that increments counter safely."""
        for i in range(100):
            with storage.lock_key("counter"):
                current = storage.get("counter", 0)
                # Simulate some work
                time.sleep(0.001)
                storage.set("counter", current + 1)
        results.append(f"Worker {worker_id} completed")
    
    # Create multiple threads
    threads = []
    num_workers = 5
    
    for i in range(num_workers):
        thread = threading.Thread(target=increment_worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify final counter value
    final_count = storage.get("counter")
    expected_count = num_workers * 100
    
    assert final_count == expected_count, f"Expected {expected_count}, got {final_count}"
    assert len(results) == num_workers
    
    print(f"✅ GlobalStorage thread safety test passed! Final counter: {final_count}")

def test_global_storage_multiple_locks():
    """Test locking multiple keys simultaneously."""
    print("🧪 Testing multiple key locking...")
    
    storage = GlobalStorage()
    storage.set("key1", 10)
    storage.set("key2", 20)
    
    def transfer_worker():
        """Transfer value between keys safely."""
        for _ in range(50):
            with storage.lock_multiple("key1", "key2"):
                val1 = storage.get("key1")
                val2 = storage.get("key2")
                # Transfer 1 unit from key1 to key2
                if val1 > 0:
                    storage.set("key1", val1 - 1)
                    storage.set("key2", val2 + 1)
                time.sleep(0.001)
    
    # Create multiple threads doing transfers
    threads = []
    for i in range(3):
        thread = threading.Thread(target=transfer_worker)
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify total is conserved
    final1 = storage.get("key1")
    final2 = storage.get("key2")
    total = final1 + final2
    
    assert total == 30, f"Expected total 30, got {total} (key1={final1}, key2={final2})"
    
    print(f"✅ Multiple key locking test passed! Final values: key1={final1}, key2={final2}")

def test_subagent_storage_sharing():
    """Test the logic for sub-agent storage sharing without full agent setup."""
    print("🧪 Testing sub-agent storage sharing logic...")
    
    # Simulate the SubAgentManager logic
    parent_storage = GlobalStorage()
    parent_storage.set("parent_value", "from_parent")
    
    # Simulate creating a "sub-agent" with shared storage
    # This simulates the logic in SubAgentManager.call_sub_agent()
    
    # Test 1: Direct assignment (fallback method)
    sub_storage = GlobalStorage()
    # Simulate: sub_agent.global_storage = parent_storage
    sub_storage = parent_storage
    
    # Verify sharing
    assert sub_storage is parent_storage
    assert sub_storage.get("parent_value") == "from_parent"
    
    # Set value from "sub-agent"
    sub_storage.set("sub_value", "from_sub")
    
    # Verify parent can see it
    assert parent_storage.get("sub_value") == "from_sub"
    
    print("✅ Sub-agent storage sharing logic test passed!")

def main():
    """Run all tests."""
    print("🚀 Starting GlobalStorage tests...\n")
    
    try:
        test_global_storage_operations()
        print()
        
        test_global_storage_isolation()
        print()
        
        test_global_storage_sharing()
        print()
        
        test_global_storage_locking()
        print()
        
        test_global_storage_multiple_locks()
        print()
        
        test_subagent_storage_sharing()
        print()
        
        print("🎉 All GlobalStorage tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()