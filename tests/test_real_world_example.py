#!/usr/bin/env python3
"""Real-world example demonstrating GlobalStorage usage in agent hierarchies."""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add the northau package to the path
sys.path.insert(0, str(Path(__file__).parent))

from northau.archs.main_sub.agent_context import GlobalStorage
from northau.archs.main_sub.agent import create_agent
from northau.archs.llm.llm_config import LLMConfig

def simulate_real_world_scenario():
    """Simulate a real-world scenario with main agent and sub-agents sharing global state."""
    print("🌍 Real-world scenario: Multi-agent task with shared state")
    print("=" * 60)
    
    with patch('northau.archs.main_sub.agent.openai') as mock_openai_module:
        mock_openai_client = Mock()
        mock_openai_module.OpenAI.return_value = mock_openai_client
        
        # Scenario: Document processing system
        # - Main agent coordinates the work
        # - Text extraction sub-agent extracts text from documents
        # - Analysis sub-agent analyzes extracted text
        # - All agents share progress and results via GlobalStorage
        
        print("📋 Setting up document processing system...")
        
        # Create shared storage for the entire system
        system_storage = GlobalStorage()
        
        # Initialize system state
        system_storage.set("documents_to_process", ["doc1.pdf", "doc2.txt", "doc3.docx"])
        system_storage.set("processed_documents", [])
        system_storage.set("analysis_results", {})
        system_storage.set("system_status", "initializing")
        
        # Sub-agent factories
        def create_text_extractor():
            return create_agent(
                name="text_extractor",
                llm_config=LLMConfig(model="gpt-4")
            )
        
        def create_analyzer():
            return create_agent(
                name="analyzer", 
                llm_config=LLMConfig(model="gpt-4")
            )
        
        # Create main coordinator agent with sub-agents
        coordinator = create_agent(
            name="document_coordinator",
            llm_config=LLMConfig(model="gpt-4"),
            global_storage=system_storage,
            sub_agents=[
                ("text_extractor", create_text_extractor),
                ("analyzer", create_analyzer)
            ]
        )
        
        print("✅ Document processing system initialized")
        print(f"📊 Initial state: {system_storage.get('documents_to_process')}")
        print()
        
        # Simulate document processing workflow
        def simulate_document_processing():
            """Simulate processing documents with shared state."""
            print("🔄 Starting document processing simulation...")
            
            # Update system status
            coordinator.global_storage.set("system_status", "processing")
            
            # Simulate processing each document
            docs_to_process = coordinator.global_storage.get("documents_to_process", [])
            processed_docs = coordinator.global_storage.get("processed_documents", [])
            
            for doc in docs_to_process:
                print(f"📄 Processing {doc}...")
                
                # Simulate text extraction (would call sub-agent in real scenario)
                with coordinator.global_storage.lock_key("processed_documents"):
                    extracted_text = f"extracted_text_from_{doc}"
                    coordinator.global_storage.set(f"text_{doc}", extracted_text)
                    
                    # Update processed documents list
                    current_processed = coordinator.global_storage.get("processed_documents", [])
                    current_processed.append(doc)
                    coordinator.global_storage.set("processed_documents", current_processed)
                
                # Simulate analysis (would call analyzer sub-agent)
                with coordinator.global_storage.lock_key("analysis_results"):
                    analysis = f"analysis_of_{doc}"
                    current_results = coordinator.global_storage.get("analysis_results", {})
                    current_results[doc] = analysis
                    coordinator.global_storage.set("analysis_results", current_results)
                
                print(f"✅ Completed {doc}")
            
            # Update final status
            coordinator.global_storage.set("system_status", "completed")
            print("🎉 All documents processed!")
        
        # Run the simulation
        simulate_document_processing()
        
        # Display final results
        print("\n📊 Final System State:")
        print("-" * 30)
        print(f"Status: {coordinator.global_storage.get('system_status')}")
        print(f"Processed: {coordinator.global_storage.get('processed_documents')}")
        print(f"Results: {len(coordinator.global_storage.get('analysis_results', {}))} analyses completed")
        
        # Demonstrate isolation - create another processing system
        print("\n🔒 Testing isolation with second system...")
        system2_storage = GlobalStorage()
        system2_storage.set("documents_to_process", ["other_doc1.pdf"])
        
        coordinator2 = create_agent(
            name="coordinator2",
            llm_config=LLMConfig(model="gpt-4"),
            global_storage=system2_storage
        )
        
        # Verify isolation
        assert coordinator.global_storage is not coordinator2.global_storage
        assert coordinator.global_storage.get("documents_to_process") != coordinator2.global_storage.get("documents_to_process")
        
        print("✅ System isolation verified")
        
        return coordinator, coordinator2

def simulate_concurrent_access():
    """Simulate concurrent access to shared storage by multiple threads."""
    print("\n🚦 Concurrent Access Simulation")
    print("=" * 60)
    
    # Create shared storage
    shared_storage = GlobalStorage()
    shared_storage.set("task_queue", list(range(1, 101)))  # 100 tasks
    shared_storage.set("completed_tasks", [])
    shared_storage.set("worker_stats", {})
    
    def worker_thread(worker_id):
        """Simulate a worker processing tasks from shared queue."""
        completed_count = 0
        
        while True:
            # Get next task safely
            with shared_storage.lock_key("task_queue"):
                task_queue = shared_storage.get("task_queue", [])
                if not task_queue:
                    break  # No more tasks
                
                # Take a task
                task = task_queue.pop(0)
                shared_storage.set("task_queue", task_queue)
            
            # Simulate task processing
            time.sleep(0.01)  # Simulate work
            completed_count += 1
            
            # Update completed tasks safely
            with shared_storage.lock_key("completed_tasks"):
                completed = shared_storage.get("completed_tasks", [])
                completed.append(f"task_{task}_by_worker_{worker_id}")
                shared_storage.set("completed_tasks", completed)
        
        # Update worker stats
        with shared_storage.lock_key("worker_stats"):
            stats = shared_storage.get("worker_stats", {})
            stats[f"worker_{worker_id}"] = completed_count
            shared_storage.set("worker_stats", stats)
        
        print(f"🔧 Worker {worker_id} completed {completed_count} tasks")
    
    # Create and start worker threads
    print("🚀 Starting 5 concurrent workers...")
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all workers to complete
    for thread in threads:
        thread.join()
    
    # Display results
    final_queue = shared_storage.get("task_queue", [])
    completed_tasks = shared_storage.get("completed_tasks", [])
    worker_stats = shared_storage.get("worker_stats", {})
    
    print(f"\n📈 Concurrent Processing Results:")
    print(f"Tasks remaining in queue: {len(final_queue)}")
    print(f"Tasks completed: {len(completed_tasks)}")
    print(f"Worker statistics: {worker_stats}")
    
    # Verify all tasks were processed
    total_completed = sum(worker_stats.values())
    assert total_completed == 100, f"Expected 100 tasks, completed {total_completed}"
    assert len(final_queue) == 0, f"Expected empty queue, got {len(final_queue)} remaining"
    
    print("✅ Concurrent access test passed!")

def main():
    """Run real-world demonstration."""
    print("🎯 GlobalStorage Real-World Demonstration")
    print("=" * 60)
    print()
    
    try:
        # Run real-world scenario
        coordinator1, coordinator2 = simulate_real_world_scenario()
        
        # Run concurrent access simulation  
        simulate_concurrent_access()
        
        print("\n" + "=" * 60)
        print("🎉 Real-world demonstration completed successfully!")
        print("\n✨ Key Features Demonstrated:")
        print("  • Agent hierarchy with shared GlobalStorage")
        print("  • Data isolation between different agent systems")  
        print("  • Thread-safe concurrent access with locking")
        print("  • Real-world document processing workflow simulation")
        print("  • Multi-threaded task processing with shared state")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()