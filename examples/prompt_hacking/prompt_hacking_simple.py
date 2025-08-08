#!/usr/bin/env python3
"""
Simplified Prompt Hacking Classification Agent Tuning Example (Refactored)

This example demonstrates prompt hacking detection with real LLM API calls.
Now using modular components for better organization and maintainability.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from modular components
from prompt_hacking import (
    run_evaluation,
    setup_logging,
    check_environment
)


def main():
    """Run the simple prompt hacking example (now using modular components)."""
    print("🤖 Simple Prompt Hacking Detection Example - Modular Version")
    print("=" * 60)
    
    # Setup using modular components
    setup_logging()
    
    if not check_environment():
        return
    
    try:
        
        # Run evaluation
        print("Running Evaluation...")

        script_dir = Path(__file__).parent
        evaluation_report = run_evaluation(str(script_dir / "sample_dataset.jsonl"))
        print("!!!!! Evaluation report", evaluation_report)
        
        print("\n🎉 Example completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()