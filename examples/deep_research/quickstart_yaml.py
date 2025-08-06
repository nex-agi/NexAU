#!/usr/bin/env python3
"""Test the quick start example loading agent from YAML configuration."""

import os
from datetime import datetime
from pathlib import Path

from northau.archs.config.config_loader import load_agent_config

def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Test the quick start example with YAML-based agent configuration."""
    print("Testing Northau Framework Quick Start Example (YAML-based)")
    print("=" * 60)
    
    try:
        # Load agent from YAML configuration
        print("Loading deep research agent from YAML configuration...")
        
        # Build LLM configuration from environment variables
        llm_config_overrides = {
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        if os.getenv("LLM_MODEL"):
            llm_config_overrides["model"] = os.getenv("LLM_MODEL")
        if os.getenv("LLM_BASE_URL"):
            llm_config_overrides["base_url"] = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_API_KEY"):
            llm_config_overrides["api_key"] = os.getenv("LLM_API_KEY")
        
        config_overrides = {
            "llm_config": llm_config_overrides
        }
        
        script_dir = Path(__file__).parent
        deep_research_agent = load_agent_config(
            str(script_dir / "deep_research_agent.yaml"),
            overrides=config_overrides
        )
        print("✓ Agent loaded successfully from YAML")

        print("\nTesting delegation with web research...")
        web_message = "What day is it today? and the stock price of Tencent on the day?"
        print(f"\nUser: {web_message}")
        print("\nAgent Response:")
        print("-" * 30)
        
        response = deep_research_agent.run(web_message, context={
            "date": get_date(),
        })
        print(response)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)