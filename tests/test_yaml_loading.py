#!/usr/bin/env python3
"""Test YAML agent loading without making LLM calls."""

import os
from northau.archs.config.config_loader import load_agent_config

def test_yaml_loading():
    """Test loading agent configuration from YAML."""
    print("Testing YAML Agent Configuration Loading")
    print("=" * 45)
    
    try:
        # Build LLM configuration from environment variables (or defaults)
        llm_config_overrides = {
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1", 
            "api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        config_overrides = {
            "llm_config": llm_config_overrides
        }
        
        print("Loading agent from YAML configuration...")
        agent = load_agent_config(
            "agents/deep_research_agent.yaml",
            overrides=config_overrides
        )
        
        print("✓ Agent loaded successfully!")
        print(f"  - Agent name: {agent.name}")
        print(f"  - Number of tools: {len(agent.tools)}")
        print(f"  - Tool names: {[tool.name for tool in agent.tools]}")
        print(f"  - LLM model: {agent.llm_config.model}")
        print(f"  - System prompt (first 100 chars): {agent.processed_system_prompt[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_yaml_loading()
    exit(0 if success else 1)