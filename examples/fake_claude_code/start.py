#!/usr/bin/env python3
"""Test the fake Claude Code agent loading from YAML configuration."""

import logging
import os
from datetime import datetime
from pathlib import Path

from northau.archs.config.config_loader import load_agent_config

logging.basicConfig(level=logging.INFO)


def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Test the fake Claude Code agent with YAML-based agent configuration."""
    print("Testing Fake Claude Code Agent (YAML-based)")
    print("=" * 60)

    try:
        # Load agent from YAML configuration
        print("Loading Claude Code agent from YAML configuration...")

        # Build LLM configuration from environment variables
        llm_config_overrides = {
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        model = os.getenv("LLM_MODEL")
        if model:
            llm_config_overrides["model"] = model
        base_url = os.getenv("LLM_BASE_URL")
        if base_url:
            llm_config_overrides["base_url"] = base_url
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            llm_config_overrides["api_key"] = api_key

        config_overrides = {
            "claude_code_agent": {
                "llm_config": llm_config_overrides,
            },
        }

        script_dir = Path(__file__).parent
        claude_code_agent = load_agent_config(
            str(script_dir / "cc_agent.yaml"),
            overrides=config_overrides,
        )
        print("✓ Agent loaded successfully from YAML")

        print("\nTesting Fake Claude Code...")
        user_message = input("Enter your task: ")
        print(f"\nUser: {user_message}")
        print("\nAgent Response:")
        print("-" * 30)

        response = claude_code_agent.run(
            user_message,
            context={
                "date": get_date(),
                "username": os.getenv("USER"),
                "working_directory": os.getcwd(),
                "env_content": {
                    "date": get_date(),
                    "username": os.getenv("USER"),
                    "working_directory": os.getcwd(),
                },
            },
        )
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
