# NexAU Standalone Agent Entry Point Template

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from nexau import Agent

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).parent


def get_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    """Entry point for the agent."""
    # 1. Load agent from YAML configuration
    agent = Agent.from_yaml(config_path=SCRIPT_DIR / "agent.yaml")  # TODO: Replace with actual YAML filename

    # 2. Get sandbox working directory
    sandbox = agent.sandbox_manager.instance
    work_dir = str(sandbox.work_dir) if sandbox else os.getcwd()

    # 3. Get user input
    user_message = input("Enter your task: ")

    # 4. Run the agent
    response = agent.run(
        message=user_message,
        context={
            "date": get_date(),
            "username": os.getenv("USER", "user"),
            "working_directory": work_dir,
        },
    )
    print(response)


if __name__ == "__main__":
    main()
