"""
Prompt Hacking Detection Example - Modular Implementation

This package provides a modular implementation for prompt hacking detection,
decomposed from the original single-file example.

Components:
- dataset_loader: Load datasets from JSONL files or create test data
- agent: The prompt hacking detection agent
- config: Configuration management  
- evaluation: Evaluation functions
- runner: Main execution logic
"""

from .dataset_loader import (
    load_dataset_from_jsonl,
)
from .agent import PromptHackingAgent, create_agent_factory
from .config import (
    create_base_config,
)
from .evaluation import (
    custom_classification_evaluation,
)
from .runner import (
    run_evaluation,
    setup_logging,
    check_environment,
)

__all__ = [
    # Dataset functions
    'load_dataset_from_jsonl',
    
    # Agent
    'PromptHackingAgent',
    'create_agent_factory',
    
    # Configuration
    'create_base_config',
    
    # Evaluation
    'custom_classification_evaluation',
    
    # Runner
    'run_evaluation',
    'setup_logging',
    'check_environment',
]