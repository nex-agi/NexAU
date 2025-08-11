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


from .agent import PromptHackingAgent
from .config import (
    create_base_config,
    create_better_config
)
from .evaluation import (
    custom_classification_evaluation,
)

__all__ = [
    # Agent
    'PromptHackingAgent',
    
    # Configuration
    'create_base_config',
    'create_better_config',
    
    # Evaluation
    'custom_classification_evaluation',
]