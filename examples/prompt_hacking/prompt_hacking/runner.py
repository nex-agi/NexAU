#!/usr/bin/env python3
"""
Main runner for prompt hacking detection examples.
"""

import os
import logging
from pathlib import Path

from northau.evaluator import ExperimentManager, Evaluator
from northau.evaluator import EvaluationConfig
from .dataset_loader import load_dataset_from_jsonl
from .agent import create_agent_factory
from .config import create_base_config, create_better_config
from .evaluation import custom_classification_evaluation


def setup_logging():
    """Configure logging to show autotuning logs but hide SQL logs."""
    # Disable SQLAlchemy logging first (before any other logging setup)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.CRITICAL)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.CRITICAL) 
    logging.getLogger('sqlalchemy.dialects').setLevel(logging.CRITICAL)
    logging.getLogger('sqlalchemy').setLevel(logging.CRITICAL)
    
    # Set up root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Enable evaluation logs
    logging.getLogger('northau.autotuning').setLevel(logging.INFO)


def check_environment():
    """Check required environment variables."""
    if not os.getenv('LLM_API_KEY'):
        print("❌ LLM_API_KEY not found in environment")
        print("Please check your .env file")
        return False
    
    print(f"🔧 Using: {os.getenv('LLM_MODEL', 'glm-4.5')} at {os.getenv('LLM_BASE_URL', 'default')}")
    return True


def run_evaluation(dataset_path=None):
    """Run the full evaluation with dataset."""
    print("\n📊 Running Evaluation")
    print("=" * 30)
    
    # Load or create dataset
    try:
        if dataset_path is None:
            dataset_path = Path("examples/sample_dataset.jsonl")
        else:
            dataset_path = Path(dataset_path)
        dataset = load_dataset_from_jsonl(dataset_path)
        print(f"✅ Loaded {len(dataset.items)} items from {dataset_path}")
    except Exception as e:
        raise ValueError(f"Dataset not found at {dataset_path}")
    
    # Setup evaluation
    config = create_base_config()
    better_config = create_better_config()
    agent_factory = create_agent_factory()
    
    # Run evaluation
    manager = ExperimentManager(
        mode="evaluation",
        session_name="prompt_hacking"
    )
    
    evaluator = Evaluator(evaluation_config=EvaluationConfig(
            default_metrics=["accuracy"],
            evaluation_methods={"accuracy": "custom_classification"},
            parallel_evaluation=True,
            timeout_seconds=60
        ), custom_eval_functions={"custom_classification": custom_classification_evaluation}
    )

    evaluation_report = manager.run_evaluation(
        dataset=dataset,
        configs=[config, better_config],
        agent_factory=agent_factory,
        evaluator=evaluator,
    )
    
    print(f"📈 Evaluation completed: {evaluation_report}")
    return evaluation_report
