# AutoTuning System

A comprehensive system for automatically optimizing agent configurations, system prompts, and tool descriptions through iterative experimentation and evaluation.

## Features

- **Automated Configuration Optimization**: Iteratively improve agent performance
- **Evaluation-Only Mode**: Benchmark and compare multiple configurations
- **Custom Evaluation Functions**: Domain-specific evaluation logic
- **Multiple Tuning Strategies**: Random search, Bayesian optimization, and more
- **Database Integration**: SQLite (dev) and PostgreSQL (production) support
- **Comprehensive Logging**: Track experiments and performance over time

## Quick Start

```python
from northau.archs.autotuning import ExperimentManager, Dataset, DatasetItem, Config

# Create dataset
items = [
    DatasetItem(
        id="test_001",
        input_data={"message": "Hello, how are you?"},
        expected_output="Hi! How can I help you?",
        evaluation_criteria={"should_be_friendly": True}
    )
]
dataset = Dataset(name="greeting_test", items=items)

# Create base configuration
config = Config(
    system_prompt="You are a helpful assistant.",
    llm_config={"temperature": 0.5, "max_tokens": 150}
)

# Run autotuning
manager = ExperimentManager(max_experiments=10, target_score=0.9)
results = manager.run_autotuning_session(
    dataset=dataset,
    initial_config=config,
    agent_factory=create_your_agent
)

print(f"Best score: {results.best_score}")
```

## Components

### Core Components

1. **ExperimentManager**: Orchestrates the autotuning process
2. **Dataset**: Manages test cases and evaluation criteria
3. **Config**: Stores tunable agent configurations
4. **ExperimentRunner**: Executes experiments with agents
5. **Evaluator**: Assesses agent performance with custom metrics
6. **AutoTuner**: Analyzes results and generates improved configurations

### Evaluation System

The evaluation system supports multiple evaluation methods:

- **Similarity Matching**: Compare outputs to expected results
- **Rule-Based**: Check specific criteria and constraints
- **LLM-Based**: Use AI to judge response quality
- **Custom Functions**: Domain-specific evaluation logic

Example custom evaluator:

```python
def custom_politeness_eval(item_result, dataset_item, context):
    agent_output = item_result.agent_output.lower()
    politeness_score = sum(1 for phrase in ['please', 'thank you'] 
                          if phrase in agent_output) / 2
    
    return ItemEvaluation(
        item_id=dataset_item.id,
        score=politeness_score,
        metric_scores={"politeness": politeness_score}
    )

# Register custom evaluator
evaluator.register_eval_function("politeness", custom_politeness_eval)
```

### Tuning Strategies

#### Random Search
- Random parameter variations
- Good for initial exploration
- Simple and robust

```python
tuning_config = {
    "mutation_rate": 0.3,
    "max_mutations_per_config": 3
}
```

#### Bayesian Optimization
- Model-based optimization
- Balances exploration vs exploitation
- More efficient for expensive evaluations

```python
tuning_config = {
    "exploration_rate": 0.3,
    "exploitation_rate": 0.7,
    "confidence_threshold": 0.6
}
```

## Database Schema

The system uses SQLModel for type-safe database operations:

- **Sessions**: Autotuning sessions with metadata
- **Experiments**: Individual experiment runs
- **ItemResults**: Detailed results for each test case
- **Configs**: Generated configurations with performance tracking
- **DatasetMetadata**: Dataset versioning and information

## Configuration Files

### Autotuning Session Config

```yaml
session:
  name: "customer_service_optimization"
  max_experiments: 50
  target_score: 0.9
  
dataset:
  path: "datasets/customer_service.json"
  validation_split: 0.2
  
evaluation:
  metrics:
    - correctness
    - helpfulness
    - efficiency
  
tuning:
  strategy: "bayesian_optimization"
  parameters:
    system_prompt:
      mutation_rate: 0.3
    llm_config:
      temperature:
        range: [0.0, 1.0]
        step: 0.1
```

### Dataset Format

```json
{
  "name": "customer_service_scenarios",
  "version": "1.0",
  "items": [
    {
      "id": "cs_001",
      "input_data": {"message": "I need help with my order"},
      "expected_output": "I'd be happy to help with your order. What's your order number?",
      "evaluation_criteria": {
        "should_offer_help": true,
        "should_ask_for_details": true
      },
      "evaluation_method": "rule_based"
    }
  ]
}
```

## Usage Modes

### AutoTuning Mode

Automatically optimize configurations through iterative experimentation:

```python
results = manager.run_autotuning_session(
    dataset=dataset,
    initial_config=config,
    agent_factory=create_agent,
    tuning_strategy="bayesian_optimization"
)
```

### Evaluation-Only Mode

Compare multiple configurations without optimization:

```python
configs = [config_a, config_b, config_c]
report = manager.run_evaluation_only(
    dataset=dataset,
    configs=configs,
    agent_factory=create_agent
)

print(report.comparison_table())
```

## Examples

See `examples/autotuning_example.py` for complete usage examples including:

- Basic autotuning session
- Evaluation-only comparison
- Custom evaluation functions
- Result analysis and reporting

## Testing

Run the test suite:

```bash
python -m pytest tests/test_autotuning_basic.py -v
```

## Architecture

The AutoTuning system follows a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Experiment     │    │   Experiment    │    │   Evaluator     │
│  Manager        │───▶│   Runner        │───▶│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AutoTuner     │    │    Dataset      │    │   Database      │
│                 │◀───│                 │    │   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

Each component is designed to be:
- **Modular**: Can be used independently
- **Extensible**: Support custom implementations
- **Type-Safe**: Full Python type hints
- **Testable**: Comprehensive test coverage
- **Performant**: Parallel execution support