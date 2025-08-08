"""Experiment execution for evaluation system."""

from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import traceback

from ..autotuning.dataset import Dataset, DatasetItem
from .config import Config

logger = logging.getLogger(__name__)

# Forward declaration for Agent type
Agent = Any  # This would be imported from the main agent system


@dataclass
class ItemResult:
    """Result of executing single dataset item with agent."""
    
    item_id: str
    agent_output: str
    execution_trace: Optional[List[Dict[str, Any]]] = None
    execution_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "agent_output": self.agent_output,
            "execution_trace": self.execution_trace,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "error": self.error
        }


@dataclass
class ExperimentResult:
    """Complete experiment execution result."""
    
    experiment_id: str
    config: Config
    item_results: List[ItemResult]
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_execution_time: float = 0.0
    total_tokens: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    error_count: int = 0
    dataset_items: List[DatasetItem] = field(default_factory=list)  # Store for evaluation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "overall_metrics": self.overall_metrics,
            "timestamp": self.timestamp.isoformat(),
            "total_execution_time": self.total_execution_time,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "item_results": [result.to_dict() for result in self.item_results]
        }


class ExperimentRunner:
    """Execute individual experiments and collect execution data."""
    
    def __init__(
        self,
        agent_factory: Callable[[Config], Agent],
        parallel_execution: bool = True,
        max_workers: int = 4,
        timeout_seconds: int = 60,
        enable_tracing: bool = False
    ):
        self.agent_factory = agent_factory
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.enable_tracing = enable_tracing
    
    def run_experiment(
        self,
        dataset: Dataset,
        config: Config,
        experiment_id: str,
        split: str = "all"
    ) -> ExperimentResult:
        """Execute single experiment with given configuration."""
        logger.info(f"Starting experiment {experiment_id} with config {config.config_id}")
        start_time = time.time()
        
        # Get dataset items
        dataset_items = dataset.get_items(split)
        if not dataset_items:
            raise ValueError(f"No items found in dataset split: {split}")
        
        # Create agent instance
        try:
            agent = self.agent_factory(config)
        except Exception as e:
            logger.error(f"Failed to create agent for experiment {experiment_id}: {e}")
            raise
        
        # Execute items
        if self.parallel_execution and len(dataset_items) > 1:
            item_results = self._run_items_parallel(dataset_items, agent, config)
        else:
            item_results = self._run_items_sequential(dataset_items, agent, config)
        
        # Calculate metrics
        total_execution_time = time.time() - start_time
        success_count = sum(1 for result in item_results if result.error is None)
        success_rate = success_count / len(item_results) if item_results else 0.0
        error_count = len(item_results) - success_count
        
        # Aggregate token usage
        total_tokens = {}
        for result in item_results:
            for token_type, count in result.token_usage.items():
                total_tokens[token_type] = total_tokens.get(token_type, 0) + count
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(item_results)
        
        experiment_result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            item_results=item_results,
            overall_metrics=overall_metrics,
            total_execution_time=total_execution_time,
            total_tokens=total_tokens,
            success_rate=success_rate,
            error_count=error_count,
            dataset_items=dataset_items
        )
        
        logger.info(
            f"Experiment {experiment_id} completed: "
            f"{success_count}/{len(item_results)} items successful, "
            f"success rate: {success_rate:.2%}, "
            f"total time: {total_execution_time:.2f}s"
        )
        
        return experiment_result
    
    def run_item(self, item: DatasetItem, config: Config, agent: Agent) -> ItemResult:
        """Execute single dataset item with agent."""
        start_time = time.time()
        
        try:
            logger.debug(f"Running item {item.id}")
            
            # Execute agent with item input
            if self.enable_tracing:
                agent_output, execution_trace, token_usage = self._run_agent_with_tracing(
                    agent, item.input_message
                )
            else:
                agent_output, token_usage = self._run_agent_simple(agent, item.input_message)
                execution_trace = None
            
            execution_time = time.time() - start_time
            
            return ItemResult(
                item_id=item.id,
                agent_output=agent_output,
                execution_trace=execution_trace,
                execution_time=execution_time,
                token_usage=token_usage
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error executing item {item.id}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            return ItemResult(
                item_id=item.id,
                agent_output="",
                execution_time=execution_time,
                error=error_msg
            )
    
    def _run_items_parallel(
        self,
        dataset_items: List[DatasetItem],
        agent: Agent,
        config: Config
    ) -> List[ItemResult]:
        """Execute dataset items in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_item = {}
            for item in dataset_items:
                # Create a new agent instance for each parallel execution
                try:
                    item_agent = self.agent_factory(config)
                    future = executor.submit(self.run_item, item, config, item_agent)
                    future_to_item[future] = item.id
                except Exception as e:
                    logger.error(f"Failed to create agent for item {item.id}: {e}")
                    results.append(ItemResult(
                        item_id=item.id,
                        agent_output="",
                        error=f"Agent creation failed: {str(e)}"
                    ))
            
            # Collect results
            for future in as_completed(future_to_item, timeout=self.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item_id = future_to_item[future]
                    logger.error(f"Parallel execution failed for item {item_id}: {e}")
                    results.append(ItemResult(
                        item_id=item_id,
                        agent_output="",
                        error=f"Execution timeout/error: {str(e)}"
                    ))
        
        # Sort results by item order in original dataset
        item_order = {item.id: i for i, item in enumerate(dataset_items)}
        results.sort(key=lambda r: item_order.get(r.item_id, float('inf')))
        
        return results
    
    def _run_items_sequential(
        self,
        dataset_items: List[DatasetItem],
        agent: Agent,
        config: Config
    ) -> List[ItemResult]:
        """Execute dataset items sequentially."""
        results = []
        
        for item in dataset_items:
            result = self.run_item(item, config, agent)
            results.append(result)
        
        return results
    
    def _run_agent_simple(self, agent: Agent, input_message: str) -> Tuple[str, Dict[str, int]]:
        """Run agent without detailed tracing."""
        # This would integrate with the actual agent system
        # For now, provide a mock implementation
        
        try:
            # Mock agent execution
            if hasattr(agent, 'process_message'):
                output = agent.process_message(input_message)
                token_usage = getattr(agent, 'last_token_usage', {})
            else:
                # Fallback mock
                output = f"Mock agent response to: {input_message}"
                token_usage = {"prompt_tokens": 100, "completion_tokens": 50}
            
            return output, token_usage
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
    
    def _run_agent_with_tracing(
        self,
        agent: Agent,
        input_message: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """Run agent with detailed execution tracing."""
        # This would integrate with the actual agent system's tracing
        # For now, provide a mock implementation
        
        try:
            output, token_usage = self._run_agent_simple(agent, input_message)
            
            # Mock execution trace
            execution_trace = [
                {
                    "step": 1,
                    "action": "process_input",
                    "timestamp": datetime.utcnow().isoformat(),
                    "input": input_message
                },
                {
                    "step": 2,
                    "action": "generate_response",
                    "timestamp": datetime.utcnow().isoformat(),
                    "output": output
                }
            ]
            
            return output, execution_trace, token_usage
            
        except Exception as e:
            logger.error(f"Agent execution with tracing failed: {e}")
            raise
    
    def _calculate_overall_metrics(self, item_results: List[ItemResult]) -> Dict[str, float]:
        """Calculate overall metrics for the experiment."""
        if not item_results:
            return {}
        
        successful_results = [r for r in item_results if r.error is None]
        
        metrics = {
            "success_rate": len(successful_results) / len(item_results),
            "error_rate": (len(item_results) - len(successful_results)) / len(item_results)
        }
        
        if successful_results:
            # Average execution time
            total_time = sum(r.execution_time for r in successful_results)
            metrics["avg_execution_time"] = total_time / len(successful_results)
            
            # Output length statistics
            output_lengths = [len(r.agent_output) for r in successful_results]
            metrics["avg_output_length"] = sum(output_lengths) / len(output_lengths)
            metrics["min_output_length"] = min(output_lengths)
            metrics["max_output_length"] = max(output_lengths)
            
            # Token usage statistics
            if any(r.token_usage for r in successful_results):
                total_prompt_tokens = sum(r.token_usage.get("prompt_tokens", 0) for r in successful_results)
                total_completion_tokens = sum(r.token_usage.get("completion_tokens", 0) for r in successful_results)
                
                if total_prompt_tokens > 0:
                    metrics["avg_prompt_tokens"] = total_prompt_tokens / len(successful_results)
                if total_completion_tokens > 0:
                    metrics["avg_completion_tokens"] = total_completion_tokens / len(successful_results)
        
        return metrics