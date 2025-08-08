"""Core evaluator for evaluation system."""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from ..autotuning.dataset import DatasetItem

# Type alias for evaluation functions
EvalFunction = Callable[["ItemResult", DatasetItem, Dict[str, Any]], "ItemEvaluation"]

logger = logging.getLogger(__name__)


@dataclass
class ItemEvaluation:
    """Evaluation result for a single item."""
    
    item_id: str
    score: float
    metric_scores: Dict[str, float] = field(default_factory=dict)
    feedback: Optional[str] = None
    evaluation_time: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "metric_scores": self.metric_scores,
            "feedback": self.feedback,
            "evaluation_time": self.evaluation_time,
            "error": self.error
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for an experiment."""
    
    experiment_id: str
    overall_score: float
    item_evaluations: List[ItemEvaluation]
    metric_scores: Dict[str, float] = field(default_factory=dict)
    evaluation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "overall_score": self.overall_score,
            "metric_scores": self.metric_scores,
            "evaluation_time": self.evaluation_time,
            "timestamp": self.timestamp.isoformat(),
            "item_evaluations": [eval.to_dict() for eval in self.item_evaluations]
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation system."""
    
    default_metrics: List[str] = field(default_factory=lambda: ["correctness", "helpfulness"])
    evaluation_methods: Dict[str, str] = field(default_factory=dict)
    llm_evaluator_config: Optional[Dict[str, Any]] = None
    custom_function_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parallel_evaluation: bool = True
    timeout_seconds: int = 30
    max_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_metrics": self.default_metrics,
            "evaluation_methods": self.evaluation_methods,
            "llm_evaluator_config": self.llm_evaluator_config,
            "custom_function_configs": self.custom_function_configs,
            "parallel_evaluation": self.parallel_evaluation,
            "timeout_seconds": self.timeout_seconds,
            "max_workers": self.max_workers
        }


# Import ItemResult from experiment runner (will be defined later)
class ItemResult:
    """Placeholder for ItemResult class."""
    
    def __init__(
        self,
        item_id: str,
        agent_output: str,
        execution_time: Optional[float] = None,
        token_usage: Dict[str, int] = None,
        error: Optional[str] = None
    ):
        self.item_id = item_id
        self.agent_output = agent_output
        self.execution_time = execution_time or 0.0
        self.token_usage = token_usage or {}
        self.error = error


class Evaluator:
    """Assess agent output quality and compute performance metrics."""
    
    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        custom_eval_functions: Dict[str, EvalFunction] = None
    ):
        self.config = evaluation_config
        self.custom_functions = custom_eval_functions or {}
        
        # Register built-in evaluation functions
        self._register_builtin_functions()
    
    def _register_builtin_functions(self) -> None:
        """Register built-in evaluation functions."""
        try:
            from .evaluation.builtin_functions import (
                similarity_evaluation,
                llm_evaluation,
                rule_based_evaluation
            )
        except ImportError:
            # Fallback for import issues
            return
        
        self.custom_functions.update({
            "similarity_evaluation": similarity_evaluation,
            "similarity_match": similarity_evaluation,
            "llm_based": llm_evaluation,
            "rule_based": rule_based_evaluation
        })
    
    def register_eval_function(self, name: str, func: EvalFunction) -> None:
        """Register a custom evaluation function."""
        if not callable(func):
            raise ValueError(f"Evaluation function {name} must be callable")
        
        self.custom_functions[name] = func
        logger.info(f"Registered custom evaluation function: {name}")
    
    def evaluate_experiment(self, experiment_result: "ExperimentResult") -> EvaluationResult:
        """Evaluate complete experiment using configured methods."""
        from .experiment_runner import ExperimentResult as ExpResult
        
        start_time = datetime.utcnow()
        
        # Get item results and dataset items (this would come from the experiment)
        item_results = experiment_result.item_results
        dataset_items = experiment_result.dataset_items  # Assume this is available
        
        # Evaluate each item
        if self.config.parallel_evaluation:
            item_evaluations = self._evaluate_items_parallel(item_results, dataset_items)
        else:
            item_evaluations = self._evaluate_items_sequential(item_results, dataset_items)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(item_evaluations)
        metric_scores = self._calculate_metric_scores(item_evaluations)
        
        evaluation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return EvaluationResult(
            experiment_id=experiment_result.experiment_id,
            overall_score=overall_score,
            item_evaluations=item_evaluations,
            metric_scores=metric_scores,
            evaluation_time=evaluation_time
        )
    
    def evaluate_item(
        self,
        item_result: ItemResult,
        dataset_item: DatasetItem,
        eval_method: str
    ) -> ItemEvaluation:
        """Evaluate single item result with specified method."""
        start_time = datetime.utcnow()
        
        try:
            # Determine evaluation method
            method = eval_method
            
            # Get evaluation function
            if method not in self.custom_functions:
                raise ValueError(f"Unknown evaluation method: {method}")
            
            eval_function = self.custom_functions[method]
            
            # Prepare context
            context = self._prepare_evaluation_context(method)
            
            # Execute evaluation
            evaluation = eval_function(item_result, dataset_item, context)
            evaluation.evaluation_time = (datetime.utcnow() - start_time).total_seconds()
            
            return evaluation
            
        except Exception as e:
            error_msg = f"Evaluation failed for item {item_result.item_id}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            return ItemEvaluation(
                item_id=item_result.item_id,
                score=0.0,
                error=error_msg,
                evaluation_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _evaluate_items_parallel(
        self,
        item_results: List[ItemResult],
        dataset_items: List[DatasetItem]
    ) -> List[ItemEvaluation]:
        """Evaluate items in parallel."""
        item_dict = {item.id: item for item in dataset_items}
        evaluations = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit evaluation tasks
            future_to_item = {}
            for item_result in item_results:
                dataset_item = item_dict.get(item_result.item_id)
                if dataset_item:
                    # Determine evaluation method
                    if self.config.default_metrics:
                        primary_metric = self.config.default_metrics[0]
                        eval_method = self.config.evaluation_methods.get(primary_metric, "similarity_evaluation")
                    else:
                        eval_method = "similarity_evaluation"
                    
                    future = executor.submit(
                        self.evaluate_item,
                        item_result,
                        dataset_item,
                        eval_method
                    )
                    future_to_item[future] = item_result.item_id
            
            # Collect results
            for future in as_completed(future_to_item, timeout=self.config.timeout_seconds):
                try:
                    evaluation = future.result()
                    evaluations.append(evaluation)
                except Exception as e:
                    item_id = future_to_item[future]
                    error_msg = f"Evaluation timeout/error for item {item_id}: {str(e)}"
                    logger.error(error_msg)
                    evaluations.append(ItemEvaluation(
                        item_id=item_id,
                        score=0.0,
                        error=error_msg
                    ))
        
        return evaluations
    
    def _evaluate_items_sequential(
        self,
        item_results: List[ItemResult],
        dataset_items: List[DatasetItem]
    ) -> List[ItemEvaluation]:
        """Evaluate items sequentially."""
        item_dict = {item.id: item for item in dataset_items}
        evaluations = []
        
        for item_result in item_results:
            dataset_item = item_dict.get(item_result.item_id)
            if dataset_item:
                # For each item, we need to get the primary evaluation method
                # Use the first metric's method as the primary evaluation
                if self.config.default_metrics:
                    primary_metric = self.config.default_metrics[0]
                    eval_method = self.config.evaluation_methods.get(primary_metric, "similarity_evaluation")
                else:
                    eval_method = "similarity_evaluation"
                
                evaluation = self.evaluate_item(item_result, dataset_item, eval_method)
                evaluations.append(evaluation)
            else:
                logger.warning(f"Dataset item not found for result: {item_result.item_id}")
                evaluations.append(ItemEvaluation(
                    item_id=item_result.item_id,
                    score=0.0,
                    error=f"Dataset item not found: {item_result.item_id}"
                ))
        
        return evaluations
    
    def _calculate_overall_score(self, evaluations: List[ItemEvaluation]) -> float:
        """Calculate overall score from item evaluations."""
        valid_scores = [e.score for e in evaluations if e.error is None and e.score is not None]
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / len(valid_scores)
    
    def _calculate_metric_scores(self, evaluations: List[ItemEvaluation]) -> Dict[str, float]:
        """Calculate aggregated metric scores."""
        metric_scores = {}
        
        # Collect all metric names
        all_metrics = set()
        for evaluation in evaluations:
            if evaluation.error is None:
                all_metrics.update(evaluation.metric_scores.keys())
        
        # Calculate average for each metric
        for metric in all_metrics:
            scores = []
            for evaluation in evaluations:
                if evaluation.error is None and metric in evaluation.metric_scores:
                    scores.append(evaluation.metric_scores[metric])
            
            if scores:
                metric_scores[metric] = sum(scores) / len(scores)
        
        return metric_scores
    
    def _get_default_method(self, dataset_item: DatasetItem) -> str:
        """Determine default evaluation method for an item."""
        # Check if item has expected output for similarity matching
        if dataset_item.expected_output:
            return "similarity_match"
        
        # Check for rule-based criteria
        if dataset_item.evaluation_criteria:
            return "rule_based"
        
        # Fall back to LLM-based if configured
        if self.config.llm_evaluator_config:
            return "llm_based"
        
        # Final fallback
        return "similarity_match"
    
    def _prepare_evaluation_context(self, method: str) -> Dict[str, Any]:
        """Prepare context for evaluation function."""
        context = {}
        
        # Add LLM evaluator config if needed
        if self.config.llm_evaluator_config and method == "llm_based":
            context["llm_evaluator_config"] = self.config.llm_evaluator_config
        
        # Add method-specific configs
        if method in self.config.custom_function_configs:
            context.update(self.config.custom_function_configs[method])
        
        return context