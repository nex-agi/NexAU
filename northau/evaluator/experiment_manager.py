"""Experiment Manager for orchestrating evaluation sessions."""

from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid
import time
from pathlib import Path

from .config import Config
from ..autotuning.dataset import Dataset
from .evaluator import Evaluator, EvaluationConfig, EvaluationResult
from .experiment_runner import ExperimentRunner, ExperimentResult
from .database.connection import get_engine, get_db_session
from .database.repositories import (
    SessionRepository,
    ExperimentRepository,
    ItemResultRepository,
    ConfigRepository
)
from .database.models import SessionBase, ExperimentBase, ItemResultBase, SessionMode, ExperimentStatus

logger = logging.getLogger(__name__)

# Forward declaration for Agent type
Agent = Any


@dataclass
class EvaluationResults:
    """Results from a complete evaluation session."""
    
    session_id: str
    best_config: Config
    best_score: float
    total_experiments: int
    convergence_reached: bool
    total_time_seconds: float
    experiment_results: List[ExperimentResult] = field(default_factory=list)
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    tuning_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "best_config": self.best_config.to_dict(),
            "best_score": self.best_score,
            "total_experiments": self.total_experiments,
            "convergence_reached": self.convergence_reached,
            "total_time_seconds": self.total_time_seconds,
            "tuning_insights": self.tuning_insights
        }


@dataclass
class EvaluationReport:
    """Report comparing multiple configurations in evaluation-only mode."""
    
    config_results: Dict[str, EvaluationResult]
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def comparison_table(self) -> str:
        """Generate comparison table of all configurations."""
        if not self.config_results:
            return "No results to compare"
        
        # Header
        table_lines = ["Configuration Comparison", "=" * 50]
        table_lines.append(f"{'Config ID':<20} {'Overall Score':<15} {'Top Metrics'}")
        table_lines.append("-" * 70)
        
        # Sort configs by score
        sorted_configs = sorted(
            self.config_results.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        for config_name, result in sorted_configs:
            # Get top 2 metrics
            top_metrics = sorted(
                result.metric_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            
            metrics_str = ", ".join([f"{k}:{v:.3f}" for k, v in top_metrics])
            
            table_lines.append(
                f"{config_name:<20} {result.overall_score:<15.4f} {metrics_str}"
            )
        
        return "\n".join(table_lines)
    
    def best_config(self) -> Tuple[str, EvaluationResult]:
        """Return best performing configuration."""
        if not self.config_results:
            return None, None
        
        best_config_name = max(
            self.config_results.keys(),
            key=lambda k: self.config_results[k].overall_score
        )
        
        return best_config_name, self.config_results[best_config_name]
    
    def save_html(self, filepath: str) -> None:
        """Save detailed HTML report."""
        html_content = self._generate_html_report()
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def save_csv(self, filepath: str) -> None:
        """Save results as CSV for analysis."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["config_id", "overall_score"]
            if self.config_results:
                first_result = next(iter(self.config_results.values()))
                header.extend(first_result.metric_scores.keys())
            writer.writerow(header)
            
            # Data
            for config_name, result in self.config_results.items():
                row = [config_name, result.overall_score]
                row.extend(result.metric_scores.values())
                writer.writerow(row)
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #e8f5e8; }}
            </style>
        </head>
        <body>
            <h1>Evaluation Report</h1>
            <p>Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Configuration Comparison</h2>
            <table>
                <tr>
                    <th>Config ID</th>
                    <th>Overall Score</th>
                    <th>Metrics</th>
                </tr>
        """
        
        # Sort and add rows
        sorted_configs = sorted(
            self.config_results.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        for i, (config_name, result) in enumerate(sorted_configs):
            row_class = "best" if i == 0 else ""
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in result.metric_scores.items()])
            
            html += f"""
                <tr class="{row_class}">
                    <td>{config_name}</td>
                    <td>{result.overall_score:.4f}</td>
                    <td>{metrics_str}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


class ExperimentManager:
    """Orchestrate the evaluation process and manage experiment lifecycle."""
    
    def __init__(
        self,
        max_experiments: Optional[int] = None,
        target_score: Optional[float] = None,
        mode: str = "evaluation",
        database_env: str = "development",
        session_name: Optional[str] = None
    ):
        self.max_experiments = max_experiments
        self.target_score = target_score
        self.mode = SessionMode(mode)
        self.session_name = session_name
        
        # Initialize database
        self.engine = get_engine(database_env)
        
        # Components (initialized later)
        self.evaluator: Optional[Evaluator] = None
        self.experiment_runner: Optional[ExperimentRunner] = None
        
        # State tracking
        self.current_session_id: Optional[str] = None
        self.experiment_count = 0
        
        logger.info(f"Initialized ExperimentManager in {mode} mode")
    
    def _initialize_components(
        self,
        agent_factory: Callable[[Config], Agent],
        evaluation_config: Optional[EvaluationConfig] = None,
        evaluator: Optional[Evaluator] = None,
        custom_eval_functions: Optional[Dict[str, Callable]] = None
    ) -> None:
        """Initialize evaluator and experiment runner components."""
        if evaluator is not None:
            # Use provided evaluator
            self.evaluator = evaluator
        else:
            # Create new evaluator
            self.evaluator = Evaluator(evaluation_config or EvaluationConfig())
            # Register custom evaluation functions if provided
            if custom_eval_functions:
                for name, func in custom_eval_functions.items():
                    self.evaluator.register_eval_function(name, func)
        
        self.experiment_runner = ExperimentRunner(agent_factory)
    
    def run_evaluation(
        self,
        dataset: Dataset,
        configs: List[Config],
        agent_factory: Callable[[Config], Agent],
        evaluation_config: Optional[EvaluationConfig] = None,
        evaluator: Optional[Evaluator] = None,
        custom_eval_functions: Optional[Dict[str, Callable]] = None
    ) -> EvaluationReport:
        """Run evaluation-only mode on multiple configurations."""
        logger.info(f"Starting evaluation-only mode with {len(configs)} configurations")
        
        # Initialize components
        self._initialize_components(agent_factory, evaluation_config, evaluator, custom_eval_functions)
        
        # Create session in database
        session_id = self._create_session(dataset, configs[0] if configs else None, {
            "mode": "evaluation_only",
            "config_count": len(configs)
        })
        
        self.current_session_id = session_id
        start_time = time.time()
        
        try:
            config_results = {}
            
            for i, config in enumerate(configs):
                logger.info(f"Evaluating configuration {i+1}/{len(configs)}: {config.config_id}")
                
                # Run experiment
                experiment_id = f"eval_{session_id}_{i:03d}"
                experiment_result = self.experiment_runner.run_experiment(
                    dataset, config, experiment_id
                )
                
                # Evaluate results
                evaluation_result = self.evaluator.evaluate_experiment(experiment_result)
                
                # Store in database
                self._store_experiment_results(experiment_result, evaluation_result, session_id)
                
                config_results[config.config_id] = evaluation_result
            
            # Generate comparative analysis
            comparative_analysis = self._generate_comparative_analysis(config_results)
            
            # Create and return report
            report = EvaluationReport(
                config_results=config_results,
                comparative_analysis=comparative_analysis
            )
            
            # Update session completion status
            best_result = max(config_results.values(), key=lambda x: x.overall_score) if config_results else None
            if best_result:
                # Find the actual best config
                best_config_id = None
                for config_id, result in config_results.items():
                    if result.overall_score == best_result.overall_score:
                        best_config_id = config_id
                        break
                
                best_config = next((c for c in configs if c.config_id == best_config_id), configs[0])
                
                # Calculate total time
                total_time = time.time() - start_time
                
                # Create EvaluationResults object for the session update
                session_results = EvaluationResults(
                    session_id=session_id,
                    best_config=best_config,
                    best_score=best_result.overall_score,
                    total_experiments=len(configs),
                    convergence_reached=True,
                    total_time_seconds=total_time
                )
                
                # Update session in database
                self._update_session_completion(session_id, session_results, total_time)
            
            logger.info("Evaluation-only session completed")
            return report
            
        except Exception as e:
            logger.error(f"Evaluation session failed: {e}")
            self._update_session_error(session_id, str(e))
            raise
    
    def should_continue(self) -> bool:
        """Check if evaluation should continue based on criteria."""
        if self.max_experiments and self.experiment_count >= self.max_experiments:
            logger.info(f"Reached maximum experiments: {self.max_experiments}")
            return False
        
        return True
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        status = {
            "session_id": self.current_session_id,
            "mode": self.mode.value,
            "experiment_count": self.experiment_count,
            "max_experiments": self.max_experiments,
            "target_score": self.target_score
        }
        
        return status
    
    def _create_session(
        self,
        dataset: Dataset,
        initial_config: Optional[Config],
        parameters: Dict[str, Any]
    ) -> str:
        """Create new session in database."""
        session_id = str(uuid.uuid4())
        
        session_data = SessionBase(
            name=self.session_name or f"Evaluation Session {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=self.mode,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            initial_config_id=initial_config.config_id if initial_config else None,
            parameters=parameters,
            status=ExperimentStatus.RUNNING
        )
        
        with get_db_session(self.engine) as db_session:
            session_repo = SessionRepository(db_session)
            session = session_repo.create_session(session_data)
            session_id = session.id  # Get ID within session context
            
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def _run_single_evaluation(
        self,
        dataset: Dataset,
        config: Config,
        session_id: str
    ) -> EvaluationResults:
        """Run a single evaluation of the given configuration."""
        logger.info("=== Running Single Evaluation ===")
        
        # Run experiment with the config
        experiment_id = f"{session_id}_001"
        experiment_result = self.experiment_runner.run_experiment(
            dataset, config, experiment_id
        )
        
        # Evaluate results
        evaluation_result = self.evaluator.evaluate_experiment(experiment_result)
        
        # Store results
        self._store_experiment_results(experiment_result, evaluation_result, session_id)
        
        # Track results
        self.experiment_count = 1
        
        logger.info("Single evaluation completed - manual tuning mode")
        
        return EvaluationResults(
            session_id=session_id,
            best_config=config,
            best_score=evaluation_result.overall_score,
            total_experiments=1,
            convergence_reached=True,  # Single evaluation in manual mode
            total_time_seconds=0.0,  # Will be set by caller
            experiment_results=[experiment_result],
            evaluation_results=[evaluation_result],
            tuning_insights=[]  # No insights in manual mode
        )
    
    def _store_experiment_results(
        self,
        experiment_result: ExperimentResult,
        evaluation_result: EvaluationResult,
        session_id: str
    ) -> None:
        """Store experiment and evaluation results in database."""
        with get_db_session(self.engine) as db_session:
            # Store experiment
            exp_repo = ExperimentRepository(db_session)
            experiment_data = ExperimentBase(
                session_id=session_id,
                config_id=experiment_result.config.config_id,
                config_data=experiment_result.config.to_dict(),
                dataset_items=len(experiment_result.item_results),
                status=ExperimentStatus.COMPLETED,
                execution_time_seconds=int(experiment_result.total_execution_time),
                overall_score=evaluation_result.overall_score,
                metric_scores=evaluation_result.metric_scores,
                token_usage=experiment_result.total_tokens,
                cost_usd=experiment_result.total_tokens.get("total_cost", 0.0)
            )
            
            experiment = exp_repo.create_experiment(experiment_data)
            
            # Store item results
            item_repo = ItemResultRepository(db_session)
            item_results_data = []
            
            for item_result in experiment_result.item_results:
                # Find corresponding evaluation
                item_eval = next(
                    (e for e in evaluation_result.item_evaluations if e.item_id == item_result.item_id),
                    None
                )
                
                item_data = ItemResultBase(
                    experiment_id=experiment.id,
                    item_id=item_result.item_id,
                    score=item_eval.score if item_eval else 0.0,
                    metric_scores=item_eval.metric_scores if item_eval else {},
                    agent_output=item_result.agent_output,
                    execution_time=item_result.execution_time,
                    token_usage=item_result.token_usage,
                    evaluation_feedback=item_eval.feedback if item_eval else None,
                    error_message=item_result.error
                )
                item_results_data.append(item_data)
            
            item_repo.batch_create_item_results(item_results_data)
    
    def _update_session_completion(
        self,
        session_id: str,
        results: EvaluationResults,
        total_time: float
    ) -> None:
        """Update session with completion data."""
        with get_db_session(self.engine) as db_session:
            session_repo = SessionRepository(db_session)
            session_repo.update_session(session_id, {
                "status": ExperimentStatus.COMPLETED,
                "completed_at": datetime.utcnow(),
                "total_experiments": results.total_experiments,
                "best_score": results.best_score,
                "best_config_id": results.best_config.config_id,
                "convergence_reached": results.convergence_reached,
                "total_time_seconds": int(total_time)
            })
    
    def _update_session_error(self, session_id: str, error_message: str) -> None:
        """Update session with error information."""
        with get_db_session(self.engine) as db_session:
            session_repo = SessionRepository(db_session)
            session_repo.update_session(session_id, {
                "status": ExperimentStatus.FAILED,
                "completed_at": datetime.utcnow()
            })
    
    def _generate_comparative_analysis(
        self,
        config_results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """Generate comparative analysis of multiple configurations."""
        if not config_results:
            return {}
        
        scores = [result.overall_score for result in config_results.values()]
        
        analysis = {
            "summary": {
                "total_configs": len(config_results),
                "best_score": max(scores),
                "worst_score": min(scores),
                "average_score": sum(scores) / len(scores),
                "score_range": max(scores) - min(scores)
            },
            "rankings": []
        }
        
        # Create rankings
        sorted_configs = sorted(
            config_results.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        for rank, (config_name, result) in enumerate(sorted_configs, 1):
            analysis["rankings"].append({
                "rank": rank,
                "config_id": config_name,
                "score": result.overall_score,
                "metrics": result.metric_scores
            })
        
        return analysis