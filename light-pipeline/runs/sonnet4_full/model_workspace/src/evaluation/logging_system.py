"""LoggingSystem for comprehensive experiment logging and data collection."""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
import os
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from loguru import logger
import pickle

from .metrics_calculator import SystemPerformanceMetrics, MetricResult
from .baseline_comparator import ComparisonAnalysis
from .accuracy_evaluator import EvaluationSummary, AccuracyResult
from ..agents.answering_agent import AgentResponse
from ..agents.challenger_agent import ChallengeReport
from ..challenge.challenge_processor import ProcessedAnalysis
from ..challenge.revision_manager import RevisionSession


class LogLevel(Enum):
    """Logging levels for different types of events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExperimentPhase(Enum):
    """Phases of an experiment run."""
    SETUP = "setup"
    DATA_PREPARATION = "data_preparation"
    AGENT_PROCESSING = "agent_processing"
    CHALLENGE_PHASE = "challenge_phase"
    REVISION_PHASE = "revision_phase"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    CLEANUP = "cleanup"


@dataclass
class LogEntry:
    """Individual log entry with structured data."""
    
    timestamp: str
    level: LogLevel
    phase: ExperimentPhase
    component: str
    event_type: str
    message: str
    data: Dict[str, Any]
    duration: Optional[float]
    error_details: Optional[str]


@dataclass
class ExperimentRun:
    """Complete experiment run with metadata and results."""
    
    run_id: str
    experiment_name: str
    start_time: str
    end_time: Optional[str]
    duration: Optional[float]
    
    # Configuration
    system_config: Dict[str, Any]
    test_dataset: Dict[str, Any]
    
    # Results
    performance_metrics: Optional[SystemPerformanceMetrics]
    comparison_analysis: Optional[ComparisonAnalysis]
    accuracy_evaluation: Optional[EvaluationSummary]
    
    # Detailed logs
    log_entries: List[LogEntry]
    error_count: int
    warning_count: int
    
    # Status
    status: str  # "running", "completed", "failed", "interrupted"
    success: bool


class LoggingSystem:
    """
    Comprehensive logging system for experiments and system evaluation.
    
    Provides structured logging, data collection, experiment tracking,
    and result analysis for the cite-and-challenge system.
    """
    
    def __init__(
        self,
        base_log_dir: Union[str, Path] = "logs",
        experiment_name: str = "cite_challenge_experiment",
        enable_file_logging: bool = True,
        enable_structured_logging: bool = True,
        auto_save_interval: int = 100  # Auto-save every N log entries
    ):
        """Initialize the logging system."""
        
        self.base_log_dir = Path(base_log_dir)
        self.experiment_name = experiment_name
        self.enable_file_logging = enable_file_logging
        self.enable_structured_logging = enable_structured_logging
        self.auto_save_interval = auto_save_interval
        
        # Create logging directory
        self.base_log_dir.mkdir(exist_ok=True)
        
        # Current experiment tracking
        self.current_run: Optional[ExperimentRun] = None
        self.log_entries: List[LogEntry] = []
        self.entry_count = 0
        
        # Performance tracking
        self.component_timings: Dict[str, List[float]] = defaultdict(list)
        self.error_tracking: Dict[str, int] = defaultdict(int)
        
        # File handles
        self.log_files: Dict[str, Any] = {}
        
        self._setup_logging()
        
        logger.info("LoggingSystem initialized")
    
    def _setup_logging(self):
        """Set up logging infrastructure."""
        
        if self.enable_file_logging:
            # Create timestamped log directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_log_dir = self.base_log_dir / f"{self.experiment_name}_{timestamp}"
            self.run_log_dir.mkdir(exist_ok=True)
            
            # Set up CSV logger for structured data
            if self.enable_structured_logging:
                csv_path = self.run_log_dir / "experiment_log.csv"
                self.log_files["csv"] = open(csv_path, 'w', newline='')
                self.csv_writer = csv.DictWriter(
                    self.log_files["csv"],
                    fieldnames=[
                        'timestamp', 'level', 'phase', 'component', 'event_type',
                        'message', 'duration', 'error_details'
                    ]
                )
                self.csv_writer.writeheader()
            
            # Set up JSON logger for detailed data
            json_path = self.run_log_dir / "detailed_log.jsonl"
            self.log_files["json"] = open(json_path, 'w')
    
    def start_experiment(
        self,
        experiment_name: str,
        system_config: Dict[str, Any],
        test_dataset: Dict[str, Any]
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            experiment_name: Name of the experiment
            system_config: System configuration parameters
            test_dataset: Test dataset information
            
        Returns:
            Unique run ID for this experiment
        """
        
        run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            start_time=self._get_timestamp(),
            end_time=None,
            duration=None,
            system_config=system_config,
            test_dataset=test_dataset,
            performance_metrics=None,
            comparison_analysis=None,
            accuracy_evaluation=None,
            log_entries=[],
            error_count=0,
            warning_count=0,
            status="running",
            success=False
        )
        
        self.log_event(
            level=LogLevel.INFO,
            phase=ExperimentPhase.SETUP,
            component="logging_system",
            event_type="experiment_start",
            message=f"Started experiment: {experiment_name}",
            data={
                "run_id": run_id,
                "config_keys": list(system_config.keys()),
                "test_dataset_size": test_dataset.get("size", 0)
            }
        )
        
        return run_id
    
    def end_experiment(
        self,
        success: bool = True,
        final_results: Optional[Dict[str, Any]] = None
    ):
        """
        End the current experiment run.
        
        Args:
            success: Whether the experiment completed successfully
            final_results: Optional final results to include
        """
        
        if not self.current_run:
            logger.warning("No active experiment to end")
            return
        
        end_time = self._get_timestamp()
        start_time = datetime.fromisoformat(self.current_run.start_time)
        end_time_dt = datetime.fromisoformat(end_time)
        duration = (end_time_dt - start_time).total_seconds()
        
        self.current_run.end_time = end_time
        self.current_run.duration = duration
        self.current_run.status = "completed" if success else "failed"
        self.current_run.success = success
        self.current_run.log_entries = self.log_entries.copy()
        
        self.log_event(
            level=LogLevel.INFO,
            phase=ExperimentPhase.CLEANUP,
            component="logging_system",
            event_type="experiment_end",
            message=f"Ended experiment: {self.current_run.experiment_name}",
            data={
                "duration": duration,
                "success": success,
                "total_log_entries": len(self.log_entries),
                "error_count": self.current_run.error_count,
                "warning_count": self.current_run.warning_count,
                **(final_results or {})
            },
            duration=duration
        )
        
        # Save experiment data
        self._save_experiment_data()
        
        # Close log files
        self._close_log_files()
        
        logger.success(f"Experiment completed in {duration:.2f}s")
    
    def log_event(
        self,
        level: LogLevel,
        phase: ExperimentPhase,
        component: str,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        error_details: Optional[str] = None
    ):
        """
        Log a structured event.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            phase: Experiment phase
            component: Component that generated the event
            event_type: Type of event (e.g., "processing_start", "error")
            message: Human-readable message
            data: Additional structured data
            duration: Duration for timed events
            error_details: Detailed error information
        """
        
        entry = LogEntry(
            timestamp=self._get_timestamp(),
            level=level,
            phase=phase,
            component=component,
            event_type=event_type,
            message=message,
            data=data or {},
            duration=duration,
            error_details=error_details
        )
        
        self.log_entries.append(entry)
        self.entry_count += 1
        
        # Update counters
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            if self.current_run:
                self.current_run.error_count += 1
            self.error_tracking[component] += 1
        elif level == LogLevel.WARNING:
            if self.current_run:
                self.current_run.warning_count += 1
        
        # Log to console (via loguru)
        log_message = f"[{phase.value}:{component}] {message}"
        if level == LogLevel.DEBUG:
            logger.debug(log_message)
        elif level == LogLevel.INFO:
            logger.info(log_message)
        elif level == LogLevel.WARNING:
            logger.warning(log_message)
        elif level == LogLevel.ERROR:
            logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            logger.critical(log_message)
        
        # Write to log files
        if self.enable_file_logging:
            self._write_to_log_files(entry)
        
        # Auto-save periodically
        if self.entry_count % self.auto_save_interval == 0:
            self._auto_save()
    
    def log_performance_metrics(
        self,
        metrics: SystemPerformanceMetrics,
        phase: ExperimentPhase = ExperimentPhase.EVALUATION
    ):
        """Log system performance metrics."""
        
        if self.current_run:
            self.current_run.performance_metrics = metrics
        
        self.log_event(
            level=LogLevel.INFO,
            phase=phase,
            component="metrics_calculator",
            event_type="performance_metrics",
            message="System performance metrics calculated",
            data={
                "overall_accuracy": metrics.overall_accuracy,
                "citation_accuracy": metrics.citation_accuracy,
                "evidence_accuracy": metrics.evidence_accuracy,
                "avg_processing_time": metrics.avg_processing_time,
                "revision_success_rate": metrics.revision_success_rate,
                "total_evaluations": metrics.total_evaluations
            }
        )
    
    def log_comparison_analysis(
        self,
        analysis: ComparisonAnalysis,
        phase: ExperimentPhase = ExperimentPhase.ANALYSIS
    ):
        """Log baseline comparison analysis."""
        
        if self.current_run:
            self.current_run.comparison_analysis = analysis
        
        self.log_event(
            level=LogLevel.INFO,
            phase=phase,
            component="baseline_comparator",
            event_type="comparison_analysis",
            message="Baseline comparison completed",
            data={
                "overall_improvement": analysis.overall_improvement,
                "accuracy_improvement": analysis.accuracy_comparison.improvement_percentage,
                "quality_improvement": analysis.quality_comparison.improvement_percentage,
                "significant_improvements": len(analysis.significant_improvements),
                "baselines_tested": len(analysis.baseline_performance)
            }
        )
    
    def log_accuracy_evaluation(
        self,
        evaluation: EvaluationSummary,
        phase: ExperimentPhase = ExperimentPhase.EVALUATION
    ):
        """Log accuracy evaluation results."""
        
        if self.current_run:
            self.current_run.accuracy_evaluation = evaluation
        
        self.log_event(
            level=LogLevel.INFO,
            phase=phase,
            component="accuracy_evaluator",
            event_type="accuracy_evaluation",
            message="Accuracy evaluation completed",
            data={
                "overall_accuracy": evaluation.overall_accuracy,
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "f1_score": evaluation.f1_score,
                "total_items": evaluation.total_items,
                "correct_items": evaluation.correct_items
            }
        )
    
    def log_agent_response(
        self,
        response: AgentResponse,
        phase: ExperimentPhase = ExperimentPhase.AGENT_PROCESSING
    ):
        """Log an agent response."""
        
        self.log_event(
            level=LogLevel.DEBUG,
            phase=phase,
            component="answering_agent",
            event_type="agent_response",
            message=f"Agent {response.agent_id} responded to claim",
            data={
                "agent_id": response.agent_id,
                "confidence_score": response.confidence_score,
                "citation_count": len(response.citations),
                "evidence_count": len(response.evidence),
                "token_usage": response.token_usage,
                "processing_time": response.processing_time
            },
            duration=response.processing_time
        )
    
    def log_challenge_report(
        self,
        report: ChallengeReport,
        phase: ExperimentPhase = ExperimentPhase.CHALLENGE_PHASE
    ):
        """Log a challenge report."""
        
        self.log_event(
            level=LogLevel.DEBUG,
            phase=phase,
            component="challenger_agent",
            event_type="challenge_report",
            message=f"Challenger {report.challenger_id} generated report",
            data={
                "challenger_id": report.challenger_id,
                "challenge_count": len(report.challenges),
                "requires_revision": report.requires_revision,
                "confidence_in_challenges": report.confidence_in_challenges,
                "priority_challenges": len(report.priority_challenges),
                "token_usage": report.token_usage,
                "processing_time": report.processing_time
            },
            duration=report.processing_time
        )
    
    def log_revision_session(
        self,
        session: RevisionSession,
        phase: ExperimentPhase = ExperimentPhase.REVISION_PHASE
    ):
        """Log a revision session."""
        
        self.log_event(
            level=LogLevel.DEBUG,
            phase=phase,
            component="revision_manager",
            event_type="revision_session",
            message=f"Revision session completed for {session.session_id}",
            data={
                "session_id": session.session_id,
                "revision_success": session.revision_success,
                "overall_improvement": session.overall_improvement,
                "revised_responses": len(session.revised_responses),
                "processing_time": session.processing_time
            },
            duration=session.processing_time
        )
    
    def start_timing(self, component: str, operation: str) -> str:
        """Start timing an operation."""
        
        timing_id = f"{component}_{operation}_{int(time.time() * 1000)}"
        self._timing_starts = getattr(self, '_timing_starts', {})
        self._timing_starts[timing_id] = time.time()
        
        return timing_id
    
    def end_timing(
        self,
        timing_id: str,
        component: str,
        operation: str,
        phase: ExperimentPhase = ExperimentPhase.AGENT_PROCESSING
    ):
        """End timing an operation and log the duration."""
        
        if not hasattr(self, '_timing_starts') or timing_id not in self._timing_starts:
            logger.warning(f"Timing ID {timing_id} not found")
            return
        
        duration = time.time() - self._timing_starts[timing_id]
        del self._timing_starts[timing_id]
        
        self.component_timings[component].append(duration)
        
        self.log_event(
            level=LogLevel.DEBUG,
            phase=phase,
            component=component,
            event_type="timing",
            message=f"Completed {operation}",
            data={"operation": operation},
            duration=duration
        )
    
    def log_error(
        self,
        component: str,
        error: Exception,
        phase: ExperimentPhase,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log an error with detailed information."""
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "component": component,
            **(additional_data or {})
        }
        
        self.log_event(
            level=LogLevel.ERROR,
            phase=phase,
            component=component,
            event_type="error",
            message=f"Error in {component}: {str(error)}",
            data=error_details,
            error_details=str(error)
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from collected metrics."""
        
        summary = {
            "total_log_entries": len(self.log_entries),
            "error_count": sum(self.error_tracking.values()),
            "component_errors": dict(self.error_tracking),
            "component_timings": {}
        }
        
        # Calculate timing statistics
        for component, timings in self.component_timings.items():
            if timings:
                summary["component_timings"][component] = {
                    "total_operations": len(timings),
                    "total_time": sum(timings),
                    "avg_time": sum(timings) / len(timings),
                    "min_time": min(timings),
                    "max_time": max(timings)
                }
        
        return summary
    
    def export_experiment_data(
        self,
        output_path: Optional[Path] = None,
        format: str = "json"
    ) -> Path:
        """Export experiment data to file."""
        
        if not self.current_run:
            raise ValueError("No active experiment to export")
        
        if output_path is None:
            output_path = self.run_log_dir / f"experiment_export.{format}"
        else:
            output_path = Path(output_path)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(asdict(self.current_run), f, indent=2, default=str)
        elif format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(self.current_run, f)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Experiment data exported to {output_path}")
        return output_path
    
    def _write_to_log_files(self, entry: LogEntry):
        """Write log entry to files."""
        
        try:
            # Write to CSV
            if "csv" in self.log_files and self.csv_writer:
                csv_row = {
                    'timestamp': entry.timestamp,
                    'level': entry.level.value,
                    'phase': entry.phase.value,
                    'component': entry.component,
                    'event_type': entry.event_type,
                    'message': entry.message,
                    'duration': entry.duration,
                    'error_details': entry.error_details
                }
                self.csv_writer.writerow(csv_row)
                self.log_files["csv"].flush()
            
            # Write to JSONL
            if "json" in self.log_files:
                json_entry = asdict(entry)
                json_entry['level'] = entry.level.value
                json_entry['phase'] = entry.phase.value
                self.log_files["json"].write(json.dumps(json_entry, default=str) + "\n")
                self.log_files["json"].flush()
        
        except Exception as e:
            logger.error(f"Failed to write to log files: {e}")
    
    def _auto_save(self):
        """Periodically save experiment state."""
        
        if self.current_run and hasattr(self, 'run_log_dir'):
            try:
                save_path = self.run_log_dir / "experiment_state.json"
                temp_run = self.current_run
                temp_run.log_entries = self.log_entries[-1000:]  # Keep last 1000 entries
                
                with open(save_path, 'w') as f:
                    json.dump(asdict(temp_run), f, indent=2, default=str)
            
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
    
    def _save_experiment_data(self):
        """Save complete experiment data."""
        
        if not self.current_run or not hasattr(self, 'run_log_dir'):
            return
        
        try:
            # Save as JSON
            json_path = self.run_log_dir / "experiment_complete.json"
            with open(json_path, 'w') as f:
                json.dump(asdict(self.current_run), f, indent=2, default=str)
            
            # Save as pickle for Python objects
            pickle_path = self.run_log_dir / "experiment_complete.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.current_run, f)
            
            # Save performance summary
            summary_path = self.run_log_dir / "performance_summary.json"
            summary = self.get_performance_summary()
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Experiment data saved to {self.run_log_dir}")
        
        except Exception as e:
            logger.error(f"Failed to save experiment data: {e}")
    
    def _close_log_files(self):
        """Close all open log files."""
        
        for file_handle in self.log_files.values():
            try:
                file_handle.close()
            except Exception as e:
                logger.error(f"Error closing log file: {e}")
        
        self.log_files.clear()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_run and self.current_run.status == "running":
            success = exc_type is None
            self.end_experiment(success=success)
        
        self._close_log_files()