"""Module 5: Evaluation and Metrics - Comprehensive system evaluation components."""

from .metrics_calculator import (
    MetricsCalculator,
    SystemPerformanceMetrics,
    MetricResult,
    MetricType
)

from .baseline_comparator import (
    BaselineComparator,
    ComparisonAnalysis,
    BaselineResult,
    BaselineType,
    ComparisonMethod
)

from .accuracy_evaluator import (
    AccuracyEvaluator,
    AccuracyResult,
    EvaluationSummary,
    GroundTruthEntry,
    EvaluationMetric
)

from .logging_system import (
    LoggingSystem,
    ExperimentRun,
    LogEntry,
    LogLevel,
    ExperimentPhase
)

__all__ = [
    # MetricsCalculator
    'MetricsCalculator',
    'SystemPerformanceMetrics',
    'MetricResult',
    'MetricType',
    
    # BaselineComparator
    'BaselineComparator',
    'ComparisonAnalysis',
    'BaselineResult',
    'BaselineType',
    'ComparisonMethod',
    
    # AccuracyEvaluator
    'AccuracyEvaluator',
    'AccuracyResult',
    'EvaluationSummary',
    'GroundTruthEntry',
    'EvaluationMetric',
    
    # LoggingSystem
    'LoggingSystem',
    'ExperimentRun',
    'LogEntry',
    'LogLevel',
    'ExperimentPhase'
]