"""MetricsCalculator for comprehensive system evaluation and performance assessment."""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
import statistics
from datetime import datetime
from collections import defaultdict, Counter
from loguru import logger
import numpy as np

from ..agents.answering_agent import AgentResponse
from ..agents.challenger_agent import ChallengeReport, Challenge
from ..challenge.challenge_processor import ProcessedAnalysis
from ..challenge.revision_manager import RevisionSession, RevisedResponse
from ..challenge.feedback_generator import StructuredFeedback


class MetricType(Enum):
    """Types of metrics that can be calculated."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    CITATION_QUALITY = "citation_quality"
    EVIDENCE_STRENGTH = "evidence_strength"
    PROCESSING_EFFICIENCY = "processing_efficiency"
    CHALLENGE_EFFECTIVENESS = "challenge_effectiveness"
    REVISION_IMPROVEMENT = "revision_improvement"
    

@dataclass
class MetricResult:
    """Individual metric calculation result."""
    
    metric_type: MetricType
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    sample_size: int
    calculation_method: str
    metadata: Dict[str, Any]
    timestamp: str
    

@dataclass
class SystemPerformanceMetrics:
    """Comprehensive system performance metrics."""
    
    # Core accuracy metrics
    overall_accuracy: float
    citation_accuracy: float
    evidence_accuracy: float
    confidence_calibration_score: float
    
    # Challenge metrics
    challenge_precision: float
    challenge_recall: float
    challenge_f1: float
    false_positive_rate: float
    
    # Quality metrics
    avg_response_quality: float
    avg_citation_quality: float
    avg_evidence_strength: float
    
    # Efficiency metrics
    avg_processing_time: float
    token_efficiency: float
    throughput_per_minute: float
    
    # Improvement metrics
    revision_success_rate: float
    avg_improvement_score: float
    issue_resolution_rate: float
    
    # Metadata
    total_evaluations: int
    evaluation_period: str
    timestamp: str
    

@dataclass
class ComparisonMetrics:
    """Metrics comparing system performance to baselines."""
    
    system_score: float
    baseline_score: float
    improvement_percentage: float
    statistical_significance: float
    confidence_level: float
    metric_name: str
    comparison_method: str
    sample_sizes: Tuple[int, int]  # (system, baseline)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for system evaluation.
    
    Calculates accuracy, quality, efficiency, and improvement metrics
    across all system components with statistical analysis.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        min_sample_size: int = 10,
        enable_statistical_tests: bool = True
    ):
        """Initialize the metrics calculator."""
        
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.enable_statistical_tests = enable_statistical_tests
        
        # Metric storage
        self.calculated_metrics: Dict[str, List[MetricResult]] = defaultdict(list)
        self.performance_history: List[SystemPerformanceMetrics] = []
        
        # Calculation statistics
        self.total_calculations = 0
        self.calculation_errors = 0
        self.last_calculation_time = 0.0
        
        logger.info("MetricsCalculator initialized")
    
    async def calculate_system_performance(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport],
        processed_analyses: List[ProcessedAnalysis],
        revision_sessions: List[RevisionSession],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> SystemPerformanceMetrics:
        """
        Calculate comprehensive system performance metrics.
        
        Args:
            agent_responses: List of agent responses to evaluate
            challenge_reports: List of challenge reports
            processed_analyses: List of processed challenge analyses
            revision_sessions: List of revision sessions
            ground_truth: Optional ground truth for accuracy calculation
            
        Returns:
            SystemPerformanceMetrics with comprehensive evaluation
        """
        
        logger.info(f"Calculating system performance metrics for {len(agent_responses)} responses")
        start_time = time.time()
        
        try:
            # Calculate core accuracy metrics
            accuracy_metrics = await self._calculate_accuracy_metrics(
                agent_responses, ground_truth
            )
            
            # Calculate challenge effectiveness metrics
            challenge_metrics = await self._calculate_challenge_metrics(
                challenge_reports, processed_analyses
            )
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                agent_responses, challenge_reports
            )
            
            # Calculate efficiency metrics
            efficiency_metrics = await self._calculate_efficiency_metrics(
                agent_responses, challenge_reports, revision_sessions
            )
            
            # Calculate improvement metrics
            improvement_metrics = await self._calculate_improvement_metrics(
                revision_sessions, processed_analyses
            )
            
            # Compile comprehensive metrics
            performance_metrics = SystemPerformanceMetrics(
                overall_accuracy=accuracy_metrics.get("overall_accuracy", 0.0),
                citation_accuracy=accuracy_metrics.get("citation_accuracy", 0.0),
                evidence_accuracy=accuracy_metrics.get("evidence_accuracy", 0.0),
                confidence_calibration_score=accuracy_metrics.get("confidence_calibration", 0.0),
                challenge_precision=challenge_metrics.get("precision", 0.0),
                challenge_recall=challenge_metrics.get("recall", 0.0),
                challenge_f1=challenge_metrics.get("f1_score", 0.0),
                false_positive_rate=challenge_metrics.get("false_positive_rate", 0.0),
                avg_response_quality=quality_metrics.get("avg_response_quality", 0.0),
                avg_citation_quality=quality_metrics.get("avg_citation_quality", 0.0),
                avg_evidence_strength=quality_metrics.get("avg_evidence_strength", 0.0),
                avg_processing_time=efficiency_metrics.get("avg_processing_time", 0.0),
                token_efficiency=efficiency_metrics.get("token_efficiency", 0.0),
                throughput_per_minute=efficiency_metrics.get("throughput_per_minute", 0.0),
                revision_success_rate=improvement_metrics.get("revision_success_rate", 0.0),
                avg_improvement_score=improvement_metrics.get("avg_improvement_score", 0.0),
                issue_resolution_rate=improvement_metrics.get("issue_resolution_rate", 0.0),
                total_evaluations=len(agent_responses),
                evaluation_period=f"{len(agent_responses)} responses",
                timestamp=self._get_timestamp()
            )
            
            # Store in history
            self.performance_history.append(performance_metrics)
            self.total_calculations += 1
            self.last_calculation_time = time.time() - start_time
            
            logger.success(
                f"System performance calculated in {self.last_calculation_time:.2f}s "
                f"(Overall accuracy: {performance_metrics.overall_accuracy:.3f})"
            )
            
            return performance_metrics
            
        except Exception as e:
            self.calculation_errors += 1
            logger.error(f"Failed to calculate system performance: {str(e)}")
            raise
    
    async def _calculate_accuracy_metrics(
        self,
        agent_responses: List[AgentResponse],
        ground_truth: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate accuracy-related metrics."""
        
        metrics = {}
        
        if not agent_responses:
            return {
                "overall_accuracy": 0.0,
                "citation_accuracy": 0.0,
                "evidence_accuracy": 0.0,
                "confidence_calibration": 0.0
            }
        
        # Overall accuracy (if ground truth available)
        if ground_truth:
            correct_responses = 0
            for response in agent_responses:
                claim = response.claim
                if claim in ground_truth:
                    expected = ground_truth[claim]
                    # Simple heuristic: check if answer sentiment matches expected
                    actual_supports = "support" in response.answer.lower()
                    expected_supports = expected.get("supports", True)
                    if actual_supports == expected_supports:
                        correct_responses += 1
            
            metrics["overall_accuracy"] = correct_responses / len(agent_responses)
        else:
            # Estimate accuracy based on confidence and evidence quality
            weighted_confidence = sum(
                r.confidence_score * self._estimate_response_reliability(r)
                for r in agent_responses
            ) / len(agent_responses)
            metrics["overall_accuracy"] = weighted_confidence
        
        # Citation accuracy
        citation_scores = []
        for response in agent_responses:
            if response.citations:
                valid_citations = sum(
                    1 for c in response.citations 
                    if c.url and c.formatted_citation and len(c.formatted_citation) > 20
                )
                citation_score = valid_citations / len(response.citations)
                citation_scores.append(citation_score)
        
        metrics["citation_accuracy"] = (
            statistics.mean(citation_scores) if citation_scores else 0.0
        )
        
        # Evidence accuracy
        evidence_scores = []
        for response in agent_responses:
            if response.evidence:
                high_quality_evidence = sum(
                    1 for e in response.evidence
                    if e.quality_score >= 0.7 and e.relevance_score >= 0.7
                )
                evidence_score = high_quality_evidence / len(response.evidence)
                evidence_scores.append(evidence_score)
        
        metrics["evidence_accuracy"] = (
            statistics.mean(evidence_scores) if evidence_scores else 0.0
        )
        
        # Confidence calibration
        calibration_scores = []
        for response in agent_responses:
            evidence_strength = self._calculate_evidence_strength(response)
            calibration_score = 1.0 - abs(response.confidence_score - evidence_strength)
            calibration_scores.append(calibration_score)
        
        metrics["confidence_calibration"] = (
            statistics.mean(calibration_scores) if calibration_scores else 0.0
        )
        
        return metrics
    
    async def _calculate_challenge_metrics(
        self,
        challenge_reports: List[ChallengeReport],
        processed_analyses: List[ProcessedAnalysis]
    ) -> Dict[str, float]:
        """Calculate challenge effectiveness metrics."""
        
        if not challenge_reports:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "false_positive_rate": 0.0
            }
        
        # Count challenge outcomes
        true_positives = 0
        false_positives = 0
        total_challenges = 0
        legitimate_issues = 0
        
        for report in challenge_reports:
            for challenge in report.challenges:
                total_challenges += 1
                
                # Heuristic: high severity challenges are more likely legitimate
                if challenge.severity >= 0.7:
                    legitimate_issues += 1
                    if report.requires_revision:
                        true_positives += 1
                else:
                    if report.requires_revision:
                        false_positives += 1
        
        # Calculate metrics
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(legitimate_issues, 1)
        f1_score = (
            2 * (precision * recall) / max(precision + recall, 0.001)
            if (precision + recall) > 0 else 0.0
        )
        false_positive_rate = false_positives / max(total_challenges, 1)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate
        }
    
    async def _calculate_quality_metrics(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> Dict[str, float]:
        """Calculate quality-related metrics."""
        
        # Response quality
        response_qualities = []
        for response in agent_responses:
            quality = self._calculate_response_quality(response)
            response_qualities.append(quality)
        
        avg_response_quality = (
            statistics.mean(response_qualities) if response_qualities else 0.0
        )
        
        # Citation quality
        citation_qualities = []
        for response in agent_responses:
            for citation in response.citations:
                quality = self._calculate_citation_quality(citation)
                citation_qualities.append(quality)
        
        avg_citation_quality = (
            statistics.mean(citation_qualities) if citation_qualities else 0.0
        )
        
        # Evidence strength
        evidence_strengths = []
        for response in agent_responses:
            strength = self._calculate_evidence_strength(response)
            evidence_strengths.append(strength)
        
        avg_evidence_strength = (
            statistics.mean(evidence_strengths) if evidence_strengths else 0.0
        )
        
        return {
            "avg_response_quality": avg_response_quality,
            "avg_citation_quality": avg_citation_quality,
            "avg_evidence_strength": avg_evidence_strength
        }
    
    async def _calculate_efficiency_metrics(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport],
        revision_sessions: List[RevisionSession]
    ) -> Dict[str, float]:
        """Calculate efficiency-related metrics."""
        
        # Processing time
        all_processing_times = []
        all_processing_times.extend(r.processing_time for r in agent_responses)
        all_processing_times.extend(r.processing_time for r in challenge_reports)
        all_processing_times.extend(r.processing_time for r in revision_sessions)
        
        avg_processing_time = (
            statistics.mean(all_processing_times) if all_processing_times else 0.0
        )
        
        # Token efficiency (outputs per token)
        total_tokens = sum(r.token_usage for r in agent_responses)
        total_tokens += sum(r.token_usage for r in challenge_reports)
        
        total_outputs = len(agent_responses) + len(challenge_reports) + len(revision_sessions)
        token_efficiency = total_outputs / max(total_tokens, 1)
        
        # Throughput
        total_time = sum(all_processing_times)
        throughput_per_minute = (
            (total_outputs * 60) / max(total_time, 0.001) if total_time > 0 else 0.0
        )
        
        return {
            "avg_processing_time": avg_processing_time,
            "token_efficiency": token_efficiency,
            "throughput_per_minute": throughput_per_minute
        }
    
    async def _calculate_improvement_metrics(
        self,
        revision_sessions: List[RevisionSession],
        processed_analyses: List[ProcessedAnalysis]
    ) -> Dict[str, float]:
        """Calculate improvement and revision metrics."""
        
        if not revision_sessions:
            return {
                "revision_success_rate": 0.0,
                "avg_improvement_score": 0.0,
                "issue_resolution_rate": 0.0
            }
        
        # Revision success rate
        successful_revisions = sum(
            1 for session in revision_sessions 
            if session.revision_success
        )
        revision_success_rate = successful_revisions / len(revision_sessions)
        
        # Average improvement score
        improvement_scores = [
            session.overall_improvement for session in revision_sessions
            if session.overall_improvement is not None
        ]
        avg_improvement_score = (
            statistics.mean(improvement_scores) if improvement_scores else 0.0
        )
        
        # Issue resolution rate
        total_issues = sum(analysis.total_issues for analysis in processed_analyses)
        resolved_issues = 0
        
        for session in revision_sessions:
            if session.revised_responses:
                resolved_issues += sum(
                    r.issues_addressed for r in session.revised_responses
                )
        
        issue_resolution_rate = resolved_issues / max(total_issues, 1)
        
        return {
            "revision_success_rate": revision_success_rate,
            "avg_improvement_score": avg_improvement_score,
            "issue_resolution_rate": issue_resolution_rate
        }
    
    def _estimate_response_reliability(self, response: AgentResponse) -> float:
        """Estimate reliability of a response based on various factors."""
        
        factors = []
        
        # Citation factor
        if response.citations:
            citation_factor = min(1.0, len(response.citations) / 3.0)
            factors.append(citation_factor)
        
        # Evidence factor
        if response.evidence:
            avg_evidence_quality = sum(e.quality_score for e in response.evidence) / len(response.evidence)
            factors.append(avg_evidence_quality)
        
        # Response length factor (longer responses often more reliable)
        length_factor = min(1.0, len(response.answer.split()) / 100.0)
        factors.append(length_factor)
        
        # Reasoning quality factor
        reasoning_factor = min(1.0, len(response.reasoning.split()) / 50.0)
        factors.append(reasoning_factor)
        
        return statistics.mean(factors) if factors else 0.5
    
    def _calculate_response_quality(self, response: AgentResponse) -> float:
        """Calculate overall quality score for a response."""
        
        quality_components = []
        
        # Citation component (30%)
        if response.citations:
            citation_quality = sum(
                self._calculate_citation_quality(c) for c in response.citations
            ) / len(response.citations)
            quality_components.append(("citation", citation_quality, 0.3))
        
        # Evidence component (30%)
        if response.evidence:
            evidence_quality = sum(
                e.quality_score * e.relevance_score for e in response.evidence
            ) / len(response.evidence)
            quality_components.append(("evidence", evidence_quality, 0.3))
        
        # Confidence component (20%)
        confidence_quality = response.confidence_score
        quality_components.append(("confidence", confidence_quality, 0.2))
        
        # Content component (20%)
        content_quality = min(1.0, len(response.answer.split()) / 150.0)  # Optimal ~150 words
        quality_components.append(("content", content_quality, 0.2))
        
        # Weighted average
        if quality_components:
            weighted_sum = sum(quality * weight for name, quality, weight in quality_components)
            total_weight = sum(weight for name, quality, weight in quality_components)
            return weighted_sum / total_weight
        
        return 0.0
    
    def _calculate_citation_quality(self, citation) -> float:
        """Calculate quality score for a citation."""
        
        quality_factors = []
        
        # URL validity
        if hasattr(citation, 'url') and citation.url:
            if citation.url.startswith(('http://', 'https://')):
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.5)
        else:
            quality_factors.append(0.0)
        
        # Formatted citation quality
        if hasattr(citation, 'formatted_citation') and citation.formatted_citation:
            citation_length = len(citation.formatted_citation)
            if citation_length >= 50:
                quality_factors.append(1.0)
            elif citation_length >= 20:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.3)
        else:
            quality_factors.append(0.0)
        
        # Source type (if available)
        if hasattr(citation, 'source_type') and citation.source_type:
            if citation.source_type in ['academic', 'government']:
                quality_factors.append(1.0)
            elif citation.source_type in ['news', 'website']:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.5)
        
        return statistics.mean(quality_factors) if quality_factors else 0.0
    
    def _calculate_evidence_strength(self, response: AgentResponse) -> float:
        """Calculate overall strength of evidence in response."""
        
        if not response.evidence:
            return 0.0
        
        # Weight by quality and relevance scores
        evidence_scores = []
        for evidence in response.evidence:
            combined_score = (
                evidence.quality_score * 0.6 +  # Quality is more important
                evidence.relevance_score * 0.4
            )
            
            # Bonus for supporting evidence
            if hasattr(evidence, 'supports_claim') and evidence.supports_claim:
                combined_score *= 1.1
            
            evidence_scores.append(min(1.0, combined_score))
        
        # Average with diminishing returns for quantity
        base_strength = statistics.mean(evidence_scores)
        
        # Quantity bonus (diminishing returns)
        quantity_factor = min(1.0, len(response.evidence) / 3.0)
        
        return base_strength * (0.8 + 0.2 * quantity_factor)
    
    def calculate_metric_trends(
        self,
        metric_name: str,
        lookback_periods: int = 10
    ) -> Dict[str, float]:
        """Calculate trends for a specific metric over time."""
        
        if len(self.performance_history) < 2:
            return {"trend": 0.0, "variance": 0.0, "latest_value": 0.0}
        
        # Get recent values
        recent_metrics = self.performance_history[-lookback_periods:]
        values = [getattr(metric, metric_name, 0.0) for metric in recent_metrics]
        
        if len(values) < 2:
            return {"trend": 0.0, "variance": 0.0, "latest_value": values[-1] if values else 0.0}
        
        # Calculate trend (simple linear regression)
        n = len(values)
        x_values = list(range(n))
        trend = self._calculate_linear_trend(x_values, values)
        
        # Calculate variance
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        
        return {
            "trend": trend,
            "variance": variance,
            "latest_value": values[-1],
            "mean_value": statistics.mean(values),
            "sample_size": len(values)
        }
    
    def _calculate_linear_trend(self, x_values: List[int], y_values: List[float]) -> float:
        """Calculate linear trend slope."""
        
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x_squared = sum(x * x for x in x_values)
        
        denominator = n * sum_x_squared - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def get_calculator_statistics(self) -> Dict[str, Any]:
        """Get calculator performance statistics."""
        
        return {
            "total_calculations": self.total_calculations,
            "calculation_errors": self.calculation_errors,
            "error_rate": (
                self.calculation_errors / max(self.total_calculations, 1)
            ),
            "last_calculation_time": self.last_calculation_time,
            "metrics_stored": len(self.calculated_metrics),
            "performance_history_size": len(self.performance_history),
            "configuration": {
                "confidence_level": self.confidence_level,
                "min_sample_size": self.min_sample_size,
                "enable_statistical_tests": self.enable_statistical_tests
            }
        }
    
    def reset_calculator(self):
        """Reset calculator statistics and history."""
        
        self.calculated_metrics.clear()
        self.performance_history.clear()
        self.total_calculations = 0
        self.calculation_errors = 0
        self.last_calculation_time = 0.0
        
        logger.info("MetricsCalculator reset")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()