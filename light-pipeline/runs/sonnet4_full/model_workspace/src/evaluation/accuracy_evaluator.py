"""AccuracyEvaluator for evaluating accuracy of responses and challenges in the system."""

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
from ..agents.challenger_agent import ChallengeReport, Challenge, ChallengeType
from ..challenge.challenge_processor import ProcessedAnalysis
from ..challenge.revision_manager import RevisionSession


class EvaluationMetric(Enum):
    """Types of accuracy evaluation metrics."""
    RESPONSE_ACCURACY = "response_accuracy"
    CITATION_ACCURACY = "citation_accuracy"
    EVIDENCE_ACCURACY = "evidence_accuracy"
    CHALLENGE_ACCURACY = "challenge_accuracy"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    FACTUAL_CONSISTENCY = "factual_consistency"


class GroundTruthType(Enum):
    """Types of ground truth sources."""
    EXPERT_ANNOTATION = "expert_annotation"
    VERIFIED_SOURCES = "verified_sources"
    CONSENSUS_JUDGMENT = "consensus_judgment"
    AUTOMATED_FACT_CHECK = "automated_fact_check"


@dataclass
class GroundTruthEntry:
    """Individual ground truth entry for evaluation."""
    
    claim: str
    correct_answer: str
    supports_claim: bool
    confidence_level: float
    authoritative_sources: List[str]
    expert_reasoning: Optional[str]
    ground_truth_type: GroundTruthType
    metadata: Dict[str, Any]


@dataclass
class AccuracyResult:
    """Result of accuracy evaluation for a single item."""
    
    item_id: str
    claim: str
    predicted_answer: str
    ground_truth_answer: str
    is_correct: bool
    confidence_score: float
    accuracy_score: float
    error_type: Optional[str]
    detailed_feedback: str
    timestamp: str


@dataclass
class EvaluationSummary:
    """Summary of accuracy evaluation across multiple items."""
    
    # Basic accuracy metrics
    overall_accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Detailed metrics
    accuracy_by_category: Dict[str, float]
    error_distribution: Dict[str, int]
    confidence_calibration: float
    
    # Sample information
    total_items: int
    correct_items: int
    evaluated_categories: List[str]
    
    # Metadata
    evaluation_timestamp: str
    ground_truth_source: str


class AccuracyEvaluator:
    """
    Evaluate accuracy of system responses and challenges against ground truth.
    
    Provides comprehensive accuracy assessment including response correctness,
    citation validation, challenge effectiveness, and confidence calibration.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        strict_citation_matching: bool = False,
        enable_semantic_matching: bool = True
    ):
        """Initialize the accuracy evaluator."""
        
        self.confidence_threshold = confidence_threshold
        self.strict_citation_matching = strict_citation_matching
        self.enable_semantic_matching = enable_semantic_matching
        
        # Evaluation history
        self.evaluation_history: List[EvaluationSummary] = []
        self.individual_results: List[AccuracyResult] = []
        
        # Statistics
        self.total_evaluations = 0
        self.correct_evaluations = 0
        self.evaluation_errors = 0
        
        logger.info("AccuracyEvaluator initialized")
    
    async def evaluate_response_accuracy(
        self,
        agent_responses: List[AgentResponse],
        ground_truth: List[GroundTruthEntry]
    ) -> EvaluationSummary:
        """
        Evaluate accuracy of agent responses against ground truth.
        
        Args:
            agent_responses: List of agent responses to evaluate
            ground_truth: List of ground truth entries for comparison
            
        Returns:
            EvaluationSummary with detailed accuracy assessment
        """
        
        logger.info(f"Evaluating accuracy of {len(agent_responses)} responses")
        start_time = time.time()
        
        try:
            # Create ground truth lookup
            gt_lookup = {gt.claim: gt for gt in ground_truth}
            
            # Evaluate each response
            results = []
            for response in agent_responses:
                if response.claim in gt_lookup:
                    result = await self._evaluate_single_response(
                        response, gt_lookup[response.claim]
                    )
                    results.append(result)
            
            # Calculate summary metrics
            summary = await self._calculate_evaluation_summary(
                results, "response_accuracy"
            )
            
            # Store results
            self.evaluation_history.append(summary)
            self.individual_results.extend(results)
            self.total_evaluations += len(results)
            self.correct_evaluations += sum(1 for r in results if r.is_correct)
            
            processing_time = time.time() - start_time
            logger.success(
                f"Response accuracy evaluation completed in {processing_time:.2f}s "
                f"(Accuracy: {summary.overall_accuracy:.3f})"
            )
            
            return summary
            
        except Exception as e:
            self.evaluation_errors += 1
            logger.error(f"Response accuracy evaluation failed: {str(e)}")
            raise
    
    async def evaluate_challenge_accuracy(
        self,
        challenge_reports: List[ChallengeReport],
        ground_truth_issues: Dict[str, List[str]]
    ) -> EvaluationSummary:
        """
        Evaluate accuracy of challenge detection against known issues.
        
        Args:
            challenge_reports: List of challenge reports to evaluate
            ground_truth_issues: Dict mapping response IDs to known issues
            
        Returns:
            EvaluationSummary for challenge accuracy
        """
        
        logger.info(f"Evaluating accuracy of {len(challenge_reports)} challenge reports")
        
        try:
            results = []
            
            for report in challenge_reports:
                response_id = getattr(report.original_response, 'agent_id', 'unknown')
                
                if response_id in ground_truth_issues:
                    result = await self._evaluate_challenge_report(
                        report, ground_truth_issues[response_id]
                    )
                    results.append(result)
            
            # Calculate summary
            summary = await self._calculate_evaluation_summary(
                results, "challenge_accuracy"
            )
            
            # Store results
            self.evaluation_history.append(summary)
            self.individual_results.extend(results)
            
            logger.success(f"Challenge accuracy: {summary.overall_accuracy:.3f}")
            return summary
            
        except Exception as e:
            self.evaluation_errors += 1
            logger.error(f"Challenge accuracy evaluation failed: {str(e)}")
            raise
    
    async def evaluate_citation_accuracy(
        self,
        agent_responses: List[AgentResponse],
        verified_sources: Dict[str, List[str]]
    ) -> EvaluationSummary:
        """
        Evaluate accuracy of citations against verified sources.
        
        Args:
            agent_responses: List of agent responses with citations
            verified_sources: Dict mapping claims to verified source URLs
            
        Returns:
            EvaluationSummary for citation accuracy
        """
        
        logger.info(f"Evaluating citation accuracy for {len(agent_responses)} responses")
        
        try:
            results = []
            
            for response in agent_responses:
                if response.claim in verified_sources:
                    result = await self._evaluate_response_citations(
                        response, verified_sources[response.claim]
                    )
                    results.append(result)
            
            # Calculate summary
            summary = await self._calculate_evaluation_summary(
                results, "citation_accuracy"
            )
            
            logger.success(f"Citation accuracy: {summary.overall_accuracy:.3f}")
            return summary
            
        except Exception as e:
            logger.error(f"Citation accuracy evaluation failed: {str(e)}")
            raise
    
    async def evaluate_confidence_calibration(
        self,
        agent_responses: List[AgentResponse],
        ground_truth: List[GroundTruthEntry]
    ) -> Dict[str, float]:
        """
        Evaluate how well confidence scores match actual accuracy.
        
        Args:
            agent_responses: List of agent responses with confidence scores
            ground_truth: Ground truth for accuracy assessment
            
        Returns:
            Dict with calibration metrics
        """
        
        logger.info(f"Evaluating confidence calibration for {len(agent_responses)} responses")
        
        try:
            # Group responses by confidence bins
            confidence_bins = {
                "very_low": [],    # 0.0-0.2
                "low": [],         # 0.2-0.4
                "moderate": [],    # 0.4-0.6
                "high": [],        # 0.6-0.8
                "very_high": []    # 0.8-1.0
            }
            
            gt_lookup = {gt.claim: gt for gt in ground_truth}
            
            for response in agent_responses:
                if response.claim not in gt_lookup:
                    continue
                
                confidence = response.confidence_score
                gt_entry = gt_lookup[response.claim]
                
                # Determine correctness
                is_correct = await self._check_response_correctness(response, gt_entry)
                
                # Assign to confidence bin
                if confidence <= 0.2:
                    confidence_bins["very_low"].append(is_correct)
                elif confidence <= 0.4:
                    confidence_bins["low"].append(is_correct)
                elif confidence <= 0.6:
                    confidence_bins["moderate"].append(is_correct)
                elif confidence <= 0.8:
                    confidence_bins["high"].append(is_correct)
                else:
                    confidence_bins["very_high"].append(is_correct)
            
            # Calculate calibration metrics
            calibration_metrics = {}
            
            for bin_name, correctness_list in confidence_bins.items():
                if correctness_list:
                    accuracy = sum(correctness_list) / len(correctness_list)
                    calibration_metrics[f"{bin_name}_accuracy"] = accuracy
                    calibration_metrics[f"{bin_name}_count"] = len(correctness_list)
                else:
                    calibration_metrics[f"{bin_name}_accuracy"] = 0.0
                    calibration_metrics[f"{bin_name}_count"] = 0
            
            # Calculate overall calibration score
            calibration_errors = []
            bin_centers = {"very_low": 0.1, "low": 0.3, "moderate": 0.5, "high": 0.7, "very_high": 0.9}
            
            for bin_name, center in bin_centers.items():
                if calibration_metrics[f"{bin_name}_count"] > 0:
                    actual_accuracy = calibration_metrics[f"{bin_name}_accuracy"]
                    error = abs(center - actual_accuracy)
                    calibration_errors.append(error)
            
            overall_calibration = 1.0 - (sum(calibration_errors) / len(calibration_errors)) if calibration_errors else 0.0
            calibration_metrics["overall_calibration"] = max(0.0, overall_calibration)
            
            logger.success(f"Confidence calibration: {overall_calibration:.3f}")
            return calibration_metrics
            
        except Exception as e:
            logger.error(f"Confidence calibration evaluation failed: {str(e)}")
            raise
    
    async def _evaluate_single_response(
        self,
        response: AgentResponse,
        ground_truth: GroundTruthEntry
    ) -> AccuracyResult:
        """Evaluate accuracy of a single response."""
        
        # Check correctness
        is_correct = await self._check_response_correctness(response, ground_truth)
        
        # Calculate accuracy score (0.0 to 1.0)
        accuracy_score = await self._calculate_response_accuracy_score(response, ground_truth)
        
        # Determine error type if incorrect
        error_type = None
        if not is_correct:
            error_type = await self._classify_error_type(response, ground_truth)
        
        # Generate detailed feedback
        feedback = await self._generate_detailed_feedback(response, ground_truth, is_correct)
        
        return AccuracyResult(
            item_id=f"{response.agent_id}_{hash(response.claim) % 1000000}",
            claim=response.claim,
            predicted_answer=response.answer,
            ground_truth_answer=ground_truth.correct_answer,
            is_correct=is_correct,
            confidence_score=response.confidence_score,
            accuracy_score=accuracy_score,
            error_type=error_type,
            detailed_feedback=feedback,
            timestamp=self._get_timestamp()
        )
    
    async def _check_response_correctness(
        self,
        response: AgentResponse,
        ground_truth: GroundTruthEntry
    ) -> bool:
        """Check if response is correct against ground truth."""
        
        # Extract support/contradiction stance from response
        response_lower = response.answer.lower()
        
        # Simple heuristic for stance detection
        supports_keywords = ["support", "confirm", "validate", "true", "correct", "yes"]
        contradicts_keywords = ["contradict", "refute", "false", "incorrect", "no", "wrong"]
        
        response_supports = any(keyword in response_lower for keyword in supports_keywords)
        response_contradicts = any(keyword in response_lower for keyword in contradicts_keywords)
        
        # Determine response stance
        if response_supports and not response_contradicts:
            predicted_stance = True
        elif response_contradicts and not response_supports:
            predicted_stance = False
        else:
            # Ambiguous - use confidence score as tiebreaker
            predicted_stance = response.confidence_score > 0.5
        
        # Compare with ground truth
        return predicted_stance == ground_truth.supports_claim
    
    async def _calculate_response_accuracy_score(
        self,
        response: AgentResponse,
        ground_truth: GroundTruthEntry
    ) -> float:
        """Calculate detailed accuracy score (0.0 to 1.0)."""
        
        scores = []
        
        # Correctness score (50% weight)
        is_correct = await self._check_response_correctness(response, ground_truth)
        correctness_score = 1.0 if is_correct else 0.0
        scores.append(("correctness", correctness_score, 0.5))
        
        # Citation quality score (30% weight)
        citation_score = await self._evaluate_citation_quality(response, ground_truth)
        scores.append(("citations", citation_score, 0.3))
        
        # Evidence alignment score (20% weight)
        evidence_score = await self._evaluate_evidence_alignment(response, ground_truth)
        scores.append(("evidence", evidence_score, 0.2))
        
        # Calculate weighted average
        weighted_sum = sum(score * weight for name, score, weight in scores)
        return weighted_sum
    
    async def _evaluate_citation_quality(
        self,
        response: AgentResponse,
        ground_truth: GroundTruthEntry
    ) -> float:
        """Evaluate quality of citations in response."""
        
        if not response.citations:
            return 0.0
        
        if not ground_truth.authoritative_sources:
            # No ground truth sources - evaluate format quality
            valid_citations = sum(
                1 for c in response.citations
                if c.url and c.formatted_citation and len(c.formatted_citation) > 20
            )
            return valid_citations / len(response.citations)
        
        # Check alignment with authoritative sources
        authoritative_domains = set()
        for source in ground_truth.authoritative_sources:
            if "://" in source:
                domain = source.split("://")[1].split("/")[0]
                authoritative_domains.add(domain.lower())
        
        aligned_citations = 0
        for citation in response.citations:
            if hasattr(citation, 'url') and citation.url:
                citation_domain = citation.url.split("://")[1].split("/")[0].lower()
                if any(auth_domain in citation_domain or citation_domain in auth_domain 
                       for auth_domain in authoritative_domains):
                    aligned_citations += 1
        
        return aligned_citations / len(response.citations)
    
    async def _evaluate_evidence_alignment(
        self,
        response: AgentResponse,
        ground_truth: GroundTruthEntry
    ) -> float:
        """Evaluate alignment of evidence with ground truth."""
        
        if not response.evidence:
            return 0.5  # Neutral score for no evidence
        
        # Count supporting vs contradicting evidence
        supporting_evidence = sum(
            1 for e in response.evidence
            if hasattr(e, 'supports_claim') and e.supports_claim == ground_truth.supports_claim
        )
        
        total_evidence = len(response.evidence)
        alignment_ratio = supporting_evidence / total_evidence
        
        # Bonus for high-quality evidence
        avg_quality = sum(
            e.quality_score for e in response.evidence
            if hasattr(e, 'quality_score')
        ) / len(response.evidence)
        
        return (alignment_ratio * 0.7) + (avg_quality * 0.3)
    
    async def _classify_error_type(
        self,
        response: AgentResponse,
        ground_truth: GroundTruthEntry
    ) -> str:
        """Classify the type of error in incorrect response."""
        
        # Simple error classification
        if not response.citations:
            return "missing_citations"
        elif not response.evidence:
            return "missing_evidence"
        elif len(response.answer) < 50:
            return "insufficient_detail"
        elif response.confidence_score < 0.3:
            return "low_confidence"
        else:
            return "stance_error"
    
    async def _generate_detailed_feedback(
        self,
        response: AgentResponse,
        ground_truth: GroundTruthEntry,
        is_correct: bool
    ) -> str:
        """Generate detailed feedback for the response."""
        
        feedback_parts = []
        
        if is_correct:
            feedback_parts.append("✓ Response correctly identifies claim stance")
        else:
            feedback_parts.append("✗ Response incorrectly identifies claim stance")
        
        # Citation feedback
        if response.citations:
            feedback_parts.append(f"Citations: {len(response.citations)} provided")
        else:
            feedback_parts.append("Citations: None provided (potential improvement area)")
        
        # Evidence feedback
        if response.evidence:
            avg_quality = sum(e.quality_score for e in response.evidence) / len(response.evidence)
            feedback_parts.append(f"Evidence: {len(response.evidence)} items (avg quality: {avg_quality:.2f})")
        else:
            feedback_parts.append("Evidence: None provided")
        
        # Confidence feedback
        confidence_level = "high" if response.confidence_score >= 0.7 else "moderate" if response.confidence_score >= 0.4 else "low"
        feedback_parts.append(f"Confidence: {response.confidence_score:.2f} ({confidence_level})")
        
        return " | ".join(feedback_parts)
    
    async def _evaluate_challenge_report(
        self,
        report: ChallengeReport,
        known_issues: List[str]
    ) -> AccuracyResult:
        """Evaluate accuracy of a challenge report."""
        
        # Check if report correctly identifies issues
        detected_issue_types = set(c.challenge_type.value for c in report.challenges)
        known_issue_types = set(known_issues)
        
        # Calculate precision and recall
        true_positives = len(detected_issue_types.intersection(known_issue_types))
        false_positives = len(detected_issue_types - known_issue_types)
        false_negatives = len(known_issue_types - detected_issue_types)
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = (2 * precision * recall) / max(precision + recall, 0.001)
        
        is_correct = f1_score >= 0.7  # Threshold for "correct" challenge
        
        feedback = f"Challenge detection - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}"
        
        return AccuracyResult(
            item_id=f"challenge_{report.challenger_id}_{hash(str(report.challenges)) % 1000000}",
            claim=getattr(report.original_response, 'claim', 'unknown'),
            predicted_answer=f"Issues: {', '.join(detected_issue_types)}",
            ground_truth_answer=f"Known issues: {', '.join(known_issues)}",
            is_correct=is_correct,
            confidence_score=report.confidence_in_challenges,
            accuracy_score=f1_score,
            error_type="detection_error" if not is_correct else None,
            detailed_feedback=feedback,
            timestamp=self._get_timestamp()
        )
    
    async def _evaluate_response_citations(
        self,
        response: AgentResponse,
        verified_sources: List[str]
    ) -> AccuracyResult:
        """Evaluate citations in a response against verified sources."""
        
        if not response.citations:
            return AccuracyResult(
                item_id=f"citation_{response.agent_id}_{hash(response.claim) % 1000000}",
                claim=response.claim,
                predicted_answer="No citations",
                ground_truth_answer=f"{len(verified_sources)} verified sources available",
                is_correct=False,
                confidence_score=0.0,
                accuracy_score=0.0,
                error_type="missing_citations",
                detailed_feedback="No citations provided",
                timestamp=self._get_timestamp()
            )
        
        # Check citation accuracy
        verified_domains = set()
        for source in verified_sources:
            if "://" in source:
                domain = source.split("://")[1].split("/")[0]
                verified_domains.add(domain.lower())
        
        accurate_citations = 0
        for citation in response.citations:
            if hasattr(citation, 'url') and citation.url:
                citation_domain = citation.url.split("://")[1].split("/")[0].lower()
                if any(verified_domain in citation_domain or citation_domain in verified_domain 
                       for verified_domain in verified_domains):
                    accurate_citations += 1
        
        accuracy_score = accurate_citations / len(response.citations)
        is_correct = accuracy_score >= 0.5
        
        feedback = f"Citations: {accurate_citations}/{len(response.citations)} from verified sources"
        
        return AccuracyResult(
            item_id=f"citation_{response.agent_id}_{hash(response.claim) % 1000000}",
            claim=response.claim,
            predicted_answer=f"{len(response.citations)} citations provided",
            ground_truth_answer=f"{len(verified_sources)} verified sources",
            is_correct=is_correct,
            confidence_score=response.confidence_score,
            accuracy_score=accuracy_score,
            error_type="citation_quality" if not is_correct else None,
            detailed_feedback=feedback,
            timestamp=self._get_timestamp()
        )
    
    async def _calculate_evaluation_summary(
        self,
        results: List[AccuracyResult],
        evaluation_type: str
    ) -> EvaluationSummary:
        """Calculate summary metrics from individual results."""
        
        if not results:
            return EvaluationSummary(
                overall_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy_by_category={},
                error_distribution={},
                confidence_calibration=0.0,
                total_items=0,
                correct_items=0,
                evaluated_categories=[],
                evaluation_timestamp=self._get_timestamp(),
                ground_truth_source="provided"
            )
        
        # Basic accuracy
        correct_items = sum(1 for r in results if r.is_correct)
        overall_accuracy = correct_items / len(results)
        
        # Calculate precision, recall, F1 (treating as binary classification)
        true_positives = correct_items
        false_positives = len(results) - correct_items
        false_negatives = 0  # Assume all ground truth items were tested
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = (2 * precision * recall) / max(precision + recall, 0.001)
        
        # Error distribution
        error_distribution = Counter(r.error_type for r in results if r.error_type)
        
        # Confidence calibration (simplified)
        high_conf_correct = sum(
            1 for r in results 
            if r.confidence_score >= 0.7 and r.is_correct
        )
        high_conf_total = sum(1 for r in results if r.confidence_score >= 0.7)
        confidence_calibration = high_conf_correct / max(high_conf_total, 1)
        
        return EvaluationSummary(
            overall_accuracy=overall_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy_by_category={evaluation_type: overall_accuracy},
            error_distribution=dict(error_distribution),
            confidence_calibration=confidence_calibration,
            total_items=len(results),
            correct_items=correct_items,
            evaluated_categories=[evaluation_type],
            evaluation_timestamp=self._get_timestamp(),
            ground_truth_source="provided"
        )
    
    def get_accuracy_trends(self, lookback_periods: int = 10) -> Dict[str, float]:
        """Get accuracy trends over recent evaluations."""
        
        if len(self.evaluation_history) < 2:
            return {"trend": 0.0, "latest_accuracy": 0.0, "variance": 0.0}
        
        recent_accuracies = [
            summary.overall_accuracy 
            for summary in self.evaluation_history[-lookback_periods:]
        ]
        
        # Simple linear trend
        n = len(recent_accuracies)
        if n < 2:
            return {"trend": 0.0, "latest_accuracy": recent_accuracies[-1], "variance": 0.0}
        
        x_values = list(range(n))
        trend_slope = self._calculate_trend_slope(x_values, recent_accuracies)
        
        return {
            "trend": trend_slope,
            "latest_accuracy": recent_accuracies[-1],
            "mean_accuracy": statistics.mean(recent_accuracies),
            "variance": statistics.variance(recent_accuracies) if n > 1 else 0.0,
            "sample_size": n
        }
    
    def _calculate_trend_slope(self, x_values: List[int], y_values: List[float]) -> float:
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
    
    def get_evaluator_statistics(self) -> Dict[str, Any]:
        """Get evaluator performance statistics."""
        
        return {
            "total_evaluations": self.total_evaluations,
            "correct_evaluations": self.correct_evaluations,
            "overall_accuracy": (
                self.correct_evaluations / max(self.total_evaluations, 1)
            ),
            "evaluation_errors": self.evaluation_errors,
            "error_rate": (
                self.evaluation_errors / max(self.total_evaluations, 1)
            ),
            "evaluation_history_size": len(self.evaluation_history),
            "individual_results_size": len(self.individual_results),
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "strict_citation_matching": self.strict_citation_matching,
                "enable_semantic_matching": self.enable_semantic_matching
            }
        }
    
    def reset_evaluator(self):
        """Reset evaluator statistics and history."""
        
        self.evaluation_history.clear()
        self.individual_results.clear()
        self.total_evaluations = 0
        self.correct_evaluations = 0
        self.evaluation_errors = 0
        
        logger.info("AccuracyEvaluator reset")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()