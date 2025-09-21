"""FeedbackGenerator for structured, specific feedback for targeted improvements."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from loguru import logger

from .challenge_processor import ProcessedAnalysis, AnalysisResult, AnalysisType
from .revision_manager import RevisionSession, RevisionPlan, RevisedResponse
from .conflict_resolver import ConflictAnalysis, ConflictResolution
from ..agents.answering_agent import AgentResponse
from ..agents.challenger_agent import ChallengeReport


class FeedbackCategory(Enum):
    """Categories of feedback that can be generated."""
    CITATION_IMPROVEMENT = "citation_improvement"
    EVIDENCE_STRENGTHENING = "evidence_strengthening"
    LOGICAL_CONSISTENCY = "logical_consistency"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    SOURCE_QUALITY = "source_quality"
    CONFLICT_RESOLUTION = "conflict_resolution"
    GENERAL_IMPROVEMENT = "general_improvement"


class FeedbackPriority(Enum):
    """Priority levels for feedback items."""
    CRITICAL = "critical"      # Must be addressed
    HIGH = "high"             # Should be addressed soon
    MEDIUM = "medium"         # Important to address
    LOW = "low"               # Nice to have improvement


class FeedbackType(Enum):
    """Types of feedback messages."""
    SPECIFIC_ACTION = "specific_action"       # Do X to Y
    GENERAL_GUIDANCE = "general_guidance"    # Consider improving X
    BEST_PRACTICE = "best_practice"          # Best practice for X is Y
    WARNING = "warning"                      # Be careful of X
    ACKNOWLEDGMENT = "acknowledgment"        # Good job with X


@dataclass
class FeedbackItem:
    """Individual feedback item with specific guidance."""
    
    feedback_id: str
    category: FeedbackCategory
    priority: FeedbackPriority
    feedback_type: FeedbackType
    
    # Core feedback content
    title: str
    description: str
    specific_action: str
    rationale: str
    
    # Context information
    affected_agent: Optional[str] = None
    affected_content: Optional[str] = None
    related_sources: List[str] = None
    
    # Improvement guidance
    example_improvement: Optional[str] = None
    resources: List[str] = None
    expected_impact: str = "moderate"
    
    # Metadata
    confidence: float = 0.8
    effort_required: str = "medium"  # low, medium, high
    created_timestamp: str = ""


@dataclass
class StructuredFeedback:
    """Complete structured feedback for a session."""
    
    session_id: str
    original_claim: str
    
    # Feedback organization
    critical_feedback: List[FeedbackItem]
    high_priority_feedback: List[FeedbackItem]
    medium_priority_feedback: List[FeedbackItem]
    low_priority_feedback: List[FeedbackItem]
    
    # Summary information
    total_feedback_items: int
    feedback_summary: Dict[FeedbackCategory, int]
    
    # Overall assessment
    overall_quality_assessment: str
    key_strengths: List[str]
    main_improvement_areas: List[str]
    
    # Implementation guidance
    suggested_priority_order: List[str]
    estimated_total_effort: str
    expected_improvement_potential: float
    
    # Metadata
    generation_confidence: float
    feedback_timestamp: str


class FeedbackGenerator:
    """
    Structured, specific feedback for targeted improvements.
    
    Synthesizes analysis from challenge processor, revision manager,
    and conflict resolver to generate comprehensive, actionable feedback
    for improving response quality.
    """
    
    def __init__(
        self,
        max_feedback_per_category: int = 5,
        min_feedback_confidence: float = 0.6,
        prioritize_actionable: bool = True,
        include_positive_feedback: bool = True
    ):
        """Initialize the feedback generator."""
        
        self.max_feedback_per_category = max_feedback_per_category
        self.min_feedback_confidence = min_feedback_confidence
        self.prioritize_actionable = prioritize_actionable
        self.include_positive_feedback = include_positive_feedback
        
        # Feedback generation statistics
        self.total_feedback_sessions = 0
        self.total_feedback_items_generated = 0
        self.feedback_acceptance_rate = 0.0
        
        logger.info("FeedbackGenerator initialized")
    
    async def generate_comprehensive_feedback(
        self,
        session_id: str,
        original_claim: str,
        processed_analysis: ProcessedAnalysis,
        revision_session: Optional[RevisionSession] = None,
        conflict_analysis: Optional[ConflictAnalysis] = None
    ) -> StructuredFeedback:
        """
        Generate comprehensive structured feedback.
        
        Args:
            session_id: Unique session identifier
            original_claim: Original claim being analyzed
            processed_analysis: Results from challenge processor
            revision_session: Optional revision session results
            conflict_analysis: Optional conflict analysis results
            
        Returns:
            StructuredFeedback with prioritized recommendations
        """
        
        logger.info(f"Generating comprehensive feedback for session {session_id}")
        
        try:
            # Generate feedback from different analysis sources
            analysis_feedback = await self._generate_analysis_feedback(processed_analysis)
            
            revision_feedback = []
            if revision_session:
                revision_feedback = await self._generate_revision_feedback(revision_session)
            
            conflict_feedback = []
            if conflict_analysis:
                conflict_feedback = await self._generate_conflict_feedback(conflict_analysis)
            
            # Combine and deduplicate feedback
            all_feedback = analysis_feedback + revision_feedback + conflict_feedback
            unique_feedback = self._deduplicate_feedback(all_feedback)
            
            # Add positive feedback if enabled
            if self.include_positive_feedback:
                positive_feedback = await self._generate_positive_feedback(processed_analysis)
                unique_feedback.extend(positive_feedback)
            
            # Filter by confidence threshold
            filtered_feedback = [
                item for item in unique_feedback 
                if item.confidence >= self.min_feedback_confidence
            ]
            
            # Prioritize and organize feedback
            prioritized_feedback = self._prioritize_feedback(filtered_feedback)
            
            # Generate summary information
            feedback_summary = self._create_feedback_summary(prioritized_feedback)
            
            # Assess overall quality
            quality_assessment, strengths, improvement_areas = self._assess_overall_quality(
                processed_analysis, prioritized_feedback
            )
            
            # Generate implementation guidance
            priority_order, total_effort, improvement_potential = self._generate_implementation_guidance(
                prioritized_feedback, processed_analysis
            )
            
            structured_feedback = StructuredFeedback(
                session_id=session_id,
                original_claim=original_claim,
                critical_feedback=[f for f in prioritized_feedback if f.priority == FeedbackPriority.CRITICAL],
                high_priority_feedback=[f for f in prioritized_feedback if f.priority == FeedbackPriority.HIGH],
                medium_priority_feedback=[f for f in prioritized_feedback if f.priority == FeedbackPriority.MEDIUM],
                low_priority_feedback=[f for f in prioritized_feedback if f.priority == FeedbackPriority.LOW],
                total_feedback_items=len(prioritized_feedback),
                feedback_summary=feedback_summary,
                overall_quality_assessment=quality_assessment,
                key_strengths=strengths,
                main_improvement_areas=improvement_areas,
                suggested_priority_order=priority_order,
                estimated_total_effort=total_effort,
                expected_improvement_potential=improvement_potential,
                generation_confidence=self._calculate_generation_confidence(prioritized_feedback),
                feedback_timestamp=self._get_timestamp()
            )
            
            # Update statistics
            self.total_feedback_sessions += 1
            self.total_feedback_items_generated += len(prioritized_feedback)
            
            logger.success(
                f"Generated {len(prioritized_feedback)} feedback items for session {session_id}"
            )
            
            return structured_feedback
            
        except Exception as e:
            logger.error(f"Feedback generation failed for session {session_id}: {str(e)}")
            raise
    
    async def _generate_analysis_feedback(
        self, 
        processed_analysis: ProcessedAnalysis
    ) -> List[FeedbackItem]:
        """Generate feedback based on challenge processor analysis."""
        
        feedback_items = []
        feedback_counter = 0
        
        for analysis_type, result in processed_analysis.analysis_results.items():
            category_feedback = await self._generate_category_feedback(
                analysis_type, result, feedback_counter
            )
            feedback_items.extend(category_feedback)
            feedback_counter += len(category_feedback)
        
        return feedback_items
    
    async def _generate_category_feedback(
        self,
        analysis_type: AnalysisType,
        result: AnalysisResult,
        base_counter: int
    ) -> List[FeedbackItem]:
        """Generate feedback for specific analysis category."""
        
        feedback_items = []
        
        if analysis_type == AnalysisType.UNSUPPORTED_CLAIMS:
            for i, issue in enumerate(result.issues_found[:3]):  # Top 3 issues
                feedback_items.append(FeedbackItem(
                    feedback_id=f"unsupported_{base_counter + i:03d}",
                    category=FeedbackCategory.CITATION_IMPROVEMENT,
                    priority=self._determine_priority_from_severity(issue.get("severity", 0.5)),
                    feedback_type=FeedbackType.SPECIFIC_ACTION,
                    title="Add Citations for Unsupported Claims",
                    description=f"Claim lacks adequate supporting evidence: {issue.get('description', 'Unknown claim')}",
                    specific_action="Add specific citations or qualify the statement with uncertainty language",
                    rationale="Unsupported factual claims reduce credibility and violate evidence standards",
                    example_improvement="Change 'X causes Y' to 'According to [Source], X may contribute to Y'",
                    resources=["Academic databases", "Government statistics", "Peer-reviewed journals"],
                    expected_impact="high",
                    confidence=0.9,
                    effort_required="medium",
                    created_timestamp=self._get_timestamp()
                ))
        
        elif analysis_type == AnalysisType.WEAK_CITATIONS:
            for i, issue in enumerate(result.issues_found[:2]):
                feedback_items.append(FeedbackItem(
                    feedback_id=f"citation_{base_counter + i:03d}",
                    category=FeedbackCategory.SOURCE_QUALITY,
                    priority=FeedbackPriority.MEDIUM,
                    feedback_type=FeedbackType.SPECIFIC_ACTION,
                    title="Upgrade Citation Sources",
                    description="Citations include low-authority or inappropriate sources",
                    specific_action="Replace blog posts, social media, and unknown sources with academic or government sources",
                    rationale="Source authority directly impacts claim credibility",
                    example_improvement="Replace blog citation with peer-reviewed journal article",
                    resources=["PubMed", "Google Scholar", "Government agency websites"],
                    expected_impact="moderate",
                    confidence=0.8,
                    effort_required="medium",
                    created_timestamp=self._get_timestamp()
                ))
        
        elif analysis_type == AnalysisType.CONTRADICTORY_EVIDENCE:
            for i, issue in enumerate(result.issues_found[:2]):
                feedback_items.append(FeedbackItem(
                    feedback_id=f"conflict_{base_counter + i:03d}",
                    category=FeedbackCategory.CONFLICT_RESOLUTION,
                    priority=FeedbackPriority.HIGH,
                    feedback_type=FeedbackType.SPECIFIC_ACTION,
                    title="Address Contradictory Evidence",
                    description="Conflicting evidence sources not adequately addressed",
                    specific_action="Explicitly acknowledge contradictions and explain how they were evaluated",
                    rationale="Unaddressed contradictions suggest incomplete analysis",
                    example_improvement="Add: 'While Source A suggests X, Source B indicates Y. Based on methodological quality...'",
                    expected_impact="high",
                    confidence=0.7,
                    effort_required="high",
                    created_timestamp=self._get_timestamp()
                ))
        
        elif analysis_type == AnalysisType.INSUFFICIENT_EVIDENCE:
            feedback_items.append(FeedbackItem(
                feedback_id=f"evidence_{base_counter:03d}",
                category=FeedbackCategory.EVIDENCE_STRENGTHENING,
                priority=FeedbackPriority.MEDIUM,
                feedback_type=FeedbackType.GENERAL_GUIDANCE,
                title="Strengthen Evidence Base",
                description="Evidence quantity or quality below recommended standards",
                specific_action="Gather additional high-quality sources or qualify conclusions appropriately",
                rationale="Insufficient evidence weakens argument strength and reliability",
                example_improvement="Add 2-3 additional authoritative sources or use hedging language",
                expected_impact="moderate",
                confidence=0.8,
                effort_required="medium",
                created_timestamp=self._get_timestamp()
            ))
        
        elif analysis_type == AnalysisType.LOGICAL_CONSISTENCY:
            for i, issue in enumerate(result.issues_found[:2]):
                if issue.get("type") == "confidence_evidence_mismatch":
                    feedback_items.append(FeedbackItem(
                        feedback_id=f"confidence_{base_counter + i:03d}",
                        category=FeedbackCategory.CONFIDENCE_CALIBRATION,
                        priority=FeedbackPriority.MEDIUM,
                        feedback_type=FeedbackType.SPECIFIC_ACTION,
                        title="Calibrate Confidence to Evidence",
                        description=f"Confidence level ({issue.get('stated_confidence', 0):.2f}) doesn't match evidence strength",
                        specific_action="Adjust confidence score to better reflect available evidence quality",
                        rationale="Well-calibrated confidence improves reliability and user trust",
                        example_improvement=f"Lower confidence to {issue.get('evidence_strength', 0.5):.2f} based on evidence analysis",
                        expected_impact="moderate",
                        confidence=0.7,
                        effort_required="low",
                        created_timestamp=self._get_timestamp()
                    ))
        
        return feedback_items[:self.max_feedback_per_category]
    
    async def _generate_revision_feedback(
        self, 
        revision_session: RevisionSession
    ) -> List[FeedbackItem]:
        """Generate feedback based on revision session results."""
        
        feedback_items = []
        
        if revision_session.revision_success:
            # Positive feedback for successful revision
            feedback_items.append(FeedbackItem(
                feedback_id="revision_success",
                category=FeedbackCategory.GENERAL_IMPROVEMENT,
                priority=FeedbackPriority.LOW,
                feedback_type=FeedbackType.ACKNOWLEDGMENT,
                title="Successful Revision Implementation",
                description=f"Revision process achieved {revision_session.overall_improvement:.2f} improvement",
                specific_action="Continue applying similar revision practices",
                rationale="Systematic revision improves response quality",
                expected_impact="sustained_improvement",
                confidence=0.9,
                effort_required="low",
                created_timestamp=self._get_timestamp()
            ))
        
        # Feedback on revision areas that need attention
        for response in revision_session.revised_responses:
            if response.instructions_skipped:
                feedback_items.append(FeedbackItem(
                    feedback_id=f"revision_incomplete_{response.original_response.agent_id}",
                    category=FeedbackCategory.GENERAL_IMPROVEMENT,
                    priority=FeedbackPriority.MEDIUM,
                    feedback_type=FeedbackType.WARNING,
                    title="Incomplete Revision Implementation",
                    description=f"Agent {response.original_response.agent_id} had {len(response.instructions_skipped)} unaddressed revision items",
                    specific_action="Review and address remaining revision instructions",
                    rationale="Complete revision implementation maximizes quality improvement",
                    affected_agent=response.original_response.agent_id,
                    expected_impact="moderate",
                    confidence=0.8,
                    effort_required="medium",
                    created_timestamp=self._get_timestamp()
                ))
        
        return feedback_items
    
    async def _generate_conflict_feedback(
        self, 
        conflict_analysis: ConflictAnalysis
    ) -> List[FeedbackItem]:
        """Generate feedback based on conflict analysis."""
        
        feedback_items = []
        
        if conflict_analysis.overall_conflict_level == "high":
            feedback_items.append(FeedbackItem(
                feedback_id="high_conflict_level",
                category=FeedbackCategory.CONFLICT_RESOLUTION,
                priority=FeedbackPriority.CRITICAL,
                feedback_type=FeedbackType.WARNING,
                title="High Level of Conflicting Evidence",
                description=f"Detected {len(conflict_analysis.detected_conflicts)} significant conflicts",
                specific_action="Systematically address all detected conflicts before finalizing conclusions",
                rationale="High conflict levels indicate need for thorough evidence reconciliation",
                expected_impact="high",
                confidence=0.9,
                effort_required="high",
                created_timestamp=self._get_timestamp()
            ))
        
        # Feedback on specific resolution strategies
        for i, resolution in enumerate(conflict_analysis.proposed_resolutions[:3]):
            feedback_items.append(FeedbackItem(
                feedback_id=f"conflict_resolution_{i:03d}",
                category=FeedbackCategory.CONFLICT_RESOLUTION,
                priority=FeedbackPriority.HIGH,
                feedback_type=FeedbackType.SPECIFIC_ACTION,
                title=f"Implement Conflict Resolution Strategy",
                description=f"Apply {resolution.resolution_strategy.value} for detected conflict",
                specific_action=resolution.resolution_text,
                rationale="Systematic conflict resolution improves response reliability",
                example_improvement=resolution.uncertainty_note or "See resolution details",
                expected_impact="high",
                confidence=resolution.resolution_confidence,
                effort_required="medium",
                created_timestamp=self._get_timestamp()
            ))
        
        return feedback_items
    
    async def _generate_positive_feedback(
        self, 
        processed_analysis: ProcessedAnalysis
    ) -> List[FeedbackItem]:
        """Generate positive feedback highlighting strengths."""
        
        feedback_items = []
        
        if processed_analysis.quality_score >= 0.8:
            feedback_items.append(FeedbackItem(
                feedback_id="high_quality",
                category=FeedbackCategory.GENERAL_IMPROVEMENT,
                priority=FeedbackPriority.LOW,
                feedback_type=FeedbackType.ACKNOWLEDGMENT,
                title="High Overall Quality",
                description=f"Response quality score of {processed_analysis.quality_score:.2f} exceeds standards",
                specific_action="Maintain current quality standards in future responses",
                rationale="Consistent high quality builds user trust and reliability",
                expected_impact="maintain_excellence",
                confidence=0.9,
                effort_required="low",
                created_timestamp=self._get_timestamp()
            ))
        
        if processed_analysis.critical_issues == 0:
            feedback_items.append(FeedbackItem(
                feedback_id="no_critical_issues",
                category=FeedbackCategory.GENERAL_IMPROVEMENT,
                priority=FeedbackPriority.LOW,
                feedback_type=FeedbackType.ACKNOWLEDGMENT,
                title="No Critical Issues Detected",
                description="Response meets basic quality and accuracy standards",
                specific_action="Continue current approach while addressing moderate improvements",
                rationale="Absence of critical issues indicates solid foundation",
                expected_impact="confidence_building",
                confidence=0.8,
                effort_required="low",
                created_timestamp=self._get_timestamp()
            ))
        
        return feedback_items
    
    def _deduplicate_feedback(self, feedback_items: List[FeedbackItem]) -> List[FeedbackItem]:
        """Remove duplicate or very similar feedback items."""
        
        unique_feedback = []
        seen_combinations = set()
        
        for item in feedback_items:
            # Create a key based on category and main action
            key = (item.category, item.title, item.specific_action[:50])
            
            if key not in seen_combinations:
                unique_feedback.append(item)
                seen_combinations.add(key)
        
        return unique_feedback
    
    def _prioritize_feedback(self, feedback_items: List[FeedbackItem]) -> List[FeedbackItem]:
        """Prioritize feedback items for optimal impact."""
        
        if self.prioritize_actionable:
            # Sort by: priority, actionability, confidence, expected impact
            def priority_key(item):
                priority_order = {
                    FeedbackPriority.CRITICAL: 0,
                    FeedbackPriority.HIGH: 1,
                    FeedbackPriority.MEDIUM: 2,
                    FeedbackPriority.LOW: 3
                }
                
                actionability_score = 2 if item.feedback_type == FeedbackType.SPECIFIC_ACTION else 1
                
                return (
                    priority_order[item.priority],
                    -actionability_score,  # Negative for reverse order
                    -item.confidence,      # Negative for reverse order
                    item.effort_required == "low"  # Low effort items first within same priority
                )
            
            return sorted(feedback_items, key=priority_key)
        
        else:
            # Simple priority-based sorting
            priority_order = {
                FeedbackPriority.CRITICAL: 0,
                FeedbackPriority.HIGH: 1,
                FeedbackPriority.MEDIUM: 2,
                FeedbackPriority.LOW: 3
            }
            
            return sorted(feedback_items, key=lambda x: priority_order[x.priority])
    
    def _determine_priority_from_severity(self, severity: float) -> FeedbackPriority:
        """Determine feedback priority based on issue severity."""
        
        if severity >= 0.8:
            return FeedbackPriority.CRITICAL
        elif severity >= 0.6:
            return FeedbackPriority.HIGH
        elif severity >= 0.4:
            return FeedbackPriority.MEDIUM
        else:
            return FeedbackPriority.LOW
    
    def _create_feedback_summary(
        self, 
        feedback_items: List[FeedbackItem]
    ) -> Dict[FeedbackCategory, int]:
        """Create summary of feedback by category."""
        
        from collections import Counter
        
        category_counts = Counter(item.category for item in feedback_items)
        return dict(category_counts)
    
    def _assess_overall_quality(
        self,
        processed_analysis: ProcessedAnalysis,
        feedback_items: List[FeedbackItem]
    ) -> Tuple[str, List[str], List[str]]:
        """Assess overall quality and identify strengths and improvement areas."""
        
        # Overall quality assessment
        quality_score = processed_analysis.quality_score
        critical_count = len([f for f in feedback_items if f.priority == FeedbackPriority.CRITICAL])
        
        if quality_score >= 0.8 and critical_count == 0:
            assessment = "Excellent - meets high quality standards with minor improvements available"
        elif quality_score >= 0.7 and critical_count <= 1:
            assessment = "Good - solid quality with some areas for improvement"
        elif quality_score >= 0.6:
            assessment = "Adequate - meets basic standards but needs improvement"
        else:
            assessment = "Needs Improvement - significant quality issues to address"
        
        # Identify strengths
        strengths = []
        if processed_analysis.critical_issues == 0:
            strengths.append("No critical quality issues")
        if processed_analysis.quality_score >= 0.7:
            strengths.append("Strong overall evidence base")
        if len(processed_analysis.agent_responses) > 1:
            strengths.append("Multi-agent perspective provides robustness")
        if processed_analysis.confidence_in_analysis >= 0.8:
            strengths.append("High confidence in analysis reliability")
        
        # Identify main improvement areas
        improvement_areas = []
        category_priorities = {}
        for item in feedback_items:
            if item.priority in [FeedbackPriority.CRITICAL, FeedbackPriority.HIGH]:
                category_priorities[item.category] = category_priorities.get(item.category, 0) + 1
        
        # Sort by frequency and convert to readable names
        top_categories = sorted(category_priorities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        category_names = {
            FeedbackCategory.CITATION_IMPROVEMENT: "Citation quality and completeness",
            FeedbackCategory.EVIDENCE_STRENGTHENING: "Evidence quantity and strength",
            FeedbackCategory.LOGICAL_CONSISTENCY: "Logical reasoning and consistency",
            FeedbackCategory.CONFIDENCE_CALIBRATION: "Confidence score calibration",
            FeedbackCategory.SOURCE_QUALITY: "Source authority and reliability",
            FeedbackCategory.CONFLICT_RESOLUTION: "Conflict resolution and evidence reconciliation"
        }
        
        improvement_areas = [category_names.get(cat, cat.value) for cat, _ in top_categories]
        
        return assessment, strengths, improvement_areas
    
    def _generate_implementation_guidance(
        self,
        feedback_items: List[FeedbackItem],
        processed_analysis: ProcessedAnalysis
    ) -> Tuple[List[str], str, float]:
        """Generate guidance for implementing feedback."""
        
        # Suggested priority order
        priority_order = []
        
        # Critical items first
        critical_items = [f for f in feedback_items if f.priority == FeedbackPriority.CRITICAL]
        if critical_items:
            priority_order.append("Address all critical issues immediately")
        
        # High priority, low effort items
        quick_wins = [
            f for f in feedback_items 
            if f.priority == FeedbackPriority.HIGH and f.effort_required == "low"
        ]
        if quick_wins:
            priority_order.append("Implement high-impact, low-effort improvements")
        
        # Medium priority items by category
        medium_items = [f for f in feedback_items if f.priority == FeedbackPriority.MEDIUM]
        if medium_items:
            priority_order.append("Systematically address moderate priority improvements")
        
        # Low priority items last
        if any(f.priority == FeedbackPriority.LOW for f in feedback_items):
            priority_order.append("Consider low-priority enhancements when time permits")
        
        # Estimate total effort
        effort_counts = {
            "low": len([f for f in feedback_items if f.effort_required == "low"]),
            "medium": len([f for f in feedback_items if f.effort_required == "medium"]),
            "high": len([f for f in feedback_items if f.effort_required == "high"])
        }
        
        if effort_counts["high"] >= 3:
            total_effort = "high"
        elif effort_counts["high"] >= 1 or effort_counts["medium"] >= 5:
            total_effort = "moderate"
        else:
            total_effort = "low"
        
        # Estimate improvement potential
        high_impact_items = len([
            f for f in feedback_items 
            if f.expected_impact in ["high", "sustained_improvement", "confidence_building"]
        ])
        
        base_potential = processed_analysis.quality_score
        improvement_potential = min(1.0, base_potential + (high_impact_items * 0.1))
        
        return priority_order, total_effort, improvement_potential
    
    def _calculate_generation_confidence(self, feedback_items: List[FeedbackItem]) -> float:
        """Calculate confidence in the generated feedback."""
        
        if not feedback_items:
            return 0.5
        
        avg_confidence = sum(item.confidence for item in feedback_items) / len(feedback_items)
        
        # Adjust for feedback diversity
        categories = set(item.category for item in feedback_items)
        diversity_bonus = min(0.1, len(categories) * 0.02)
        
        final_confidence = avg_confidence + diversity_bonus
        
        return min(1.0, final_confidence)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """Get feedback generator statistics."""
        
        return {
            "total_feedback_sessions": self.total_feedback_sessions,
            "total_feedback_items_generated": self.total_feedback_items_generated,
            "avg_feedback_per_session": (
                self.total_feedback_items_generated / self.total_feedback_sessions
                if self.total_feedback_sessions > 0 else 0.0
            ),
            "feedback_acceptance_rate": self.feedback_acceptance_rate,
            "configuration": {
                "max_feedback_per_category": self.max_feedback_per_category,
                "min_feedback_confidence": self.min_feedback_confidence,
                "prioritize_actionable": self.prioritize_actionable,
                "include_positive_feedback": self.include_positive_feedback
            }
        }
    
    def reset_statistics(self):
        """Reset feedback generator statistics."""
        
        self.total_feedback_sessions = 0
        self.total_feedback_items_generated = 0
        self.feedback_acceptance_rate = 0.0
        
        logger.info("FeedbackGenerator statistics reset")