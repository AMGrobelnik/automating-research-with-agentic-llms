"""RevisionManager for single-round revision process with no additional search capability."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from loguru import logger

from .challenge_processor import ProcessedAnalysis, AnalysisResult, AnalysisType
from ..agents.answering_agent import AgentResponse
from ..agents.challenger_agent import ChallengeReport
from ..schemas.citation_schemas import CitationSchema, EvidenceSchema


class RevisionStatus(Enum):
    """Status of revision process."""
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RevisionType(Enum):
    """Types of revisions that can be performed."""
    MINOR_IMPROVEMENT = "minor_improvement"
    MODERATE_REVISION = "moderate_revision"
    MAJOR_REVISION = "major_revision"
    COMPLETE_REWRITE = "complete_rewrite"


@dataclass
class RevisionInstruction:
    """Specific instruction for revising content."""
    
    priority: str  # "high", "medium", "low"
    category: str  # e.g., "citation", "evidence", "logic"
    target: str    # What to revise (claim, citation, etc.)
    action: str    # What action to take
    reason: str    # Why this revision is needed
    example: Optional[str] = None  # Example of improvement


@dataclass
class RevisionPlan:
    """Comprehensive plan for revision."""
    
    session_id: str
    revision_type: RevisionType
    revision_status: RevisionStatus
    
    # Instructions organized by priority
    high_priority_instructions: List[RevisionInstruction]
    medium_priority_instructions: List[RevisionInstruction]
    low_priority_instructions: List[RevisionInstruction]
    
    # Planning metadata
    total_instructions: int
    estimated_effort: str  # "low", "medium", "high"
    expected_improvement: str  # Description of expected outcome
    created_timestamp: str
    
    # Revision constraints
    no_additional_search: bool = True
    single_round_only: bool = True
    preserve_original_structure: bool = True


@dataclass
class RevisedResponse:
    """Response after revision process."""
    
    original_response: AgentResponse
    revised_answer: str
    revised_reasoning: str
    revised_confidence: float
    
    # Revision tracking
    instructions_followed: List[RevisionInstruction]
    instructions_skipped: List[RevisionInstruction]
    revision_notes: str
    
    # Quality assessment
    improvement_score: float  # 0.0 to 1.0
    issues_addressed: int
    issues_remaining: int
    
    revision_timestamp: str


@dataclass
class RevisionSession:
    """Complete revision session results."""
    
    session_id: str
    original_analysis: ProcessedAnalysis
    revision_plan: RevisionPlan
    
    # Revision results
    revised_responses: List[RevisedResponse]
    overall_improvement: float
    revision_success: bool
    
    # Session metadata
    processing_time: float
    revision_status: RevisionStatus
    completion_timestamp: str
    
    # Summary
    summary_report: str


class RevisionManager:
    """
    Single-round revision process with no additional search capability.
    
    Manages the systematic revision of agent responses based on challenge
    analysis, ensuring improvements address identified issues without
    conducting additional research.
    """
    
    def __init__(
        self,
        max_instructions_per_response: int = 10,
        min_improvement_threshold: float = 0.2,
        preserve_agent_voice: bool = True,
        strict_single_round: bool = True
    ):
        """Initialize the revision manager."""
        
        self.max_instructions_per_response = max_instructions_per_response
        self.min_improvement_threshold = min_improvement_threshold
        self.preserve_agent_voice = preserve_agent_voice
        self.strict_single_round = strict_single_round
        
        # Revision statistics
        self.total_revision_sessions = 0
        self.successful_revisions = 0
        self.failed_revisions = 0
        self.total_instructions_generated = 0
        self.total_instructions_followed = 0
        
        logger.info("RevisionManager initialized with single-round constraint")
    
    async def create_revision_plan(
        self, 
        processed_analysis: ProcessedAnalysis
    ) -> RevisionPlan:
        """
        Create a comprehensive revision plan based on challenge analysis.
        
        Args:
            processed_analysis: Results from challenge processor
            
        Returns:
            RevisionPlan with prioritized instructions
        """
        
        logger.info(f"Creating revision plan for session {processed_analysis.session_id}")
        
        # Determine revision type based on analysis
        revision_type = self._determine_revision_type(processed_analysis)
        
        # Determine if revision is required
        revision_status = self._determine_revision_status(processed_analysis, revision_type)
        
        if revision_status == RevisionStatus.NOT_REQUIRED:
            logger.info(f"No revision required for session {processed_analysis.session_id}")
            return RevisionPlan(
                session_id=processed_analysis.session_id,
                revision_type=RevisionType.MINOR_IMPROVEMENT,
                revision_status=RevisionStatus.NOT_REQUIRED,
                high_priority_instructions=[],
                medium_priority_instructions=[],
                low_priority_instructions=[],
                total_instructions=0,
                estimated_effort="none",
                expected_improvement="No significant improvements needed",
                created_timestamp=self._get_timestamp()
            )
        
        # Generate revision instructions
        instructions = await self._generate_revision_instructions(processed_analysis)
        
        # Prioritize instructions
        high_priority, medium_priority, low_priority = self._prioritize_instructions(instructions)
        
        # Estimate effort and expected improvement
        estimated_effort = self._estimate_revision_effort(len(instructions), revision_type)
        expected_improvement = self._describe_expected_improvement(processed_analysis, instructions)
        
        revision_plan = RevisionPlan(
            session_id=processed_analysis.session_id,
            revision_type=revision_type,
            revision_status=RevisionStatus.REQUIRED,
            high_priority_instructions=high_priority,
            medium_priority_instructions=medium_priority,
            low_priority_instructions=low_priority,
            total_instructions=len(instructions),
            estimated_effort=estimated_effort,
            expected_improvement=expected_improvement,
            created_timestamp=self._get_timestamp()
        )
        
        self.total_instructions_generated += len(instructions)
        
        logger.success(
            f"Revision plan created for session {processed_analysis.session_id} "
            f"({len(instructions)} instructions, {revision_type.value})"
        )
        
        return revision_plan
    
    async def execute_revision(
        self,
        revision_plan: RevisionPlan,
        processed_analysis: ProcessedAnalysis
    ) -> RevisionSession:
        """
        Execute the revision plan on agent responses.
        
        Args:
            revision_plan: Plan with prioritized instructions
            processed_analysis: Original analysis with responses
            
        Returns:
            RevisionSession with results
        """
        
        import time
        start_time = time.time()
        
        logger.info(f"Executing revision for session {revision_plan.session_id}")
        
        if revision_plan.revision_status == RevisionStatus.NOT_REQUIRED:
            logger.info("Skipping revision - not required")
            return RevisionSession(
                session_id=revision_plan.session_id,
                original_analysis=processed_analysis,
                revision_plan=revision_plan,
                revised_responses=[],
                overall_improvement=0.0,
                revision_success=True,
                processing_time=time.time() - start_time,
                revision_status=RevisionStatus.SKIPPED,
                completion_timestamp=self._get_timestamp(),
                summary_report="No revision required - responses meet quality standards"
            )
        
        try:
            # Revise each agent response
            revised_responses = []
            total_improvement = 0.0
            
            for response in processed_analysis.agent_responses:
                revised_response = await self._revise_single_response(
                    response, revision_plan, processed_analysis
                )
                revised_responses.append(revised_response)
                total_improvement += revised_response.improvement_score
            
            # Calculate overall improvement
            overall_improvement = total_improvement / len(revised_responses) if revised_responses else 0.0
            
            # Determine revision success
            revision_success = overall_improvement >= self.min_improvement_threshold
            
            # Generate summary report
            summary_report = self._generate_summary_report(
                revision_plan, revised_responses, overall_improvement
            )
            
            revision_session = RevisionSession(
                session_id=revision_plan.session_id,
                original_analysis=processed_analysis,
                revision_plan=revision_plan,
                revised_responses=revised_responses,
                overall_improvement=overall_improvement,
                revision_success=revision_success,
                processing_time=time.time() - start_time,
                revision_status=RevisionStatus.COMPLETED,
                completion_timestamp=self._get_timestamp(),
                summary_report=summary_report
            )
            
            # Update statistics
            self.total_revision_sessions += 1
            if revision_success:
                self.successful_revisions += 1
            else:
                self.failed_revisions += 1
            
            logger.success(
                f"Revision completed for session {revision_plan.session_id} "
                f"(improvement: {overall_improvement:.2f})"
            )
            
            return revision_session
            
        except Exception as e:
            logger.error(f"Revision execution failed for session {revision_plan.session_id}: {str(e)}")
            
            return RevisionSession(
                session_id=revision_plan.session_id,
                original_analysis=processed_analysis,
                revision_plan=revision_plan,
                revised_responses=[],
                overall_improvement=0.0,
                revision_success=False,
                processing_time=time.time() - start_time,
                revision_status=RevisionStatus.FAILED,
                completion_timestamp=self._get_timestamp(),
                summary_report=f"Revision failed: {str(e)}"
            )
    
    async def _revise_single_response(
        self,
        response: AgentResponse,
        revision_plan: RevisionPlan,
        processed_analysis: ProcessedAnalysis
    ) -> RevisedResponse:
        """Revise a single agent response based on revision plan."""
        
        logger.info(f"Revising response from {response.agent_id}")
        
        # Collect all instructions (prioritized)
        all_instructions = (
            revision_plan.high_priority_instructions +
            revision_plan.medium_priority_instructions +
            revision_plan.low_priority_instructions
        )
        
        # Filter instructions relevant to this response
        relevant_instructions = [
            inst for inst in all_instructions
            if self._instruction_applies_to_response(inst, response)
        ][:self.max_instructions_per_response]
        
        # Apply revisions
        revised_answer = response.answer
        revised_reasoning = response.reasoning
        revised_confidence = response.confidence_score
        
        instructions_followed = []
        instructions_skipped = []
        revision_notes = []
        
        for instruction in relevant_instructions:
            try:
                if instruction.category == "citation":
                    revised_answer = self._apply_citation_revision(revised_answer, instruction)
                    instructions_followed.append(instruction)
                    
                elif instruction.category == "evidence":
                    revised_answer = self._apply_evidence_revision(revised_answer, instruction)
                    instructions_followed.append(instruction)
                    
                elif instruction.category == "logic":
                    revised_answer = self._apply_logic_revision(revised_answer, instruction)
                    instructions_followed.append(instruction)
                    
                elif instruction.category == "confidence":
                    revised_confidence = self._apply_confidence_revision(
                        revised_confidence, instruction, response
                    )
                    instructions_followed.append(instruction)
                    
                elif instruction.category == "reasoning":
                    revised_reasoning = self._apply_reasoning_revision(revised_reasoning, instruction)
                    instructions_followed.append(instruction)
                    
                else:
                    instructions_skipped.append(instruction)
                    revision_notes.append(f"Skipped unknown category: {instruction.category}")
                    
            except Exception as e:
                instructions_skipped.append(instruction)
                revision_notes.append(f"Failed to apply instruction: {str(e)}")
                logger.warning(f"Failed to apply instruction {instruction.action}: {str(e)}")
        
        # Count instructions followed for statistics
        self.total_instructions_followed += len(instructions_followed)
        
        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(
            response, revised_answer, revised_reasoning, revised_confidence,
            instructions_followed, processed_analysis
        )
        
        # Count issues addressed
        issues_addressed = len(instructions_followed)
        issues_remaining = len(instructions_skipped)
        
        revised_response = RevisedResponse(
            original_response=response,
            revised_answer=revised_answer,
            revised_reasoning=revised_reasoning,
            revised_confidence=revised_confidence,
            instructions_followed=instructions_followed,
            instructions_skipped=instructions_skipped,
            revision_notes="; ".join(revision_notes),
            improvement_score=improvement_score,
            issues_addressed=issues_addressed,
            issues_remaining=issues_remaining,
            revision_timestamp=self._get_timestamp()
        )
        
        logger.info(
            f"Revised response from {response.agent_id} "
            f"({issues_addressed} issues addressed, improvement: {improvement_score:.2f})"
        )
        
        return revised_response
    
    def _determine_revision_type(self, processed_analysis: ProcessedAnalysis) -> RevisionType:
        """Determine the type of revision needed based on analysis."""
        
        if processed_analysis.critical_issues >= 3:
            return RevisionType.COMPLETE_REWRITE
        elif processed_analysis.critical_issues >= 1 or processed_analysis.needs_major_revision:
            return RevisionType.MAJOR_REVISION
        elif processed_analysis.moderate_issues >= 2 or processed_analysis.needs_moderate_revision:
            return RevisionType.MODERATE_REVISION
        else:
            return RevisionType.MINOR_IMPROVEMENT
    
    def _determine_revision_status(
        self, 
        processed_analysis: ProcessedAnalysis,
        revision_type: RevisionType
    ) -> RevisionStatus:
        """Determine if revision is required."""
        
        # Quality-based decision
        if processed_analysis.quality_score >= 0.8 and processed_analysis.total_issues <= 2:
            return RevisionStatus.NOT_REQUIRED
        
        # Issue-based decision
        if processed_analysis.critical_issues > 0 or processed_analysis.moderate_issues > 1:
            return RevisionStatus.REQUIRED
        
        # Type-based decision
        if revision_type in [RevisionType.MAJOR_REVISION, RevisionType.COMPLETE_REWRITE]:
            return RevisionStatus.REQUIRED
        
        return RevisionStatus.NOT_REQUIRED
    
    async def _generate_revision_instructions(
        self, 
        processed_analysis: ProcessedAnalysis
    ) -> List[RevisionInstruction]:
        """Generate specific revision instructions based on analysis."""
        
        instructions = []
        
        for analysis_type, result in processed_analysis.analysis_results.items():
            category_instructions = self._generate_category_instructions(analysis_type, result)
            instructions.extend(category_instructions)
        
        # Add general improvement instructions if few specific issues found
        if len(instructions) < 3:
            general_instructions = self._generate_general_instructions(processed_analysis)
            instructions.extend(general_instructions)
        
        return instructions[:20]  # Limit total instructions
    
    def _generate_category_instructions(
        self,
        analysis_type: AnalysisType,
        result: AnalysisResult
    ) -> List[RevisionInstruction]:
        """Generate instructions for a specific analysis category."""
        
        instructions = []
        
        if analysis_type == AnalysisType.UNSUPPORTED_CLAIMS:
            for issue in result.issues_found:
                if issue.get("severity", 0) >= 0.6:
                    instructions.append(RevisionInstruction(
                        priority="high",
                        category="citation",
                        target=issue.get("claim", "unsupported claim"),
                        action="add_citation",
                        reason="Claim lacks supporting evidence",
                        example="Add: According to [Source], ..."
                    ))
        
        elif analysis_type == AnalysisType.WEAK_CITATIONS:
            for issue in result.issues_found:
                instructions.append(RevisionInstruction(
                    priority="medium",
                    category="citation",
                    target="citation quality",
                    action="upgrade_sources",
                    reason="Citations from low-authority sources",
                    example="Replace blog/social media with academic/government sources"
                ))
        
        elif analysis_type == AnalysisType.CONTRADICTORY_EVIDENCE:
            for issue in result.issues_found:
                instructions.append(RevisionInstruction(
                    priority="high",
                    category="logic",
                    target="contradictory evidence",
                    action="acknowledge_contradiction",
                    reason="Conflicting evidence not addressed",
                    example="Note: While X suggests..., Y indicates... Further research needed."
                ))
        
        elif analysis_type == AnalysisType.INSUFFICIENT_EVIDENCE:
            instructions.append(RevisionInstruction(
                priority="medium",
                category="evidence",
                target="evidence quantity",
                action="strengthen_evidence_statements",
                reason="Insufficient supporting evidence",
                example="Qualify claims with 'preliminary evidence suggests' or 'limited data indicates'"
            ))
        
        elif analysis_type == AnalysisType.LOGICAL_CONSISTENCY:
            for issue in result.issues_found:
                if issue.get("type") == "confidence_evidence_mismatch":
                    instructions.append(RevisionInstruction(
                        priority="medium",
                        category="confidence",
                        target="confidence calibration",
                        action="adjust_confidence",
                        reason=f"Confidence ({issue.get('stated_confidence', 0):.2f}) doesn't match evidence strength",
                        example="Lower confidence to match available evidence quality"
                    ))
        
        return instructions[:5]  # Limit per category
    
    def _generate_general_instructions(
        self, 
        processed_analysis: ProcessedAnalysis
    ) -> List[RevisionInstruction]:
        """Generate general improvement instructions."""
        
        instructions = []
        
        if processed_analysis.quality_score < 0.6:
            instructions.append(RevisionInstruction(
                priority="medium",
                category="reasoning",
                target="overall reasoning",
                action="strengthen_argumentation",
                reason="Overall argument could be strengthened",
                example="Add more detailed explanation of evidence evaluation"
            ))
        
        if len(processed_analysis.agent_responses) > 1:
            instructions.append(RevisionInstruction(
                priority="low",
                category="logic",
                target="consistency",
                action="ensure_consistency",
                reason="Multiple responses should be consistent",
                example="Align terminology and conclusions across responses"
            ))
        
        return instructions
    
    def _prioritize_instructions(
        self, 
        instructions: List[RevisionInstruction]
    ) -> Tuple[List[RevisionInstruction], List[RevisionInstruction], List[RevisionInstruction]]:
        """Prioritize instructions into high, medium, and low priority groups."""
        
        high_priority = [inst for inst in instructions if inst.priority == "high"]
        medium_priority = [inst for inst in instructions if inst.priority == "medium"]
        low_priority = [inst for inst in instructions if inst.priority == "low"]
        
        # Limit each category
        return high_priority[:5], medium_priority[:8], low_priority[:5]
    
    def _instruction_applies_to_response(
        self, 
        instruction: RevisionInstruction, 
        response: AgentResponse
    ) -> bool:
        """Check if instruction applies to specific response."""
        
        # Simple heuristic - could be more sophisticated
        target_lower = instruction.target.lower()
        response_text = (response.answer + " " + response.reasoning).lower()
        
        # Check if target content is in response
        if any(word in response_text for word in target_lower.split()):
            return True
        
        # Category-based applicability
        if instruction.category == "confidence":
            return True  # Confidence always applies
        
        if instruction.category == "citation" and len(response.citations) > 0:
            return True
        
        if instruction.category == "evidence" and len(response.evidence) > 0:
            return True
        
        return False
    
    def _apply_citation_revision(self, answer: str, instruction: RevisionInstruction) -> str:
        """Apply citation-related revision to answer."""
        
        if instruction.action == "add_citation":
            # Add placeholder citation markers
            if "[citation needed]" not in answer:
                answer = answer.replace(".", ". [citation needed]", 1)
        
        elif instruction.action == "upgrade_sources":
            # Add note about source quality
            answer += " Note: Citations should be upgraded to more authoritative sources."
        
        return answer
    
    def _apply_evidence_revision(self, answer: str, instruction: RevisionInstruction) -> str:
        """Apply evidence-related revision to answer."""
        
        if instruction.action == "strengthen_evidence_statements":
            # Add qualifying language
            answer = answer.replace("studies show", "preliminary studies suggest")
            answer = answer.replace("research indicates", "limited research indicates")
            answer = answer.replace("data shows", "available data suggests")
        
        return answer
    
    def _apply_logic_revision(self, answer: str, instruction: RevisionInstruction) -> str:
        """Apply logic-related revision to answer."""
        
        if instruction.action == "acknowledge_contradiction":
            answer += " Note: There is conflicting evidence on this topic that requires further investigation."
        
        elif instruction.action == "ensure_consistency":
            # This would require more sophisticated text processing
            pass
        
        return answer
    
    def _apply_confidence_revision(
        self, 
        confidence: float, 
        instruction: RevisionInstruction, 
        response: AgentResponse
    ) -> float:
        """Apply confidence-related revision."""
        
        if instruction.action == "adjust_confidence":
            # Calculate evidence-based confidence
            if response.evidence:
                evidence_strength = sum(e.quality_score * e.relevance_score for e in response.evidence) / len(response.evidence)
                return min(confidence, evidence_strength + 0.1)  # Cap confidence based on evidence
        
        return confidence
    
    def _apply_reasoning_revision(self, reasoning: str, instruction: RevisionInstruction) -> str:
        """Apply reasoning-related revision."""
        
        if instruction.action == "strengthen_argumentation":
            reasoning += " The analysis considers multiple perspectives and acknowledges limitations in available evidence."
        
        return reasoning
    
    def _calculate_improvement_score(
        self,
        original_response: AgentResponse,
        revised_answer: str,
        revised_reasoning: str,
        revised_confidence: float,
        instructions_followed: List[RevisionInstruction],
        processed_analysis: ProcessedAnalysis
    ) -> float:
        """Calculate improvement score for revised response."""
        
        # Base improvement from following instructions
        instruction_score = len(instructions_followed) * 0.1
        
        # Improvement from answer length/quality (simple heuristic)
        answer_improvement = (len(revised_answer) - len(original_response.answer)) / 1000.0
        answer_improvement = max(0, min(0.2, answer_improvement))  # Cap at 0.2
        
        # Confidence calibration improvement
        confidence_improvement = 0.0
        if abs(revised_confidence - original_response.confidence_score) > 0.1:
            # Assume confidence adjustment is an improvement
            confidence_improvement = 0.1
        
        total_improvement = instruction_score + answer_improvement + confidence_improvement
        
        return min(1.0, total_improvement)
    
    def _estimate_revision_effort(self, num_instructions: int, revision_type: RevisionType) -> str:
        """Estimate the effort required for revision."""
        
        if revision_type == RevisionType.COMPLETE_REWRITE:
            return "high"
        elif revision_type == RevisionType.MAJOR_REVISION or num_instructions > 8:
            return "high"
        elif revision_type == RevisionType.MODERATE_REVISION or num_instructions > 4:
            return "medium"
        else:
            return "low"
    
    def _describe_expected_improvement(
        self,
        processed_analysis: ProcessedAnalysis,
        instructions: List[RevisionInstruction]
    ) -> str:
        """Describe expected improvement from revision."""
        
        high_priority_count = len([i for i in instructions if i.priority == "high"])
        
        if high_priority_count >= 3:
            return "Significant improvement expected - addressing major quality issues"
        elif high_priority_count >= 1:
            return "Moderate improvement expected - fixing critical issues"
        else:
            return "Minor improvements to enhance overall quality"
    
    def _generate_summary_report(
        self,
        revision_plan: RevisionPlan,
        revised_responses: List[RevisedResponse],
        overall_improvement: float
    ) -> str:
        """Generate summary report of revision session."""
        
        total_instructions = sum(len(r.instructions_followed) for r in revised_responses)
        total_issues_addressed = sum(r.issues_addressed for r in revised_responses)
        
        report_parts = []
        report_parts.append(f"Revision Summary for Session {revision_plan.session_id}")
        report_parts.append(f"Revision Type: {revision_plan.revision_type.value}")
        report_parts.append(f"Instructions Generated: {revision_plan.total_instructions}")
        report_parts.append(f"Instructions Followed: {total_instructions}")
        report_parts.append(f"Issues Addressed: {total_issues_addressed}")
        report_parts.append(f"Overall Improvement: {overall_improvement:.2f}")
        
        if overall_improvement >= 0.3:
            report_parts.append("Result: Significant improvement achieved")
        elif overall_improvement >= 0.2:
            report_parts.append("Result: Moderate improvement achieved")
        else:
            report_parts.append("Result: Minor improvements made")
        
        return " | ".join(report_parts)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_revision_statistics(self) -> Dict[str, Any]:
        """Get revision manager statistics."""
        
        return {
            "total_revision_sessions": self.total_revision_sessions,
            "successful_revisions": self.successful_revisions,
            "failed_revisions": self.failed_revisions,
            "success_rate": (
                self.successful_revisions / self.total_revision_sessions
                if self.total_revision_sessions > 0 else 0.0
            ),
            "total_instructions_generated": self.total_instructions_generated,
            "total_instructions_followed": self.total_instructions_followed,
            "instruction_follow_rate": (
                self.total_instructions_followed / self.total_instructions_generated
                if self.total_instructions_generated > 0 else 0.0
            ),
            "configuration": {
                "max_instructions_per_response": self.max_instructions_per_response,
                "min_improvement_threshold": self.min_improvement_threshold,
                "preserve_agent_voice": self.preserve_agent_voice,
                "strict_single_round": self.strict_single_round
            }
        }
    
    def reset_statistics(self):
        """Reset revision manager statistics."""
        
        self.total_revision_sessions = 0
        self.successful_revisions = 0
        self.failed_revisions = 0
        self.total_instructions_generated = 0
        self.total_instructions_followed = 0
        
        logger.info("RevisionManager statistics reset")