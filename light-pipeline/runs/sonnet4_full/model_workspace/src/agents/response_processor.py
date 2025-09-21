"""ResponseProcessor for standardizing and processing agent outputs."""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
import time
from datetime import datetime
from loguru import logger
import hashlib

from .answering_agent import AgentResponse
from .challenger_agent import ChallengeReport, Challenge, ChallengeType
from ..schemas.citation_schemas import CitationSchema, EvidenceSchema


class ProcessingStatus(Enum):
    """Status values for processed responses."""
    PENDING = "pending"
    PROCESSED = "processed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"


class ConfidenceCategory(Enum):
    """Categories for confidence scores."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class ProcessedResponse:
    """Standardized processed response from any agent."""
    
    # Original response metadata
    response_id: str
    agent_id: str
    agent_type: str  # 'answering' or 'challenger'
    processing_timestamp: str
    
    # Content standardization
    original_claim: str
    processed_content: str
    confidence_score: float
    confidence_category: ConfidenceCategory
    
    # Quality metrics
    citation_count: int
    evidence_count: int
    token_usage: int
    processing_time: float
    
    # Validation results
    validation_status: ProcessingStatus
    validation_errors: List[str]
    quality_score: float  # Computed overall quality (0.0 to 1.0)
    
    # Additional metadata
    metadata: Dict[str, Any]
    raw_response: Dict[str, Any]  # Original response serialized


@dataclass
class ProcessedChallenge:
    """Standardized processed challenge report."""
    
    # Challenge metadata
    challenge_id: str
    challenger_id: str
    target_agent_id: str
    processing_timestamp: str
    
    # Challenge analysis
    total_challenges: int
    critical_challenges: int
    challenge_types: List[str]
    overall_severity: float
    
    # Processing results
    requires_revision: bool
    priority_actions: List[str]
    confidence_in_challenges: float
    
    # Quality assessment
    assessment_quality: float
    processing_status: ProcessingStatus
    
    # References
    original_report: Dict[str, Any]
    affected_response_id: str


class ResponseProcessor:
    """
    Standardizes agent outputs and applies quality validation.
    
    Processes responses from both answering and challenger agents,
    ensuring consistent formatting and comprehensive quality assessment.
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.5,
        citation_weight: float = 0.3,
        evidence_weight: float = 0.3,
        confidence_weight: float = 0.4
    ):
        """Initialize the response processor."""
        
        self.min_quality_threshold = min_quality_threshold
        self.citation_weight = citation_weight
        self.evidence_weight = evidence_weight
        self.confidence_weight = confidence_weight
        
        # Processing statistics
        self.processed_responses = 0
        self.processed_challenges = 0
        self.rejected_responses = 0
        self.total_processing_time = 0.0
        
        # Validation patterns
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.citation_pattern = re.compile(r'\[\d+\]')
        
        logger.info("ResponseProcessor initialized")
    
    async def process_agent_response(
        self,
        response: AgentResponse,
        original_claim: str
    ) -> ProcessedResponse:
        """
        Process and standardize an answering agent response.
        
        Args:
            response: The agent response to process
            original_claim: Original claim being evaluated
            
        Returns:
            ProcessedResponse with standardized format and quality metrics
        """
        
        logger.info(f"Processing response from {response.agent_id}")
        start_time = time.time()
        
        try:
            # Generate unique response ID
            response_id = self._generate_response_id(response)
            
            # Standardize content
            processed_content = await self._standardize_response_content(response)
            
            # Categorize confidence
            confidence_category = self._categorize_confidence(response.confidence_score)
            
            # Validate response
            validation_status, validation_errors = await self._validate_response(response)
            
            # Calculate quality score
            quality_score = await self._calculate_response_quality(response)
            
            # Extract metadata
            metadata = self._extract_response_metadata(response)
            
            # Serialize raw response
            raw_response = self._serialize_response(response)
            
            processed = ProcessedResponse(
                response_id=response_id,
                agent_id=response.agent_id,
                agent_type="answering",
                processing_timestamp=self._get_timestamp(),
                original_claim=original_claim,
                processed_content=processed_content,
                confidence_score=response.confidence_score,
                confidence_category=confidence_category,
                citation_count=len(response.citations),
                evidence_count=len(response.evidence),
                token_usage=response.token_usage,
                processing_time=response.processing_time,
                validation_status=validation_status,
                validation_errors=validation_errors,
                quality_score=quality_score,
                metadata=metadata,
                raw_response=raw_response
            )
            
            self.processed_responses += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.success(
                f"Processed response {response_id} "
                f"(quality: {quality_score:.2f}, status: {validation_status.value})"
            )
            
            return processed
            
        except Exception as e:
            self.rejected_responses += 1
            logger.error(f"Failed to process response from {response.agent_id}: {str(e)}")
            raise
    
    async def process_challenge_report(
        self,
        report: ChallengeReport,
        target_response_id: str
    ) -> ProcessedChallenge:
        """
        Process and standardize a challenger agent report.
        
        Args:
            report: The challenge report to process
            target_response_id: ID of the response being challenged
            
        Returns:
            ProcessedChallenge with standardized format and analysis
        """
        
        logger.info(f"Processing challenge report from {report.challenger_id}")
        
        try:
            # Generate unique challenge ID
            challenge_id = self._generate_challenge_id(report)
            
            # Analyze challenges
            critical_challenges = len([c for c in report.challenges if c.severity >= 0.8])
            challenge_types = list(set(c.challenge_type.value for c in report.challenges))
            overall_severity = (
                sum(c.severity for c in report.challenges) / len(report.challenges)
                if report.challenges else 0.0
            )
            
            # Extract priority actions
            priority_actions = self._extract_priority_actions(report)
            
            # Assess challenge quality
            assessment_quality = await self._assess_challenge_quality(report)
            
            # Determine processing status
            if assessment_quality < self.min_quality_threshold:
                processing_status = ProcessingStatus.REJECTED
            elif report.requires_revision:
                processing_status = ProcessingStatus.REQUIRES_REVISION
            else:
                processing_status = ProcessingStatus.PROCESSED
            
            # Serialize original report
            original_report = self._serialize_challenge_report(report)
            
            processed = ProcessedChallenge(
                challenge_id=challenge_id,
                challenger_id=report.challenger_id,
                target_agent_id=report.original_response.agent_id,
                processing_timestamp=self._get_timestamp(),
                total_challenges=len(report.challenges),
                critical_challenges=critical_challenges,
                challenge_types=challenge_types,
                overall_severity=overall_severity,
                requires_revision=report.requires_revision,
                priority_actions=priority_actions,
                confidence_in_challenges=report.confidence_in_challenges,
                assessment_quality=assessment_quality,
                processing_status=processing_status,
                original_report=original_report,
                affected_response_id=target_response_id
            )
            
            self.processed_challenges += 1
            
            logger.success(
                f"Processed challenge {challenge_id} "
                f"(severity: {overall_severity:.2f}, {len(report.challenges)} challenges)"
            )
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process challenge report from {report.challenger_id}: {str(e)}")
            raise
    
    async def _standardize_response_content(self, response: AgentResponse) -> str:
        """Standardize the formatting of response content."""
        
        content_parts = []
        
        # Main answer
        content_parts.append(f"ANSWER: {response.answer}")
        
        # Citations section
        if response.citations:
            content_parts.append("\nCITATIONS:")
            for i, citation in enumerate(response.citations, 1):
                content_parts.append(f"[{i}] {citation.formatted_citation}")
        
        # Evidence section
        if response.evidence:
            content_parts.append("\nSUPPORTING EVIDENCE:")
            for i, evidence in enumerate(response.evidence[:5], 1):  # Top 5
                support_indicator = "✓" if evidence.supports_claim else "✗"
                content_parts.append(
                    f"{i}. {support_indicator} {evidence.evidence_text[:200]}... "
                    f"(Quality: {evidence.quality_score:.2f}, Relevance: {evidence.relevance_score:.2f})"
                )
        
        # Reasoning
        content_parts.append(f"\nREASONING: {response.reasoning}")
        
        # Confidence
        content_parts.append(f"\nCONFIDENCE: {response.confidence_score:.3f}")
        
        return "\n".join(content_parts)
    
    def _categorize_confidence(self, confidence: float) -> ConfidenceCategory:
        """Categorize confidence score into discrete levels."""
        
        if confidence >= 0.8:
            return ConfidenceCategory.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceCategory.HIGH
        elif confidence >= 0.4:
            return ConfidenceCategory.MODERATE
        elif confidence >= 0.2:
            return ConfidenceCategory.LOW
        else:
            return ConfidenceCategory.VERY_LOW
    
    async def _validate_response(self, response: AgentResponse) -> Tuple[ProcessingStatus, List[str]]:
        """Validate response content and structure."""
        
        errors = []
        
        # Check basic structure
        if not response.answer or len(response.answer.strip()) < 50:
            errors.append("Answer too short or empty")
        
        if not response.reasoning or len(response.reasoning.strip()) < 20:
            errors.append("Insufficient reasoning provided")
        
        # Check confidence score validity
        if not (0.0 <= response.confidence_score <= 1.0):
            errors.append(f"Invalid confidence score: {response.confidence_score}")
        
        # Check citations
        if response.citations:
            for i, citation in enumerate(response.citations):
                if not citation.formatted_citation:
                    errors.append(f"Citation {i+1} missing formatted text")
                if not citation.url or not citation.url.startswith(('http://', 'https://')):
                    errors.append(f"Citation {i+1} has invalid URL")
        
        # Check evidence
        if response.evidence:
            for i, evidence in enumerate(response.evidence):
                if not evidence.evidence_text or len(evidence.evidence_text) < 10:
                    errors.append(f"Evidence {i+1} too short or empty")
                if not (0.0 <= evidence.quality_score <= 1.0):
                    errors.append(f"Evidence {i+1} invalid quality score")
                if not (0.0 <= evidence.relevance_score <= 1.0):
                    errors.append(f"Evidence {i+1} invalid relevance score")
        
        # Check token usage
        if response.token_usage <= 0:
            errors.append("Invalid token usage reported")
        
        # Determine status
        if not errors:
            status = ProcessingStatus.VALIDATED
        elif len(errors) <= 2:
            status = ProcessingStatus.PROCESSED
        else:
            status = ProcessingStatus.REJECTED
        
        return status, errors
    
    async def _calculate_response_quality(self, response: AgentResponse) -> float:
        """Calculate overall quality score for the response."""
        
        # Citation quality component
        citation_score = 0.0
        if response.citations:
            valid_citations = sum(
                1 for c in response.citations 
                if c.formatted_citation and c.url and c.url.startswith(('http://', 'https://'))
            )
            citation_score = valid_citations / len(response.citations)
        
        # Evidence quality component
        evidence_score = 0.0
        if response.evidence:
            evidence_score = sum(
                (e.quality_score * e.relevance_score) for e in response.evidence
            ) / len(response.evidence)
        
        # Confidence component (scaled to reward appropriate confidence)
        confidence_score = response.confidence_score
        
        # Weighted combination
        quality_score = (
            self.citation_weight * citation_score +
            self.evidence_weight * evidence_score +
            self.confidence_weight * confidence_score
        )
        
        # Bonus for comprehensive responses
        if len(response.citations) >= 3 and len(response.evidence) >= 3:
            quality_score += 0.1
        
        # Penalty for very short responses
        if len(response.answer) < 200:
            quality_score -= 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _extract_response_metadata(self, response: AgentResponse) -> Dict[str, Any]:
        """Extract additional metadata from the response."""
        
        # Text analysis
        word_count = len(response.answer.split())
        sentence_count = len([s for s in response.answer.split('.') if s.strip()])
        url_count = len(self.url_pattern.findall(response.answer))
        citation_ref_count = len(self.citation_pattern.findall(response.answer))
        
        # Evidence analysis
        supporting_evidence = sum(1 for e in response.evidence if e.supports_claim)
        contradicting_evidence = len(response.evidence) - supporting_evidence
        
        avg_evidence_quality = (
            sum(e.quality_score for e in response.evidence) / len(response.evidence)
            if response.evidence else 0.0
        )
        
        avg_evidence_relevance = (
            sum(e.relevance_score for e in response.evidence) / len(response.evidence)
            if response.evidence else 0.0
        )
        
        return {
            "text_analysis": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "url_count": url_count,
                "citation_references": citation_ref_count
            },
            "evidence_analysis": {
                "supporting_count": supporting_evidence,
                "contradicting_count": contradicting_evidence,
                "avg_quality": avg_evidence_quality,
                "avg_relevance": avg_evidence_relevance
            },
            "processing_info": {
                "tokens_per_second": response.token_usage / response.processing_time if response.processing_time > 0 else 0,
                "evidence_per_citation": len(response.evidence) / len(response.citations) if response.citations else 0
            }
        }
    
    def _extract_priority_actions(self, report: ChallengeReport) -> List[str]:
        """Extract priority actions from challenge report."""
        
        actions = []
        
        # Add actions from priority challenges
        for challenge in report.priority_challenges:
            if challenge.suggested_improvement:
                actions.append(f"{challenge.challenge_type.value}: {challenge.suggested_improvement}")
        
        # Add general revision requirement
        if report.requires_revision:
            actions.append("General revision required based on challenge analysis")
        
        return actions[:5]  # Limit to top 5 actions
    
    async def _assess_challenge_quality(self, report: ChallengeReport) -> float:
        """Assess the quality of the challenge analysis."""
        
        quality_factors = []
        
        # Challenge specificity
        if report.challenges:
            avg_severity = sum(c.severity for c in report.challenges) / len(report.challenges)
            quality_factors.append(avg_severity)
            
            # Check for specific suggestions
            specific_suggestions = sum(
                1 for c in report.challenges 
                if c.suggested_improvement and len(c.suggested_improvement) > 20
            )
            suggestion_ratio = specific_suggestions / len(report.challenges)
            quality_factors.append(suggestion_ratio)
        
        # Confidence in analysis
        quality_factors.append(report.confidence_in_challenges)
        
        # Assessment comprehensiveness
        assessment_length = len(report.overall_assessment.split())
        comprehensiveness = min(1.0, assessment_length / 50)  # Target: 50+ words
        quality_factors.append(comprehensiveness)
        
        # Diversity of challenge types
        unique_types = len(set(c.challenge_type for c in report.challenges))
        diversity = min(1.0, unique_types / 4)  # Max diversity with 4+ types
        quality_factors.append(diversity)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    def _generate_response_id(self, response: AgentResponse) -> str:
        """Generate unique ID for processed response."""
        
        # Create hash from agent ID, claim, and timestamp
        hash_input = f"{response.agent_id}_{response.claim}_{self._get_timestamp()}"
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"resp_{response.agent_id}_{hash_digest}"
    
    def _generate_challenge_id(self, report: ChallengeReport) -> str:
        """Generate unique ID for processed challenge."""
        
        hash_input = f"{report.challenger_id}_{report.original_response.agent_id}_{self._get_timestamp()}"
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"chal_{report.challenger_id}_{hash_digest}"
    
    def _serialize_response(self, response: AgentResponse) -> Dict[str, Any]:
        """Serialize agent response for storage."""
        
        return {
            "agent_id": response.agent_id,
            "claim": response.claim,
            "answer": response.answer,
            "reasoning": response.reasoning,
            "confidence_score": response.confidence_score,
            "token_usage": response.token_usage,
            "processing_time": response.processing_time,
            "citations": [c.model_dump() if hasattr(c, 'model_dump') else c.__dict__ for c in response.citations],
            "evidence": [e.model_dump() if hasattr(e, 'model_dump') else e.__dict__ for e in response.evidence]
        }
    
    def _serialize_challenge_report(self, report: ChallengeReport) -> Dict[str, Any]:
        """Serialize challenge report for storage."""
        
        return {
            "challenger_id": report.challenger_id,
            "overall_assessment": report.overall_assessment,
            "confidence_in_challenges": report.confidence_in_challenges,
            "requires_revision": report.requires_revision,
            "token_usage": report.token_usage,
            "processing_time": report.processing_time,
            "challenges": [
                {
                    "type": c.challenge_type.value,
                    "description": c.description,
                    "severity": c.severity,
                    "affected_claims": c.affected_claims,
                    "suggested_improvement": c.suggested_improvement,
                    "evidence_against": c.evidence_against,
                    "source_quality_issues": c.source_quality_issues
                } for c in report.challenges
            ]
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()
    
    async def batch_process_responses(
        self,
        responses: List[AgentResponse],
        original_claim: str
    ) -> List[ProcessedResponse]:
        """Process multiple responses in batch."""
        
        logger.info(f"Batch processing {len(responses)} responses")
        
        processed_responses = []
        for response in responses:
            try:
                processed = await self.process_agent_response(response, original_claim)
                processed_responses.append(processed)
            except Exception as e:
                logger.error(f"Failed to process response from {response.agent_id}: {str(e)}")
                continue
        
        logger.success(f"Batch processed {len(processed_responses)}/{len(responses)} responses")
        return processed_responses
    
    async def batch_process_challenges(
        self,
        reports: List[ChallengeReport],
        response_ids: List[str]
    ) -> List[ProcessedChallenge]:
        """Process multiple challenge reports in batch."""
        
        logger.info(f"Batch processing {len(reports)} challenge reports")
        
        processed_challenges = []
        for report, response_id in zip(reports, response_ids):
            try:
                processed = await self.process_challenge_report(report, response_id)
                processed_challenges.append(processed)
            except Exception as e:
                logger.error(f"Failed to process challenge from {report.challenger_id}: {str(e)}")
                continue
        
        logger.success(f"Batch processed {len(processed_challenges)}/{len(reports)} challenges")
        return processed_challenges
    
    def get_processor_statistics(self) -> Dict[str, Any]:
        """Get processor performance statistics."""
        
        total_processed = self.processed_responses + self.processed_challenges
        
        return {
            "processing_counts": {
                "processed_responses": self.processed_responses,
                "processed_challenges": self.processed_challenges,
                "rejected_responses": self.rejected_responses,
                "total_processed": total_processed
            },
            "performance_metrics": {
                "total_processing_time": self.total_processing_time,
                "avg_processing_time": (
                    self.total_processing_time / total_processed 
                    if total_processed > 0 else 0.0
                ),
                "success_rate": (
                    (total_processed - self.rejected_responses) / total_processed 
                    if total_processed > 0 else 0.0
                )
            },
            "configuration": {
                "min_quality_threshold": self.min_quality_threshold,
                "citation_weight": self.citation_weight,
                "evidence_weight": self.evidence_weight,
                "confidence_weight": self.confidence_weight
            }
        }
    
    def reset_processor(self):
        """Reset processor statistics."""
        
        self.processed_responses = 0
        self.processed_challenges = 0
        self.rejected_responses = 0
        self.total_processing_time = 0.0
        
        logger.info("ResponseProcessor statistics reset")