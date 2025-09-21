"""ChallengeProcessor for systematic analysis of agent responses."""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
from loguru import logger
from collections import Counter

from ..agents.answering_agent import AgentResponse
from ..agents.challenger_agent import ChallengeReport, Challenge, ChallengeType
from ..schemas.citation_schemas import CitationSchema, EvidenceSchema


class AnalysisType(Enum):
    """Types of analysis performed by the challenge processor."""
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    WEAK_CITATIONS = "weak_citations"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SOURCE_QUALITY = "source_quality"
    LOGICAL_CONSISTENCY = "logical_consistency"


@dataclass
class AnalysisResult:
    """Result of a specific type of analysis."""
    
    analysis_type: AnalysisType
    issues_found: List[Dict[str, Any]]
    severity_scores: List[float]
    confidence: float
    recommendations: List[str]
    processing_time: float


@dataclass
class ProcessedAnalysis:
    """Complete processed analysis of agent responses."""
    
    session_id: str
    original_claim: str
    agent_responses: List[AgentResponse]
    challenge_reports: List[ChallengeReport]
    
    # Analysis results
    analysis_results: Dict[AnalysisType, AnalysisResult]
    
    # Summary metrics
    total_issues: int
    critical_issues: int
    moderate_issues: int
    minor_issues: int
    
    # Overall assessment
    needs_major_revision: bool
    needs_moderate_revision: bool
    quality_score: float  # 0.0 to 1.0
    confidence_in_analysis: float
    
    # Processing metadata
    processing_time: float
    timestamp: str


class ChallengeProcessor:
    """
    Systematic analysis for unsupported claims, weak citations, and conflicts.
    
    Processes challenge reports from challenger agents to identify patterns,
    prioritize issues, and generate comprehensive improvement recommendations.
    """
    
    def __init__(
        self,
        critical_severity_threshold: float = 0.8,
        moderate_severity_threshold: float = 0.5,
        min_confidence_threshold: float = 0.6,
        max_issues_per_category: int = 10
    ):
        """Initialize the challenge processor."""
        
        self.critical_severity_threshold = critical_severity_threshold
        self.moderate_severity_threshold = moderate_severity_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.max_issues_per_category = max_issues_per_category
        
        # Processing statistics
        self.total_sessions_processed = 0
        self.total_issues_identified = 0
        self.total_processing_time = 0.0
        
        logger.info("ChallengeProcessor initialized")
    
    async def process_challenges(
        self,
        session_id: str,
        original_claim: str,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> ProcessedAnalysis:
        """
        Process challenge reports systematically to identify patterns and issues.
        
        Args:
            session_id: Unique session identifier
            original_claim: Original factual claim being analyzed
            agent_responses: List of agent responses to analyze
            challenge_reports: List of challenge reports from challenger agents
            
        Returns:
            ProcessedAnalysis with systematic analysis and recommendations
        """
        
        import time
        start_time = time.time()
        
        logger.info(f"Processing challenges for session {session_id}")
        
        try:
            # Perform systematic analysis across all categories
            analysis_results = {}
            
            # Analysis 1: Unsupported claims analysis
            analysis_results[AnalysisType.UNSUPPORTED_CLAIMS] = await self._analyze_unsupported_claims(
                agent_responses, challenge_reports
            )
            
            # Analysis 2: Weak citations analysis
            analysis_results[AnalysisType.WEAK_CITATIONS] = await self._analyze_weak_citations(
                agent_responses, challenge_reports
            )
            
            # Analysis 3: Contradictory evidence analysis
            analysis_results[AnalysisType.CONTRADICTORY_EVIDENCE] = await self._analyze_contradictory_evidence(
                agent_responses, challenge_reports
            )
            
            # Analysis 4: Insufficient evidence analysis
            analysis_results[AnalysisType.INSUFFICIENT_EVIDENCE] = await self._analyze_insufficient_evidence(
                agent_responses, challenge_reports
            )
            
            # Analysis 5: Source quality analysis
            analysis_results[AnalysisType.SOURCE_QUALITY] = await self._analyze_source_quality(
                agent_responses, challenge_reports
            )
            
            # Analysis 6: Logical consistency analysis
            analysis_results[AnalysisType.LOGICAL_CONSISTENCY] = await self._analyze_logical_consistency(
                agent_responses, challenge_reports
            )
            
            # Calculate summary metrics
            total_issues, critical_issues, moderate_issues, minor_issues = self._calculate_issue_metrics(
                analysis_results
            )
            
            # Determine revision requirements
            needs_major_revision, needs_moderate_revision = self._determine_revision_needs(
                critical_issues, moderate_issues, total_issues
            )
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                analysis_results, total_issues, len(agent_responses)
            )
            
            # Calculate confidence in analysis
            confidence_in_analysis = self._calculate_analysis_confidence(
                analysis_results, len(challenge_reports)
            )
            
            processing_time = time.time() - start_time
            
            processed_analysis = ProcessedAnalysis(
                session_id=session_id,
                original_claim=original_claim,
                agent_responses=agent_responses,
                challenge_reports=challenge_reports,
                analysis_results=analysis_results,
                total_issues=total_issues,
                critical_issues=critical_issues,
                moderate_issues=moderate_issues,
                minor_issues=minor_issues,
                needs_major_revision=needs_major_revision,
                needs_moderate_revision=needs_moderate_revision,
                quality_score=quality_score,
                confidence_in_analysis=confidence_in_analysis,
                processing_time=processing_time,
                timestamp=self._get_timestamp()
            )
            
            # Update statistics
            self.total_sessions_processed += 1
            self.total_issues_identified += total_issues
            self.total_processing_time += processing_time
            
            logger.success(
                f"Challenge processing completed for session {session_id} "
                f"({total_issues} issues found, {processing_time:.2f}s)"
            )
            
            return processed_analysis
            
        except Exception as e:
            logger.error(f"Challenge processing failed for session {session_id}: {str(e)}")
            raise
    
    async def _analyze_unsupported_claims(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> AnalysisResult:
        """Analyze unsupported claims across all responses."""
        
        import time
        start_time = time.time()
        
        issues_found = []
        severity_scores = []
        recommendations = []
        
        # Collect all unsupported claim challenges
        unsupported_challenges = []
        for report in challenge_reports:
            for challenge in report.challenges:
                if challenge.challenge_type == ChallengeType.UNSUPPORTED_CLAIM:
                    unsupported_challenges.append(challenge)
        
        # Analyze patterns in unsupported claims
        claim_patterns = self._identify_claim_patterns(unsupported_challenges)
        
        for pattern, challenges in claim_patterns.items():
            if len(challenges) >= 2:  # Pattern appears in multiple challenges
                issue = {
                    "pattern": pattern,
                    "occurrences": len(challenges),
                    "affected_responses": list(set(c.affected_claims for c in challenges)),
                    "avg_severity": sum(c.severity for c in challenges) / len(challenges),
                    "description": f"Recurring unsupported claim pattern: {pattern}"
                }
                issues_found.append(issue)
                severity_scores.append(issue["avg_severity"])
        
        # Individual high-severity unsupported claims
        for challenge in unsupported_challenges:
            if challenge.severity >= self.critical_severity_threshold:
                issue = {
                    "type": "critical_unsupported_claim",
                    "claim": challenge.affected_claims[0] if challenge.affected_claims else "Unknown",
                    "severity": challenge.severity,
                    "description": challenge.description,
                    "suggestion": challenge.suggested_improvement
                }
                issues_found.append(issue)
                severity_scores.append(challenge.severity)
        
        # Generate recommendations
        if len(unsupported_challenges) > len(agent_responses):
            recommendations.append("Multiple agents are making unsupported claims - review evidence standards")
        
        if any(s >= 0.9 for s in severity_scores):
            recommendations.append("Critical unsupported claims identified - immediate revision required")
        
        recommendations.append("Add specific citations for all factual assertions")
        recommendations.append("Qualify uncertain statements with appropriate hedging language")
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_type=AnalysisType.UNSUPPORTED_CLAIMS,
            issues_found=issues_found[:self.max_issues_per_category],
            severity_scores=severity_scores,
            confidence=0.8,  # High confidence in unsupported claim detection
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    async def _analyze_weak_citations(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> AnalysisResult:
        """Analyze citation quality issues across responses."""
        
        import time
        start_time = time.time()
        
        issues_found = []
        severity_scores = []
        recommendations = []
        
        # Collect citation-related challenges
        citation_challenges = []
        for report in challenge_reports:
            for challenge in report.challenges:
                if challenge.challenge_type == ChallengeType.WEAK_CITATION:
                    citation_challenges.append(challenge)
        
        # Analyze citation quality by source type
        source_type_issues = Counter()
        for response in agent_responses:
            for citation in response.citations:
                if hasattr(citation, 'source_type'):
                    source_type = citation.source_type
                    if source_type in ['blog', 'social_media', 'unknown']:
                        source_type_issues[source_type] += 1
        
        if source_type_issues:
            issue = {
                "type": "low_quality_source_types",
                "source_issues": dict(source_type_issues),
                "total_low_quality": sum(source_type_issues.values()),
                "description": f"Found {sum(source_type_issues.values())} citations from low-quality source types"
            }
            issues_found.append(issue)
            severity_scores.append(0.6)
        
        # Analyze individual citation challenges
        for challenge in citation_challenges:
            if challenge.source_quality_issues:
                issue = {
                    "type": "citation_quality_issues",
                    "issues": challenge.source_quality_issues,
                    "severity": challenge.severity,
                    "affected_citations": challenge.affected_claims,
                    "description": challenge.description
                }
                issues_found.append(issue)
                severity_scores.append(challenge.severity)
        
        # Generate recommendations
        if source_type_issues['blog'] + source_type_issues['social_media'] > 2:
            recommendations.append("Replace blog and social media sources with academic or government sources")
        
        if any(s >= 0.7 for s in severity_scores):
            recommendations.append("Significant citation quality issues - upgrade to authoritative sources")
        
        recommendations.append("Prioritize peer-reviewed academic sources")
        recommendations.append("Verify all URLs are accessible and properly formatted")
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_type=AnalysisType.WEAK_CITATIONS,
            issues_found=issues_found[:self.max_issues_per_category],
            severity_scores=severity_scores,
            confidence=0.7,  # Good confidence in citation analysis
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    async def _analyze_contradictory_evidence(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> AnalysisResult:
        """Analyze contradictory evidence across responses."""
        
        import time
        start_time = time.time()
        
        issues_found = []
        severity_scores = []
        recommendations = []
        
        # Collect contradiction challenges
        contradiction_challenges = []
        for report in challenge_reports:
            for challenge in report.challenges:
                if challenge.challenge_type == ChallengeType.CONTRADICTORY_EVIDENCE:
                    contradiction_challenges.append(challenge)
        
        # Analyze inter-response contradictions
        for i, response1 in enumerate(agent_responses):
            for j, response2 in enumerate(agent_responses[i+1:], i+1):
                contradictions = self._find_response_contradictions(response1, response2)
                if contradictions:
                    issue = {
                        "type": "inter_response_contradiction",
                        "agent1": response1.agent_id,
                        "agent2": response2.agent_id,
                        "contradictions": contradictions,
                        "severity": len(contradictions) * 0.2  # Scale with number
                    }
                    issues_found.append(issue)
                    severity_scores.append(issue["severity"])
        
        # Individual contradiction challenges
        for challenge in contradiction_challenges:
            issue = {
                "type": "evidence_contradiction",
                "severity": challenge.severity,
                "description": challenge.description,
                "evidence_against": challenge.evidence_against,
                "affected_claims": challenge.affected_claims
            }
            issues_found.append(issue)
            severity_scores.append(challenge.severity)
        
        # Generate recommendations
        if len(contradiction_challenges) > 2:
            recommendations.append("Multiple contradictions found - resolve conflicting evidence")
        
        if any(s >= 0.8 for s in severity_scores):
            recommendations.append("Strong contradictions present - additional research needed")
        
        recommendations.append("Acknowledge conflicting evidence explicitly when present")
        recommendations.append("Seek additional sources to resolve contradictions")
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_type=AnalysisType.CONTRADICTORY_EVIDENCE,
            issues_found=issues_found[:self.max_issues_per_category],
            severity_scores=severity_scores,
            confidence=0.6,  # Moderate confidence - contradictions can be subtle
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    async def _analyze_insufficient_evidence(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> AnalysisResult:
        """Analyze evidence sufficiency across responses."""
        
        import time
        start_time = time.time()
        
        issues_found = []
        severity_scores = []
        recommendations = []
        
        # Analyze evidence quantity and quality
        for response in agent_responses:
            evidence_count = len(response.evidence)
            citation_count = len(response.citations)
            
            if evidence_count < 3:
                issue = {
                    "type": "insufficient_evidence_count",
                    "agent_id": response.agent_id,
                    "evidence_count": evidence_count,
                    "minimum_expected": 3,
                    "severity": 0.7
                }
                issues_found.append(issue)
                severity_scores.append(0.7)
            
            if citation_count < 3:
                issue = {
                    "type": "insufficient_citation_count",
                    "agent_id": response.agent_id,
                    "citation_count": citation_count,
                    "minimum_expected": 3,
                    "severity": 0.6
                }
                issues_found.append(issue)
                severity_scores.append(0.6)
            
            # Check evidence quality
            if response.evidence:
                avg_quality = sum(e.quality_score for e in response.evidence) / len(response.evidence)
                avg_relevance = sum(e.relevance_score for e in response.evidence) / len(response.evidence)
                
                if avg_quality < 0.6:
                    issue = {
                        "type": "low_evidence_quality",
                        "agent_id": response.agent_id,
                        "avg_quality": avg_quality,
                        "minimum_expected": 0.6,
                        "severity": 0.5
                    }
                    issues_found.append(issue)
                    severity_scores.append(0.5)
                
                if avg_relevance < 0.7:
                    issue = {
                        "type": "low_evidence_relevance",
                        "agent_id": response.agent_id,
                        "avg_relevance": avg_relevance,
                        "minimum_expected": 0.7,
                        "severity": 0.4
                    }
                    issues_found.append(issue)
                    severity_scores.append(0.4)
        
        # Collect insufficient evidence challenges
        insufficient_challenges = []
        for report in challenge_reports:
            for challenge in report.challenges:
                if challenge.challenge_type == ChallengeType.INSUFFICIENT_EVIDENCE:
                    insufficient_challenges.append(challenge)
        
        for challenge in insufficient_challenges:
            issue = {
                "type": "challenger_identified_insufficient",
                "severity": challenge.severity,
                "description": challenge.description,
                "suggestion": challenge.suggested_improvement
            }
            issues_found.append(issue)
            severity_scores.append(challenge.severity)
        
        # Generate recommendations
        min_evidence = min(len(r.evidence) for r in agent_responses) if agent_responses else 0
        if min_evidence < 3:
            recommendations.append(f"Gather additional evidence - minimum found: {min_evidence}, recommended: 3+")
        
        recommendations.append("Focus on high-quality, authoritative sources")
        recommendations.append("Ensure evidence directly supports the specific claim being made")
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_type=AnalysisType.INSUFFICIENT_EVIDENCE,
            issues_found=issues_found[:self.max_issues_per_category],
            severity_scores=severity_scores,
            confidence=0.8,  # High confidence in evidence quantity analysis
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    async def _analyze_source_quality(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> AnalysisResult:
        """Analyze source quality and bias across responses."""
        
        import time
        start_time = time.time()
        
        issues_found = []
        severity_scores = []
        recommendations = []
        
        # Analyze source diversity and authority
        all_sources = []
        for response in agent_responses:
            for citation in response.citations:
                if hasattr(citation, 'url'):
                    all_sources.append(citation.url)
        
        unique_sources = len(set(all_sources))
        total_sources = len(all_sources)
        
        if unique_sources < total_sources * 0.8:  # Too much duplication
            issue = {
                "type": "source_duplication",
                "unique_sources": unique_sources,
                "total_sources": total_sources,
                "duplication_rate": 1 - (unique_sources / total_sources),
                "severity": 0.5
            }
            issues_found.append(issue)
            severity_scores.append(0.5)
        
        # Collect source quality challenges
        source_challenges = []
        for report in challenge_reports:
            for challenge in report.challenges:
                if challenge.challenge_type == ChallengeType.BIASED_SOURCE:
                    source_challenges.append(challenge)
        
        for challenge in source_challenges:
            issue = {
                "type": "biased_or_low_quality_source",
                "severity": challenge.severity,
                "description": challenge.description,
                "affected_sources": challenge.affected_claims,
                "suggestion": challenge.suggested_improvement
            }
            issues_found.append(issue)
            severity_scores.append(challenge.severity)
        
        # Generate recommendations
        if len(source_challenges) > 1:
            recommendations.append("Multiple source quality issues - review source selection criteria")
        
        recommendations.append("Diversify source types (academic, government, news, expert analysis)")
        recommendations.append("Verify source authority and potential conflicts of interest")
        recommendations.append("Prefer primary sources over secondary sources when available")
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_type=AnalysisType.SOURCE_QUALITY,
            issues_found=issues_found[:self.max_issues_per_category],
            severity_scores=severity_scores,
            confidence=0.7,  # Good confidence in source analysis
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    async def _analyze_logical_consistency(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> AnalysisResult:
        """Analyze logical consistency in responses."""
        
        import time
        start_time = time.time()
        
        issues_found = []
        severity_scores = []
        recommendations = []
        
        # Analyze confidence vs evidence consistency
        for response in agent_responses:
            confidence = response.confidence_score
            evidence_strength = self._calculate_evidence_strength(response)
            
            confidence_evidence_gap = abs(confidence - evidence_strength)
            if confidence_evidence_gap > 0.3:  # Significant mismatch
                issue = {
                    "type": "confidence_evidence_mismatch",
                    "agent_id": response.agent_id,
                    "stated_confidence": confidence,
                    "evidence_strength": evidence_strength,
                    "gap": confidence_evidence_gap,
                    "severity": confidence_evidence_gap
                }
                issues_found.append(issue)
                severity_scores.append(confidence_evidence_gap)
        
        # Collect logical consistency challenges
        logic_challenges = []
        for report in challenge_reports:
            for challenge in report.challenges:
                if challenge.challenge_type == ChallengeType.LOGICAL_INCONSISTENCY:
                    logic_challenges.append(challenge)
        
        for challenge in logic_challenges:
            issue = {
                "type": "logical_inconsistency",
                "severity": challenge.severity,
                "description": challenge.description,
                "suggestion": challenge.suggested_improvement
            }
            issues_found.append(issue)
            severity_scores.append(challenge.severity)
        
        # Generate recommendations
        if any(i["type"] == "confidence_evidence_mismatch" and i["gap"] > 0.4 for i in issues_found):
            recommendations.append("Calibrate confidence levels to match evidence strength")
        
        if len(logic_challenges) > 0:
            recommendations.append("Review reasoning for logical consistency and coherence")
        
        recommendations.append("Ensure conclusions follow logically from presented evidence")
        recommendations.append("Address potential counter-arguments or alternative interpretations")
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_type=AnalysisType.LOGICAL_CONSISTENCY,
            issues_found=issues_found[:self.max_issues_per_category],
            severity_scores=severity_scores,
            confidence=0.6,  # Moderate confidence - logical analysis is complex
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    def _identify_claim_patterns(self, challenges: List[Challenge]) -> Dict[str, List[Challenge]]:
        """Identify common patterns in unsupported claim challenges."""
        
        patterns = {}
        
        for challenge in challenges:
            # Extract key words from the challenge description
            words = challenge.description.lower().split()
            
            # Look for pattern indicators
            if "statistics" in words or "data" in words or "study" in words:
                pattern = "unsupported_statistics"
            elif "always" in words or "never" in words or "all" in words:
                pattern = "absolute_claims"
            elif "causes" in words or "leads to" in challenge.description.lower():
                pattern = "causal_claims"
            else:
                pattern = "general_factual"
            
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(challenge)
        
        return patterns
    
    def _find_response_contradictions(
        self, 
        response1: AgentResponse, 
        response2: AgentResponse
    ) -> List[Dict[str, Any]]:
        """Find contradictions between two responses."""
        
        contradictions = []
        
        # Simple contradiction detection based on keywords
        answer1_words = set(response1.answer.lower().split())
        answer2_words = set(response2.answer.lower().split())
        
        # Look for opposing conclusions
        if ("supported" in answer1_words and "contradicted" in answer2_words) or \
           ("contradicted" in answer1_words and "supported" in answer2_words):
            contradictions.append({
                "type": "conclusion_contradiction",
                "response1_conclusion": "supported" if "supported" in answer1_words else "contradicted",
                "response2_conclusion": "supported" if "supported" in answer2_words else "contradicted"
            })
        
        # Check confidence discrepancy
        confidence_gap = abs(response1.confidence_score - response2.confidence_score)
        if confidence_gap > 0.4:
            contradictions.append({
                "type": "confidence_discrepancy",
                "agent1_confidence": response1.confidence_score,
                "agent2_confidence": response2.confidence_score,
                "gap": confidence_gap
            })
        
        return contradictions
    
    def _calculate_evidence_strength(self, response: AgentResponse) -> float:
        """Calculate overall evidence strength for a response."""
        
        if not response.evidence:
            return 0.0
        
        # Weighted combination of quality and relevance
        evidence_scores = []
        for evidence in response.evidence:
            score = (evidence.quality_score * 0.6) + (evidence.relevance_score * 0.4)
            evidence_scores.append(score)
        
        # Average with bonus for quantity (up to 5 pieces of evidence)
        avg_score = sum(evidence_scores) / len(evidence_scores)
        quantity_bonus = min(0.1, len(evidence_scores) * 0.02)
        
        return min(1.0, avg_score + quantity_bonus)
    
    def _calculate_issue_metrics(
        self, 
        analysis_results: Dict[AnalysisType, AnalysisResult]
    ) -> Tuple[int, int, int, int]:
        """Calculate summary metrics for issues found."""
        
        total_issues = 0
        critical_issues = 0
        moderate_issues = 0
        minor_issues = 0
        
        for result in analysis_results.values():
            for severity in result.severity_scores:
                total_issues += 1
                if severity >= self.critical_severity_threshold:
                    critical_issues += 1
                elif severity >= self.moderate_severity_threshold:
                    moderate_issues += 1
                else:
                    minor_issues += 1
        
        return total_issues, critical_issues, moderate_issues, minor_issues
    
    def _determine_revision_needs(
        self, 
        critical_issues: int, 
        moderate_issues: int, 
        total_issues: int
    ) -> Tuple[bool, bool]:
        """Determine if major or moderate revision is needed."""
        
        needs_major_revision = critical_issues >= 2 or total_issues >= 8
        needs_moderate_revision = moderate_issues >= 3 or total_issues >= 5
        
        return needs_major_revision, needs_moderate_revision
    
    def _calculate_quality_score(
        self,
        analysis_results: Dict[AnalysisType, AnalysisResult],
        total_issues: int,
        num_responses: int
    ) -> float:
        """Calculate overall quality score based on analysis results."""
        
        # Base score starts high
        base_score = 1.0
        
        # Penalty for issues (scaled by severity)
        issue_penalty = 0.0
        for result in analysis_results.values():
            for severity in result.severity_scores:
                issue_penalty += severity * 0.1  # Max 0.1 penalty per issue
        
        # Bonus for multiple responses (consensus)
        consensus_bonus = min(0.1, (num_responses - 1) * 0.05)
        
        final_score = base_score - issue_penalty + consensus_bonus
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_analysis_confidence(
        self,
        analysis_results: Dict[AnalysisType, AnalysisResult],
        num_challenge_reports: int
    ) -> float:
        """Calculate confidence in the analysis results."""
        
        # Average confidence across all analysis types
        confidences = [result.confidence for result in analysis_results.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Bonus for multiple challenge reports (more data)
        data_bonus = min(0.2, num_challenge_reports * 0.05)
        
        final_confidence = avg_confidence + data_bonus
        
        return min(1.0, final_confidence)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_processor_statistics(self) -> Dict[str, Any]:
        """Get challenge processor performance statistics."""
        
        return {
            "total_sessions_processed": self.total_sessions_processed,
            "total_issues_identified": self.total_issues_identified,
            "avg_issues_per_session": (
                self.total_issues_identified / self.total_sessions_processed
                if self.total_sessions_processed > 0 else 0.0
            ),
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (
                self.total_processing_time / self.total_sessions_processed
                if self.total_sessions_processed > 0 else 0.0
            ),
            "configuration": {
                "critical_severity_threshold": self.critical_severity_threshold,
                "moderate_severity_threshold": self.moderate_severity_threshold,
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_issues_per_category": self.max_issues_per_category
            }
        }
    
    def reset_statistics(self):
        """Reset processor statistics."""
        
        self.total_sessions_processed = 0
        self.total_issues_identified = 0
        self.total_processing_time = 0.0
        
        logger.info("ChallengeProcessor statistics reset")