"""ChallengerAgent implementation for adversarial review of factual claims."""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .answering_agent import AgentResponse
from ..schemas.citation_schemas import CitationSchema, EvidenceSchema


class ChallengeType(Enum):
    """Types of challenges that can be identified."""
    UNSUPPORTED_CLAIM = "unsupported_claim"
    WEAK_CITATION = "weak_citation"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BIASED_SOURCE = "biased_source"
    OUTDATED_INFORMATION = "outdated_information"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"


@dataclass
class Challenge:
    """Represents a specific challenge identified by the challenger agent."""
    
    challenge_type: ChallengeType
    description: str
    severity: float  # 0.0 to 1.0, where 1.0 is most severe
    affected_claims: List[str]  # Specific claims or spans affected
    suggested_improvement: str
    evidence_against: Optional[str] = None
    source_quality_issues: Optional[List[str]] = None


@dataclass
class ChallengeReport:
    """Complete challenge report from the challenger agent."""
    
    challenger_id: str
    original_response: AgentResponse
    challenges: List[Challenge]
    overall_assessment: str
    confidence_in_challenges: float  # 0.0 to 1.0
    requires_revision: bool
    priority_challenges: List[Challenge]  # Top priority challenges to address
    token_usage: int
    processing_time: float


class ChallengerAgent:
    """
    Specialized adversarial review agent for identifying unsupported claims.
    
    The challenger agent analyzes responses from answering agents to identify
    weak citations, unsupported claims, and contradictory evidence.
    """
    
    def __init__(
        self,
        challenger_id: str,
        min_challenge_severity: float = 0.3,
        max_challenges_per_response: int = 10,
        citation_quality_threshold: float = 0.6
    ):
        """Initialize the challenger agent."""
        self.challenger_id = challenger_id
        self.min_challenge_severity = min_challenge_severity
        self.max_challenges_per_response = max_challenges_per_response
        self.citation_quality_threshold = citation_quality_threshold
        
        # Challenge statistics
        self.total_challenges = 0
        self.total_reviews = 0
        self.total_token_usage = 0
        
        logger.info(f"ChallengerAgent {challenger_id} initialized")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def challenge_response(
        self, 
        response: AgentResponse,
        original_claim: str,
        additional_context: Optional[str] = None
    ) -> ChallengeReport:
        """
        Conduct adversarial review of an agent's response.
        
        Args:
            response: The agent response to challenge
            original_claim: Original claim being evaluated
            additional_context: Additional context for evaluation
            
        Returns:
            ChallengeReport with identified issues and suggestions
        """
        import time
        start_time = time.time()
        
        logger.info(
            f"Challenger {self.challenger_id} reviewing response from {response.agent_id}"
        )
        
        try:
            challenges = []
            
            # Challenge 1: Analyze unsupported claims
            unsupported_challenges = await self._identify_unsupported_claims(
                response, original_claim
            )
            challenges.extend(unsupported_challenges)
            
            # Challenge 2: Evaluate citation quality
            citation_challenges = await self._evaluate_citations(response)
            challenges.extend(citation_challenges)
            
            # Challenge 3: Check for contradictory evidence
            contradiction_challenges = await self._identify_contradictions(response)
            challenges.extend(contradiction_challenges)
            
            # Challenge 4: Assess evidence sufficiency
            sufficiency_challenges = await self._assess_evidence_sufficiency(
                response, original_claim
            )
            challenges.extend(sufficiency_challenges)
            
            # Challenge 5: Check source quality and bias
            source_challenges = await self._evaluate_source_quality(response)
            challenges.extend(source_challenges)
            
            # Filter and prioritize challenges
            significant_challenges = [
                c for c in challenges 
                if c.severity >= self.min_challenge_severity
            ]
            
            significant_challenges.sort(key=lambda x: x.severity, reverse=True)
            significant_challenges = significant_challenges[:self.max_challenges_per_response]
            
            # Determine if revision is required
            requires_revision = self._determine_revision_requirement(significant_challenges)
            
            # Generate overall assessment
            overall_assessment = await self._generate_overall_assessment(
                response, significant_challenges, original_claim
            )
            
            # Calculate confidence in challenges
            challenge_confidence = self._calculate_challenge_confidence(significant_challenges)
            
            # Identify priority challenges
            priority_challenges = self._identify_priority_challenges(significant_challenges)
            
            # Estimate token usage
            token_usage = self._estimate_token_usage(response, significant_challenges)
            self.total_token_usage += token_usage
            self.total_challenges += len(significant_challenges)
            self.total_reviews += 1
            
            processing_time = time.time() - start_time
            
            challenge_report = ChallengeReport(
                challenger_id=self.challenger_id,
                original_response=response,
                challenges=significant_challenges,
                overall_assessment=overall_assessment,
                confidence_in_challenges=challenge_confidence,
                requires_revision=requires_revision,
                priority_challenges=priority_challenges,
                token_usage=token_usage,
                processing_time=processing_time
            )
            
            logger.success(
                f"Challenger {self.challenger_id} completed review in {processing_time:.2f}s "
                f"({len(significant_challenges)} challenges, revision: {requires_revision})"
            )
            
            return challenge_report
            
        except Exception as e:
            logger.error(f"Challenger {self.challenger_id} review failed: {str(e)}")
            raise
    
    async def _identify_unsupported_claims(
        self, 
        response: AgentResponse, 
        original_claim: str
    ) -> List[Challenge]:
        """Identify claims in the response that lack adequate support."""
        challenges = []
        
        # Extract specific claims from the answer
        answer_sentences = response.answer.split('.')
        
        for sentence in answer_sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence makes a factual claim
            if self._is_factual_claim(sentence):
                # Check if this claim has supporting evidence
                has_support = self._has_supporting_evidence(sentence, response.evidence)
                
                if not has_support:
                    severity = self._calculate_unsupported_severity(sentence, response)
                    
                    challenges.append(Challenge(
                        challenge_type=ChallengeType.UNSUPPORTED_CLAIM,
                        description=f"Claim lacks adequate supporting evidence: '{sentence[:100]}...'",
                        severity=severity,
                        affected_claims=[sentence],
                        suggested_improvement=(
                            "Provide specific evidence or citations to support this claim, "
                            "or qualify the statement with appropriate uncertainty."
                        )
                    ))
        
        return challenges
    
    async def _evaluate_citations(self, response: AgentResponse) -> List[Challenge]:
        """Evaluate the quality and appropriateness of citations."""
        challenges = []
        
        for i, citation in enumerate(response.citations):
            citation_issues = []
            
            # Check citation format
            if not citation.formatted_citation or len(citation.formatted_citation) < 20:
                citation_issues.append("Improperly formatted citation")
            
            # Check URL accessibility (simulated)
            if not citation.url or not citation.url.startswith(('http://', 'https://')):
                citation_issues.append("Invalid or missing URL")
            
            # Check source authority
            if citation.source_type in ['blog', 'social_media', 'unknown']:
                citation_issues.append("Low-authority source type")
            
            # Check relevance to claim
            relevance_score = self._assess_citation_relevance(citation, response)
            if relevance_score < self.citation_quality_threshold:
                citation_issues.append(f"Low relevance to claim (score: {relevance_score:.2f})")
            
            if citation_issues:
                severity = min(0.9, len(citation_issues) * 0.3)  # Max severity 0.9
                
                challenges.append(Challenge(
                    challenge_type=ChallengeType.WEAK_CITATION,
                    description=f"Citation {i+1} has quality issues: {', '.join(citation_issues)}",
                    severity=severity,
                    affected_claims=[citation.url],
                    suggested_improvement=(
                        "Replace with higher-quality sources from academic journals, "
                        "government agencies, or reputable news organizations."
                    ),
                    source_quality_issues=citation_issues
                ))
        
        return challenges
    
    async def _identify_contradictions(self, response: AgentResponse) -> List[Challenge]:
        """Identify contradictory evidence within the response."""
        challenges = []
        
        # Group evidence by support/contradiction
        supporting = [e for e in response.evidence if e.supports_claim]
        contradicting = [e for e in response.evidence if not e.supports_claim]
        
        # If both exist, identify specific contradictions
        if supporting and contradicting:
            for support_ev in supporting:
                for contra_ev in contradicting:
                    contradiction_strength = self._assess_contradiction_strength(
                        support_ev, contra_ev
                    )
                    
                    if contradiction_strength > 0.5:
                        challenges.append(Challenge(
                            challenge_type=ChallengeType.CONTRADICTORY_EVIDENCE,
                            description=(
                                f"Strong contradiction between evidence sources: "
                                f"'{support_ev.evidence_text[:100]}...' vs "
                                f"'{contra_ev.evidence_text[:100]}...'"
                            ),
                            severity=contradiction_strength,
                            affected_claims=[support_ev.source_url, contra_ev.source_url],
                            suggested_improvement=(
                                "Resolve the contradiction by finding additional sources, "
                                "or acknowledge the conflicting evidence explicitly."
                            ),
                            evidence_against=contra_ev.evidence_text
                        ))
        
        return challenges
    
    async def _assess_evidence_sufficiency(
        self, 
        response: AgentResponse, 
        original_claim: str
    ) -> List[Challenge]:
        """Assess whether evidence is sufficient to support the claim."""
        challenges = []
        
        # Calculate evidence metrics
        total_evidence = len(response.evidence)
        avg_quality = (
            sum(e.quality_score for e in response.evidence) / total_evidence
            if total_evidence > 0 else 0.0
        )
        avg_relevance = (
            sum(e.relevance_score for e in response.evidence) / total_evidence
            if total_evidence > 0 else 0.0
        )
        
        # Check for insufficient evidence
        if total_evidence < 3:
            challenges.append(Challenge(
                challenge_type=ChallengeType.INSUFFICIENT_EVIDENCE,
                description=f"Only {total_evidence} pieces of evidence provided (minimum: 3)",
                severity=0.8,
                affected_claims=[original_claim],
                suggested_improvement="Gather additional evidence from diverse sources."
            ))
        
        # Check evidence quality
        if avg_quality < 0.6:
            challenges.append(Challenge(
                challenge_type=ChallengeType.INSUFFICIENT_EVIDENCE,
                description=f"Low average evidence quality: {avg_quality:.2f}/1.0",
                severity=0.7,
                affected_claims=[original_claim],
                suggested_improvement="Seek higher-quality sources with better evidence."
            ))
        
        # Check evidence relevance
        if avg_relevance < 0.7:
            challenges.append(Challenge(
                challenge_type=ChallengeType.INSUFFICIENT_EVIDENCE,
                description=f"Low average evidence relevance: {avg_relevance:.2f}/1.0",
                severity=0.6,
                affected_claims=[original_claim],
                suggested_improvement="Focus on more directly relevant evidence."
            ))
        
        return challenges
    
    async def _evaluate_source_quality(self, response: AgentResponse) -> List[Challenge]:
        """Evaluate the quality and bias of sources used."""
        challenges = []
        
        source_types = [c.source_type for c in response.citations if c.source_type]
        low_quality_types = ['blog', 'social_media', 'unknown', 'personal_website']
        
        low_quality_count = sum(1 for st in source_types if st in low_quality_types)
        total_sources = len(source_types)
        
        if total_sources > 0 and (low_quality_count / total_sources) > 0.5:
            challenges.append(Challenge(
                challenge_type=ChallengeType.BIASED_SOURCE,
                description=(
                    f"{low_quality_count}/{total_sources} sources are from "
                    f"low-authority types: {low_quality_types}"
                ),
                severity=0.7,
                affected_claims=[c.url for c in response.citations if c.source_type in low_quality_types],
                suggested_improvement=(
                    "Replace low-authority sources with peer-reviewed articles, "
                    "government publications, or established news organizations."
                )
            ))
        
        return challenges
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if a sentence contains a factual claim."""
        factual_indicators = [
            'is', 'are', 'was', 'were', 'has', 'have', 'will', 'can', 'cannot',
            'studies show', 'research indicates', 'evidence suggests', 'data shows'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)
    
    def _has_supporting_evidence(self, claim: str, evidence: List[EvidenceSchema]) -> bool:
        """Check if a claim has supporting evidence."""
        claim_words = set(claim.lower().split())
        
        for ev in evidence:
            if ev.supports_claim:
                evidence_words = set(ev.evidence_text.lower().split())
                overlap = len(claim_words.intersection(evidence_words))
                if overlap >= 2:  # Minimum word overlap
                    return True
        
        return False
    
    def _calculate_unsupported_severity(self, sentence: str, response: AgentResponse) -> float:
        """Calculate severity of an unsupported claim."""
        base_severity = 0.5
        
        # Increase severity for definitive statements
        if any(word in sentence.lower() for word in ['definitely', 'certainly', 'always', 'never']):
            base_severity += 0.2
        
        # Increase severity if overall confidence is high but this claim is unsupported
        if response.confidence_score > 0.8:
            base_severity += 0.2
        
        return min(1.0, base_severity)
    
    def _assess_citation_relevance(self, citation: CitationSchema, response: AgentResponse) -> float:
        """Assess how relevant a citation is to the response."""
        # Simple relevance based on word overlap
        citation_words = set(citation.title.lower().split() + citation.description.lower().split())
        claim_words = set(response.claim.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = len(citation_words.intersection(claim_words))
        relevance = overlap / len(claim_words)
        
        return min(1.0, relevance)
    
    def _assess_contradiction_strength(
        self, 
        evidence1: EvidenceSchema, 
        evidence2: EvidenceSchema
    ) -> float:
        """Assess the strength of contradiction between two pieces of evidence."""
        
        # If one supports and the other doesn't, there's potential contradiction
        if evidence1.supports_claim == evidence2.supports_claim:
            return 0.0
        
        # Check for opposing keywords
        opposing_pairs = [
            (['true', 'correct', 'accurate'], ['false', 'incorrect', 'inaccurate']),
            (['increase', 'rise', 'higher'], ['decrease', 'fall', 'lower']),
            (['effective', 'works'], ['ineffective', 'fails']),
            (['safe'], ['dangerous', 'harmful'])
        ]
        
        text1 = evidence1.evidence_text.lower()
        text2 = evidence2.evidence_text.lower()
        
        contradiction_score = 0.0
        
        for positive_words, negative_words in opposing_pairs:
            has_positive_1 = any(word in text1 for word in positive_words)
            has_negative_1 = any(word in text1 for word in negative_words)
            has_positive_2 = any(word in text2 for word in positive_words)
            has_negative_2 = any(word in text2 for word in negative_words)
            
            if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                contradiction_score += 0.25
        
        return min(1.0, contradiction_score)
    
    def _determine_revision_requirement(self, challenges: List[Challenge]) -> bool:
        """Determine if the response requires revision based on challenges."""
        if not challenges:
            return False
        
        # High severity challenges require revision
        critical_challenges = [c for c in challenges if c.severity >= 0.8]
        if critical_challenges:
            return True
        
        # Multiple moderate challenges require revision
        moderate_challenges = [c for c in challenges if c.severity >= 0.6]
        if len(moderate_challenges) >= 3:
            return True
        
        return False
    
    async def _generate_overall_assessment(
        self,
        response: AgentResponse,
        challenges: List[Challenge],
        original_claim: str
    ) -> str:
        """Generate overall assessment of the response."""
        
        if not challenges:
            return f"Response from {response.agent_id} appears well-supported with no significant issues identified."
        
        challenge_types = list(set(c.challenge_type for c in challenges))
        avg_severity = sum(c.severity for c in challenges) / len(challenges)
        
        assessment_parts = [
            f"Response from {response.agent_id} has {len(challenges)} significant issues identified.",
            f"Challenge types: {', '.join([ct.value for ct in challenge_types])}",
            f"Average challenge severity: {avg_severity:.2f}/1.0"
        ]
        
        if avg_severity >= 0.7:
            assessment_parts.append("Major revision recommended before acceptance.")
        elif avg_severity >= 0.5:
            assessment_parts.append("Moderate revision suggested to improve quality.")
        else:
            assessment_parts.append("Minor improvements could enhance the response.")
        
        return " ".join(assessment_parts)
    
    def _calculate_challenge_confidence(self, challenges: List[Challenge]) -> float:
        """Calculate confidence in the identified challenges."""
        if not challenges:
            return 1.0  # High confidence in finding no issues
        
        # Confidence based on challenge severity and consistency
        avg_severity = sum(c.severity for c in challenges) / len(challenges)
        challenge_consistency = len(set(c.challenge_type for c in challenges)) / len(challenges)
        
        # Higher severity and more diverse challenges increase confidence
        confidence = (avg_severity + challenge_consistency) / 2
        
        return min(1.0, max(0.0, confidence))
    
    def _identify_priority_challenges(self, challenges: List[Challenge]) -> List[Challenge]:
        """Identify the most critical challenges to address first."""
        priority_threshold = 0.7
        priority_challenges = [c for c in challenges if c.severity >= priority_threshold]
        
        # Sort by severity, limit to top 5
        priority_challenges.sort(key=lambda x: x.severity, reverse=True)
        return priority_challenges[:5]
    
    def _estimate_token_usage(
        self, 
        response: AgentResponse, 
        challenges: List[Challenge]
    ) -> int:
        """Estimate token usage for the challenge process."""
        
        # Estimate based on response analysis and challenge generation
        base_tokens = len(response.answer) // 4  # Rough character-to-token ratio
        challenge_tokens = sum(len(c.description) + len(c.suggested_improvement) for c in challenges) // 4
        
        total_tokens = base_tokens + challenge_tokens + 500  # Base processing overhead
        return total_tokens
    
    def get_challenger_stats(self) -> Dict[str, Any]:
        """Get challenger performance statistics."""
        return {
            "challenger_id": self.challenger_id,
            "total_reviews": self.total_reviews,
            "total_challenges": self.total_challenges,
            "avg_challenges_per_review": (
                self.total_challenges / self.total_reviews 
                if self.total_reviews > 0 else 0
            ),
            "total_token_usage": self.total_token_usage,
            "min_challenge_severity": self.min_challenge_severity,
            "citation_quality_threshold": self.citation_quality_threshold
        }
    
    async def reset_challenger(self):
        """Reset challenger state for new session."""
        self.total_challenges = 0
        self.total_reviews = 0
        self.total_token_usage = 0
        logger.info(f"ChallengerAgent {self.challenger_id} reset")