"""AnsweringAgent implementation for independent research and claim answering."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..research.web_search_api import WebSearchAPI, SearchResult
from ..research.citation_formatter import CitationFormatter
from ..research.evidence_extractor import EvidenceExtractor
from ..schemas.citation_schemas import CitationSchema, EvidenceSchema


@dataclass
class AgentResponse:
    """Standardized response from an answering agent."""
    
    agent_id: str
    claim: str
    answer: str
    citations: List[CitationSchema]
    evidence: List[EvidenceSchema]
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    token_usage: int
    processing_time: float


class AnsweringAgent:
    """
    Independent research agent with web search capabilities.
    
    Each agent operates independently, conducting web searches and providing
    comprehensive answers with proper citations and evidence.
    """
    
    def __init__(
        self,
        agent_id: str,
        max_search_results: int = 10,
        max_tokens_per_response: int = 2000,
        confidence_threshold: float = 0.7
    ):
        """Initialize the answering agent."""
        self.agent_id = agent_id
        self.max_search_results = max_search_results
        self.max_tokens_per_response = max_tokens_per_response
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.web_search = WebSearchAPI()
        self.citation_formatter = CitationFormatter()
        self.evidence_extractor = EvidenceExtractor()
        
        # Agent state
        self.total_token_usage = 0
        self.response_count = 0
        
        logger.info(f"AnsweringAgent {agent_id} initialized")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def research_claim(
        self,
        claim: str,
        domain: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> AgentResponse:
        """
        Research a factual claim and provide comprehensive answer with citations.
        
        Args:
            claim: The factual claim to research
            domain: Domain context (science, health, history, finance)
            additional_context: Any additional context to guide research
            
        Returns:
            AgentResponse with answer, citations, and evidence
        """
        import time
        start_time = time.time()
        
        logger.info(f"Agent {self.agent_id} researching claim: {claim[:100]}...")
        
        try:
            # Step 1: Conduct web searches
            search_results = await self._conduct_web_search(claim, domain)
            
            # Step 2: Extract and evaluate evidence
            evidence = await self._extract_evidence(claim, search_results)
            
            # Step 3: Generate citations
            citations = await self._generate_citations(search_results, evidence)
            
            # Step 4: Formulate answer with reasoning
            answer, reasoning, confidence = await self._formulate_answer(
                claim, evidence, citations, additional_context
            )
            
            # Step 5: Calculate token usage (estimated)
            token_usage = self._estimate_token_usage(claim, answer, citations, evidence)
            self.total_token_usage += token_usage
            self.response_count += 1
            
            processing_time = time.time() - start_time
            
            response = AgentResponse(
                agent_id=self.agent_id,
                claim=claim,
                answer=answer,
                citations=citations,
                evidence=evidence,
                confidence_score=confidence,
                reasoning=reasoning,
                token_usage=token_usage,
                processing_time=processing_time
            )
            
            logger.success(
                f"Agent {self.agent_id} completed research in {processing_time:.2f}s "
                f"(confidence: {confidence:.3f})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} research failed: {str(e)}")
            raise
    
    async def _conduct_web_search(
        self, 
        claim: str, 
        domain: Optional[str] = None
    ) -> List[SearchResult]:
        """Conduct comprehensive web search for the claim."""
        search_queries = self._generate_search_queries(claim, domain)
        all_results = []
        
        for query in search_queries[:3]:  # Limit to 3 queries per claim
            try:
                results = await self.web_search.search(
                    query=query,
                    max_results=self.max_search_results // len(search_queries)
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search query '{query}' failed: {str(e)}")
                continue
        
        # Remove duplicates and limit results
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result.url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result.url)
                if len(unique_results) >= self.max_search_results:
                    break
        
        logger.info(f"Found {len(unique_results)} unique search results")
        return unique_results
    
    def _generate_search_queries(self, claim: str, domain: Optional[str] = None) -> List[str]:
        """Generate diverse search queries for comprehensive research."""
        queries = [claim]  # Original claim as base query
        
        # Add domain-specific queries
        if domain:
            queries.append(f"{claim} {domain} research")
            queries.append(f"{domain} {claim} evidence")
        
        # Add verification queries
        queries.append(f"is {claim} true")
        queries.append(f"{claim} fact check")
        queries.append(f"{claim} scientific evidence")
        
        return queries[:5]  # Limit to 5 queries maximum
    
    async def _extract_evidence(
        self, 
        claim: str, 
        search_results: List[SearchResult]
    ) -> List[EvidenceSchema]:
        """Extract and evaluate evidence from search results."""
        evidence_list = []
        
        for result in search_results:
            try:
                evidence = await self.evidence_extractor.extract_evidence(
                    claim=claim,
                    source_url=result.url,
                    source_title=result.title,
                    source_content=result.snippet
                )
                if evidence and evidence.relevance_score >= 0.3:  # Minimum relevance
                    evidence_list.append(evidence)
            except Exception as e:
                logger.warning(f"Evidence extraction failed for {result.url}: {str(e)}")
                continue
        
        # Sort by relevance and quality
        evidence_list.sort(
            key=lambda x: (x.relevance_score * x.quality_score), 
            reverse=True
        )
        
        return evidence_list[:10]  # Top 10 pieces of evidence
    
    async def _generate_citations(
        self,
        search_results: List[SearchResult],
        evidence: List[EvidenceSchema]
    ) -> List[CitationSchema]:
        """Generate properly formatted citations."""
        citations = []
        
        # Create citations for evidence sources
        evidence_urls = {ev.source_url for ev in evidence}
        
        for result in search_results:
            if result.url in evidence_urls:
                try:
                    citation = await self.citation_formatter.format_citation(
                        url=result.url,
                        title=result.title,
                        description=result.snippet
                    )
                    if citation:
                        citations.append(citation)
                except Exception as e:
                    logger.warning(f"Citation formatting failed for {result.url}: {str(e)}")
                    continue
        
        return citations[:8]  # Limit to 8 citations
    
    async def _formulate_answer(
        self,
        claim: str,
        evidence: List[EvidenceSchema],
        citations: List[CitationSchema],
        additional_context: Optional[str] = None
    ) -> tuple[str, str, float]:
        """Formulate comprehensive answer with reasoning and confidence score."""
        
        # Analyze evidence strength
        supporting_evidence = [e for e in evidence if e.supports_claim]
        contradicting_evidence = [e for e in evidence if not e.supports_claim]
        
        avg_quality = sum(e.quality_score for e in evidence) / len(evidence) if evidence else 0.0
        avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence) if evidence else 0.0
        
        # Calculate confidence based on evidence
        confidence = self._calculate_confidence(
            supporting_evidence, contradicting_evidence, avg_quality, avg_relevance
        )
        
        # Construct answer
        answer_parts = []
        
        if len(supporting_evidence) > len(contradicting_evidence):
            answer_parts.append(f"The claim '{claim}' appears to be SUPPORTED by available evidence.")
        elif len(contradicting_evidence) > len(supporting_evidence):
            answer_parts.append(f"The claim '{claim}' appears to be CONTRADICTED by available evidence.")
        else:
            answer_parts.append(f"The evidence for the claim '{claim}' is MIXED or INCONCLUSIVE.")
        
        # Add evidence summary
        if supporting_evidence:
            answer_parts.append(f"\nSupporting evidence ({len(supporting_evidence)} sources):")
            for i, ev in enumerate(supporting_evidence[:3], 1):  # Top 3
                answer_parts.append(f"{i}. {ev.evidence_text[:200]}...")
        
        if contradicting_evidence:
            answer_parts.append(f"\nContradicting evidence ({len(contradicting_evidence)} sources):")
            for i, ev in enumerate(contradicting_evidence[:3], 1):  # Top 3
                answer_parts.append(f"{i}. {ev.evidence_text[:200]}...")
        
        # Add citation references
        if citations:
            answer_parts.append(f"\nSources: {', '.join([f'[{i+1}]' for i in range(len(citations))])}")
        
        answer = "\n".join(answer_parts)
        
        # Construct reasoning
        reasoning_parts = [
            f"Analysis based on {len(evidence)} pieces of evidence from {len(citations)} sources.",
            f"Evidence quality: {avg_quality:.2f}/1.0, relevance: {avg_relevance:.2f}/1.0",
            f"Supporting: {len(supporting_evidence)}, contradicting: {len(contradicting_evidence)}",
            f"Final confidence: {confidence:.3f}/1.0"
        ]
        
        if additional_context:
            reasoning_parts.append(f"Additional context considered: {additional_context}")
        
        reasoning = " ".join(reasoning_parts)
        
        return answer, reasoning, confidence
    
    def _calculate_confidence(
        self,
        supporting: List[EvidenceSchema],
        contradicting: List[EvidenceSchema],
        avg_quality: float,
        avg_relevance: float
    ) -> float:
        """Calculate confidence score based on evidence analysis."""
        
        if not supporting and not contradicting:
            return 0.1  # Very low confidence with no evidence
        
        # Base confidence from evidence balance
        total_evidence = len(supporting) + len(contradicting)
        evidence_ratio = abs(len(supporting) - len(contradicting)) / total_evidence
        
        base_confidence = 0.5 + (evidence_ratio * 0.3)  # 0.5 to 0.8 range
        
        # Adjust for quality and relevance
        quality_factor = avg_quality  # 0.0 to 1.0
        relevance_factor = avg_relevance  # 0.0 to 1.0
        
        final_confidence = base_confidence * quality_factor * relevance_factor
        
        # Ensure within valid range
        return max(0.0, min(1.0, final_confidence))
    
    def _estimate_token_usage(
        self,
        claim: str,
        answer: str,
        citations: List[CitationSchema],
        evidence: List[EvidenceSchema]
    ) -> int:
        """Estimate token usage for this response."""
        
        # Rough estimation: ~4 characters per token
        total_chars = (
            len(claim) +
            len(answer) +
            sum(len(c.formatted_citation) for c in citations) +
            sum(len(e.evidence_text) for e in evidence)
        )
        
        estimated_tokens = total_chars // 4
        return min(estimated_tokens, self.max_tokens_per_response)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "agent_id": self.agent_id,
            "total_token_usage": self.total_token_usage,
            "response_count": self.response_count,
            "avg_tokens_per_response": (
                self.total_token_usage / self.response_count 
                if self.response_count > 0 else 0
            ),
            "max_tokens_per_response": self.max_tokens_per_response,
            "confidence_threshold": self.confidence_threshold
        }
    
    async def reset_agent(self):
        """Reset agent state for new research session."""
        self.total_token_usage = 0
        self.response_count = 0
        logger.info(f"AnsweringAgent {self.agent_id} reset")