"""Specialized prompts for answering agents in the Cite-and-Challenge protocol."""

from typing import Dict, List, Optional


class AnsweringPrompts:
    """Specialized prompts for independent answering agents."""
    
    @staticmethod
    def get_research_prompt(
        claim: str, 
        domain: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """Get main research prompt for answering agents."""
        
        base_prompt = f"""
You are an expert research agent tasked with evaluating the following factual claim:

CLAIM: "{claim}"
"""
        
        if domain:
            base_prompt += f"""
DOMAIN: {domain.title()}
"""
        
        if additional_context:
            base_prompt += f"""
ADDITIONAL CONTEXT: {additional_context}
"""
        
        base_prompt += """

YOUR TASK:
1. Research this claim thoroughly using web searches
2. Gather diverse, high-quality evidence from reliable sources
3. Evaluate the claim based on the evidence you find
4. Provide a comprehensive answer with proper citations
5. Assess your confidence in your conclusion

RESEARCH GUIDELINES:
- Use multiple search queries to find comprehensive evidence
- Prioritize authoritative sources (academic papers, government agencies, reputable news)
- Look for both supporting and contradicting evidence
- Verify information across multiple independent sources
- Consider the recency and reliability of your sources

OUTPUT FORMAT:
Provide your response with:
1. CLEAR CONCLUSION: State whether the claim is SUPPORTED, CONTRADICTED, or MIXED based on evidence
2. EVIDENCE SUMMARY: Summarize key supporting and contradicting evidence
3. SOURCE CITATIONS: Properly formatted citations for all sources used
4. CONFIDENCE ASSESSMENT: Your confidence level (0.0 to 1.0) with justification
5. REASONING: Explain your analytical process and key factors in your conclusion

QUALITY STANDARDS:
- Minimum 3 high-quality sources
- Balanced consideration of multiple perspectives
- Specific evidence rather than general statements
- Transparency about limitations and uncertainties
- Professional, objective tone throughout

Begin your research and provide your comprehensive analysis.
"""
        
        return base_prompt
    
    @staticmethod
    def get_search_query_prompt(claim: str, domain: Optional[str] = None) -> str:
        """Get prompt for generating search queries."""
        
        prompt = f"""
Generate 3-5 diverse search queries to thoroughly research this claim:

CLAIM: "{claim}"
"""
        
        if domain:
            prompt += f"DOMAIN: {domain}\n"
        
        prompt += """

QUERY GENERATION GUIDELINES:
1. Create queries that approach the claim from different angles
2. Include both direct and indirect verification approaches
3. Consider potential counter-arguments or alternative perspectives
4. Use specific terminology and keywords relevant to the domain
5. Balance broad context with specific details

QUERY TYPES TO INCLUDE:
- Direct factual verification
- Academic/scientific research
- Historical context or precedent
- Expert opinions or analysis
- Statistical or empirical data

Provide each query on a separate line, numbered 1-5.
"""
        
        return prompt
    
    @staticmethod
    def get_evidence_evaluation_prompt(
        claim: str,
        search_results: List[str],
        source_info: List[str]
    ) -> str:
        """Get prompt for evaluating evidence from search results."""
        
        prompt = f"""
Evaluate the following search results for relevance to this claim:

CLAIM: "{claim}"

SEARCH RESULTS:
"""
        
        for i, (result, info) in enumerate(zip(search_results, source_info), 1):
            prompt += f"""
{i}. SOURCE: {info}
   CONTENT: {result}
   ---
"""
        
        prompt += """

EVALUATION CRITERIA:
1. RELEVANCE: How directly does this evidence relate to the claim?
2. QUALITY: How reliable and authoritative is the source?
3. SUPPORT: Does this evidence support or contradict the claim?
4. SPECIFICITY: How specific and detailed is the evidence?
5. RECENCY: How current is this information?

For each piece of evidence, provide:
- Relevance score (0.0 to 1.0)
- Quality score (0.0 to 1.0)  
- Support determination (SUPPORTS/CONTRADICTS/NEUTRAL)
- Brief explanation of your evaluation

Then summarize your overall assessment of the evidence set.
"""
        
        return prompt
    
    @staticmethod
    def get_confidence_assessment_prompt(
        claim: str,
        supporting_evidence: List[str],
        contradicting_evidence: List[str],
        evidence_quality: float
    ) -> str:
        """Get prompt for confidence assessment."""
        
        prompt = f"""
Assess your confidence in your conclusion about this claim:

CLAIM: "{claim}"

EVIDENCE SUMMARY:
- Supporting evidence pieces: {len(supporting_evidence)}
- Contradicting evidence pieces: {len(contradicting_evidence)}
- Average evidence quality: {evidence_quality:.2f}/1.0

SUPPORTING EVIDENCE:
"""
        
        for i, evidence in enumerate(supporting_evidence, 1):
            prompt += f"{i}. {evidence[:200]}...\n"
        
        if contradicting_evidence:
            prompt += "\nCONTRADICTING EVIDENCE:\n"
            for i, evidence in enumerate(contradicting_evidence, 1):
                prompt += f"{i}. {evidence[:200]}...\n"
        
        prompt += """

CONFIDENCE FACTORS TO CONSIDER:
1. Quantity and quality of supporting evidence
2. Presence and strength of contradicting evidence
3. Authority and reliability of sources
4. Consistency across multiple sources
5. Specificity and detail of evidence
6. Potential for bias or incomplete information
7. Your domain expertise limitations

Provide:
1. Final confidence score (0.0 to 1.0)
2. Detailed justification for this confidence level
3. Key factors that increase or decrease your confidence
4. Acknowledgment of any significant uncertainties

Be honest about limitations and avoid overconfidence.
"""
        
        return prompt
    
    @staticmethod
    def get_citation_formatting_prompt(sources: List[str]) -> str:
        """Get prompt for proper citation formatting."""
        
        prompt = """
Format the following sources into proper APA-style citations:

SOURCES TO CITE:
"""
        
        for i, source in enumerate(sources, 1):
            prompt += f"{i}. {source}\n"
        
        prompt += """

CITATION REQUIREMENTS:
1. Use proper APA format for each source type
2. Include all available information (author, date, title, URL, etc.)
3. Ensure URLs are accessible and properly formatted
4. Maintain consistent formatting across all citations
5. Order citations logically (by relevance or alphabetically)

EXAMPLE FORMATS:
- Journal article: Author, A. A. (Year). Title of article. Title of Journal, Volume(Issue), pages. DOI or URL
- Website: Author/Organization. (Year). Title. Website Name. URL
- News article: Author, A. A. (Year, Month Day). Article title. Newspaper Name. URL

Provide each citation on a separate line, numbered for reference.
"""
        
        return prompt
    
    @staticmethod
    def get_response_synthesis_prompt(
        claim: str,
        evidence_analysis: str,
        citations: List[str],
        confidence_score: float
    ) -> str:
        """Get prompt for synthesizing final response."""
        
        prompt = f"""
Synthesize your research into a comprehensive final response:

CLAIM: "{claim}"

YOUR EVIDENCE ANALYSIS:
{evidence_analysis}

YOUR CITATIONS:
"""
        
        for i, citation in enumerate(citations, 1):
            prompt += f"[{i}] {citation}\n"
        
        prompt += f"""
YOUR CONFIDENCE SCORE: {confidence_score:.3f}

SYNTHESIS REQUIREMENTS:
1. Start with a clear, definitive conclusion about the claim
2. Summarize the most compelling evidence for your conclusion
3. Acknowledge any significant contradicting evidence
4. Reference your citations appropriately throughout
5. Explain your reasoning and analytical approach
6. Be transparent about limitations and uncertainties
7. Maintain an objective, professional tone

STRUCTURE YOUR RESPONSE:
1. CONCLUSION: Clear statement of whether claim is supported/contradicted/mixed
2. KEY EVIDENCE: Most important supporting points with citations
3. CONSIDERATIONS: Any contradicting evidence or limitations
4. CONFIDENCE: Your confidence level with justification
5. REASONING: Brief explanation of your analytical approach

Aim for clarity, objectivity, and appropriate nuance in your final response.
"""
        
        return prompt
    
    @staticmethod
    def get_domain_specific_prompts(domain: str) -> Dict[str, str]:
        """Get domain-specific prompt modifications."""
        
        domain_prompts = {
            "science": {
                "focus": "Prioritize peer-reviewed research, scientific studies, and expert consensus. Look for replication, methodology quality, and statistical significance.",
                "sources": "Academic journals, research institutions, scientific organizations, government science agencies",
                "keywords": "study, research, peer-reviewed, methodology, statistics, replication, scientific consensus"
            },
            
            "health": {
                "focus": "Emphasize clinical evidence, medical consensus, and regulatory guidance. Be cautious of preliminary or single studies.",
                "sources": "Medical journals, health organizations (WHO, CDC), clinical trials, medical institutions",
                "keywords": "clinical trial, medical study, health organization, FDA approval, medical consensus, treatment efficacy"
            },
            
            "history": {
                "focus": "Seek primary sources, scholarly consensus, and multiple historical accounts. Consider historical context and interpretation.",
                "sources": "Academic historians, historical societies, archived documents, scholarly books, museum sources",
                "keywords": "historical record, primary source, scholarly consensus, historical analysis, archival evidence"
            },
            
            "finance": {
                "focus": "Look for official data, regulatory information, and expert analysis. Consider market context and timing.",
                "sources": "Financial institutions, regulatory bodies (SEC, Fed), economic data agencies, financial news",
                "keywords": "financial data, regulatory filing, economic indicator, market analysis, official statistics"
            }
        }
        
        return domain_prompts.get(domain, {
            "focus": "Seek authoritative sources and expert consensus in the relevant field.",
            "sources": "Authoritative organizations, expert analysis, official data sources",
            "keywords": "expert analysis, authoritative source, official data, professional consensus"
        })
    
    @staticmethod
    def get_quality_check_prompt(response_draft: str) -> str:
        """Get prompt for quality checking response before submission."""
        
        return f"""
Review your draft response for quality and completeness:

YOUR DRAFT RESPONSE:
{response_draft}

QUALITY CHECKLIST:
1. CLARITY: Is your conclusion clear and unambiguous?
2. EVIDENCE: Have you included sufficient supporting evidence?
3. CITATIONS: Are all sources properly cited and accessible?
4. BALANCE: Have you fairly considered contradicting evidence?
5. CONFIDENCE: Is your confidence level appropriately calibrated?
6. REASONING: Have you explained your analytical process clearly?
7. OBJECTIVITY: Is your tone professional and unbiased?
8. COMPLETENESS: Have you addressed all aspects of the claim?

IMPROVEMENT AREAS TO CHECK:
- Add more specific evidence if claims are too general
- Include additional citations if sources are insufficient
- Adjust confidence if it seems over/under-calibrated
- Clarify reasoning if logic is unclear
- Balance perspective if analysis seems one-sided

Provide:
1. Overall quality assessment (1-10 scale)
2. Top 3 strengths of your response
3. Top 3 areas for potential improvement
4. Final recommendation (SUBMIT AS IS / REVISE SPECIFIC AREAS / MAJOR REVISION NEEDED)

Be self-critical and objective in your assessment.
"""