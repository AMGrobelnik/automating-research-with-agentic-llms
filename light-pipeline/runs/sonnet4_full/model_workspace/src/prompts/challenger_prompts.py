"""Specialized adversarial prompts for challenger agents in the Cite-and-Challenge protocol."""

from typing import Dict, List, Optional, Any


class ChallengerPrompts:
    """Specialized prompts for adversarial challenger agents."""
    
    @staticmethod
    def get_main_challenge_prompt(
        agent_response: str,
        original_claim: str,
        agent_id: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Get main adversarial challenge prompt."""
        
        base_prompt = f"""
You are an expert adversarial reviewer tasked with critically analyzing this response to identify weaknesses, unsupported claims, and areas requiring improvement.

ORIGINAL CLAIM: "{original_claim}"

AGENT RESPONSE TO ANALYZE (from {agent_id}):
{agent_response}
"""
        
        if additional_context:
            base_prompt += f"""
ADDITIONAL CONTEXT: {additional_context}
"""
        
        base_prompt += """

YOUR ROLE AS CHALLENGER:
You are a skeptical expert whose job is to find legitimate flaws, weaknesses, and areas for improvement. Be thorough, fair, but appropriately critical. Your goal is to improve the quality of factual claims through rigorous analysis.

CHALLENGE AREAS TO EXAMINE:

1. UNSUPPORTED CLAIMS
   - Identify statements that lack adequate evidence
   - Look for assertions presented as fact without proper backing
   - Flag definitive statements that should be qualified with uncertainty

2. CITATION QUALITY ISSUES
   - Evaluate source authority and reliability
   - Check for appropriate source types for the claims made
   - Identify weak, outdated, or inappropriate citations
   - Look for missing citations where evidence is needed

3. CONTRADICTORY EVIDENCE
   - Identify internal contradictions in the response
   - Look for conflicts between cited sources
   - Find evidence that undermines the stated conclusion

4. EVIDENCE SUFFICIENCY
   - Assess whether evidence quantity is adequate
   - Evaluate evidence quality and relevance
   - Identify gaps in the evidence base

5. LOGICAL INCONSISTENCIES
   - Find flaws in reasoning or argumentation
   - Identify unjustified leaps in logic
   - Look for bias or selective presentation

6. SOURCE QUALITY PROBLEMS
   - Identify low-authority or biased sources
   - Flag outdated information
   - Point out potential conflicts of interest

FOR EACH CHALLENGE YOU IDENTIFY:
- Specify the exact problem location (quote the problematic text)
- Explain why this is problematic
- Assess severity (0.0 to 1.0, where 1.0 = critical flaw)
- Provide specific improvement suggestions
- Include alternative evidence if available

CRITICAL ANALYSIS GUIDELINES:
- Be fair but thorough - look for real problems, not trivial issues
- Focus on substantive concerns that affect accuracy or reliability
- Provide constructive suggestions for improvement
- Consider whether a revision would be beneficial
- Maintain high standards for factual accuracy

OUTPUT STRUCTURE:
1. EXECUTIVE SUMMARY: Brief overview of your analysis
2. MAJOR CHALLENGES: List critical issues (severity ≥ 0.7)
3. MODERATE CHALLENGES: List moderate issues (severity 0.4-0.7)
4. MINOR CONCERNS: List minor issues (severity < 0.4)
5. OVERALL ASSESSMENT: Your verdict on response quality
6. REVISION RECOMMENDATION: Whether revision is needed and why
7. CONFIDENCE: Your confidence in this challenge analysis (0.0-1.0)

Be rigorous, specific, and constructive in your analysis.
"""
        
        return base_prompt
    
    @staticmethod
    def get_citation_analysis_prompt(citations: List[str], claim: str) -> str:
        """Get prompt for detailed citation quality analysis."""
        
        prompt = f"""
Conduct a thorough analysis of these citations for the claim: "{claim}"

CITATIONS TO ANALYZE:
"""
        
        for i, citation in enumerate(citations, 1):
            prompt += f"[{i}] {citation}\n"
        
        prompt += """

CITATION QUALITY CRITERIA:

1. SOURCE AUTHORITY
   - Is this a credible, authoritative source for this type of claim?
   - Does the source have relevant expertise or institutional backing?
   - Is the source known for accuracy and reliability?

2. RELEVANCE TO CLAIM
   - How directly does this source address the specific claim?
   - Does the source provide evidence for the exact statement made?
   - Is the connection between source and claim clear and logical?

3. CURRENCY AND TIMELINESS
   - Is the information current enough for this claim?
   - Has newer information potentially superseded this source?
   - Is the publication date appropriate for the topic?

4. SOURCE TYPE APPROPRIATENESS
   - Is this the right type of source for this claim (academic, news, government, etc.)?
   - Would a different source type be more appropriate?
   - Does the source type match the nature of the evidence needed?

5. ACCESSIBILITY AND VERIFIABILITY
   - Is the citation complete and properly formatted?
   - Can the source be easily located and verified?
   - Are URLs functional and properly formatted?

6. POTENTIAL BIAS OR CONFLICTS
   - Does the source have potential conflicts of interest?
   - Is there apparent bias that affects credibility?
   - Are there alternative perspectives not represented?

For each citation, provide:
- Authority assessment (0.0-1.0)
- Relevance score (0.0-1.0)  
- Currency rating (current/acceptable/outdated)
- Source type evaluation (appropriate/marginal/inappropriate)
- Overall citation quality score (0.0-1.0)
- Specific concerns or red flags
- Improvement suggestions

Then provide an overall assessment of the citation set's quality.
"""
        
        return prompt
    
    @staticmethod
    def get_evidence_contradiction_prompt(
        supporting_evidence: List[str],
        contradicting_evidence: List[str],
        claim: str
    ) -> str:
        """Get prompt for identifying evidence contradictions."""
        
        prompt = f"""
Analyze potential contradictions in the evidence presented for this claim:

CLAIM: "{claim}"

SUPPORTING EVIDENCE:
"""
        
        for i, evidence in enumerate(supporting_evidence, 1):
            prompt += f"{i}. {evidence}\n"
        
        if contradicting_evidence:
            prompt += "\nCONTRADICTING EVIDENCE:\n"
            for i, evidence in enumerate(contradicting_evidence, 1):
                prompt += f"{i}. {evidence}\n"
        
        prompt += """

CONTRADICTION ANALYSIS:

1. DIRECT CONTRADICTIONS
   - Are there pieces of evidence that directly oppose each other?
   - Do any sources make conflicting factual claims?
   - Are there contradictory statistics or data points?

2. METHODOLOGICAL CONFLICTS
   - Do different studies use conflicting methodologies that could explain differences?
   - Are there temporal differences that might explain contradictions?
   - Do sample sizes or populations differ significantly?

3. CONTEXTUAL CONTRADICTIONS  
   - Are evidence pieces from different contexts being inappropriately compared?
   - Do time periods, locations, or conditions differ in ways that matter?
   - Are there scope or definition differences causing apparent contradictions?

4. QUALITY DISCREPANCIES
   - Is higher-quality evidence being contradicted by lower-quality evidence?
   - Are preliminary findings conflicting with more established research?
   - Do peer-reviewed sources contradict non-peer-reviewed sources?

5. INTERPRETATION ISSUES
   - Are contradictions due to different interpretations of the same data?
   - Are authors drawing different conclusions from similar evidence?
   - Are there nuances being missed that resolve apparent contradictions?

For each potential contradiction identified:
- Describe the specific conflict
- Assess the strength of the contradiction (0.0-1.0)
- Evaluate which evidence is more reliable
- Suggest how the contradiction should be resolved
- Recommend additional evidence needed

Determine overall impact on claim credibility.
"""
        
        return prompt
    
    @staticmethod
    def get_logical_consistency_prompt(response_text: str, claim: str) -> str:
        """Get prompt for analyzing logical consistency."""
        
        return f"""
Analyze the logical consistency of this response about the claim: "{claim}"

RESPONSE TO ANALYZE:
{response_text}

LOGICAL ANALYSIS FRAMEWORK:

1. ARGUMENT STRUCTURE
   - Is the reasoning clearly laid out and easy to follow?
   - Are conclusions properly supported by the evidence presented?
   - Are there logical gaps or leaps in the argumentation?

2. CONSISTENCY CHECK
   - Are all statements internally consistent with each other?
   - Does the conclusion align with the evidence presented?
   - Are there contradictory statements within the response?

3. EVIDENCE-TO-CONCLUSION ALIGNMENT
   - Does the evidence actually support the stated conclusion?
   - Is the confidence level appropriate given the evidence?
   - Are alternative conclusions properly considered?

4. CAUSAL REASONING
   - Are cause-and-effect relationships properly established?
   - Are correlations inappropriately presented as causation?
   - Are confounding factors adequately addressed?

5. SCOPE AND GENERALIZABILITY
   - Are conclusions appropriately limited to the evidence scope?
   - Are generalizations beyond the data properly qualified?
   - Are limitations and exceptions acknowledged?

6. BIAS AND SELECTIVITY
   - Is evidence presented fairly and completely?
   - Are counterarguments adequately addressed?
   - Is there evidence of confirmation bias in source selection?

IDENTIFY LOGICAL FLAWS:
- Circular reasoning
- False dichotomies
- Hasty generalizations  
- Ad hominem attacks on sources
- Appeal to authority without substance
- Cherry-picking evidence
- Straw man arguments

For each logical issue found:
- Quote the specific problematic text
- Explain the logical flaw
- Assess severity (0.0-1.0)
- Suggest how to correct the reasoning
- Recommend additional analysis needed

Provide overall assessment of logical soundness.
"""
    
    @staticmethod
    def get_evidence_sufficiency_prompt(
        evidence_count: int,
        evidence_quality_scores: List[float],
        claim: str,
        domain: Optional[str] = None
    ) -> str:
        """Get prompt for assessing evidence sufficiency."""
        
        avg_quality = sum(evidence_quality_scores) / len(evidence_quality_scores) if evidence_quality_scores else 0.0
        
        prompt = f"""
Assess whether the evidence provided is sufficient to support conclusions about this claim:

CLAIM: "{claim}"
DOMAIN: {domain or "General"}
EVIDENCE METRICS:
- Total evidence pieces: {evidence_count}
- Average evidence quality: {avg_quality:.2f}/1.0
- Quality distribution: {evidence_quality_scores}

SUFFICIENCY ANALYSIS CRITERIA:

1. QUANTITY ASSESSMENT
   - Is the number of evidence sources adequate for this type of claim?
   - Are there enough independent sources to establish reliability?
   - Would additional evidence significantly strengthen the analysis?

2. QUALITY EVALUATION
   - Is the overall evidence quality sufficient for confident conclusions?
   - Are there enough high-quality sources (>0.8 quality)?
   - Do low-quality sources undermine the analysis?

3. DIVERSITY CHECK
   - Are evidence sources sufficiently diverse in type and origin?
   - Is there appropriate geographic, temporal, or methodological diversity?
   - Are different perspectives and approaches represented?

4. DOMAIN-SPECIFIC REQUIREMENTS
   - Does the evidence meet standards typical for this domain?
   - Are the right types of sources being used for this claim type?
   - Is the evidence appropriate for the level of certainty expressed?

5. COVERAGE ANALYSIS
   - Do the sources cover all key aspects of the claim?
   - Are there important dimensions left unaddressed?
   - Is the scope of evidence appropriate for the scope of the claim?

6. INDEPENDENCE VERIFICATION
   - Are the sources truly independent of each other?
   - Is there potential for the same underlying data/study to be cited multiple times?
   - Are primary sources distinguished from secondary sources?

SUFFICIENCY STANDARDS BY CLAIM TYPE:
- Factual/historical claims: Multiple independent authoritative sources
- Scientific claims: Peer-reviewed research, preferably with replication
- Statistical claims: Official data sources with appropriate methodology  
- Expert consensus claims: Multiple independent expert sources

IDENTIFY SUFFICIENCY PROBLEMS:
- Too few total sources
- Lack of high-quality sources
- Insufficient diversity
- Over-reliance on single source types
- Missing critical perspectives
- Inadequate coverage of claim scope

For each sufficiency issue:
- Describe the specific gap
- Assess impact on conclusion reliability (0.0-1.0)
- Recommend additional evidence needed
- Suggest specific source types to seek

Provide overall sufficiency rating and improvement recommendations.
"""
        
        return prompt
    
    @staticmethod
    def get_confidence_calibration_prompt(
        stated_confidence: float,
        evidence_analysis: str,
        conclusion: str
    ) -> str:
        """Get prompt for evaluating confidence calibration."""
        
        return f"""
Evaluate whether the stated confidence level is appropriately calibrated:

STATED CONFIDENCE: {stated_confidence:.3f} (on 0.0-1.0 scale)
CONCLUSION: {conclusion}

EVIDENCE ANALYSIS:
{evidence_analysis}

CONFIDENCE CALIBRATION ASSESSMENT:

1. EVIDENCE STRENGTH ALIGNMENT
   - Does the confidence level match the strength of evidence?
   - Is high confidence justified by strong evidence?
   - Is low confidence appropriate when evidence is weak?

2. UNCERTAINTY ACKNOWLEDGMENT
   - Are significant uncertainties properly reflected in confidence?
   - Is overconfidence evident despite limitations?
   - Are appropriate caveats and qualifications included?

3. PRECISION VS. ACCURACY
   - Is excessive precision claimed (e.g., 0.847 vs. ~0.8)?
   - Is false precision masking genuine uncertainty?
   - Are confidence intervals or ranges more appropriate?

4. DOMAIN CONSIDERATIONS
   - Is the confidence appropriate for this type of claim and domain?
   - Are domain-specific uncertainty factors considered?
   - Is the confidence consistent with expert consensus in the field?

5. COMPARATIVE CALIBRATION
   - How does this confidence compare to similar claims?
   - Is the confidence consistent across different parts of the response?
   - Are there internal confidence discrepancies?

CONFIDENCE CALIBRATION ISSUES:

OVERCONFIDENCE INDICATORS:
- High confidence (>0.8) with limited evidence
- Precise confidence values without justification
- Insufficient acknowledgment of limitations
- Dismissal of contradictory evidence
- Claims of certainty in uncertain domains

UNDERCONFIDENCE INDICATORS:
- Very low confidence (<0.4) with strong evidence
- Excessive hedging despite clear evidence
- Overemphasis on minor contradictions
- Failure to recognize evidence strength

APPROPRIATE CONFIDENCE RANGES:
- 0.9-1.0: Near certainty, overwhelming evidence, established facts
- 0.7-0.9: High confidence, strong evidence, expert consensus
- 0.5-0.7: Moderate confidence, mixed evidence, ongoing debate
- 0.3-0.5: Low confidence, weak evidence, high uncertainty
- 0.0-0.3: Very low confidence, insufficient/contradictory evidence

ANALYSIS REQUIRED:
- Is stated confidence justified by evidence presented?
- What confidence level would be more appropriate?
- What factors were overlooked in confidence assessment?
- How should uncertainty be better communicated?

Provide:
- Calibration assessment (well-calibrated/overconfident/underconfident)
- Suggested confidence range with justification
- Key factors affecting appropriate confidence level
- Recommendations for better uncertainty communication
"""
    
    @staticmethod
    def get_revision_recommendation_prompt(
        challenges_identified: List[Dict[str, Any]],
        overall_quality_score: float,
        claim: str
    ) -> str:
        """Get prompt for making revision recommendations."""
        
        prompt = f"""
Based on your analysis, provide revision recommendations for this response about: "{claim}"

CHALLENGES IDENTIFIED:
"""
        
        for i, challenge in enumerate(challenges_identified, 1):
            prompt += f"{i}. {challenge.get('type', 'Unknown')}: {challenge.get('description', 'No description')} (Severity: {challenge.get('severity', 0.0):.2f})\n"
        
        prompt += f"""
OVERALL QUALITY SCORE: {overall_quality_score:.2f}/1.0

REVISION DECISION FRAMEWORK:

1. CRITICAL ISSUES (Severity ≥ 0.8):
   - Major factual errors or unsupported claims
   - Severely inadequate or inappropriate sources
   - Fundamental logical flaws
   - Significant bias or misrepresentation
   → RECOMMENDATION: Major revision required

2. MODERATE ISSUES (Severity 0.5-0.8):
   - Weak or insufficient evidence
   - Minor logical inconsistencies
   - Citation format or quality problems
   - Overconfidence or underconfidence
   → RECOMMENDATION: Moderate revision suggested

3. MINOR ISSUES (Severity < 0.5):
   - Formatting inconsistencies
   - Minor source quality improvements
   - Clarification needs
   - Style or presentation issues
   → RECOMMENDATION: Minor improvements beneficial

REVISION RECOMMENDATIONS:

NO REVISION NEEDED:
- High overall quality (>0.8)
- No critical or moderate issues
- Minor issues only that don't affect conclusions
- Well-supported, well-reasoned response

MINOR REVISION:
- Good overall quality (0.6-0.8)
- Few moderate issues, no critical issues
- Clear improvement path available
- Core analysis remains sound

MODERATE REVISION:
- Fair overall quality (0.4-0.6)
- Multiple moderate issues or one critical issue
- Significant improvements needed
- May require additional research

MAJOR REVISION:
- Poor overall quality (<0.4)
- Multiple critical issues
- Fundamental problems with analysis
- Complete rework recommended

SPECIFIC REVISION GUIDANCE:
For each identified challenge, provide:
- Priority level (high/medium/low)
- Specific actions needed
- Additional research required
- Expected improvement impact

REVISION DECISION:
Based on the analysis:
- Overall revision necessity: [NONE/MINOR/MODERATE/MAJOR]
- Key priorities for improvement
- Estimated effort required
- Timeline recommendations

Justify your recommendation thoroughly.
"""
        
        return prompt
    
    @staticmethod
    def get_final_challenge_summary_prompt(
        challenge_details: List[str],
        revision_recommendation: str,
        confidence_in_analysis: float
    ) -> str:
        """Get prompt for final challenge report summary."""
        
        return f"""
Synthesize your challenger analysis into a comprehensive final report.

YOUR DETAILED CHALLENGES:
{chr(10).join(challenge_details)}

YOUR REVISION RECOMMENDATION:
{revision_recommendation}

YOUR CONFIDENCE IN THIS ANALYSIS: {confidence_in_analysis:.3f}

FINAL REPORT STRUCTURE:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - Overall assessment of response quality
   - Key problems identified
   - Primary recommendation

2. CRITICAL FINDINGS (if any)
   - Most serious issues requiring immediate attention
   - Impact on reliability and accuracy
   - Urgency of correction needed

3. CONSTRUCTIVE FEEDBACK
   - Specific, actionable improvement suggestions
   - Priority order for addressing issues
   - Resources or approaches that might help

4. STRENGTHS ACKNOWLEDGED
   - Positive aspects of the response
   - Areas where analysis was strong
   - Good practices to maintain

5. OVERALL RECOMMENDATION
   - Clear verdict on revision necessity
   - Expected impact of recommended changes
   - Confidence in your assessment

REPORT GUIDELINES:
- Be fair and balanced - acknowledge strengths as well as weaknesses
- Provide specific, actionable feedback rather than general criticism
- Maintain a constructive, professional tone
- Focus on substance over style unless style affects comprehension
- Be confident in your analysis but acknowledge any limitations

Your role is to improve the quality of factual analysis through rigorous but fair criticism.

Provide your comprehensive final challenge report.
"""