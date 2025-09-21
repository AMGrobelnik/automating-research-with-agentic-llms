"""Text span identification for citation support requirements."""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from loguru import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..schemas.citation_schemas import TextSpan, EvidenceItem
from ..config.config_manager import get_config_manager

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


@dataclass
class SpanFeatures:
    """Features extracted from a text span for citation requirement analysis."""
    has_statistics: bool = False
    has_specific_claims: bool = False
    has_technical_terms: bool = False
    has_proper_nouns: bool = False
    has_temporal_references: bool = False
    has_quantitative_data: bool = False
    has_attribution: bool = False
    sentence_complexity: float = 0.0
    factual_confidence: float = 0.0


@dataclass
class SpanAnalysis:
    """Analysis result for a text span."""
    span: TextSpan
    features: SpanFeatures
    citation_necessity_score: float
    reasoning: str


class PatternMatcher:
    """Pattern matching for identifying citation-worthy content."""
    
    def __init__(self):
        # Statistical patterns that typically require citation
        self.statistical_patterns = [
            r'\b\d+(?:\.\d+)?\s*(%|percent|percentage)',
            r'\b\d+(?:\.\d+)?\s*(million|billion|thousand|trillion)\b',
            r'\b(approximately|about|nearly|roughly|over|under|more than|less than|up to)\s+\d+',
            r'\b\d+(?:\.\d+)?\s*(times|fold|x)\s+(more|less|higher|lower|greater|smaller)',
            r'\b(increased|decreased|rose|fell|dropped|climbed)\s+by\s+\d+',
            r'\b\d+(?:\.\d+)?\s*(in|out of)\s+\d+',
        ]
        
        # Specific factual claim patterns
        self.factual_claim_patterns = [
            r'\b(research|study|studies|survey|analysis|investigation)\s+(shows?|indicates?|suggests?|found|reveals?|demonstrates?)',
            r'\b(according to|based on)\s+([^,\.]+)',
            r'\b(scientists?|researchers?|experts?|doctors?|economists?)\s+(say|report|claim|argue|suggest)',
            r'\b(data|evidence|findings|results)\s+(shows?|indicates?|suggests?|reveals?)',
            r'\b(the\s+)?(latest|recent|new)\s+(study|research|report|survey)',
        ]
        
        # Technical/specialized terms that often need citation
        self.technical_term_patterns = [
            r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # CamelCase terms (often technical)
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w*-\w*-\w*\b',  # Hyphenated technical terms
            r'\b(algorithm|methodology|protocol|syndrome|disease|disorder|therapy|treatment)\b',
        ]
        
        # Proper nouns and names
        self.proper_noun_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?:\s+(?:University|Institute|Organization|Foundation|Company|Corporation|Inc|Ltd))?',
            r'\b(?:Dr|Prof|Professor|Mr|Ms|Mrs)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        ]
        
        # Temporal references
        self.temporal_patterns = [
            r'\bin\s+(19|20)\d{2}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}',
            r'\b(last|past)\s+(year|decade|century|month|week)',
            r'\b(since|until|from)\s+(19|20)\d{2}',
        ]
        
        # Quantitative data
        self.quantitative_patterns = [
            r'\b\d+\.?\d*\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b',
            r'\b\d+\.?\d*\s*(grams?|kilograms?|pounds?|ounces?|tons?)\b',
            r'\b\d+\.?\d*\s*(meters?|kilometers?|feet|inches?|miles?)\b',
            r'\b\d+\.?\d*\s*(degrees?|celsius|fahrenheit)\b',
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
        ]
        
        # Attribution indicators
        self.attribution_patterns = [
            r'\baccording to\b',
            r'\bas reported by\b',
            r'\bciting\b',
            r'\bquoted from\b',
            r'\bsource:\b',
            r'\breference:\b',
        ]


class SpanMarker:
    """Identify text spans that require citation support."""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.pattern_matcher = PatternMatcher()
        self.min_span_length = 10
        self.max_span_length = 200
        
    def identify_citation_spans(
        self, 
        text: str, 
        min_confidence: float = 0.6
    ) -> List[TextSpan]:
        """
        Identify text spans that require citation support.
        
        Args:
            text: Input text to analyze
            min_confidence: Minimum confidence threshold for span identification
            
        Returns:
            List of TextSpan objects requiring citations
        """
        logger.info(f"{BLUE}Identifying citation spans in text ({len(text)} chars){END}")
        
        # Split text into sentences for initial analysis
        sentences = self._split_into_sentences(text)
        
        spans = []
        current_position = 0
        
        for sentence in sentences:
            # Find the sentence position in the original text
            sentence_start = text.find(sentence, current_position)
            if sentence_start == -1:
                continue
            
            sentence_end = sentence_start + len(sentence)
            current_position = sentence_end
            
            # Analyze the sentence for citation requirements
            analysis = self._analyze_span(sentence, sentence_start, sentence_end)
            
            if analysis.citation_necessity_score >= min_confidence:
                spans.append(analysis.span)
        
        # Look for multi-sentence spans that form coherent claims
        multi_sentence_spans = self._identify_multi_sentence_spans(text, sentences)
        
        for span_info in multi_sentence_spans:
            analysis = self._analyze_span(span_info['text'], span_info['start'], span_info['end'])
            if analysis.citation_necessity_score >= min_confidence:
                spans.append(analysis.span)
        
        # Remove overlapping spans, keeping the ones with higher confidence
        spans = self._remove_overlapping_spans(spans)
        
        logger.success(f"{GREEN}Identified {len(spans)} citation spans{END}")
        return spans
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with more sophisticated NLP
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    def _analyze_span(self, span_text: str, start_pos: int, end_pos: int) -> SpanAnalysis:
        """Analyze a text span for citation requirements."""
        
        features = self._extract_span_features(span_text)
        
        # Calculate citation necessity score
        necessity_score = self._calculate_citation_necessity_score(features, span_text)
        
        # Generate reasoning for why citation is needed
        reasoning = self._generate_reasoning(features, span_text)
        
        # Determine span type
        span_type = self._determine_span_type(features, span_text)
        
        span = TextSpan(
            text=span_text,
            start_position=start_pos,
            end_position=end_pos,
            confidence=necessity_score,
            span_type=span_type,
            supporting_evidence=[]
        )
        
        return SpanAnalysis(
            span=span,
            features=features,
            citation_necessity_score=necessity_score,
            reasoning=reasoning
        )
    
    def _extract_span_features(self, span_text: str) -> SpanFeatures:
        """Extract features from a text span."""
        
        features = SpanFeatures()
        span_lower = span_text.lower()
        
        # Check for statistics
        features.has_statistics = any(
            re.search(pattern, span_text, re.IGNORECASE) 
            for pattern in self.pattern_matcher.statistical_patterns
        )
        
        # Check for specific factual claims
        features.has_specific_claims = any(
            re.search(pattern, span_text, re.IGNORECASE) 
            for pattern in self.pattern_matcher.factual_claim_patterns
        )
        
        # Check for technical terms
        features.has_technical_terms = any(
            re.search(pattern, span_text) 
            for pattern in self.pattern_matcher.technical_term_patterns
        )
        
        # Check for proper nouns
        features.has_proper_nouns = any(
            re.search(pattern, span_text) 
            for pattern in self.pattern_matcher.proper_noun_patterns
        )
        
        # Check for temporal references
        features.has_temporal_references = any(
            re.search(pattern, span_text, re.IGNORECASE) 
            for pattern in self.pattern_matcher.temporal_patterns
        )
        
        # Check for quantitative data
        features.has_quantitative_data = any(
            re.search(pattern, span_text, re.IGNORECASE) 
            for pattern in self.pattern_matcher.quantitative_patterns
        )
        
        # Check for attribution
        features.has_attribution = any(
            re.search(pattern, span_text, re.IGNORECASE) 
            for pattern in self.pattern_matcher.attribution_patterns
        )
        
        # Calculate sentence complexity
        features.sentence_complexity = self._calculate_sentence_complexity(span_text)
        
        # Calculate factual confidence
        features.factual_confidence = self._calculate_factual_confidence(span_text)
        
        return features
    
    def _calculate_citation_necessity_score(self, features: SpanFeatures, span_text: str) -> float:
        """Calculate how much a span needs citation support."""
        
        score = 0.0
        
        # High-weight indicators
        if features.has_statistics:
            score += 0.3
        
        if features.has_specific_claims:
            score += 0.25
        
        # Medium-weight indicators
        if features.has_quantitative_data:
            score += 0.15
        
        if features.has_temporal_references:
            score += 0.1
        
        if features.has_technical_terms:
            score += 0.1
        
        # Lower-weight indicators
        if features.has_proper_nouns:
            score += 0.05
        
        # Complexity and factual confidence modifiers
        score += features.sentence_complexity * 0.1
        score += features.factual_confidence * 0.15
        
        # Attribution modifier (if already attributed, reduce necessity)
        if features.has_attribution:
            score *= 0.7  # Reduce score if already attributed
        
        # Specific keyword-based scoring
        span_lower = span_text.lower()
        
        # High-necessity keywords
        high_necessity_keywords = [
            'research shows', 'studies indicate', 'according to experts',
            'data reveals', 'analysis found', 'survey results',
            'statistics show', 'evidence suggests', 'findings indicate'
        ]
        
        for keyword in high_necessity_keywords:
            if keyword in span_lower:
                score += 0.2
                break
        
        # Domain-specific indicators
        domain_indicators = {
            'medical': ['patients', 'treatment', 'clinical', 'symptoms', 'diagnosis'],
            'scientific': ['experiment', 'hypothesis', 'theory', 'methodology', 'analysis'],
            'economic': ['market', 'investment', 'financial', 'economic', 'revenue'],
            'historical': ['century', 'historical', 'ancient', 'civilization', 'period']
        }
        
        for domain, keywords in domain_indicators.items():
            if any(keyword in span_lower for keyword in keywords):
                score += 0.05
                break
        
        # Ensure score is within bounds
        return min(1.0, max(0.0, score))
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        """Calculate sentence complexity (0-1 scale)."""
        
        complexity_score = 0.0
        
        # Word count factor
        word_count = len(text.split())
        if word_count > 15:
            complexity_score += min(0.3, (word_count - 15) / 50)
        
        # Clause indicators
        clause_indicators = [',', ';', 'which', 'that', 'because', 'although', 'however', 'therefore']
        clause_count = sum(text.lower().count(indicator) for indicator in clause_indicators)
        complexity_score += min(0.3, clause_count / 10)
        
        # Technical language indicators
        if any(char.isupper() for char in text):  # Has uppercase (acronyms, proper nouns)
            complexity_score += 0.1
        
        if re.search(r'\d', text):  # Contains numbers
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    def _calculate_factual_confidence(self, text: str) -> float:
        """Calculate how factual/objective the text appears (0-1 scale)."""
        
        factual_score = 0.5  # Base score
        text_lower = text.lower()
        
        # Factual indicators
        factual_indicators = [
            'research', 'study', 'data', 'evidence', 'analysis', 
            'findings', 'results', 'survey', 'report', 'investigation'
        ]
        
        factual_count = sum(1 for indicator in factual_indicators if indicator in text_lower)
        factual_score += min(0.3, factual_count * 0.1)
        
        # Objective language indicators
        objective_indicators = [
            'indicates', 'suggests', 'shows', 'demonstrates', 'reveals',
            'found', 'observed', 'measured', 'recorded', 'documented'
        ]
        
        objective_count = sum(1 for indicator in objective_indicators if indicator in text_lower)
        factual_score += min(0.2, objective_count * 0.1)
        
        # Subjective language penalties
        subjective_indicators = [
            'i think', 'i believe', 'in my opinion', 'seems like', 
            'probably', 'maybe', 'perhaps', 'might', 'could be'
        ]
        
        subjective_count = sum(1 for indicator in subjective_indicators if indicator in text_lower)
        factual_score -= min(0.3, subjective_count * 0.15)
        
        return min(1.0, max(0.0, factual_score))
    
    def _determine_span_type(self, features: SpanFeatures, span_text: str) -> str:
        """Determine the type of span based on its features."""
        
        if features.has_statistics or features.has_quantitative_data:
            return "statistical"
        elif features.has_specific_claims:
            return "factual_claim"  
        elif features.has_technical_terms:
            return "technical"
        elif features.has_temporal_references:
            return "historical"
        else:
            return "general"
    
    def _generate_reasoning(self, features: SpanFeatures, span_text: str) -> str:
        """Generate reasoning for why this span needs citation."""
        
        reasons = []
        
        if features.has_statistics:
            reasons.append("contains statistical data")
        
        if features.has_specific_claims:
            reasons.append("makes specific factual claims")
        
        if features.has_quantitative_data:
            reasons.append("includes quantitative information")
        
        if features.has_technical_terms:
            reasons.append("uses technical terminology")
        
        if features.has_temporal_references:
            reasons.append("references specific time periods")
        
        if features.factual_confidence > 0.7:
            reasons.append("appears to state objective facts")
        
        if features.sentence_complexity > 0.6:
            reasons.append("contains complex information")
        
        if not reasons:
            reasons.append("contains information that may require verification")
        
        return f"Citation needed because this span {', '.join(reasons)}."
    
    def _identify_multi_sentence_spans(self, text: str, sentences: List[str]) -> List[Dict]:
        """Identify spans that cross sentence boundaries."""
        
        multi_spans = []
        
        # Look for consecutive sentences that together form a coherent claim
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i]
            next_sentence = sentences[i + 1]
            
            # Check if sentences are thematically related
            if self._are_sentences_related(current_sentence, next_sentence):
                combined_text = current_sentence + " " + next_sentence
                
                # Find position in original text
                start_pos = text.find(current_sentence)
                end_pos = text.find(next_sentence) + len(next_sentence)
                
                if start_pos != -1 and end_pos > start_pos:
                    multi_spans.append({
                        'text': combined_text,
                        'start': start_pos,
                        'end': end_pos
                    })
        
        return multi_spans
    
    def _are_sentences_related(self, sentence1: str, sentence2: str) -> bool:
        """Check if two sentences are thematically related."""
        
        # Simple keyword overlap approach
        words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return False
        
        overlap_ratio = len(words1.intersection(words2)) / min(len(words1), len(words2))
        
        return overlap_ratio > 0.2  # At least 20% keyword overlap
    
    def _remove_overlapping_spans(self, spans: List[TextSpan]) -> List[TextSpan]:
        """Remove overlapping spans, keeping those with higher confidence."""
        
        if not spans:
            return []
        
        # Sort by confidence (descending)
        sorted_spans = sorted(spans, key=lambda x: x.confidence, reverse=True)
        
        non_overlapping = []
        
        for span in sorted_spans:
            overlaps = False
            
            for existing_span in non_overlapping:
                if self._spans_overlap(span, existing_span):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(span)
        
        return non_overlapping
    
    def _spans_overlap(self, span1: TextSpan, span2: TextSpan) -> bool:
        """Check if two spans overlap."""
        return not (span1.end_position <= span2.start_position or 
                   span2.end_position <= span1.start_position)
    
    def annotate_text_with_spans(self, text: str, spans: List[TextSpan], citation_marker: str = "[Citation needed]") -> str:
        """
        Annotate text with citation markers at identified spans.
        
        Args:
            text: Original text
            spans: Identified citation spans
            citation_marker: Marker to insert for citations
            
        Returns:
            Text annotated with citation markers
        """
        if not spans:
            return text
        
        # Sort spans by position (descending) to avoid position shifting
        sorted_spans = sorted(spans, key=lambda x: x.start_position, reverse=True)
        
        annotated_text = text
        
        for span in sorted_spans:
            # Insert citation marker at the end of the span
            insertion_point = span.end_position
            annotated_text = (annotated_text[:insertion_point] + 
                            f" {citation_marker}" + 
                            annotated_text[insertion_point:])
        
        return annotated_text
    
    def match_spans_with_evidence(
        self, 
        spans: List[TextSpan], 
        evidence_items: List[EvidenceItem]
    ) -> List[TextSpan]:
        """
        Match citation spans with available evidence items.
        
        Args:
            spans: Citation spans needing evidence
            evidence_items: Available evidence items
            
        Returns:
            Spans with matched evidence items
        """
        for span in spans:
            # Find evidence items that are relevant to this span
            relevant_evidence = []
            
            for evidence in evidence_items:
                # Calculate relevance between span and evidence
                relevance = self._calculate_span_evidence_relevance(span, evidence)
                
                if relevance > 0.5:  # Threshold for relevance
                    relevant_evidence.append(evidence)
            
            # Sort evidence by relevance and take top items
            relevant_evidence.sort(key=lambda x: x.relevance_score, reverse=True)
            span.supporting_evidence = relevant_evidence[:3]  # Top 3 evidence items
        
        return spans
    
    def _calculate_span_evidence_relevance(self, span: TextSpan, evidence: EvidenceItem) -> float:
        """Calculate relevance between a span and an evidence item."""
        
        # Simple keyword overlap approach
        span_words = set(re.findall(r'\b\w+\b', span.text.lower()))
        evidence_words = set(re.findall(r'\b\w+\b', evidence.text.lower()))
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        span_words -= stop_words
        evidence_words -= stop_words
        
        if not span_words or not evidence_words:
            return 0.0
        
        overlap = len(span_words.intersection(evidence_words))
        union = len(span_words.union(evidence_words))
        
        jaccard_similarity = overlap / union if union > 0 else 0.0
        
        # Boost relevance if evidence is high quality
        quality_boost = evidence.quality_score * 0.2
        
        return min(1.0, jaccard_similarity + quality_boost)
    
    def get_span_statistics(self, spans: List[TextSpan]) -> Dict[str, any]:
        """Get statistics about identified citation spans."""
        
        if not spans:
            return {
                'total_spans': 0,
                'span_types': {},
                'average_confidence': 0.0,
                'average_length': 0.0
            }
        
        span_types = {}
        for span in spans:
            span_types[span.span_type] = span_types.get(span.span_type, 0) + 1
        
        avg_confidence = sum(span.confidence for span in spans) / len(spans)
        avg_length = sum(len(span.text) for span in spans) / len(spans)
        
        return {
            'total_spans': len(spans),
            'span_types': span_types,
            'average_confidence': avg_confidence,
            'average_length': avg_length,
            'spans_with_evidence': sum(1 for span in spans if span.supporting_evidence)
        }