"""Evidence extraction with relevance scoring and ranking of search results."""

import re
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ..schemas.citation_schemas import (
    SearchResult, EvidenceItem, CitationSource, SearchResponse, CitationType
)
from ..config.config_manager import get_config_manager

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning(f"{YELLOW}Could not download NLTK data, some features may be limited{END}")


@dataclass
class RelevanceScores:
    """Container for various relevance scores."""
    semantic_similarity: float = 0.0
    keyword_overlap: float = 0.0
    domain_relevance: float = 0.0
    source_credibility: float = 0.0
    content_quality: float = 0.0
    overall_score: float = 0.0


@dataclass
class ExtractedEvidence:
    """Evidence extracted from a single source with scoring."""
    text_snippets: List[str]
    relevance_scores: RelevanceScores
    supporting_quotes: List[str]
    contradicting_quotes: List[str]
    factual_claims: List[str]
    statistical_data: List[str]


class TextProcessor:
    """Text processing utilities for evidence extraction."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep periods and commas for sentence structure
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text."""
        try:
            words = word_tokenize(self.preprocess_text(text))
            
            # Filter out stop words and short words
            keywords = [
                self.lemmatizer.lemmatize(word) 
                for word in words 
                if word.lower() not in self.stop_words 
                and len(word) >= min_length
                and word.isalpha()
            ]
            
            return list(set(keywords))  # Remove duplicates
        except:
            # Fallback if NLTK fails
            words = text.split()
            return [word for word in words if len(word) >= min_length and word.isalpha()]
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        try:
            return sent_tokenize(text)
        except:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def extract_factual_patterns(self, text: str) -> List[str]:
        """Extract patterns that likely contain factual information."""
        patterns = [
            r'\b\d+\.?\d*\s*(percent|%|million|billion|thousand|years?|days?|months?)\b',  # Statistics
            r'\baccording to\s+[^\.]+',  # Attribution
            r'\bresearch\s+(shows?|indicates?|suggests?|found)[^\.]+',  # Research findings
            r'\bstudy\s+(shows?|indicates?|suggests?|found)[^\.]+',  # Study findings
            r'\b(approximately|about|nearly|over|under|more than|less than)\s+\d+',  # Quantities
            r'\bin\s+\d{4}[^\.]*',  # Year references
        ]
        
        factual_claims = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            factual_claims.extend(matches)
        
        return factual_claims


class SourceCredibilityAnalyzer:
    """Analyze the credibility of sources."""
    
    def __init__(self):
        # Trusted domain patterns
        self.high_credibility_domains = {
            'academic': ['.edu', 'scholar.google', 'pubmed', 'ncbi.nlm.nih.gov', 'doi.org'],
            'government': ['.gov'],
            'reputable_news': ['bbc.com', 'reuters.com', 'ap.org', 'npr.org'],
            'medical': ['who.int', 'cdc.gov', 'nih.gov', 'mayo.edu', 'webmd.com'],
            'financial': ['federalreserve.gov', 'sec.gov', 'treasury.gov']
        }
        
        # Lower credibility indicators
        self.low_credibility_indicators = [
            'blog', 'wordpress', 'tumblr', 'personal', 'opinion'
        ]
    
    def analyze_source_credibility(self, url: str, source_type: CitationType) -> float:
        """
        Analyze source credibility based on URL and type.
        
        Returns:
            Credibility score between 0.0 and 1.0
        """
        score = 0.5  # Base score
        url_lower = url.lower()
        
        # Check high credibility domains
        for category, domains in self.high_credibility_domains.items():
            for domain in domains:
                if domain in url_lower:
                    if category == 'academic':
                        score += 0.4
                    elif category == 'government':
                        score += 0.35
                    elif category in ['medical', 'reputable_news']:
                        score += 0.3
                    else:
                        score += 0.2
                    break
        
        # Check low credibility indicators
        for indicator in self.low_credibility_indicators:
            if indicator in url_lower:
                score -= 0.2
        
        # Type-based scoring
        if source_type in [CitationType.ACADEMIC_PAPER, CitationType.JOURNAL_ARTICLE]:
            score += 0.2
        elif source_type == CitationType.GOVERNMENT_REPORT:
            score += 0.15
        elif source_type == CitationType.NEWS_ARTICLE:
            score += 0.1
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))


class EvidenceExtractor:
    """Extract and score evidence from search results with relevance ranking."""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.text_processor = TextProcessor()
        self.credibility_analyzer = SourceCredibilityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def extract_evidence_from_results(
        self, 
        claim_text: str, 
        search_results: List[SearchResult],
        max_evidence_items: int = 10
    ) -> List[EvidenceItem]:
        """
        Extract evidence items from search results with relevance scoring.
        
        Args:
            claim_text: Original claim that needs evidence
            search_results: Search results to extract evidence from
            max_evidence_items: Maximum number of evidence items to return
            
        Returns:
            List of EvidenceItem objects ranked by relevance
        """
        if not search_results:
            return []
        
        logger.info(f"{BLUE}Extracting evidence from {len(search_results)} search results{END}")
        
        evidence_items = []
        
        # Process each search result
        for result in search_results:
            try:
                evidence = self._extract_evidence_from_result(claim_text, result)
                if evidence:
                    evidence_items.append(evidence)
            except Exception as e:
                logger.warning(f"{YELLOW}Error extracting evidence from result: {e}{END}")
                continue
        
        # Rank evidence by overall relevance score
        evidence_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Return top evidence items
        top_evidence = evidence_items[:max_evidence_items]
        
        logger.success(f"{GREEN}Extracted {len(top_evidence)} evidence items{END}")
        return top_evidence
    
    def _extract_evidence_from_result(
        self, 
        claim_text: str, 
        search_result: SearchResult
    ) -> Optional[EvidenceItem]:
        """Extract evidence from a single search result."""
        
        # Calculate relevance scores
        relevance_scores = self._calculate_relevance_scores(claim_text, search_result)
        
        # Skip if overall relevance is too low
        min_relevance = self.config_manager.get_config().evaluation.citation_precision_target
        if relevance_scores.overall_score < min_relevance * 0.8:  # Allow slightly lower threshold
            return None
        
        # Create citation source
        citation_source = CitationSource(
            url=search_result.url,
            title=search_result.title,
            citation_type=self._infer_citation_type(str(search_result.url)),
            access_date=None  # Will be set when actually accessed
        )
        
        # Extract text snippets and analyze content
        extracted_content = self._analyze_search_result_content(search_result, claim_text)
        
        # Determine if evidence supports or contradicts the claim
        supporting = self._determine_evidence_stance(claim_text, search_result.snippet)
        
        # Create evidence item
        evidence_item = EvidenceItem(
            text=search_result.snippet,
            source=citation_source,
            relevance_score=relevance_scores.overall_score,
            quality_score=relevance_scores.source_credibility,
            supporting=supporting,
            text_span=extracted_content.get('key_span')
        )
        
        return evidence_item
    
    def _calculate_relevance_scores(
        self, 
        claim_text: str, 
        search_result: SearchResult
    ) -> RelevanceScores:
        """Calculate comprehensive relevance scores."""
        
        scores = RelevanceScores()
        
        # Semantic similarity using TF-IDF
        scores.semantic_similarity = self._calculate_semantic_similarity(
            claim_text, 
            search_result.title + " " + search_result.snippet
        )
        
        # Keyword overlap
        scores.keyword_overlap = self._calculate_keyword_overlap(
            claim_text,
            search_result.title + " " + search_result.snippet
        )
        
        # Domain relevance (if domain classification is available)
        scores.domain_relevance = self._calculate_domain_relevance(claim_text, search_result)
        
        # Source credibility
        citation_type = self._infer_citation_type(str(search_result.url))
        scores.source_credibility = self.credibility_analyzer.analyze_source_credibility(
            str(search_result.url), 
            citation_type
        )
        
        # Content quality (based on snippet length, structure, etc.)
        scores.content_quality = self._calculate_content_quality(search_result)
        
        # Calculate overall score as weighted average
        weights = {
            'semantic_similarity': 0.3,
            'keyword_overlap': 0.25,
            'domain_relevance': 0.15,
            'source_credibility': 0.2,
            'content_quality': 0.1
        }
        
        scores.overall_score = (
            scores.semantic_similarity * weights['semantic_similarity'] +
            scores.keyword_overlap * weights['keyword_overlap'] +
            scores.domain_relevance * weights['domain_relevance'] +
            scores.source_credibility * weights['source_credibility'] +
            scores.content_quality * weights['content_quality']
        )
        
        return scores
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using TF-IDF."""
        try:
            # Prepare texts
            texts = [
                self.text_processor.preprocess_text(text1),
                self.text_processor.preprocess_text(text2)
            ]
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.debug(f"{YELLOW}Error calculating semantic similarity: {e}{END}")
            return 0.0
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap between two texts."""
        try:
            keywords1 = set(self.text_processor.extract_keywords(text1))
            keywords2 = set(self.text_processor.extract_keywords(text2))
            
            if not keywords1 or not keywords2:
                return 0.0
            
            overlap = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            
            return overlap / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_domain_relevance(self, claim_text: str, search_result: SearchResult) -> float:
        """Calculate domain-specific relevance."""
        # For now, use a simple heuristic based on URL and content
        # This could be enhanced with domain classification
        
        url_lower = str(search_result.url).lower()
        title_lower = search_result.title.lower()
        snippet_lower = search_result.snippet.lower()
        claim_lower = claim_text.lower()
        
        relevance_score = 0.5  # Base score
        
        # Check for domain-specific terms
        domain_keywords = {
            'science': ['research', 'study', 'experiment', 'scientific', 'analysis'],
            'health': ['medical', 'health', 'disease', 'treatment', 'clinical'],
            'history': ['historical', 'century', 'past', 'ancient', 'history'],
            'finance': ['economic', 'financial', 'market', 'investment', 'economic']
        }
        
        # Find dominant domain in claim
        claim_domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in claim_lower)
            claim_domain_scores[domain] = score
        
        if claim_domain_scores:
            dominant_domain = max(claim_domain_scores, key=claim_domain_scores.get)
            
            # Check if search result content matches the domain
            result_content = title_lower + " " + snippet_lower
            domain_match_score = sum(1 for keyword in domain_keywords[dominant_domain] if keyword in result_content)
            
            if domain_match_score > 0:
                relevance_score += min(0.3, domain_match_score * 0.1)
        
        return min(1.0, relevance_score)
    
    def _calculate_content_quality(self, search_result: SearchResult) -> float:
        """Calculate content quality based on various factors."""
        quality_score = 0.5  # Base score
        
        # Title quality
        if search_result.title and len(search_result.title) > 10:
            quality_score += 0.1
        
        # Snippet quality
        if search_result.snippet:
            snippet_len = len(search_result.snippet)
            if snippet_len > 50:
                quality_score += 0.2
            if snippet_len > 100:
                quality_score += 0.1
        
        # Search result rank (higher rank = higher quality)
        if hasattr(search_result, 'rank'):
            rank_score = max(0, (10 - search_result.rank) / 10 * 0.2)
            quality_score += rank_score
        
        return min(1.0, quality_score)
    
    def _infer_citation_type(self, url: str) -> CitationType:
        """Infer citation type from URL."""
        url_lower = url.lower()
        
        if any(domain in url_lower for domain in ['doi.org', 'pubmed', 'scholar.google']):
            return CitationType.ACADEMIC_PAPER
        elif '.gov' in url_lower:
            return CitationType.GOVERNMENT_REPORT
        elif any(domain in url_lower for domain in ['news', 'cnn', 'bbc', 'reuters']):
            return CitationType.NEWS_ARTICLE
        else:
            return CitationType.WEBSITE
    
    def _analyze_search_result_content(self, search_result: SearchResult, claim_text: str) -> Dict[str, any]:
        """Analyze search result content for key information."""
        content = search_result.title + " " + search_result.snippet
        
        # Extract factual patterns
        factual_claims = self.text_processor.extract_factual_patterns(content)
        
        # Find the most relevant sentence/span
        sentences = self.text_processor.extract_sentences(content)
        if sentences:
            # Find sentence with highest keyword overlap with claim
            claim_keywords = set(self.text_processor.extract_keywords(claim_text))
            best_sentence = ""
            best_overlap = 0
            
            for sentence in sentences:
                sentence_keywords = set(self.text_processor.extract_keywords(sentence))
                overlap = len(claim_keywords.intersection(sentence_keywords))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sentence
            
            key_span = best_sentence if best_overlap > 0 else sentences[0]
        else:
            key_span = search_result.snippet[:100] + "..." if len(search_result.snippet) > 100 else search_result.snippet
        
        return {
            'factual_claims': factual_claims,
            'key_span': key_span,
            'sentences': sentences
        }
    
    def _determine_evidence_stance(self, claim_text: str, evidence_text: str) -> bool:
        """
        Determine if evidence supports or contradicts the claim.
        
        Returns:
            True if evidence supports the claim, False if it contradicts
        """
        # Simple heuristic approach - can be enhanced with more sophisticated NLP
        evidence_lower = evidence_text.lower()
        claim_lower = claim_text.lower()
        
        # Look for contradiction indicators
        contradiction_terms = [
            'however', 'but', 'although', 'despite', 'contrary', 'opposite',
            'not', 'no', 'false', 'incorrect', 'wrong', 'debunked'
        ]
        
        # Look for support indicators
        support_terms = [
            'confirms', 'supports', 'proves', 'shows', 'demonstrates',
            'according to', 'research shows', 'studies indicate'
        ]
        
        contradiction_score = sum(1 for term in contradiction_terms if term in evidence_lower)
        support_score = sum(1 for term in support_terms if term in evidence_lower)
        
        # Also check keyword overlap - high overlap usually means support
        claim_keywords = set(self.text_processor.extract_keywords(claim_text))
        evidence_keywords = set(self.text_processor.extract_keywords(evidence_text))
        overlap_ratio = len(claim_keywords.intersection(evidence_keywords)) / len(claim_keywords) if claim_keywords else 0
        
        # Decision logic
        if contradiction_score > support_score and overlap_ratio < 0.3:
            return False  # Likely contradictory
        else:
            return True   # Assume supportive by default
    
    def rank_evidence_by_relevance(self, evidence_items: List[EvidenceItem]) -> List[EvidenceItem]:
        """Rank evidence items by their relevance scores."""
        return sorted(evidence_items, key=lambda x: x.relevance_score, reverse=True)
    
    def filter_evidence_by_quality(
        self, 
        evidence_items: List[EvidenceItem], 
        min_relevance: float = 0.6,
        min_quality: float = 0.5
    ) -> List[EvidenceItem]:
        """Filter evidence items by minimum quality thresholds."""
        filtered = [
            item for item in evidence_items 
            if item.relevance_score >= min_relevance and item.quality_score >= min_quality
        ]
        
        logger.info(f"{CYAN}Filtered {len(evidence_items)} to {len(filtered)} evidence items{END}")
        return filtered
    
    def get_evidence_summary(self, evidence_items: List[EvidenceItem]) -> Dict[str, any]:
        """Generate summary statistics for evidence items."""
        if not evidence_items:
            return {
                'total_items': 0,
                'supporting_items': 0,
                'contradicting_items': 0,
                'average_relevance': 0.0,
                'average_quality': 0.0
            }
        
        supporting = sum(1 for item in evidence_items if item.supporting)
        contradicting = len(evidence_items) - supporting
        
        avg_relevance = sum(item.relevance_score for item in evidence_items) / len(evidence_items)
        avg_quality = sum(item.quality_score for item in evidence_items) / len(evidence_items)
        
        return {
            'total_items': len(evidence_items),
            'supporting_items': supporting,
            'contradicting_items': contradicting,
            'average_relevance': avg_relevance,
            'average_quality': avg_quality
        }