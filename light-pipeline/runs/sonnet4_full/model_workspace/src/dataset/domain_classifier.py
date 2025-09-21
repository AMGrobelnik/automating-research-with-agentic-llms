"""Domain classification for factual claims in the Cite-and-Challenge Protocol system."""

import re
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..config.config_manager import get_config_manager, DomainConfig

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


@dataclass
class ClassificationResult:
    """Result of domain classification."""
    domain: str
    confidence: float
    complexity_score: float
    keyword_matches: List[str]
    reasoning: str


class DomainClassifier:
    """Automated categorization of factual claims by domain with complexity scoring."""
    
    def __init__(self):
        """Initialize DomainClassifier with configuration."""
        self.config_manager = get_config_manager()
        self.domains = self.config_manager.get_domains()
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)
        self._setup_classification()
    
    def _setup_classification(self) -> None:
        """Set up classification components."""
        # Create domain keyword corpus
        domain_texts = []
        self.domain_names = []
        
        for domain in self.domains:
            domain_text = " ".join(domain.keywords)
            domain_texts.append(domain_text)
            self.domain_names.append(domain.name)
        
        # Fit vectorizer on domain keywords
        self.domain_vectors = self.vectorizer.fit_transform(domain_texts)
        
        logger.info(f"{BLUE}Initialized domain classifier with {len(self.domains)} domains{END}")
    
    def classify_claim(self, claim_text: str) -> ClassificationResult:
        """
        Classify a claim into one of the predefined domains.
        
        Args:
            claim_text: The factual claim to classify
            
        Returns:
            ClassificationResult with domain, confidence, and complexity score
        """
        # Preprocess claim text
        processed_claim = self._preprocess_text(claim_text)
        
        # Extract keyword matches for each domain
        keyword_scores = self._calculate_keyword_scores(processed_claim)
        
        # Calculate TF-IDF similarity scores
        tfidf_scores = self._calculate_tfidf_scores(processed_claim)
        
        # Combine scores
        combined_scores = self._combine_scores(keyword_scores, tfidf_scores)
        
        # Find best domain
        best_domain_idx = np.argmax(combined_scores)
        best_domain = self.domain_names[best_domain_idx]
        confidence = combined_scores[best_domain_idx]
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(claim_text, best_domain)
        
        # Get keyword matches for the best domain
        domain_config = next(d for d in self.domains if d.name == best_domain)
        keyword_matches = self._get_keyword_matches(processed_claim, domain_config.keywords)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(claim_text, best_domain, keyword_matches, confidence, complexity_score)
        
        result = ClassificationResult(
            domain=best_domain,
            confidence=confidence,
            complexity_score=complexity_score,
            keyword_matches=keyword_matches,
            reasoning=reasoning
        )
        
        logger.debug(f"{CYAN}Classified claim to domain: {best_domain} (confidence: {confidence:.3f}){END}")
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except periods and commas
        text = re.sub(r'[^\w\s\.,]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_keyword_scores(self, claim_text: str) -> np.ndarray:
        """Calculate keyword-based scores for each domain."""
        scores = np.zeros(len(self.domains))
        
        for i, domain in enumerate(self.domains):
            score = 0.0
            matches = []
            
            for keyword in domain.keywords:
                # Count occurrences of keyword (with word boundaries)
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches_count = len(re.findall(pattern, claim_text))
                if matches_count > 0:
                    matches.append(keyword)
                    # Weight by keyword importance and frequency
                    score += matches_count * (1.0 / len(domain.keywords))
            
            # Apply domain complexity weight
            scores[i] = score * domain.complexity_weight
        
        # Normalize scores
        if scores.sum() > 0:
            scores = scores / scores.sum()
        
        return scores
    
    def _calculate_tfidf_scores(self, claim_text: str) -> np.ndarray:
        """Calculate TF-IDF similarity scores for each domain."""
        try:
            # Transform claim text
            claim_vector = self.vectorizer.transform([claim_text])
            
            # Calculate cosine similarity with each domain
            similarities = cosine_similarity(claim_vector, self.domain_vectors).flatten()
            
            # Apply softmax normalization
            exp_similarities = np.exp(similarities - np.max(similarities))
            softmax_similarities = exp_similarities / exp_similarities.sum()
            
            return softmax_similarities
            
        except Exception as e:
            logger.warning(f"{YELLOW}TF-IDF scoring failed: {e}, using uniform distribution{END}")
            return np.ones(len(self.domains)) / len(self.domains)
    
    def _combine_scores(self, keyword_scores: np.ndarray, tfidf_scores: np.ndarray) -> np.ndarray:
        """Combine keyword and TF-IDF scores."""
        # Weight combination (70% keyword, 30% TF-IDF)
        combined = 0.7 * keyword_scores + 0.3 * tfidf_scores
        
        # Apply softmax for final normalization
        exp_combined = np.exp(combined - np.max(combined))
        return exp_combined / exp_combined.sum()
    
    def _calculate_complexity_score(self, claim_text: str, domain: str) -> float:
        """
        Calculate complexity score for a claim based on various factors.
        
        Factors considered:
        - Text length
        - Number of technical terms
        - Sentence complexity
        - Domain-specific complexity weight
        """
        # Base complexity from text length
        length_score = min(len(claim_text) / 500.0, 1.0)  # Normalize to max 500 chars
        
        # Technical terms complexity (presence of numbers, technical words)
        technical_patterns = [
            r'\d+\.?\d*%',  # Percentages
            r'\d+\.?\d*',   # Numbers
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'(research|study|analysis|data|statistics|evidence|correlation|causation)',  # Technical terms
        ]
        
        technical_score = 0.0
        for pattern in technical_patterns:
            matches = len(re.findall(pattern, claim_text, re.IGNORECASE))
            technical_score += matches * 0.1
        
        technical_score = min(technical_score, 1.0)  # Cap at 1.0
        
        # Sentence complexity (number of clauses, conjunctions)
        complexity_indicators = [
            r'\b(however|although|because|therefore|moreover|furthermore|nevertheless)\b',
            r'\b(which|that|who|where|when)\b',  # Relative clauses
            r'[,;:]',  # Punctuation indicating complexity
        ]
        
        sentence_score = 0.0
        for pattern in complexity_indicators:
            matches = len(re.findall(pattern, claim_text, re.IGNORECASE))
            sentence_score += matches * 0.05
        
        sentence_score = min(sentence_score, 1.0)  # Cap at 1.0
        
        # Domain-specific weight
        domain_config = next(d for d in self.domains if d.name == domain)
        domain_weight = domain_config.complexity_weight
        
        # Combine all factors
        base_complexity = (length_score * 0.3 + technical_score * 0.4 + sentence_score * 0.3)
        final_complexity = base_complexity * domain_weight
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_complexity))
    
    def _get_keyword_matches(self, claim_text: str, keywords: List[str]) -> List[str]:
        """Get list of keywords that match in the claim text."""
        matches = []
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, claim_text):
                matches.append(keyword)
        return matches
    
    def _generate_reasoning(self, claim_text: str, domain: str, keyword_matches: List[str], 
                          confidence: float, complexity_score: float) -> str:
        """Generate reasoning for the classification decision."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Classified as '{domain}' domain with {confidence:.1%} confidence.")
        
        if keyword_matches:
            reasoning_parts.append(f"Key indicators: {', '.join(keyword_matches[:3])}.")
        
        if complexity_score > 0.7:
            reasoning_parts.append("High complexity due to technical language and detailed information.")
        elif complexity_score > 0.4:
            reasoning_parts.append("Moderate complexity with some technical elements.")
        else:
            reasoning_parts.append("Relatively simple claim with clear language.")
        
        return " ".join(reasoning_parts)
    
    def classify_multiple_claims(self, claims: List[str]) -> List[ClassificationResult]:
        """Classify multiple claims efficiently."""
        results = []
        for claim in claims:
            result = self.classify_claim(claim)
            results.append(result)
        
        logger.info(f"{GREEN}Classified {len(claims)} claims{END}")
        return results
    
    def get_domain_distribution(self, claims: List[str]) -> Dict[str, int]:
        """Get domain distribution for a list of claims."""
        results = self.classify_multiple_claims(claims)
        distribution = {}
        
        for result in results:
            domain = result.domain
            distribution[domain] = distribution.get(domain, 0) + 1
        
        return distribution
    
    def validate_domain_balance(self, claims: List[str], target_per_domain: int = 75) -> Dict[str, Any]:
        """Validate that claims are balanced across domains."""
        distribution = self.get_domain_distribution(claims)
        
        validation_result = {
            "is_balanced": True,
            "distribution": distribution,
            "target_per_domain": target_per_domain,
            "total_claims": len(claims),
            "issues": []
        }
        
        for domain_name in self.domain_names:
            count = distribution.get(domain_name, 0)
            if count < target_per_domain * 0.8:  # Allow 20% deviation
                validation_result["is_balanced"] = False
                validation_result["issues"].append(f"Domain '{domain_name}' has only {count} claims (target: {target_per_domain})")
            elif count > target_per_domain * 1.2:
                validation_result["is_balanced"] = False
                validation_result["issues"].append(f"Domain '{domain_name}' has {count} claims (target: {target_per_domain})")
        
        return validation_result
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification system statistics."""
        return {
            "total_domains": len(self.domains),
            "domain_names": self.domain_names,
            "vectorizer_features": len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'get_feature_names_out') else "N/A",
            "domain_configs": [
                {
                    "name": d.name,
                    "keywords_count": len(d.keywords),
                    "complexity_weight": d.complexity_weight
                }
                for d in self.domains
            ]
        }