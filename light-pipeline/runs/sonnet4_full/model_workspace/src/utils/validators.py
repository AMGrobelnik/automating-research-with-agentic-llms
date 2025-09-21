"""Input validation utilities for the Cite-and-Challenge Protocol system."""

import re
import validators
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from loguru import logger

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ClaimValidator:
    """Validator for factual claims."""
    
    @staticmethod
    def validate_claim_text(claim_text: str, max_length: int = 500, min_length: int = 10) -> bool:
        """
        Validate claim text format and length.
        
        Args:
            claim_text: The claim text to validate
            max_length: Maximum allowed length
            min_length: Minimum required length
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(claim_text, str):
            raise ValidationError("Claim text must be a string")
        
        claim_text = claim_text.strip()
        
        if len(claim_text) < min_length:
            raise ValidationError(f"Claim text too short: {len(claim_text)} < {min_length}")
        
        if len(claim_text) > max_length:
            raise ValidationError(f"Claim text too long: {len(claim_text)} > {max_length}")
        
        # Check for minimum content (not just punctuation/whitespace)
        content_chars = re.sub(r'[^\w\s]', '', claim_text)
        if len(content_chars.strip()) < min_length // 2:
            raise ValidationError("Claim text lacks sufficient content")
        
        return True
    
    @staticmethod
    def validate_domain(domain: str, valid_domains: List[str] = None) -> bool:
        """
        Validate domain name.
        
        Args:
            domain: Domain name to validate
            valid_domains: List of valid domain names
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(domain, str):
            raise ValidationError("Domain must be a string")
        
        domain = domain.strip().lower()
        
        if not domain:
            raise ValidationError("Domain cannot be empty")
        
        # Check against valid domains if provided
        if valid_domains:
            if domain not in [d.lower() for d in valid_domains]:
                raise ValidationError(f"Invalid domain '{domain}'. Valid options: {valid_domains}")
        
        # Basic format validation
        if not re.match(r'^[a-z][a-z_]*[a-z]$', domain) and len(domain) > 1:
            if not re.match(r'^[a-z]$', domain):  # Single character domains allowed
                raise ValidationError("Domain must contain only lowercase letters and underscores")
        
        return True
    
    @staticmethod
    def validate_complexity_score(score: Union[int, float]) -> bool:
        """
        Validate complexity score.
        
        Args:
            score: Complexity score to validate (should be 0-1)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(score, (int, float)):
            raise ValidationError("Complexity score must be a number")
        
        if not (0.0 <= score <= 1.0):
            raise ValidationError(f"Complexity score must be between 0 and 1, got: {score}")
        
        return True


class CitationValidator:
    """Validator for citations and URLs."""
    
    @staticmethod
    def validate_url(url: str, allow_local: bool = False) -> bool:
        """
        Validate URL format and accessibility.
        
        Args:
            url: URL to validate
            allow_local: Whether to allow localhost/local URLs
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")
        
        url = url.strip()
        
        if not url:
            raise ValidationError("URL cannot be empty")
        
        # Basic URL format validation
        if not validators.url(url):
            raise ValidationError(f"Invalid URL format: {url}")
        
        # Parse URL components
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValidationError(f"URL must use HTTP or HTTPS protocol: {url}")
        
        # Check for local URLs if not allowed
        if not allow_local:
            local_patterns = [
                r'^localhost',
                r'^127\.',
                r'^192\.168\.',
                r'^10\.',
                r'^172\.(1[6-9]|2[0-9]|3[0-1])\.'
            ]
            
            for pattern in local_patterns:
                if re.match(pattern, parsed.hostname or ''):
                    raise ValidationError(f"Local URLs not allowed: {url}")
        
        return True
    
    @staticmethod
    def validate_citation_format(citation: str, format_type: str = "apa") -> bool:
        """
        Validate citation format.
        
        Args:
            citation: Citation text to validate
            format_type: Citation format type ("apa", "mla", etc.)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(citation, str):
            raise ValidationError("Citation must be a string")
        
        citation = citation.strip()
        
        if not citation:
            raise ValidationError("Citation cannot be empty")
        
        if len(citation) < 10:
            raise ValidationError("Citation too short to be valid")
        
        # Basic APA format validation (simplified)
        if format_type.lower() == "apa":
            # Should contain year in parentheses
            year_pattern = r'\(\d{4}\)'
            if not re.search(year_pattern, citation):
                logger.warning(f"{YELLOW}APA citation missing year in parentheses: {citation[:50]}...{END}")
            
            # Should contain a period
            if '.' not in citation:
                raise ValidationError("APA citation should contain at least one period")
        
        return True
    
    @staticmethod
    def validate_citation_list(citations: List[str], max_count: int = 20) -> bool:
        """
        Validate list of citations.
        
        Args:
            citations: List of citation strings
            max_count: Maximum allowed number of citations
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(citations, list):
            raise ValidationError("Citations must be a list")
        
        if len(citations) > max_count:
            raise ValidationError(f"Too many citations: {len(citations)} > {max_count}")
        
        # Validate each citation
        for i, citation in enumerate(citations):
            try:
                CitationValidator.validate_citation_format(citation)
            except ValidationError as e:
                raise ValidationError(f"Invalid citation at index {i}: {e}")
        
        return True


class ResponseValidator:
    """Validator for agent responses."""
    
    @staticmethod
    def validate_confidence_score(score: Union[int, float]) -> bool:
        """
        Validate confidence score.
        
        Args:
            score: Confidence score (should be 0-1)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(score, (int, float)):
            raise ValidationError("Confidence score must be a number")
        
        if not (0.0 <= score <= 1.0):
            raise ValidationError(f"Confidence score must be between 0 and 1, got: {score}")
        
        return True
    
    @staticmethod
    def validate_agent_response(response_text: str, min_length: int = 50, max_length: int = 5000) -> bool:
        """
        Validate agent response text.
        
        Args:
            response_text: Response text to validate
            min_length: Minimum required length
            max_length: Maximum allowed length
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(response_text, str):
            raise ValidationError("Response text must be a string")
        
        response_text = response_text.strip()
        
        if len(response_text) < min_length:
            raise ValidationError(f"Response text too short: {len(response_text)} < {min_length}")
        
        if len(response_text) > max_length:
            raise ValidationError(f"Response text too long: {len(response_text)} > {max_length}")
        
        # Check for actual content
        if len(response_text.replace(' ', '').replace('\n', '')) < min_length // 2:
            raise ValidationError("Response lacks sufficient content")
        
        return True
    
    @staticmethod
    def validate_agent_type(agent_type: str, valid_types: List[str] = None) -> bool:
        """
        Validate agent type.
        
        Args:
            agent_type: Agent type to validate
            valid_types: List of valid agent types
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(agent_type, str):
            raise ValidationError("Agent type must be a string")
        
        agent_type = agent_type.strip().lower()
        
        if not agent_type:
            raise ValidationError("Agent type cannot be empty")
        
        # Default valid types
        if valid_types is None:
            valid_types = ["answering", "challenger", "baseline"]
        
        valid_types = [t.lower() for t in valid_types]
        
        if agent_type not in valid_types:
            raise ValidationError(f"Invalid agent type '{agent_type}'. Valid options: {valid_types}")
        
        return True


class ConfigValidator:
    """Validator for configuration values."""
    
    @staticmethod
    def validate_api_key(api_key: str, key_type: str = "generic") -> bool:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            key_type: Type of API key for specific validation
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(api_key, str):
            raise ValidationError("API key must be a string")
        
        api_key = api_key.strip()
        
        if not api_key:
            raise ValidationError("API key cannot be empty")
        
        # Basic length check
        if len(api_key) < 10:
            raise ValidationError("API key too short to be valid")
        
        # Type-specific validation
        if key_type.lower() == "openai":
            if not api_key.startswith("sk-"):
                logger.warning(f"{YELLOW}OpenAI API key should start with 'sk-'{END}")
        
        # Check for obvious placeholder values
        placeholder_patterns = [
            r'^(your|my)[-_]?api[-_]?key$',
            r'^(replace|enter|insert)[-_]?this$',
            r'^(xxx|placeholder|token)$',
            r'^(test|demo|example)[-_]?key$'
        ]
        
        for pattern in placeholder_patterns:
            if re.match(pattern, api_key.lower()):
                raise ValidationError(f"API key appears to be a placeholder: {api_key}")
        
        return True
    
    @staticmethod
    def validate_model_name(model_name: str, valid_models: List[str] = None) -> bool:
        """
        Validate AI model name.
        
        Args:
            model_name: Model name to validate
            valid_models: List of valid model names
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(model_name, str):
            raise ValidationError("Model name must be a string")
        
        model_name = model_name.strip()
        
        if not model_name:
            raise ValidationError("Model name cannot be empty")
        
        # Common valid model patterns (if no specific list provided)
        if valid_models is None:
            valid_patterns = [
                r'^gpt-[34]',  # GPT models
                r'^claude-',   # Claude models
                r'^text-',     # OpenAI text models
                r'^gemini-',   # Gemini models
            ]
            
            if not any(re.match(pattern, model_name.lower()) for pattern in valid_patterns):
                logger.warning(f"{YELLOW}Model name '{model_name}' doesn't match common patterns{END}")
        else:
            # Check against provided valid models
            if model_name not in valid_models:
                raise ValidationError(f"Invalid model '{model_name}'. Valid options: {valid_models}")
        
        return True


def validate_experiment_config(config: Dict[str, Any]) -> bool:
    """
    Validate complete experiment configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ["database", "api_keys", "search", "agents", "system", "domains", "evaluation"]
    
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required configuration field: {field}")
    
    # Validate specific subsections
    if "agents" in config:
        agents_config = config["agents"]
        if "max_tokens" in agents_config:
            max_tokens = agents_config["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens < 100 or max_tokens > 32000:
                raise ValidationError(f"Invalid max_tokens: {max_tokens} (should be 100-32000)")
    
    logger.success(f"{GREEN}Configuration validation passed{END}")
    return True