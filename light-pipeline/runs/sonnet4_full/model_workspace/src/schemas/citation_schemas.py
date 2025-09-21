"""Pydantic schemas for citation validation and data structures."""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, field_validator
from enum import Enum


class CitationType(str, Enum):
    """Types of citations supported."""
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    WEBSITE = "website"
    NEWS_ARTICLE = "news_article"
    ACADEMIC_PAPER = "academic_paper"
    GOVERNMENT_REPORT = "government_report"
    OTHER = "other"


class SearchProvider(str, Enum):
    """Search providers supported."""
    GOOGLE = "google"
    BING = "bing" 
    DUCKDUCKGO = "duckduckgo"


class CitationSource(BaseModel):
    """Individual citation source with metadata."""
    url: HttpUrl = Field(..., description="URL of the source")
    title: str = Field(..., min_length=1, max_length=500, description="Title of the source")
    author: Optional[str] = Field(None, description="Author(s) of the source")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    publication_name: Optional[str] = Field(None, description="Name of publication/journal/website")
    citation_type: CitationType = Field(CitationType.OTHER, description="Type of citation")
    access_date: Optional[datetime] = Field(None, description="Date when URL was accessed")
    
    @field_validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()


class FormattedCitation(BaseModel):
    """APA-formatted citation with metadata."""
    formatted_text: str = Field(..., description="Full APA-formatted citation")
    source: CitationSource = Field(..., description="Source information")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in citation accuracy")
    
    @field_validator('formatted_text')
    def validate_formatted_text(cls, v):
        if len(v) < 10:
            raise ValueError("Formatted citation too short")
        if not any(char in v for char in '.!?'):
            raise ValueError("Citation should end with punctuation")
        return v


class SearchResult(BaseModel):
    """Individual search result from web search."""
    title: str = Field(..., description="Title of the search result")
    url: HttpUrl = Field(..., description="URL of the search result")
    snippet: str = Field(..., description="Snippet/description from search")
    provider: SearchProvider = Field(..., description="Search provider that returned this result")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score for the query")
    rank: int = Field(..., ge=1, description="Rank in search results")
    
    @field_validator('snippet')
    def validate_snippet(cls, v):
        if len(v) > 1000:
            return v[:997] + "..."
        return v


class SearchQuery(BaseModel):
    """Search query with parameters."""
    query: str = Field(..., min_length=1, description="Search query text")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    providers: List[SearchProvider] = Field([SearchProvider.DUCKDUCKGO], description="Search providers to use")
    timeout: int = Field(30, ge=5, le=120, description="Timeout in seconds")
    
    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResponse(BaseModel):
    """Response from web search API."""
    query: SearchQuery = Field(..., description="Original search query")
    results: List[SearchResult] = Field([], description="Search results")
    total_results: int = Field(0, ge=0, description="Total number of results found")
    search_time: float = Field(0.0, ge=0.0, description="Time taken for search in seconds")
    provider_used: SearchProvider = Field(..., description="Primary provider used")
    error: Optional[str] = Field(None, description="Error message if search failed")


class EvidenceItem(BaseModel):
    """Individual piece of evidence extracted from search results."""
    text: str = Field(..., description="Text content of the evidence")
    source: CitationSource = Field(..., description="Source of the evidence")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance to the original claim")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality/credibility of the evidence")
    supporting: bool = Field(True, description="Whether evidence supports or contradicts the claim")
    text_span: Optional[str] = Field(None, description="Specific text span that needs citation")
    
    @field_validator('text')
    def validate_text(cls, v):
        if len(v) < 10:
            raise ValueError("Evidence text too short")
        if len(v) > 2000:
            return v[:1997] + "..."
        return v


class TextSpan(BaseModel):
    """Text span that requires citation support."""
    text: str = Field(..., description="Text span content")
    start_position: int = Field(..., ge=0, description="Start position in original text")
    end_position: int = Field(..., ge=0, description="End position in original text")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence that this span needs citation")
    span_type: str = Field("factual", description="Type of span (factual, statistical, claim)")
    supporting_evidence: List[EvidenceItem] = Field([], description="Evidence that supports this span")
    
    @field_validator('end_position')
    def validate_positions(cls, v, info):
        if 'start_position' in info.data and v <= info.data['start_position']:
            raise ValueError("End position must be greater than start position")
        return v


class CitationRequest(BaseModel):
    """Request for citation research and formatting."""
    claim_text: str = Field(..., description="Original claim text that needs citations")
    domain: Optional[str] = Field(None, description="Domain of the claim")
    max_citations: int = Field(5, ge=1, le=20, description="Maximum number of citations to find")
    search_queries: List[str] = Field([], description="Specific search queries to use")
    required_spans: List[TextSpan] = Field([], description="Specific text spans that need citations")
    
    @field_validator('claim_text')
    def validate_claim_text(cls, v):
        if len(v) < 10:
            raise ValueError("Claim text too short")
        if len(v) > 1000:
            raise ValueError("Claim text too long")
        return v.strip()


class CitationResponse(BaseModel):
    """Response with formatted citations and evidence."""
    request: CitationRequest = Field(..., description="Original citation request")
    citations: List[FormattedCitation] = Field([], description="Formatted citations found")
    evidence: List[EvidenceItem] = Field([], description="Supporting evidence items")
    annotated_text: str = Field("", description="Original text with citation markers")
    search_queries_used: List[str] = Field([], description="Search queries that were executed")
    processing_time: float = Field(0.0, ge=0.0, description="Total processing time in seconds")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence in citations")
    
    @field_validator('annotated_text')
    def validate_annotated_text(cls, v, info):
        if v and 'request' in info.data:
            # Basic validation that annotated text contains original claim
            original_words = set(info.data['request'].claim_text.lower().split())
            annotated_words = set(v.lower().split())
            if len(original_words.intersection(annotated_words)) < len(original_words) * 0.5:
                raise ValueError("Annotated text should contain most of the original claim")
        return v


class RateLimitInfo(BaseModel):
    """Rate limiting information for search APIs."""
    provider: SearchProvider = Field(..., description="Search provider")
    requests_made: int = Field(0, ge=0, description="Number of requests made")
    requests_limit: int = Field(100, ge=1, description="Request limit")
    reset_time: Optional[datetime] = Field(None, description="When the rate limit resets")
    
    def can_make_request(self) -> bool:
        """Check if we can make another request."""
        return self.requests_made < self.requests_limit


class APIError(BaseModel):
    """API error information."""
    provider: SearchProvider = Field(..., description="Provider that generated the error")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")
    retryable: bool = Field(False, description="Whether the request can be retried")


class SearchMetrics(BaseModel):
    """Metrics for search operations."""
    total_queries: int = Field(0, ge=0, description="Total number of queries executed")
    successful_queries: int = Field(0, ge=0, description="Number of successful queries")
    failed_queries: int = Field(0, ge=0, description="Number of failed queries")
    average_response_time: float = Field(0.0, ge=0.0, description="Average response time in seconds")
    total_results_found: int = Field(0, ge=0, description="Total results found across all queries")
    providers_used: Dict[SearchProvider, int] = Field({}, description="Count of queries per provider")
    
    def success_rate(self) -> float:
        """Calculate success rate of queries."""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries


# Configuration schemas for citation module
class CitationConfig(BaseModel):
    """Configuration for citation and research functionality."""
    default_search_provider: SearchProvider = Field(SearchProvider.DUCKDUCKGO, description="Default search provider")
    max_citations_per_claim: int = Field(5, ge=1, le=20, description="Maximum citations per claim")
    citation_format: str = Field("apa", description="Citation format to use")
    search_timeout: int = Field(30, ge=5, le=120, description="Search timeout in seconds")
    enable_span_detection: bool = Field(True, description="Whether to detect text spans needing citations")
    quality_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum quality score for evidence")
    relevance_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum relevance score for evidence")


# Aliases for agent compatibility
CitationSchema = FormattedCitation
EvidenceSchema = EvidenceItem

# Additional schemas for agent use
class SimpleCitationSchema(BaseModel):
    """Simplified citation schema for agent responses."""
    url: str = Field(..., description="URL of the citation")
    title: str = Field(..., description="Title of the source")
    description: str = Field("", description="Description or snippet")
    formatted_citation: str = Field(..., description="APA-formatted citation text")
    source_type: str = Field("unknown", description="Type of source")
    access_date: str = Field("", description="Access date string")


class SimpleEvidenceSchema(BaseModel):
    """Simplified evidence schema for agent responses."""
    evidence_text: str = Field(..., description="Text content of evidence")
    source_url: str = Field(..., description="Source URL")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score")
    supports_claim: bool = Field(True, description="Whether evidence supports claim")
    confidence_level: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in evidence")


# Override aliases for backward compatibility
CitationSchema = SimpleCitationSchema
EvidenceSchema = SimpleEvidenceSchema

# Export all schemas
__all__ = [
    'CitationType',
    'SearchProvider', 
    'CitationSource',
    'FormattedCitation',
    'SearchResult',
    'SearchQuery',
    'SearchResponse',
    'EvidenceItem',
    'TextSpan',
    'CitationRequest',
    'CitationResponse',
    'RateLimitInfo',
    'APIError',
    'SearchMetrics',
    'CitationConfig',
    'CitationSchema',
    'EvidenceSchema',
    'SimpleCitationSchema',
    'SimpleEvidenceSchema'
]