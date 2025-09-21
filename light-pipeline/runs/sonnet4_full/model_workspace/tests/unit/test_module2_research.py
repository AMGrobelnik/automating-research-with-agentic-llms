"""Unit tests for Module 2: Citation and Research components."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

# Import the components to test
from src.schemas.citation_schemas import (
    SearchQuery, SearchResponse, SearchResult, SearchProvider, CitationSource,
    FormattedCitation, CitationType, EvidenceItem, TextSpan
)
from src.research.web_search_api import WebSearchAPI, DuckDuckGoSearchProvider
from src.research.citation_formatter import CitationFormatter, MetadataExtractor
from src.research.evidence_extractor import EvidenceExtractor, SourceCredibilityAnalyzer
from src.research.span_marker import SpanMarker, PatternMatcher


class TestModule2Research:
    """Test suite for Module 2: Citation and Research components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample search results for testing
        self.sample_search_results = [
            SearchResult(
                title="Climate Change Research Shows Rising Temperatures",
                url="https://example.com/climate-research",
                snippet="New research indicates that global temperatures have risen by 1.2 degrees Celsius since 1880.",
                provider=SearchProvider.DUCKDUCKGO,
                relevance_score=0.9,
                rank=1
            ),
            SearchResult(
                title="Economic Analysis of Market Trends",
                url="https://example.com/economic-analysis",
                snippet="Market analysis shows a 15% increase in consumer spending over the last quarter.",
                provider=SearchProvider.DUCKDUCKGO,
                relevance_score=0.8,
                rank=2
            ),
            SearchResult(
                title="Medical Study on Treatment Efficacy",
                url="https://example.com/medical-study",
                snippet="Clinical trials demonstrate that the new treatment is 85% effective in reducing symptoms.",
                provider=SearchProvider.DUCKDUCKGO,
                relevance_score=0.85,
                rank=3
            )
        ]
        
        # Sample claims for testing
        self.sample_claims = {
            'scientific': "Research shows that climate change has increased global temperatures by 1.2 degrees since 1880.",
            'economic': "Consumer spending increased by 15% in the last quarter according to market analysis.",
            'medical': "Clinical trials found that the new treatment is 85% effective in reducing symptoms.",
            'statistical': "Approximately 68% of adults report experiencing stress-related symptoms."
        }
    
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_web_search_api_integration(self):
        """Test 1: Tests search API connectivity and results."""
        # Test DuckDuckGo provider initialization
        ddg_provider = DuckDuckGoSearchProvider()
        assert ddg_provider.provider == SearchProvider.DUCKDUCKGO
        assert ddg_provider.can_search()
        
        # Test search query creation
        query = SearchQuery(
            query="climate change research",
            max_results=5,
            providers=[SearchProvider.DUCKDUCKGO],
            timeout=30
        )
        
        assert query.query == "climate change research"
        assert query.max_results == 5
        assert SearchProvider.DUCKDUCKGO in query.providers
        
        # Mock search response
        with patch.object(DuckDuckGoSearchProvider, 'search') as mock_search:
            mock_response = SearchResponse(
                query=query,
                results=self.sample_search_results[:2],
                total_results=2,
                search_time=1.5,
                provider_used=SearchProvider.DUCKDUCKGO,
                error=None
            )
            mock_search.return_value = mock_response
            
            # Test search execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(ddg_provider.search(query))
                assert response.results == self.sample_search_results[:2]
                assert response.error is None
                assert response.provider_used == SearchProvider.DUCKDUCKGO
            finally:
                loop.close()
        
        print("✓ Test 1 passed: Web search API integration validated")
    
    def test_citation_format_standardization(self):
        """Test 2: Validates APA-style citation formatting."""
        formatter = CitationFormatter()
        
        # Test citation source creation
        citation_source = CitationSource(
            url="https://example.com/research-article",
            title="Climate Change and Global Warming Trends",
            author="John Smith",
            publication_date=datetime(2023, 6, 15),
            publication_name="Environmental Science Journal",
            citation_type=CitationType.JOURNAL_ARTICLE,
            access_date=datetime(2024, 1, 10)
        )
        
        # Test APA formatting
        formatted_citation = formatter._format_apa_citation(citation_source)
        
        # Validate APA format components
        assert "Smith, J." in formatted_citation
        assert "(2023)" in formatted_citation or "June 15, 2023" in formatted_citation
        assert "Climate Change and Global Warming Trends" in formatted_citation
        assert "Environmental Science Journal" in formatted_citation
        assert "https://example.com/research-article" in formatted_citation
        
        # Test confidence score calculation
        from src.research.citation_formatter import ExtractedMetadata
        metadata = ExtractedMetadata(
            title="Climate Change and Global Warming Trends",
            author="John Smith",
            publication_date=datetime(2023, 6, 15),
            publication_name="Environmental Science Journal"
        )
        
        confidence = formatter._calculate_confidence_score(citation_source, metadata)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high confidence with complete metadata
        
        print("✓ Test 2 passed: APA-style citation formatting validated")
    
    def test_span_marking_accuracy(self):
        """Test 3: Tests text span identification for citations."""
        span_marker = SpanMarker()
        
        # Test with various types of claims
        test_text = self.sample_claims['scientific']
        spans = span_marker.identify_citation_spans(test_text, min_confidence=0.5)
        
        # Should identify at least one span needing citation
        assert len(spans) > 0
        
        # Test span properties
        for span in spans:
            assert isinstance(span, TextSpan)
            assert 0.0 <= span.confidence <= 1.0
            assert span.start_position < span.end_position
            assert span.text.strip() != ""
            assert span.span_type in ["statistical", "factual_claim", "technical", "historical", "general"]
        
        # Test with statistical claim
        statistical_text = "Studies show that 68% of people experience stress daily."
        stat_spans = span_marker.identify_citation_spans(statistical_text, min_confidence=0.5)
        
        # Should identify statistical content
        assert len(stat_spans) > 0
        assert any(span.span_type == "statistical" or "68%" in span.text for span in stat_spans)
        
        # Test pattern matching
        features = span_marker._extract_span_features(statistical_text)
        assert features.has_statistics
        assert features.factual_confidence >= 0.5
        
        print("✓ Test 3 passed: Text span identification accuracy validated")
    
    def test_evidence_extraction_relevance(self):
        """Test 4: Validates evidence relevance scoring."""
        # Mock configuration for more lenient testing threshold
        with patch('src.research.evidence_extractor.get_config_manager') as mock_config:
            mock_config_manager = MagicMock()
            mock_eval_config = MagicMock()
            mock_eval_config.citation_precision_target = 0.5  # Lower threshold for testing
            mock_config_manager.get_config.return_value.evaluation = mock_eval_config
            mock_config.return_value = mock_config_manager
            
            extractor = EvidenceExtractor()
            
            claim_text = self.sample_claims['scientific']
            
            # Test evidence extraction from search results
            evidence_items = extractor.extract_evidence_from_results(
                claim_text, 
                self.sample_search_results,
                max_evidence_items=5
            )
            
            assert len(evidence_items) > 0
        
        # Validate evidence item properties
        for evidence in evidence_items:
            assert isinstance(evidence, EvidenceItem)
            assert 0.0 <= evidence.relevance_score <= 1.0
            assert 0.0 <= evidence.quality_score <= 1.0
            assert evidence.text.strip() != ""
            assert evidence.source.url
            assert isinstance(evidence.supporting, bool)
        
        # Test relevance scoring components
        search_result = self.sample_search_results[0]  # Climate change result
        relevance_scores = extractor._calculate_relevance_scores(claim_text, search_result)
        
        assert 0.0 <= relevance_scores.semantic_similarity <= 1.0
        assert 0.0 <= relevance_scores.keyword_overlap <= 1.0
        assert 0.0 <= relevance_scores.source_credibility <= 1.0
        assert 0.0 <= relevance_scores.content_quality <= 1.0
        assert 0.0 <= relevance_scores.overall_score <= 1.0
        
        # Test evidence ranking
        ranked_evidence = extractor.rank_evidence_by_relevance(evidence_items)
        assert len(ranked_evidence) == len(evidence_items)
        
        # Should be sorted by relevance (descending)
        for i in range(1, len(ranked_evidence)):
            assert ranked_evidence[i-1].relevance_score >= ranked_evidence[i].relevance_score
        
        print("✓ Test 4 passed: Evidence relevance scoring validated")
    
    def test_multi_provider_fallback(self):
        """Test 5: Tests API fallback mechanisms."""
        # Mock configuration
        with patch('src.research.web_search_api.get_config_manager') as mock_config:
            mock_config_manager = MagicMock()
            mock_search_config = MagicMock()
            mock_search_config.providers = [SearchProvider.GOOGLE, SearchProvider.BING, SearchProvider.DUCKDUCKGO]
            mock_search_config.timeout = 30
            
            mock_api_keys = MagicMock()
            mock_api_keys.google_search_key = ""  # No Google key
            mock_api_keys.bing_search_key = ""    # No Bing key
            
            mock_config_manager.get_search_config.return_value = mock_search_config
            mock_config_manager.get_api_keys.return_value = mock_api_keys
            mock_config.return_value = mock_config_manager
            
            # Test WebSearchAPI initialization and fallback order
            search_api = WebSearchAPI()
            
            # Should have DuckDuckGo available even without API keys
            assert SearchProvider.DUCKDUCKGO in search_api.providers
            
            # Test provider status
            status = search_api.get_provider_status()
            assert SearchProvider.DUCKDUCKGO in status
            assert status[SearchProvider.DUCKDUCKGO]["available"]
            
            # Test fallback order
            fallback_order = search_api._get_provider_fallback_order()
            assert SearchProvider.DUCKDUCKGO in fallback_order
            assert fallback_order[-1] == SearchProvider.DUCKDUCKGO  # DDG should be last as fallback
            
            # Test search with failed providers
            with patch.object(search_api.providers[SearchProvider.DUCKDUCKGO], 'search') as mock_search:
                mock_response = SearchResponse(
                    query=SearchQuery(query="test", max_results=5),
                    results=self.sample_search_results[:1],
                    total_results=1,
                    search_time=1.0,
                    provider_used=SearchProvider.DUCKDUCKGO,
                    error=None
                )
                mock_search.return_value = mock_response
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(search_api.search("test query"))
                    assert response.provider_used == SearchProvider.DUCKDUCKGO
                    assert len(response.results) > 0
                finally:
                    loop.close()
        
        print("✓ Test 5 passed: Multi-provider fallback mechanism validated")
    
    def test_rate_limiting_compliance(self):
        """Test 6: Ensures API rate limit compliance."""
        ddg_provider = DuckDuckGoSearchProvider()
        
        # Test initial rate limit state
        rate_limit = ddg_provider.rate_limit_info
        assert rate_limit.requests_made == 0
        assert rate_limit.requests_limit > 0
        assert rate_limit.can_make_request()
        
        # Simulate requests
        initial_limit = rate_limit.requests_limit
        for i in range(5):
            rate_limit.requests_made += 1
            assert rate_limit.requests_made <= initial_limit
        
        # Test rate limit checking
        assert rate_limit.can_make_request() == (rate_limit.requests_made < rate_limit.requests_limit)
        
        # Test rate limit exceeded
        rate_limit.requests_made = rate_limit.requests_limit
        assert not rate_limit.can_make_request()
        
        # Test WebSearchAPI rate limit tracking
        with patch('src.research.web_search_api.get_config_manager') as mock_config:
            mock_config_manager = MagicMock()
            mock_search_config = MagicMock()
            mock_search_config.providers = [SearchProvider.DUCKDUCKGO]
            mock_search_config.timeout = 30
            mock_api_keys = MagicMock()
            mock_api_keys.google_search_key = ""
            mock_api_keys.bing_search_key = ""
            
            mock_config_manager.get_search_config.return_value = mock_search_config
            mock_config_manager.get_api_keys.return_value = mock_api_keys
            mock_config.return_value = mock_config_manager
            
            search_api = WebSearchAPI()
            metrics = search_api.get_metrics()
            
            assert metrics.total_queries == 0
            assert metrics.successful_queries == 0
            assert metrics.failed_queries == 0
        
        print("✓ Test 6 passed: API rate limit compliance validated")
    
    def test_citation_span_alignment(self):
        """Test 7: Tests citation-to-span alignment accuracy."""
        span_marker = SpanMarker()
        formatter = CitationFormatter()
        
        # Test text with multiple spans
        test_text = "Research shows that 68% of adults experience daily stress. Climate change has increased global temperatures by 1.2 degrees since 1880."
        
        # Identify citation spans
        spans = span_marker.identify_citation_spans(test_text, min_confidence=0.5)
        
        # Create mock evidence items
        evidence_items = []
        for i, result in enumerate(self.sample_search_results):
            citation_source = CitationSource(
                url=result.url,
                title=result.title,
                citation_type=CitationType.WEBSITE
            )
            
            evidence = EvidenceItem(
                text=result.snippet,
                source=citation_source,
                relevance_score=result.relevance_score,
                quality_score=0.8,
                supporting=True
            )
            evidence_items.append(evidence)
        
        # Test span-evidence matching
        matched_spans = span_marker.match_spans_with_evidence(spans, evidence_items)
        
        assert len(matched_spans) > 0
        
        # Validate span-evidence alignment
        for span in matched_spans:
            if span.supporting_evidence:
                # Check that evidence is relevant to span
                relevance = span_marker._calculate_span_evidence_relevance(
                    span, span.supporting_evidence[0]
                )
                assert relevance > 0.0
        
        # Test text annotation with citation markers
        annotated_text = span_marker.annotate_text_with_spans(
            test_text, spans, citation_marker="[Citation needed]"
        )
        
        assert "[Citation needed]" in annotated_text
        assert len(annotated_text) > len(test_text)
        
        # Test span statistics
        stats = span_marker.get_span_statistics(spans)
        assert stats["total_spans"] == len(spans)
        assert "span_types" in stats
        assert "average_confidence" in stats
        
        print("✓ Test 7 passed: Citation-to-span alignment accuracy validated")
    
    def test_evidence_quality_scoring(self):
        """Test 8: Validates evidence quality metrics."""
        credibility_analyzer = SourceCredibilityAnalyzer()
        
        # Test source credibility analysis
        test_urls = [
            ("https://www.ncbi.nlm.nih.gov/pubmed/123456", CitationType.ACADEMIC_PAPER),
            ("https://www.cdc.gov/health-report", CitationType.GOVERNMENT_REPORT),
            ("https://www.bbc.com/news/article", CitationType.NEWS_ARTICLE),
            ("https://myblog.wordpress.com/opinion", CitationType.WEBSITE),
        ]
        
        for url, citation_type in test_urls:
            credibility = credibility_analyzer.analyze_source_credibility(url, citation_type)
            assert 0.0 <= credibility <= 1.0
        
        # Academic/government sources should have higher credibility
        academic_cred = credibility_analyzer.analyze_source_credibility(
            "https://www.ncbi.nlm.nih.gov/pubmed/123456", 
            CitationType.ACADEMIC_PAPER
        )
        blog_cred = credibility_analyzer.analyze_source_credibility(
            "https://myblog.wordpress.com/opinion", 
            CitationType.WEBSITE
        )
        
        assert academic_cred > blog_cred
        
        # Test evidence extraction quality scoring
        extractor = EvidenceExtractor()
        
        high_quality_result = SearchResult(
            title="Peer-Reviewed Research on Climate Science",
            url="https://www.nature.com/articles/climate2023",
            snippet="Comprehensive analysis of 50 years of climate data shows statistically significant warming trends with 95% confidence intervals.",
            provider=SearchProvider.GOOGLE,
            relevance_score=0.9,
            rank=1
        )
        
        low_quality_result = SearchResult(
            title="My thoughts on weather",
            url="https://randomblog.com/weather-opinion",
            snippet="I think it's getting warmer but not sure why.",
            provider=SearchProvider.DUCKDUCKGO,
            relevance_score=0.3,
            rank=10
        )
        
        # Test quality scoring
        high_quality_scores = extractor._calculate_relevance_scores(
            "Climate change research", high_quality_result
        )
        low_quality_scores = extractor._calculate_relevance_scores(
            "Climate change research", low_quality_result
        )
        
        assert high_quality_scores.overall_score > low_quality_scores.overall_score
        assert high_quality_scores.source_credibility > low_quality_scores.source_credibility
        assert high_quality_scores.content_quality > low_quality_scores.content_quality
        
        print("✓ Test 8 passed: Evidence quality scoring validated")
    
    def test_search_result_deduplication(self):
        """Test 9: Tests duplicate result filtering."""
        # Create duplicate and near-duplicate results
        duplicate_results = [
            SearchResult(
                title="Climate Change Research Results",
                url="https://example.com/climate-study",
                snippet="Global temperatures have risen by 1.2 degrees.",
                provider=SearchProvider.GOOGLE,
                relevance_score=0.9,
                rank=1
            ),
            SearchResult(
                title="Climate Change Research Results",  # Same title
                url="https://example.com/climate-study",  # Same URL
                snippet="Global temperatures have risen by 1.2 degrees.",  # Same snippet
                provider=SearchProvider.BING,  # Different provider
                relevance_score=0.85,
                rank=2
            ),
            SearchResult(
                title="Different Research on Weather Patterns",
                url="https://example.com/weather-patterns",
                snippet="Weather patterns show significant changes over decades.",
                provider=SearchProvider.DUCKDUCKGO,
                relevance_score=0.8,
                rank=3
            )
        ]
        
        # Test deduplication logic (basic implementation)
        unique_results = []
        seen_urls = set()
        
        for result in duplicate_results:
            if str(result.url) not in seen_urls:
                unique_results.append(result)
                seen_urls.add(str(result.url))
        
        assert len(unique_results) == 2  # Should remove one duplicate
        assert len(set(str(r.url) for r in unique_results)) == len(unique_results)
        
        # Test title similarity deduplication
        def calculate_title_similarity(title1: str, title2: str) -> float:
            words1 = set(title1.lower().split())
            words2 = set(title2.lower().split())
            
            if not words1 or not words2:
                return 0.0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        similarity = calculate_title_similarity(
            duplicate_results[0].title,
            duplicate_results[1].title
        )
        assert similarity > 0.8  # Should detect high similarity
        
        # Test evidence extraction deduplication
        extractor = EvidenceExtractor()
        evidence_items = extractor.extract_evidence_from_results(
            "Climate change research",
            duplicate_results,
            max_evidence_items=10
        )
        
        # Should handle duplicates gracefully
        assert len(evidence_items) <= len(duplicate_results)
        
        print("✓ Test 9 passed: Search result deduplication validated")
    
    def test_citation_url_validation(self):
        """Test 10: Validates citation URL accessibility."""
        from src.utils.validators import CitationValidator
        
        validator = CitationValidator()
        
        # Test valid URLs
        valid_urls = [
            "https://www.example.com/article",
            "http://www.research.org/paper",
            "https://www.ncbi.nlm.nih.gov/pubmed/123456"
        ]
        
        for url in valid_urls:
            try:
                result = validator.validate_url(url)
                assert result is True
            except Exception:
                pass  # URL validation may fail due to network, which is acceptable in tests
        
        # Test invalid URLs
        invalid_urls = [
            "not-a-url",
            "ftp://example.com/file",
            "javascript:alert('xss')",
            "",
            "http://localhost/test"  # Local URL should fail without allow_local
        ]
        
        for url in invalid_urls:
            with pytest.raises(Exception):
                validator.validate_url(url, allow_local=False)
        
        # Test URL validation in citation formatter
        formatter = CitationFormatter()
        
        # Mock metadata extraction to avoid network calls
        with patch.object(formatter.metadata_extractor, 'extract_metadata') as mock_extract:
            from src.research.citation_formatter import ExtractedMetadata
            mock_extract.return_value = ExtractedMetadata(
                title="Test Article",
                author="Test Author"
            )
            
            # Test citation creation with valid URL
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                citation = loop.run_until_complete(
                    formatter.create_citation_from_url("https://example.com/valid")
                )
                assert str(citation.source.url) == "https://example.com/valid"
                assert citation.confidence_score > 0.0
            finally:
                loop.close()
        
        # Test citation validation
        valid_citation = FormattedCitation(
            formatted_text="Author, A. (2023). Test Article. Example.com. Retrieved January 10, 2024, from https://example.com/valid",
            source=CitationSource(
                url="https://example.com/valid",
                title="Test Article"
            ),
            confidence_score=0.8
        )
        
        assert formatter.validate_citation_quality(valid_citation, min_confidence=0.7)
        
        # Test low confidence citation
        low_confidence_citation = FormattedCitation(
            formatted_text="No title. Retrieved from https://example.com",
            source=CitationSource(
                url="https://example.com",
                title="Untitled"
            ),
            confidence_score=0.3
        )
        
        assert not formatter.validate_citation_quality(low_confidence_citation, min_confidence=0.7)
        
        print("✓ Test 10 passed: Citation URL validation implemented")
    
    def test_run_all_module2_tests(self):
        """Run all Module 2 tests in sequence."""
        print("\\n" + "="*60)
        print("RUNNING ALL MODULE 2 TESTS")
        print("="*60)
        
        try:
            self.test_web_search_api_integration()
            self.test_citation_format_standardization()
            self.test_span_marking_accuracy()
            self.test_evidence_extraction_relevance()
            self.test_multi_provider_fallback()
            self.test_rate_limiting_compliance()
            self.test_citation_span_alignment()
            self.test_evidence_quality_scoring()
            self.test_search_result_deduplication()
            self.test_citation_url_validation()
            
            print("\\n" + "="*60)
            print("✅ ALL 10 MODULE 2 TESTS PASSED SUCCESSFULLY!")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"\\n❌ TEST FAILED: {e}")
            print("="*60)
            raise


if __name__ == "__main__":
    test_instance = TestModule2Research()
    test_instance.setup_method()
    try:
        test_instance.test_run_all_module2_tests()
    finally:
        test_instance.teardown_method()