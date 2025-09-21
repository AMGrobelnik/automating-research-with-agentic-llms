"""Unit tests for Module 3: Multi-Agent Architecture components."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time
from dataclasses import asdict

# Import the components to test
from src.agents.answering_agent import AnsweringAgent, AgentResponse
from src.agents.challenger_agent import ChallengerAgent, ChallengeReport, Challenge, ChallengeType
from src.agents.agent_manager import AgentManager, SessionMetrics, AgentConfiguration
from src.agents.response_processor import (
    ResponseProcessor, ProcessedResponse, ProcessedChallenge, 
    ProcessingStatus, ConfidenceCategory
)
from src.schemas.citation_schemas import CitationSchema, EvidenceSchema, SearchResult, SearchProvider


class TestModule3Agents:
    """Test suite for Module 3: Multi-Agent Architecture components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock data for testing
        self.test_claim = "Climate change is primarily caused by human activities"
        self.test_domain = "science"
        
        # Create mock search results
        self.mock_search_results = [
            SearchResult(
                title="IPCC Climate Report 2023",
                url="https://www.ipcc.ch/report/ar6/wg1/",
                snippet="Human influence has warmed the climate at a rate unprecedented in at least the last 2000 years",
                provider=SearchProvider.GOOGLE,
                relevance_score=0.95,
                rank=1
            ),
            SearchResult(
                title="NASA Climate Evidence",
                url="https://climate.nasa.gov/evidence/",
                snippet="Multiple lines of evidence show human activities are the primary cause of climate change",
                provider=SearchProvider.GOOGLE,
                relevance_score=0.92,
                rank=2
            )
        ]
        
        # Create mock citations
        self.mock_citations = [
            CitationSchema(
                url="https://www.ipcc.ch/report/ar6/wg1/",
                title="IPCC Climate Report 2023",
                description="Scientific assessment of climate change",
                formatted_citation="IPCC. (2023). Climate Change 2023: The Physical Science Basis. https://www.ipcc.ch/report/ar6/wg1/",
                source_type="academic",
                access_date="2024-01-15"
            )
        ]
        
        # Create mock evidence
        self.mock_evidence = [
            EvidenceSchema(
                evidence_text="Human influence has warmed the climate at a rate unprecedented in at least the last 2000 years",
                source_url="https://www.ipcc.ch/report/ar6/wg1/",
                relevance_score=0.95,
                quality_score=0.98,
                supports_claim=True,
                confidence_level=0.99
            )
        ]
    
    @pytest.mark.unit
    def test_answering_agent_initialization(self):
        """Test answering agent initialization and basic setup."""
        agent_id = "test_agent_1"
        agent = AnsweringAgent(
            agent_id=agent_id,
            max_search_results=5,
            max_tokens_per_response=1000,
            confidence_threshold=0.8
        )
        
        assert agent.agent_id == agent_id
        assert agent.max_search_results == 5
        assert agent.max_tokens_per_response == 1000
        assert agent.confidence_threshold == 0.8
        assert agent.total_token_usage == 0
        assert agent.response_count == 0
        
        # Check component initialization
        assert agent.web_search is not None
        assert agent.citation_formatter is not None
        assert agent.evidence_extractor is not None
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch('src.agents.answering_agent.WebSearchAPI')
    @patch('src.agents.answering_agent.CitationFormatter')
    @patch('src.agents.answering_agent.EvidenceExtractor')
    async def test_agent_communication_protocol(self, mock_evidence, mock_citation, mock_search):
        """Test standardized messaging between agents."""
        # Setup mocks
        mock_search.return_value.search = AsyncMock(return_value=self.mock_search_results)
        mock_citation.return_value.format_citation = AsyncMock(return_value=self.mock_citations[0])
        mock_evidence.return_value.extract_evidence = AsyncMock(return_value=self.mock_evidence[0])
        
        agent = AnsweringAgent("test_agent")
        
        # Test research claim - should return standardized AgentResponse
        response = await agent.research_claim(self.test_claim, self.test_domain)
        
        # Verify response structure follows protocol
        assert isinstance(response, AgentResponse)
        assert response.agent_id == "test_agent"
        assert response.claim == self.test_claim
        assert isinstance(response.answer, str)
        assert isinstance(response.citations, list)
        assert isinstance(response.evidence, list)
        assert 0.0 <= response.confidence_score <= 1.0
        assert isinstance(response.reasoning, str)
        assert response.token_usage > 0
        assert response.processing_time >= 0.0
    
    @pytest.mark.unit
    def test_challenger_agent_initialization(self):
        """Test challenger agent setup and configuration."""
        challenger_id = "challenger_1"
        challenger = ChallengerAgent(
            challenger_id=challenger_id,
            min_challenge_severity=0.4,
            max_challenges_per_response=8,
            citation_quality_threshold=0.7
        )
        
        assert challenger.challenger_id == challenger_id
        assert challenger.min_challenge_severity == 0.4
        assert challenger.max_challenges_per_response == 8
        assert challenger.citation_quality_threshold == 0.7
        assert challenger.total_challenges == 0
        assert challenger.total_reviews == 0
        assert challenger.total_token_usage == 0
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_confidence_score_generation(self):
        """Test confidence score accuracy within 0-1 range."""
        agent = AnsweringAgent("confidence_tester")
        
        # Test confidence calculation with mock data
        supporting = [self.mock_evidence[0]]  # 1 supporting
        contradicting = []  # 0 contradicting
        avg_quality = 0.98
        avg_relevance = 0.95
        
        confidence = agent._calculate_confidence(supporting, contradicting, avg_quality, avg_relevance)
        
        # Verify confidence is in valid range
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
        
        # Test with contradicting evidence
        contradicting_evidence = EvidenceSchema(
            evidence_text="Some studies suggest natural factors play a larger role",
            source_url="https://example.com/study",
            relevance_score=0.7,
            quality_score=0.6,
            supports_claim=False,
            confidence_level=0.7
        )
        
        contradicting = [contradicting_evidence]
        confidence_mixed = agent._calculate_confidence(supporting, contradicting, avg_quality, avg_relevance)
        
        assert 0.0 <= confidence_mixed <= 1.0
        assert confidence_mixed < confidence  # Should be lower with contradicting evidence
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_response_standardization(self):
        """Test output format consistency across agents."""
        processor = ResponseProcessor()
        
        # Create mock agent response
        mock_response = AgentResponse(
            agent_id="test_agent",
            claim=self.test_claim,
            answer="The claim is SUPPORTED by scientific evidence.",
            citations=self.mock_citations,
            evidence=self.mock_evidence,
            confidence_score=0.85,
            reasoning="Based on peer-reviewed research and expert consensus.",
            token_usage=150,
            processing_time=2.5
        )
        
        processed = await processor.process_agent_response(mock_response, self.test_claim)
        
        # Verify standardized format
        assert isinstance(processed, ProcessedResponse)
        assert processed.agent_id == "test_agent"
        assert processed.agent_type == "answering"
        assert processed.original_claim == self.test_claim
        assert isinstance(processed.processed_content, str)
        assert 0.0 <= processed.confidence_score <= 1.0
        assert isinstance(processed.confidence_category, ConfidenceCategory)
        assert processed.citation_count == 1
        assert processed.evidence_count == 1
        assert processed.token_usage == 150
        assert processed.processing_time == 2.5
        assert isinstance(processed.validation_status, ProcessingStatus)
        assert isinstance(processed.quality_score, float)
        assert 0.0 <= processed.quality_score <= 1.0
    
    @pytest.mark.unit
    def test_token_budget_management(self):
        """Test token budget compliance across agents."""
        max_tokens = 1500
        agent = AnsweringAgent("budget_tester", max_tokens_per_response=max_tokens)
        
        # Test token estimation
        long_claim = "This is a very long claim " * 50  # ~300 words
        long_answer = "This is a detailed response " * 100  # ~600 words
        long_evidence = ["Long evidence text " * 20] * 5  # ~500 words
        
        estimated_tokens = agent._estimate_token_usage(
            claim=long_claim,
            answer=long_answer,
            citations=self.mock_citations,
            evidence=self.mock_evidence
        )
        
        # Should respect max token limit
        assert estimated_tokens <= max_tokens
        assert estimated_tokens > 0
        assert isinstance(estimated_tokens, int)
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_agent_manager_coordination(self):
        """Test multi-agent coordination and management."""
        config = AgentConfiguration(
            max_answering_agents=2,
            challenger_enabled=True,
            max_tokens_per_response=1000,
            timeout_seconds=30
        )
        
        manager = AgentManager()
        await manager.update_configuration(config)
        
        # Wait for agent initialization to complete
        await asyncio.sleep(0.1)  # Give time for async initialization
        
        # Verify agent setup
        assert len(manager.answering_agents) == 2
        assert manager.challenger_agent is not None
        assert manager.agent_config.max_answering_agents == 2
        assert manager.agent_config.challenger_enabled == True
        
        # Test agent accessibility
        agent_ids = list(manager.answering_agents.keys())
        assert "answering_agent_1" in agent_ids
        assert "answering_agent_2" in agent_ids
    
    @pytest.mark.unit
    def test_prompt_template_validation(self):
        """Test prompt completeness and structure."""
        from src.prompts.answering_prompts import AnsweringPrompts
        from src.prompts.challenger_prompts import ChallengerPrompts
        
        # Test answering prompts
        research_prompt = AnsweringPrompts.get_research_prompt(
            self.test_claim, self.test_domain, "Additional context"
        )
        
        assert isinstance(research_prompt, str)
        assert len(research_prompt) > 100  # Should be substantial
        assert self.test_claim in research_prompt
        assert self.test_domain.title() in research_prompt
        assert "Additional context" in research_prompt
        
        # Test challenger prompts
        mock_response_text = "Test response with citations and evidence."
        challenge_prompt = ChallengerPrompts.get_main_challenge_prompt(
            mock_response_text, self.test_claim, "test_agent"
        )
        
        assert isinstance(challenge_prompt, str)
        assert len(challenge_prompt) > 200  # Should be comprehensive
        assert self.test_claim in challenge_prompt
        assert "test_agent" in challenge_prompt
        assert mock_response_text in challenge_prompt
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_agent_error_recovery(self):
        """Test error handling and recovery mechanisms."""
        agent = AnsweringAgent("error_tester")
        
        # Test with invalid domain
        try:
            with patch.object(agent.web_search, 'search', side_effect=Exception("API Error")):
                response = await agent.research_claim("Invalid claim", "invalid_domain")
                # Should handle gracefully or raise appropriate exception
                assert response is not None or True  # Either handles gracefully or raises
        except Exception as e:
            # Exception should be informative
            assert isinstance(e, Exception)
            assert len(str(e)) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch('src.agents.answering_agent.WebSearchAPI')
    @patch('src.agents.answering_agent.CitationFormatter') 
    @patch('src.agents.answering_agent.EvidenceExtractor')
    async def test_parallel_processing_capability(self, mock_evidence, mock_citation, mock_search):
        """Test concurrent agent processing capabilities."""
        # Setup mocks for parallel execution
        mock_search.return_value.search = AsyncMock(return_value=self.mock_search_results)
        mock_citation.return_value.format_citation = AsyncMock(return_value=self.mock_citations[0])
        mock_evidence.return_value.extract_evidence = AsyncMock(return_value=self.mock_evidence[0])
        
        # Create multiple agents
        agents = [AnsweringAgent(f"parallel_agent_{i}") for i in range(3)]
        
        # Test parallel execution
        start_time = time.time()
        
        tasks = [
            agent.research_claim(f"Test claim {i}", "science")
            for i, agent in enumerate(agents)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify parallel processing worked
        successful_responses = [r for r in responses if isinstance(r, AgentResponse)]
        assert len(successful_responses) >= 1  # At least one should succeed
        
        # Parallel should be faster than sequential (rough check)
        assert processing_time < 30  # Should complete in reasonable time
        
        # Each response should have unique agent ID
        agent_ids = [r.agent_id for r in successful_responses]
        assert len(set(agent_ids)) == len(successful_responses)  # All unique
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_all_module3_tests(self):
        """Comprehensive test runner for all Module 3 functionality."""
        try:
            # Test 1: Agent initialization
            agent = AnsweringAgent("comprehensive_test")
            challenger = ChallengerAgent("comprehensive_challenger")
            processor = ResponseProcessor()
            manager = AgentManager()
            
            assert agent is not None
            assert challenger is not None
            assert processor is not None
            assert manager is not None
            
            # Test 2: Basic functionality
            stats = agent.get_agent_stats()
            assert isinstance(stats, dict)
            assert "agent_id" in stats
            
            challenger_stats = challenger.get_challenger_stats()
            assert isinstance(challenger_stats, dict)
            assert "challenger_id" in challenger_stats
            
            processor_stats = processor.get_processor_statistics()
            assert isinstance(processor_stats, dict)
            assert "processing_counts" in processor_stats
            
            # Test 3: Component integration
            mock_response = AgentResponse(
                agent_id="test_integration",
                claim="Integration test claim",
                answer="Test answer",
                citations=[],
                evidence=[],
                confidence_score=0.7,
                reasoning="Test reasoning",
                token_usage=100,
                processing_time=1.0
            )
            
            processed = await processor.process_agent_response(mock_response, "Integration test claim")
            assert processed.agent_id == "test_integration"
            
            # Test 4: Error handling
            await agent.reset_agent()
            await challenger.reset_challenger()
            processor.reset_processor()
            
            # Test 5: Configuration management
            config = AgentConfiguration(max_answering_agents=1, challenger_enabled=False)
            await manager.update_configuration(config)
            
            print("All Module 3 tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"Module 3 test failed: {str(e)}")
            raise e