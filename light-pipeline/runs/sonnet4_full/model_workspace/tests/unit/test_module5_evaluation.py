"""Unit tests for Module 5: Evaluation and Metrics components."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time
from datetime import datetime
from typing import List, Dict, Any

# Import the components to test
from src.evaluation.metrics_calculator import (
    MetricsCalculator, SystemPerformanceMetrics, MetricResult, MetricType
)
from src.evaluation.baseline_comparator import (
    BaselineComparator, ComparisonAnalysis, BaselineResult, BaselineType, ComparisonMethod
)
from src.evaluation.accuracy_evaluator import (
    AccuracyEvaluator, AccuracyResult, EvaluationSummary, GroundTruthEntry, 
    EvaluationMetric, GroundTruthType
)
from src.evaluation.logging_system import (
    LoggingSystem, ExperimentRun, LogEntry, LogLevel, ExperimentPhase
)

# Import supporting types
from src.agents.answering_agent import AgentResponse
from src.agents.challenger_agent import ChallengeReport, Challenge, ChallengeType
from src.challenge.challenge_processor import ProcessedAnalysis, AnalysisResult, AnalysisType
from src.challenge.revision_manager import RevisionSession, RevisionStatus
from src.schemas.citation_schemas import CitationSchema, EvidenceSchema


class TestModule5Evaluation:
    """Test suite for Module 5: Evaluation and Metrics components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        
        # Create test data
        self.test_claim = "Exercise reduces cardiovascular disease risk"
        
        # Create mock citations and evidence
        self.mock_citation = CitationSchema(
            url="https://example.com/study",
            title="Exercise and Heart Health",
            description="Comprehensive study on exercise benefits",
            formatted_citation="Author, A. (2023). Exercise and Heart Health. Journal of Health.",
            source_type="academic",
            access_date="2024-01-01"
        )
        
        self.mock_evidence = EvidenceSchema(
            evidence_text="Regular exercise reduces cardiovascular disease risk by 30%",
            source_url="https://example.com/study",
            relevance_score=0.9,
            quality_score=0.8,
            supports_claim=True,
            confidence_level=0.85
        )
        
        # Create mock agent response
        self.mock_agent_response = AgentResponse(
            agent_id="test_agent",
            claim=self.test_claim,
            answer="The claim is SUPPORTED by evidence",
            citations=[self.mock_citation],
            evidence=[self.mock_evidence],
            confidence_score=0.8,
            reasoning="Based on multiple studies",
            token_usage=150,
            processing_time=2.0
        )
        
        # Create mock challenge
        self.mock_challenge = Challenge(
            challenge_type=ChallengeType.WEAK_CITATION,
            description="Citation quality could be improved",
            severity=0.6,
            affected_claims=["exercise claim"],
            suggested_improvement="Add more authoritative sources"
        )
        
        self.mock_challenge_report = ChallengeReport(
            challenger_id="test_challenger",
            original_response=self.mock_agent_response,
            challenges=[self.mock_challenge],
            overall_assessment="Minor issues identified",
            confidence_in_challenges=0.7,
            requires_revision=False,
            priority_challenges=[],
            token_usage=100,
            processing_time=1.5
        )
        
        # Create mock processed analysis
        self.mock_processed_analysis = ProcessedAnalysis(
            session_id="test_session",
            original_claim=self.test_claim,
            agent_responses=[self.mock_agent_response],
            challenge_reports=[self.mock_challenge_report],
            analysis_results={},
            total_issues=1,
            critical_issues=0,
            moderate_issues=1,
            minor_issues=0,
            needs_major_revision=False,
            needs_moderate_revision=False,
            quality_score=0.8,
            confidence_in_analysis=0.7,
            processing_time=1.0,
            timestamp="2024-01-01T00:00:00"
        )
        
        # Create mock revision session
        self.mock_revision_session = RevisionSession(
            session_id="test_session",
            original_analysis=self.mock_processed_analysis,
            revision_plan=None,
            revised_responses=[],
            overall_improvement=0.1,
            revision_success=True,
            processing_time=1.0,
            revision_status=RevisionStatus.COMPLETED,
            completion_timestamp="2024-01-01T00:00:00",
            summary_report="Minor improvements made"
        )
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metrics_calculator_initialization(self):
        """Test MetricsCalculator initialization and configuration."""
        
        calculator = MetricsCalculator(
            confidence_level=0.95,
            min_sample_size=10,
            enable_statistical_tests=True
        )
        
        assert calculator.confidence_level == 0.95
        assert calculator.min_sample_size == 10
        assert calculator.enable_statistical_tests == True
        assert calculator.total_calculations == 0
        assert calculator.calculation_errors == 0
        assert len(calculator.calculated_metrics) == 0
        assert len(calculator.performance_history) == 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_system_performance_calculation(self):
        """Test comprehensive system performance metrics calculation."""
        
        calculator = MetricsCalculator()
        
        # Prepare test data
        agent_responses = [self.mock_agent_response]
        challenge_reports = [self.mock_challenge_report]
        processed_analyses = [self.mock_processed_analysis]
        revision_sessions = [self.mock_revision_session]
        
        # Calculate performance metrics
        performance = await calculator.calculate_system_performance(
            agent_responses=agent_responses,
            challenge_reports=challenge_reports,
            processed_analyses=processed_analyses,
            revision_sessions=revision_sessions
        )
        
        # Verify performance metrics structure
        assert isinstance(performance, SystemPerformanceMetrics)
        assert 0.0 <= performance.overall_accuracy <= 1.0
        assert 0.0 <= performance.citation_accuracy <= 1.0
        assert 0.0 <= performance.evidence_accuracy <= 1.0
        assert 0.0 <= performance.confidence_calibration_score <= 1.0
        
        # Verify challenge metrics
        assert 0.0 <= performance.challenge_precision <= 1.0
        assert 0.0 <= performance.challenge_recall <= 1.0
        assert 0.0 <= performance.challenge_f1 <= 1.0
        assert 0.0 <= performance.false_positive_rate <= 1.0
        
        # Verify quality metrics
        assert 0.0 <= performance.avg_response_quality <= 1.0
        assert 0.0 <= performance.avg_citation_quality <= 1.0
        assert 0.0 <= performance.avg_evidence_strength <= 1.0
        
        # Verify efficiency metrics
        assert performance.avg_processing_time >= 0.0
        assert performance.token_efficiency >= 0.0
        assert performance.throughput_per_minute >= 0.0
        
        # Verify improvement metrics
        assert 0.0 <= performance.revision_success_rate <= 1.0
        assert performance.avg_improvement_score >= 0.0
        assert 0.0 <= performance.issue_resolution_rate <= 1.0
        
        # Verify metadata
        assert performance.total_evaluations == 1
        assert isinstance(performance.timestamp, str)
    
    @pytest.mark.unit
    def test_baseline_comparator_initialization(self):
        """Test BaselineComparator initialization and baseline setup."""
        
        comparator = BaselineComparator(
            confidence_level=0.95,
            min_effect_size=0.3,
            enable_bootstrap=True,
            bootstrap_samples=1000
        )
        
        assert comparator.confidence_level == 0.95
        assert comparator.min_effect_size == 0.3
        assert comparator.enable_bootstrap == True
        assert comparator.bootstrap_samples == 1000
        assert comparator.total_comparisons == 0
        assert comparator.successful_comparisons == 0
        assert len(comparator.comparison_history) == 0
        
        # Verify baseline implementations are set up
        assert BaselineType.RANDOM_BASELINE in comparator.baseline_implementations
        assert BaselineType.SIMPLE_SEARCH in comparator.baseline_implementations
        assert BaselineType.NO_CHALLENGE in comparator.baseline_implementations
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_baseline_comparison(self):
        """Test baseline comparison against system performance."""
        
        comparator = BaselineComparator()
        
        # Create mock system performance
        system_performance = SystemPerformanceMetrics(
            overall_accuracy=0.85,
            citation_accuracy=0.80,
            evidence_accuracy=0.75,
            confidence_calibration_score=0.70,
            challenge_precision=0.65,
            challenge_recall=0.60,
            challenge_f1=0.62,
            false_positive_rate=0.15,
            avg_response_quality=0.75,
            avg_citation_quality=0.70,
            avg_evidence_strength=0.72,
            avg_processing_time=2.5,
            token_efficiency=0.8,
            throughput_per_minute=20.0,
            revision_success_rate=0.80,
            avg_improvement_score=0.15,
            issue_resolution_rate=0.70,
            total_evaluations=100,
            evaluation_period="100 responses",
            timestamp="2024-01-01T00:00:00"
        )
        
        test_claims = ["claim1", "claim2", "claim3"]
        system_responses = [self.mock_agent_response] * 3
        
        # Run comparison
        analysis = await comparator.compare_with_baselines(
            system_performance=system_performance,
            test_claims=test_claims,
            system_responses=system_responses,
            baseline_types=[BaselineType.RANDOM_BASELINE, BaselineType.SIMPLE_SEARCH]
        )
        
        # Verify analysis structure
        assert isinstance(analysis, ComparisonAnalysis)
        assert analysis.system_performance == system_performance
        assert len(analysis.baseline_performance) >= 1
        
        # Verify comparison metrics
        assert isinstance(analysis.accuracy_comparison.system_score, float)
        assert isinstance(analysis.accuracy_comparison.baseline_score, float)
        assert isinstance(analysis.accuracy_comparison.improvement_percentage, float)
        
        # Verify summary data
        assert analysis.sample_size == len(test_claims)
        assert isinstance(analysis.overall_improvement, float)
        assert isinstance(analysis.significant_improvements, list)
        assert isinstance(analysis.areas_for_improvement, list)
    
    @pytest.mark.unit
    def test_accuracy_evaluator_initialization(self):
        """Test AccuracyEvaluator initialization and configuration."""
        
        evaluator = AccuracyEvaluator(
            confidence_threshold=0.5,
            strict_citation_matching=False,
            enable_semantic_matching=True
        )
        
        assert evaluator.confidence_threshold == 0.5
        assert evaluator.strict_citation_matching == False
        assert evaluator.enable_semantic_matching == True
        assert evaluator.total_evaluations == 0
        assert evaluator.correct_evaluations == 0
        assert evaluator.evaluation_errors == 0
        assert len(evaluator.evaluation_history) == 0
        assert len(evaluator.individual_results) == 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_response_accuracy_evaluation(self):
        """Test accuracy evaluation of agent responses."""
        
        evaluator = AccuracyEvaluator()
        
        # Create ground truth
        ground_truth = [
            GroundTruthEntry(
                claim=self.test_claim,
                correct_answer="SUPPORTED",
                supports_claim=True,
                confidence_level=0.9,
                authoritative_sources=["https://example.com/study"],
                expert_reasoning="Well-established medical consensus",
                ground_truth_type=GroundTruthType.EXPERT_ANNOTATION,
                metadata={"domain": "health"}
            )
        ]
        
        agent_responses = [self.mock_agent_response]
        
        # Evaluate accuracy
        summary = await evaluator.evaluate_response_accuracy(
            agent_responses=agent_responses,
            ground_truth=ground_truth
        )
        
        # Verify evaluation summary
        assert isinstance(summary, EvaluationSummary)
        assert 0.0 <= summary.overall_accuracy <= 1.0
        assert 0.0 <= summary.precision <= 1.0
        assert 0.0 <= summary.recall <= 1.0
        assert 0.0 <= summary.f1_score <= 1.0
        assert summary.total_items == 1
        assert summary.correct_items >= 0
        assert isinstance(summary.accuracy_by_category, dict)
        assert isinstance(summary.error_distribution, dict)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_confidence_calibration_evaluation(self):
        """Test confidence calibration assessment."""
        
        evaluator = AccuracyEvaluator()
        
        # Create responses with different confidence levels
        high_conf_response = AgentResponse(
            agent_id="high_conf",
            claim="High confidence claim",
            answer="SUPPORTED",
            citations=[self.mock_citation],
            evidence=[self.mock_evidence],
            confidence_score=0.9,
            reasoning="Strong evidence",
            token_usage=150,
            processing_time=2.0
        )
        
        low_conf_response = AgentResponse(
            agent_id="low_conf",
            claim="Low confidence claim",
            answer="CONTRADICTED",
            citations=[],
            evidence=[],
            confidence_score=0.3,
            reasoning="Limited evidence",
            token_usage=50,
            processing_time=1.0
        )
        
        agent_responses = [high_conf_response, low_conf_response]
        
        # Create corresponding ground truth
        ground_truth = [
            GroundTruthEntry(
                claim="High confidence claim",
                correct_answer="SUPPORTED",
                supports_claim=True,
                confidence_level=0.95,
                authoritative_sources=[],
                expert_reasoning="Correct",
                ground_truth_type=GroundTruthType.EXPERT_ANNOTATION,
                metadata={}
            ),
            GroundTruthEntry(
                claim="Low confidence claim",
                correct_answer="CONTRADICTED",
                supports_claim=False,
                confidence_level=0.9,
                authoritative_sources=[],
                expert_reasoning="Correct",
                ground_truth_type=GroundTruthType.EXPERT_ANNOTATION,
                metadata={}
            )
        ]
        
        # Evaluate confidence calibration
        calibration = await evaluator.evaluate_confidence_calibration(
            agent_responses=agent_responses,
            ground_truth=ground_truth
        )
        
        # Verify calibration metrics
        assert isinstance(calibration, dict)
        assert "overall_calibration" in calibration
        assert 0.0 <= calibration["overall_calibration"] <= 1.0
        
        # Check confidence bins
        for bin_name in ["very_low", "low", "moderate", "high", "very_high"]:
            assert f"{bin_name}_accuracy" in calibration
            assert f"{bin_name}_count" in calibration
    
    @pytest.mark.unit
    def test_logging_system_initialization(self):
        """Test LoggingSystem initialization and setup."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logging_system = LoggingSystem(
                base_log_dir=temp_dir,
                experiment_name="test_experiment",
                enable_file_logging=True,
                enable_structured_logging=True
            )
            
            assert logging_system.base_log_dir == Path(temp_dir)
            assert logging_system.experiment_name == "test_experiment"
            assert logging_system.enable_file_logging == True
            assert logging_system.enable_structured_logging == True
            assert logging_system.current_run is None
            assert len(logging_system.log_entries) == 0
            assert logging_system.entry_count == 0
    
    @pytest.mark.unit
    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle with logging."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logging_system = LoggingSystem(base_log_dir=temp_dir)
            
            # Start experiment
            run_id = logging_system.start_experiment(
                experiment_name="test_lifecycle",
                system_config={"param1": "value1"},
                test_dataset={"size": 100}
            )
            
            # Verify experiment started
            assert logging_system.current_run is not None
            assert logging_system.current_run.run_id == run_id
            assert logging_system.current_run.status == "running"
            assert logging_system.current_run.success == False
            
            # Log some events
            logging_system.log_event(
                level=LogLevel.INFO,
                phase=ExperimentPhase.AGENT_PROCESSING,
                component="test_component",
                event_type="test_event",
                message="Test event occurred",
                data={"test_data": "test_value"}
            )
            
            # End experiment
            logging_system.end_experiment(success=True)
            
            # Verify experiment ended
            assert logging_system.current_run.status == "completed"
            assert logging_system.current_run.success == True
            assert logging_system.current_run.end_time is not None
            assert logging_system.current_run.duration is not None
            assert len(logging_system.current_run.log_entries) > 0
    
    @pytest.mark.unit
    def test_performance_data_logging(self):
        """Test logging of performance metrics and analysis results."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logging_system = LoggingSystem(base_log_dir=temp_dir)
            
            run_id = logging_system.start_experiment(
                experiment_name="performance_test",
                system_config={},
                test_dataset={}
            )
            
            # Log performance metrics
            performance_metrics = SystemPerformanceMetrics(
                overall_accuracy=0.85,
                citation_accuracy=0.80,
                evidence_accuracy=0.75,
                confidence_calibration_score=0.70,
                challenge_precision=0.65,
                challenge_recall=0.60,
                challenge_f1=0.62,
                false_positive_rate=0.15,
                avg_response_quality=0.75,
                avg_citation_quality=0.70,
                avg_evidence_strength=0.72,
                avg_processing_time=2.5,
                token_efficiency=0.8,
                throughput_per_minute=20.0,
                revision_success_rate=0.80,
                avg_improvement_score=0.15,
                issue_resolution_rate=0.70,
                total_evaluations=100,
                evaluation_period="100 responses",
                timestamp="2024-01-01T00:00:00"
            )
            
            logging_system.log_performance_metrics(performance_metrics)
            
            # Verify metrics were stored
            assert logging_system.current_run.performance_metrics == performance_metrics
            
            # Log agent response
            logging_system.log_agent_response(self.mock_agent_response)
            
            # Log challenge report
            logging_system.log_challenge_report(self.mock_challenge_report)
            
            # Log revision session
            logging_system.log_revision_session(self.mock_revision_session)
            
            # Verify events were logged
            assert len(logging_system.log_entries) >= 4  # At least 4 events logged
            
            logging_system.end_experiment(success=True)
    
    @pytest.mark.unit
    def test_error_logging(self):
        """Test error logging and tracking."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logging_system = LoggingSystem(base_log_dir=temp_dir)
            
            run_id = logging_system.start_experiment(
                experiment_name="error_test",
                system_config={},
                test_dataset={}
            )
            
            # Log an error
            test_error = ValueError("Test error message")
            logging_system.log_error(
                component="test_component",
                error=test_error,
                phase=ExperimentPhase.AGENT_PROCESSING,
                additional_data={"context": "test_context"}
            )
            
            # Verify error was logged
            assert logging_system.current_run.error_count == 1
            assert "test_component" in logging_system.error_tracking
            assert logging_system.error_tracking["test_component"] == 1
            
            # Check log entry
            error_entries = [
                entry for entry in logging_system.log_entries 
                if entry.level == LogLevel.ERROR
            ]
            assert len(error_entries) == 1
            assert error_entries[0].component == "test_component"
            assert "Test error message" in error_entries[0].message
            
            logging_system.end_experiment(success=False)
    
    @pytest.mark.unit
    def test_metrics_trend_calculation(self):
        """Test calculation of metric trends over time."""
        
        calculator = MetricsCalculator()
        
        # Add some mock performance history
        for i in range(5):
            performance = SystemPerformanceMetrics(
                overall_accuracy=0.7 + i * 0.05,  # Increasing trend
                citation_accuracy=0.8,
                evidence_accuracy=0.75,
                confidence_calibration_score=0.70,
                challenge_precision=0.65,
                challenge_recall=0.60,
                challenge_f1=0.62,
                false_positive_rate=0.15,
                avg_response_quality=0.75,
                avg_citation_quality=0.70,
                avg_evidence_strength=0.72,
                avg_processing_time=2.5,
                token_efficiency=0.8,
                throughput_per_minute=20.0,
                revision_success_rate=0.80,
                avg_improvement_score=0.15,
                issue_resolution_rate=0.70,
                total_evaluations=100,
                evaluation_period="100 responses",
                timestamp=f"2024-01-0{i+1}T00:00:00"
            )
            calculator.performance_history.append(performance)
        
        # Calculate trends
        trends = calculator.calculate_metric_trends("overall_accuracy", lookback_periods=5)
        
        # Verify trend structure
        assert "trend" in trends
        assert "variance" in trends
        assert "latest_value" in trends
        assert "mean_value" in trends
        assert "sample_size" in trends
        
        # Should show positive trend (increasing accuracy)
        assert trends["trend"] > 0
        assert trends["sample_size"] == 5
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_all_module5_tests(self):
        """Comprehensive test runner for all Module 5 functionality."""
        
        try:
            # Test 1: Component initialization
            calculator = MetricsCalculator()
            comparator = BaselineComparator()
            evaluator = AccuracyEvaluator()
            
            assert calculator is not None
            assert comparator is not None
            assert evaluator is not None
            
            # Test 2: Basic functionality
            calc_stats = calculator.get_calculator_statistics()
            comp_stats = comparator.get_comparator_statistics()
            eval_stats = evaluator.get_evaluator_statistics()
            
            assert isinstance(calc_stats, dict)
            assert isinstance(comp_stats, dict)
            assert isinstance(eval_stats, dict)
            
            # Test 3: Integration test with temporary logging
            with tempfile.TemporaryDirectory() as temp_dir:
                logging_system = LoggingSystem(base_log_dir=temp_dir)
                
                run_id = logging_system.start_experiment(
                    experiment_name="integration_test",
                    system_config={"test": True},
                    test_dataset={"size": 1}
                )
                
                # Test metrics calculation
                performance = await calculator.calculate_system_performance(
                    agent_responses=[self.mock_agent_response],
                    challenge_reports=[self.mock_challenge_report],
                    processed_analyses=[self.mock_processed_analysis],
                    revision_sessions=[self.mock_revision_session]
                )
                
                assert isinstance(performance, SystemPerformanceMetrics)
                
                # Test baseline comparison
                analysis = await comparator.compare_with_baselines(
                    system_performance=performance,
                    test_claims=["test_claim"],
                    system_responses=[self.mock_agent_response],
                    baseline_types=[BaselineType.RANDOM_BASELINE]
                )
                
                assert isinstance(analysis, ComparisonAnalysis)
                
                # Test accuracy evaluation
                ground_truth = [
                    GroundTruthEntry(
                        claim=self.test_claim,
                        correct_answer="SUPPORTED",
                        supports_claim=True,
                        confidence_level=0.9,
                        authoritative_sources=[],
                        expert_reasoning="Test",
                        ground_truth_type=GroundTruthType.EXPERT_ANNOTATION,
                        metadata={}
                    )
                ]
                
                eval_summary = await evaluator.evaluate_response_accuracy(
                    agent_responses=[self.mock_agent_response],
                    ground_truth=ground_truth
                )
                
                assert isinstance(eval_summary, EvaluationSummary)
                
                # Log all results
                logging_system.log_performance_metrics(performance)
                logging_system.log_comparison_analysis(analysis)
                logging_system.log_accuracy_evaluation(eval_summary)
                
                logging_system.end_experiment(success=True)
                
                # Verify logging completed
                assert logging_system.current_run.status == "completed"
            
            # Test 4: Reset functionality
            calculator.reset_calculator()
            comparator.reset_comparator()
            evaluator.reset_evaluator()
            
            assert calculator.total_calculations == 0
            assert comparator.total_comparisons == 0
            assert evaluator.total_evaluations == 0
            
            print("All Module 5 tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"Module 5 test failed: {str(e)}")
            raise e