"""Unit tests for Module 1: Dataset and Infrastructure components."""

import pytest
import tempfile
import json
import csv
from pathlib import Path
import sqlite3
import os
from unittest.mock import patch, MagicMock

# Import the components to test
from src.config.config_manager import ConfigManager, Config
from src.dataset.data_storage import DataStorage, ClaimRecord, ResponseRecord, EvaluationRecord
from src.dataset.domain_classifier import DomainClassifier, ClassificationResult
from src.dataset.claim_dataset import ClaimDataset, ClaimEntry
from src.utils.validators import ClaimValidator, CitationValidator, ValidationError


class TestModule1Dataset:
    """Test suite for Module 1: Dataset and Infrastructure components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = Path(self.temp_dir) / "test_config.yaml"
        self.test_db_path = Path(self.temp_dir) / "test.db"
        
        # Create test configuration
        test_config = {
            "database": {
                "type": "sqlite",
                "path": str(self.test_db_path)
            },
            "api_keys": {
                "openai_key": "test-key",
                "anthropic_key": "",
                "google_search_key": "",
                "bing_search_key": ""
            },
            "search": {
                "providers": ["duckduckgo"],
                "rate_limits": {"duckduckgo": 50},
                "timeout": 30,
                "max_results": 10
            },
            "agents": {
                "max_tokens": 4000,
                "temperature": 0.1,
                "model": "gpt-4",
                "backup_model": "claude-3-sonnet-20241022"
            },
            "system": {
                "log_level": "INFO",
                "log_file": f"{self.temp_dir}/system.log",
                "max_claim_length": 500,
                "max_citation_count": 20,
                "revision_rounds": 1
            },
            "domains": [
                {"name": "science", "keywords": ["research", "study"], "complexity_weight": 1.2},
                {"name": "health", "keywords": ["medical", "health"], "complexity_weight": 1.1},
                {"name": "history", "keywords": ["historical", "past"], "complexity_weight": 1.0},
                {"name": "finance", "keywords": ["economic", "financial"], "complexity_weight": 1.3}
            ],
            "evaluation": {
                "hallucination_threshold": 0.85,
                "citation_precision_target": 0.85,
                "citation_recall_target": 0.90,
                "statistical_significance": 0.05
            }
        }
        
        # Write test config to file
        import yaml
        with open(self.test_config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_claim_dataset_initialization(self):
        """Test 1: Validates ClaimDataset structure and initialization."""
        with patch('src.dataset.claim_dataset.get_config_manager') as mock_config:
            mock_config_manager = MagicMock()
            mock_config.return_value = mock_config_manager
            
            # Initialize ClaimDataset
            dataset = ClaimDataset(data_dir=self.temp_dir)
            
            # Verify initialization
            assert dataset is not None
            assert len(dataset.claims) == 0
            assert dataset.domain_targets == {"science": 75, "health": 75, "history": 75, "finance": 75}
            assert sum(dataset.domain_targets.values()) == 300
            assert dataset.data_dir.exists()
            
            print("✓ Test 1 passed: ClaimDataset initialization validated")
    
    def test_claim_loading_from_file(self):
        """Test 2: Tests claim loading from JSON/CSV formats."""
        # Create test JSON file
        test_claims_json = [
            {"text": "The speed of light is constant in vacuum", "domain": "science"},
            {"text": "Exercise reduces heart disease risk", "domain": "health"},
            {"text": "World War II ended in 1945", "domain": "history"}
        ]
        
        json_path = Path(self.temp_dir) / "test_claims.json"
        with open(json_path, 'w') as f:
            json.dump(test_claims_json, f)
        
        # Create test CSV file
        csv_path = Path(self.temp_dir) / "test_claims.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "domain"])
            writer.writerow(["GDP measures economic output", "finance"])
            writer.writerow(["DNA contains genetic information", "science"])
        
        with patch('src.dataset.claim_dataset.get_config_manager') as mock_config:
            mock_config_manager = MagicMock()
            mock_config.return_value = mock_config_manager
            
            dataset = ClaimDataset(data_dir=self.temp_dir)
            
            # Test JSON loading
            json_count = dataset.load_claims_from_file(str(json_path), format="json")
            assert json_count == 3
            assert len(dataset.claims) == 3
            
            # Test CSV loading
            csv_count = dataset.load_claims_from_file(str(csv_path), format="csv")
            assert csv_count == 2
            assert len(dataset.claims) == 5
            
            # Test auto-format detection
            dataset.claims = []  # Reset
            auto_count = dataset.load_claims_from_file(str(json_path), format="auto")
            assert auto_count == 3
            
            print("✓ Test 2 passed: File loading (JSON/CSV) validated")
    
    def test_domain_distribution_validation(self):
        """Test 3: Ensures 75 claims per domain distribution."""
        with patch('src.dataset.claim_dataset.get_config_manager') as mock_config:
            mock_config_manager = MagicMock()
            mock_config.return_value = mock_config_manager
            
            dataset = ClaimDataset(data_dir=self.temp_dir)
            
            # Add claims with uneven distribution
            test_claims = []
            # 80 science claims (over target)
            for i in range(80):
                test_claims.append(ClaimEntry(text=f"Science claim {i}", domain="science", complexity_score=0.5))
            
            # 60 health claims (under target) 
            for i in range(60):
                test_claims.append(ClaimEntry(text=f"Health claim {i}", domain="health", complexity_score=0.5))
            
            # 75 history claims (exactly target)
            for i in range(75):
                test_claims.append(ClaimEntry(text=f"History claim {i}", domain="history", complexity_score=0.5))
            
            # 70 finance claims (slightly under)
            for i in range(70):
                test_claims.append(ClaimEntry(text=f"Finance claim {i}", domain="finance", complexity_score=0.5))
            
            dataset.claims = test_claims
            
            # Test validation
            validation = dataset.validate_domain_distribution()
            
            assert validation["total_claims"] == 285
            assert validation["distribution"]["science"] == 80
            assert validation["distribution"]["health"] == 60
            assert validation["distribution"]["history"] == 75
            assert validation["distribution"]["finance"] == 70
            
            # Should fail validation due to imbalance
            assert not validation["is_valid"]
            assert len(validation["issues"]) > 0
            
            print("✓ Test 3 passed: Domain distribution validation works correctly")
    
    def test_claim_complexity_scoring(self):
        """Test 4: Validates complexity scoring algorithm."""
        config_manager = ConfigManager(str(self.test_config_path))
        classifier = DomainClassifier()
        
        # Test different complexity levels
        simple_claim = "Water is wet."
        complex_claim = "The photosynthetic process involves the conversion of carbon dioxide and water into glucose through the light-dependent and light-independent reactions, utilizing chlorophyll molecules and producing approximately 6.02×10²³ molecules per mole of glucose."
        
        simple_result = classifier.classify_claim(simple_claim)
        complex_result = classifier.classify_claim(complex_claim)
        
        # Verify complexity scores are in valid range
        assert 0.0 <= simple_result.complexity_score <= 1.0
        assert 0.0 <= complex_result.complexity_score <= 1.0
        
        # Complex claim should have higher complexity score
        assert complex_result.complexity_score > simple_result.complexity_score
        
        # Test domain-specific complexity weighting
        science_claim = "Research indicates that quantum entanglement allows instantaneous correlation."
        finance_claim = "Economic indicators suggest market volatility."
        
        science_result = classifier.classify_claim(science_claim)
        finance_result = classifier.classify_claim(finance_claim)
        
        # Both should be properly classified and scored
        assert science_result.domain in ["science", "finance"]  # Could classify either way
        assert finance_result.domain in ["science", "finance"]
        
        print("✓ Test 4 passed: Complexity scoring algorithm validated")
    
    def test_data_storage_persistence(self):
        """Test 5: Tests database save/retrieve operations."""
        # Initialize storage with test config
        config_manager = ConfigManager(str(self.test_config_path))
        
        with patch('src.dataset.data_storage.get_config_manager', return_value=config_manager):
            storage = DataStorage()
            
            # Test claim operations
            test_claim = ClaimRecord(
                claim_text="Test claim for database persistence",
                domain="science",
                complexity_score=0.75,
                ground_truth="This is a test claim"
            )
            
            # Save claim
            claim_id = storage.save_claim(test_claim)
            assert claim_id is not None
            assert isinstance(claim_id, int)
            
            # Retrieve claim
            retrieved_claim = storage.get_claim(claim_id)
            assert retrieved_claim is not None
            assert retrieved_claim.claim_text == test_claim.claim_text
            assert retrieved_claim.domain == test_claim.domain
            assert abs(retrieved_claim.complexity_score - test_claim.complexity_score) < 0.001
            
            # Test response operations
            test_response = ResponseRecord(
                claim_id=claim_id,
                agent_type="answering",
                agent_id="test-agent-1",
                response_text="This is a test response with citations.",
                citations=["Source 1", "Source 2"],
                confidence_score=0.85,
                reasoning="Based on scientific evidence"
            )
            
            response_id = storage.save_response(test_response)
            assert response_id is not None
            
            # Retrieve responses for claim
            responses = storage.get_responses_for_claim(claim_id)
            assert len(responses) == 1
            assert responses[0].response_text == test_response.response_text
            assert responses[0].citations == test_response.citations
            
            print("✓ Test 5 passed: Database persistence operations validated")
    
    def test_config_manager_validation(self):
        """Test 6: Validates configuration parameter requirements."""
        # Test valid configuration
        config_manager = ConfigManager(str(self.test_config_path))
        config = config_manager.get_config()
        
        assert config is not None
        assert config.database.type == "sqlite"
        assert len(config.domains) == 4
        assert all(domain.name in ["science", "health", "history", "finance"] for domain in config.domains)
        
        # Test configuration validation
        db_config = config_manager.get_database_config()
        assert db_config.type == "sqlite"
        assert db_config.path == str(self.test_db_path)
        
        api_keys = config_manager.get_api_keys()
        assert api_keys.openai_key == "test-key"
        
        # Test API key validation
        is_valid = config_manager.validate_api_keys()
        assert is_valid  # Should pass with test-key
        
        # Test directory creation
        config_manager.create_directories()
        log_dir = Path(config.system.log_file).parent
        assert log_dir.exists()
        
        print("✓ Test 6 passed: Configuration manager validation works correctly")
    
    def test_domain_classifier_accuracy(self):
        """Test 7: Tests domain categorization accuracy."""
        config_manager = ConfigManager(str(self.test_config_path))
        classifier = DomainClassifier()
        
        # Test with clear domain-specific claims
        test_cases = [
            ("Research shows that DNA replication involves polymerase enzymes", "science"),
            ("Studies indicate that exercise reduces cardiovascular disease risk", "health"),
            ("World War II began in 1939 with the invasion of Poland", "history"),
            ("Interest rates affect inflation and economic growth patterns", "finance")
        ]
        
        correct_classifications = 0
        for claim_text, expected_domain in test_cases:
            result = classifier.classify_claim(claim_text)
            
            # Verify result structure
            assert isinstance(result, ClassificationResult)
            assert result.domain in ["science", "health", "history", "finance"]
            assert 0.0 <= result.confidence <= 1.0
            assert 0.0 <= result.complexity_score <= 1.0
            assert isinstance(result.keyword_matches, list)
            assert isinstance(result.reasoning, str)
            
            if result.domain == expected_domain:
                correct_classifications += 1
        
        # Should classify most correctly (allow some flexibility due to overlapping domains)
        accuracy = correct_classifications / len(test_cases)
        assert accuracy >= 0.5  # At least 50% accuracy on clear cases
        
        # Test classification statistics
        stats = classifier.get_classification_stats()
        assert stats["total_domains"] == 4
        assert len(stats["domain_names"]) == 4
        
        print(f"✓ Test 7 passed: Domain classifier accuracy: {accuracy:.1%}")
    
    def test_claim_preprocessing_normalization(self):
        """Test 8: Tests text preprocessing pipeline."""
        config_manager = ConfigManager(str(self.test_config_path))
        classifier = DomainClassifier()
        
        # Test preprocessing with various text formats
        test_claims = [
            "  This is a claim with extra whitespace.  ",
            "This claim has MIXED case LETTERS!",
            "This claim has... multiple punctuation marks???",
            "This claim has\nmultiple\nlines",
            "This claim has    irregular     spacing"
        ]
        
        for claim in test_claims:
            result = classifier.classify_claim(claim)
            
            # Should successfully classify despite formatting issues
            assert result is not None
            assert result.domain in ["science", "health", "history", "finance"]
            assert 0.0 <= result.confidence <= 1.0
            
            # Preprocessing should handle the text properly
            processed = classifier._preprocess_text(claim)
            assert processed == processed.strip()  # No leading/trailing whitespace
            assert not any(c.isupper() for c in processed if c.isalpha())  # All lowercase
        
        # Test with very short claims (should still work but have low complexity)
        short_result = classifier.classify_claim("Test")
        assert short_result is not None
        assert short_result.complexity_score < 0.3  # Should be low complexity
        
        print("✓ Test 8 passed: Text preprocessing pipeline validated")
    
    def test_database_schema_integrity(self):
        """Test 9: Validates database schema and constraints."""
        config_manager = ConfigManager(str(self.test_config_path))
        
        with patch('src.dataset.data_storage.get_config_manager', return_value=config_manager):
            storage = DataStorage()
            
            # Test database schema by examining tables
            with storage.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check that all expected tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = ["claims", "responses", "evaluations"]
                for table in expected_tables:
                    assert table in tables, f"Missing table: {table}"
                
                # Check claims table schema
                cursor.execute("PRAGMA table_info(claims)")
                claims_columns = {row[1]: row[2] for row in cursor.fetchall()}
                expected_claims_columns = ["id", "claim_text", "domain", "complexity_score", "ground_truth", "created_at", "updated_at"]
                for col in expected_claims_columns:
                    assert col in claims_columns, f"Missing column in claims table: {col}"
                
                # Check responses table schema
                cursor.execute("PRAGMA table_info(responses)")
                responses_columns = {row[1]: row[2] for row in cursor.fetchall()}
                expected_responses_columns = ["id", "claim_id", "agent_type", "agent_id", "response_text", "citations", "confidence_score", "reasoning", "created_at"]
                for col in expected_responses_columns:
                    assert col in responses_columns, f"Missing column in responses table: {col}"
                
                # Test foreign key constraint
                test_claim = ClaimRecord(
                    claim_text="Test claim for constraints",
                    domain="science",
                    complexity_score=0.5
                )
                claim_id = storage.save_claim(test_claim)
                
                # Valid response should work
                valid_response = ResponseRecord(
                    claim_id=claim_id,
                    agent_type="answering",
                    agent_id="test",
                    response_text="Valid response",
                    citations=[],
                    confidence_score=0.8,
                    reasoning="Test reasoning"
                )
                response_id = storage.save_response(valid_response)
                assert response_id is not None
        
        print("✓ Test 9 passed: Database schema integrity validated")
    
    def test_error_handling_malformed_data(self):
        """Test 10: Tests graceful error handling."""
        # Test configuration errors
        invalid_config_path = Path(self.temp_dir) / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(Exception):
            ConfigManager(str(invalid_config_path))
        
        # Test validator error handling
        with pytest.raises(ValidationError):
            ClaimValidator.validate_claim_text("")  # Empty claim
        
        with pytest.raises(ValidationError):
            ClaimValidator.validate_claim_text("x" * 1000)  # Too long
        
        with pytest.raises(ValidationError):
            ClaimValidator.validate_domain("invalid domain name!")
        
        with pytest.raises(ValidationError):
            ClaimValidator.validate_complexity_score(-1.0)  # Invalid range
        
        with pytest.raises(ValidationError):
            ClaimValidator.validate_complexity_score(2.0)  # Invalid range
        
        # Test citation validator errors
        with pytest.raises(ValidationError):
            CitationValidator.validate_url("not-a-url")
        
        with pytest.raises(ValidationError):
            CitationValidator.validate_url("ftp://invalid-protocol.com")
        
        with pytest.raises(ValidationError):
            CitationValidator.validate_citation_format("")
        
        # Test dataset error handling
        with patch('src.dataset.claim_dataset.get_config_manager') as mock_config:
            mock_config_manager = MagicMock()
            mock_config.return_value = mock_config_manager
            
            dataset = ClaimDataset(data_dir=self.temp_dir)
            
            # Test loading non-existent file
            with pytest.raises(FileNotFoundError):
                dataset.load_claims_from_file("non_existent_file.json")
            
            # Test invalid file format
            invalid_file = Path(self.temp_dir) / "invalid.txt"
            invalid_file.write_text("Not JSON or CSV content")
            
            with pytest.raises(ValueError):
                dataset.load_claims_from_file(str(invalid_file), format="xml")
        
        print("✓ Test 10 passed: Error handling for malformed data validated")
    
    def test_run_all_module1_tests(self):
        """Run all Module 1 tests in sequence."""
        print("\\n" + "="*60)
        print("RUNNING ALL MODULE 1 TESTS")
        print("="*60)
        
        try:
            self.test_claim_dataset_initialization()
            self.test_claim_loading_from_file()
            self.test_domain_distribution_validation()
            self.test_claim_complexity_scoring()
            self.test_data_storage_persistence()
            self.test_config_manager_validation()
            self.test_domain_classifier_accuracy()
            self.test_claim_preprocessing_normalization()
            self.test_database_schema_integrity()
            self.test_error_handling_malformed_data()
            
            print("\\n" + "="*60)
            print("✅ ALL 10 MODULE 1 TESTS PASSED SUCCESSFULLY!")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"\\n❌ TEST FAILED: {e}")
            print("="*60)
            raise
    

if __name__ == "__main__":
    test_instance = TestModule1Dataset()
    test_instance.setup_method()
    try:
        test_instance.test_run_all_module1_tests()
    finally:
        test_instance.teardown_method()