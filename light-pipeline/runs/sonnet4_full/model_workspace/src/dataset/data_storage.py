"""Data storage and database management for the Cite-and-Challenge Protocol system."""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
from loguru import logger
from pydantic import BaseModel

from ..config.config_manager import get_config_manager

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


class ClaimRecord(BaseModel):
    """Pydantic model for claim database records."""
    id: Optional[int] = None
    claim_text: str
    domain: str
    complexity_score: float
    ground_truth: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ResponseRecord(BaseModel):
    """Pydantic model for agent response database records."""
    id: Optional[int] = None
    claim_id: int
    agent_type: str  # "answering" or "challenger"
    agent_id: str
    response_text: str
    citations: List[str]
    confidence_score: float
    reasoning: str
    created_at: Optional[datetime] = None


class EvaluationRecord(BaseModel):
    """Pydantic model for evaluation results database records."""
    id: Optional[int] = None
    claim_id: int
    experiment_id: str
    hallucination_score: float
    citation_precision: float
    citation_recall: float
    accuracy_score: float
    baseline_comparison: float
    created_at: Optional[datetime] = None


class DataStorage:
    """Database management for storing claims, responses, and evaluation results."""
    
    def __init__(self):
        """Initialize DataStorage with configuration."""
        self.config_manager = get_config_manager()
        self.db_config = self.config_manager.get_database_config()
        self._setup_database()
    
    def _setup_database(self) -> None:
        """Set up database connection and create tables."""
        if self.db_config.type == "sqlite":
            self._setup_sqlite()
        else:
            raise NotImplementedError("PostgreSQL support not yet implemented")
    
    def _setup_sqlite(self) -> None:
        """Set up SQLite database."""
        db_path = Path(self.db_config.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{BLUE}Setting up SQLite database at: {db_path}{END}")
        
        with sqlite3.connect(str(db_path)) as conn:
            self._create_tables(conn)
        
        logger.success(f"{GREEN}Database setup completed{END}")
    
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create database tables."""
        cursor = conn.cursor()
        
        # Create claims table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_text TEXT NOT NULL,
                domain TEXT NOT NULL,
                complexity_score REAL NOT NULL,
                ground_truth TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                agent_type TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                response_text TEXT NOT NULL,
                citations TEXT,  -- JSON array of citations
                confidence_score REAL NOT NULL,
                reasoning TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claim_id) REFERENCES claims (id)
            )
        """)
        
        # Create evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                experiment_id TEXT NOT NULL,
                hallucination_score REAL NOT NULL,
                citation_precision REAL NOT NULL,
                citation_recall REAL NOT NULL,
                accuracy_score REAL NOT NULL,
                baseline_comparison REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claim_id) REFERENCES claims (id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_domain ON claims (domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_responses_claim_id ON responses (claim_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_experiment ON evaluations (experiment_id)")
        
        conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        if self.db_config.type == "sqlite":
            conn = sqlite3.connect(self.db_config.path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        else:
            raise NotImplementedError("PostgreSQL support not yet implemented")
    
    def save_claim(self, claim: ClaimRecord) -> int:
        """Save a claim to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO claims (claim_text, domain, complexity_score, ground_truth)
                VALUES (?, ?, ?, ?)
            """, (claim.claim_text, claim.domain, claim.complexity_score, claim.ground_truth))
            conn.commit()
            claim_id = cursor.lastrowid
            
        logger.debug(f"{CYAN}Saved claim with ID: {claim_id}{END}")
        return claim_id
    
    def get_claim(self, claim_id: int) -> Optional[ClaimRecord]:
        """Retrieve a claim by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM claims WHERE id = ?", (claim_id,))
            row = cursor.fetchone()
            
            if row:
                return ClaimRecord(
                    id=row["id"],
                    claim_text=row["claim_text"],
                    domain=row["domain"],
                    complexity_score=row["complexity_score"],
                    ground_truth=row["ground_truth"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
        return None
    
    def get_claims_by_domain(self, domain: str) -> List[ClaimRecord]:
        """Retrieve all claims for a specific domain."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM claims WHERE domain = ? ORDER BY id", (domain,))
            rows = cursor.fetchall()
            
            return [
                ClaimRecord(
                    id=row["id"],
                    claim_text=row["claim_text"],
                    domain=row["domain"],
                    complexity_score=row["complexity_score"],
                    ground_truth=row["ground_truth"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                for row in rows
            ]
    
    def get_all_claims(self) -> List[ClaimRecord]:
        """Retrieve all claims from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM claims ORDER BY id")
            rows = cursor.fetchall()
            
            return [
                ClaimRecord(
                    id=row["id"],
                    claim_text=row["claim_text"],
                    domain=row["domain"],
                    complexity_score=row["complexity_score"],
                    ground_truth=row["ground_truth"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                for row in rows
            ]
    
    def save_response(self, response: ResponseRecord) -> int:
        """Save an agent response to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            citations_json = json.dumps(response.citations)
            cursor.execute("""
                INSERT INTO responses (claim_id, agent_type, agent_id, response_text, citations, confidence_score, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                response.claim_id, 
                response.agent_type, 
                response.agent_id,
                response.response_text,
                citations_json,
                response.confidence_score,
                response.reasoning
            ))
            conn.commit()
            response_id = cursor.lastrowid
            
        logger.debug(f"{CYAN}Saved response with ID: {response_id}{END}")
        return response_id
    
    def get_responses_for_claim(self, claim_id: int) -> List[ResponseRecord]:
        """Retrieve all responses for a specific claim."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM responses WHERE claim_id = ? ORDER BY created_at", (claim_id,))
            rows = cursor.fetchall()
            
            return [
                ResponseRecord(
                    id=row["id"],
                    claim_id=row["claim_id"],
                    agent_type=row["agent_type"],
                    agent_id=row["agent_id"],
                    response_text=row["response_text"],
                    citations=json.loads(row["citations"]) if row["citations"] else [],
                    confidence_score=row["confidence_score"],
                    reasoning=row["reasoning"],
                    created_at=row["created_at"]
                )
                for row in rows
            ]
    
    def save_evaluation(self, evaluation: EvaluationRecord) -> int:
        """Save evaluation results to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluations (claim_id, experiment_id, hallucination_score, citation_precision, 
                                       citation_recall, accuracy_score, baseline_comparison)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation.claim_id,
                evaluation.experiment_id,
                evaluation.hallucination_score,
                evaluation.citation_precision,
                evaluation.citation_recall,
                evaluation.accuracy_score,
                evaluation.baseline_comparison
            ))
            conn.commit()
            eval_id = cursor.lastrowid
            
        logger.debug(f"{CYAN}Saved evaluation with ID: {eval_id}{END}")
        return eval_id
    
    def get_experiment_results(self, experiment_id: str) -> List[EvaluationRecord]:
        """Retrieve all evaluation results for a specific experiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluations WHERE experiment_id = ? ORDER BY claim_id", (experiment_id,))
            rows = cursor.fetchall()
            
            return [
                EvaluationRecord(
                    id=row["id"],
                    claim_id=row["claim_id"],
                    experiment_id=row["experiment_id"],
                    hallucination_score=row["hallucination_score"],
                    citation_precision=row["citation_precision"],
                    citation_recall=row["citation_recall"],
                    accuracy_score=row["accuracy_score"],
                    baseline_comparison=row["baseline_comparison"],
                    created_at=row["created_at"]
                )
                for row in rows
            ]
    
    def get_domain_statistics(self) -> Dict[str, int]:
        """Get claim count statistics by domain."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT domain, COUNT(*) as count FROM claims GROUP BY domain")
            rows = cursor.fetchall()
            
            return {row["domain"]: row["count"] for row in rows}
    
    def bulk_insert_claims(self, claims: List[ClaimRecord]) -> List[int]:
        """Bulk insert multiple claims."""
        claim_ids = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for claim in claims:
                cursor.execute("""
                    INSERT INTO claims (claim_text, domain, complexity_score, ground_truth)
                    VALUES (?, ?, ?, ?)
                """, (claim.claim_text, claim.domain, claim.complexity_score, claim.ground_truth))
                claim_ids.append(cursor.lastrowid)
            conn.commit()
        
        logger.info(f"{GREEN}Bulk inserted {len(claims)} claims{END}")
        return claim_ids
    
    def clear_experiment_data(self, experiment_id: str) -> None:
        """Clear all data for a specific experiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evaluations WHERE experiment_id = ?", (experiment_id,))
            conn.commit()
            
        logger.info(f"{YELLOW}Cleared data for experiment: {experiment_id}{END}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count tables
            cursor.execute("SELECT COUNT(*) as claims_count FROM claims")
            claims_count = cursor.fetchone()["claims_count"]
            
            cursor.execute("SELECT COUNT(*) as responses_count FROM responses")
            responses_count = cursor.fetchone()["responses_count"]
            
            cursor.execute("SELECT COUNT(*) as evaluations_count FROM evaluations")
            evaluations_count = cursor.fetchone()["evaluations_count"]
            
            # Get domain distribution
            domain_stats = self.get_domain_statistics()
            
            return {
                "database_type": self.db_config.type,
                "database_path": self.db_config.path,
                "claims_count": claims_count,
                "responses_count": responses_count,
                "evaluations_count": evaluations_count,
                "domain_distribution": domain_stats
            }