"""Claim dataset management for the Cite-and-Challenge Protocol system."""

import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
import pandas as pd

from .data_storage import DataStorage, ClaimRecord
from .domain_classifier import DomainClassifier, ClassificationResult
from ..config.config_manager import get_config_manager

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


@dataclass
class ClaimEntry:
    """Individual claim entry with metadata."""
    text: str
    domain: str
    complexity_score: float
    ground_truth: Optional[str] = None
    source: Optional[str] = None
    verified: bool = False


class ClaimDataset:
    """
    Manages 300 curated factual claims across 4 domains (75 claims each: science, health, history, finance).
    Provides functionality for loading, storing, and accessing claims with domain balance validation.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize ClaimDataset with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_manager = get_config_manager()
        self.storage = DataStorage()
        self.classifier = DomainClassifier()
        
        self.claims: List[ClaimEntry] = []
        self.domain_targets = {"science": 75, "health": 75, "history": 75, "finance": 75}
        
        logger.info(f"{BLUE}Initialized ClaimDataset with target: {sum(self.domain_targets.values())} claims{END}")
    
    def load_claims_from_file(self, file_path: str, format: str = "auto") -> int:
        """
        Load claims from JSON or CSV file.
        
        Args:
            file_path: Path to the claims file
            format: File format ("json", "csv", or "auto" to detect)
            
        Returns:
            Number of claims loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Claims file not found: {file_path}")
        
        # Auto-detect format if needed
        if format == "auto":
            format = file_path.suffix.lower().lstrip('.')
        
        logger.info(f"{BLUE}Loading claims from {file_path} (format: {format}){END}")
        
        loaded_claims = []
        
        if format == "json":
            loaded_claims = self._load_from_json(file_path)
        elif format == "csv":
            loaded_claims = self._load_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {format}")
        
        # Add to internal list
        self.claims.extend(loaded_claims)
        
        logger.success(f"{GREEN}Loaded {len(loaded_claims)} claims from file{END}")
        return len(loaded_claims)
    
    def _load_from_json(self, file_path: Path) -> List[ClaimEntry]:
        """Load claims from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        claims = []
        if isinstance(data, list):
            # Array of claim objects
            for item in data:
                claims.append(self._parse_claim_dict(item))
        elif isinstance(data, dict):
            # Single object or nested structure
            if "claims" in data:
                for item in data["claims"]:
                    claims.append(self._parse_claim_dict(item))
            else:
                claims.append(self._parse_claim_dict(data))
        
        return claims
    
    def _load_from_csv(self, file_path: Path) -> List[ClaimEntry]:
        """Load claims from CSV file."""
        df = pd.read_csv(file_path)
        claims = []
        
        for _, row in df.iterrows():
            claim_dict = row.to_dict()
            claims.append(self._parse_claim_dict(claim_dict))
        
        return claims
    
    def _parse_claim_dict(self, item: Dict[str, Any]) -> ClaimEntry:
        """Parse a dictionary into a ClaimEntry."""
        # Extract text (try different possible field names)
        text_fields = ["text", "claim", "claim_text", "statement", "content"]
        text = None
        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field]).strip()
                break
        
        if not text:
            raise ValueError(f"No valid text field found in claim: {item}")
        
        # Extract or classify domain
        domain = item.get("domain", "").lower()
        if not domain or domain not in self.domain_targets:
            # Auto-classify domain
            classification = self.classifier.classify_claim(text)
            domain = classification.domain
            complexity_score = classification.complexity_score
        else:
            # Calculate complexity for provided domain
            classification = self.classifier.classify_claim(text)
            complexity_score = classification.complexity_score
        
        return ClaimEntry(
            text=text,
            domain=domain,
            complexity_score=complexity_score,
            ground_truth=item.get("ground_truth"),
            source=item.get("source"),
            verified=item.get("verified", False)
        )
    
    def generate_sample_claims(self, claims_per_domain: int = 75) -> int:
        """
        Generate sample claims for testing purposes.
        
        Args:
            claims_per_domain: Number of claims to generate per domain
            
        Returns:
            Number of claims generated
        """
        logger.info(f"{BLUE}Generating {claims_per_domain * 4} sample claims{END}")
        
        sample_claims = {
            "science": [
                "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                "DNA is composed of four nucleotide bases: adenine, guanine, cytosine, and thymine.",
                "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
                "The human brain contains approximately 86 billion neurons.",
                "Quantum entanglement allows particles to be instantaneously connected regardless of distance.",
                # Add more sample science claims...
            ],
            "health": [
                "Regular exercise can reduce the risk of cardiovascular disease by up to 35%.",
                "The recommended daily water intake for adults is about 8 glasses or 2 liters.",
                "Vitamin D deficiency affects approximately 1 billion people worldwide.",
                "Sleep deprivation can significantly impact immune system function.",
                "The human body requires essential amino acids that cannot be produced internally.",
                # Add more sample health claims...
            ],
            "history": [
                "The Great Wall of China was built over several dynasties spanning more than 2,000 years.",
                "World War II ended in 1945 with the surrender of Japan in September.",
                "The printing press was invented by Johannes Gutenberg around 1440.",
                "The Roman Empire at its peak controlled territory across three continents.",
                "The Industrial Revolution began in Britain in the late 18th century.",
                # Add more sample history claims...
            ],
            "finance": [
                "Compound interest allows investments to grow exponentially over time.",
                "The S&P 500 has historically provided average annual returns of about 10%.",
                "Inflation reduces the purchasing power of currency over time.",
                "Diversification can help reduce investment portfolio risk.",
                "Central banks use interest rates to control monetary policy and inflation.",
                # Add more sample finance claims...
            ]
        }
        
        generated_count = 0
        for domain, base_claims in sample_claims.items():
            # Generate variations and expand to reach target count
            domain_claims = self._expand_claims_list(base_claims, claims_per_domain)
            
            for claim_text in domain_claims:
                classification = self.classifier.classify_claim(claim_text)
                claim_entry = ClaimEntry(
                    text=claim_text,
                    domain=domain,
                    complexity_score=classification.complexity_score,
                    ground_truth=None,  # Would need manual verification
                    source="generated",
                    verified=False
                )
                self.claims.append(claim_entry)
                generated_count += 1
        
        logger.success(f"{GREEN}Generated {generated_count} sample claims{END}")
        return generated_count
    
    def _expand_claims_list(self, base_claims: List[str], target_count: int) -> List[str]:
        """Expand a list of base claims to reach target count."""
        if len(base_claims) >= target_count:
            return base_claims[:target_count]
        
        expanded = base_claims.copy()
        
        # Create variations of existing claims
        variation_templates = [
            "Studies show that {}",
            "Research indicates that {}",
            "According to experts, {}",
            "Evidence suggests that {}",
            "Scientific data confirms that {}",
        ]
        
        while len(expanded) < target_count:
            base_claim = random.choice(base_claims)
            # Remove leading "The" or similar to make template work
            modified_claim = base_claim
            if base_claim.lower().startswith(('the ', 'a ', 'an ')):
                words = base_claim.split(' ', 1)
                if len(words) > 1:
                    modified_claim = words[1].lower()
            
            template = random.choice(variation_templates)
            new_claim = template.format(modified_claim)
            if new_claim not in expanded:
                expanded.append(new_claim)
        
        return expanded[:target_count]
    
    def save_to_database(self) -> int:
        """Save all loaded claims to the database."""
        if not self.claims:
            logger.warning(f"{YELLOW}No claims to save to database{END}")
            return 0
        
        # Convert to database records
        db_records = []
        for claim in self.claims:
            record = ClaimRecord(
                claim_text=claim.text,
                domain=claim.domain,
                complexity_score=claim.complexity_score,
                ground_truth=claim.ground_truth
            )
            db_records.append(record)
        
        # Bulk insert
        claim_ids = self.storage.bulk_insert_claims(db_records)
        
        logger.success(f"{GREEN}Saved {len(claim_ids)} claims to database{END}")
        return len(claim_ids)
    
    def load_from_database(self) -> int:
        """Load claims from the database."""
        db_claims = self.storage.get_all_claims()
        
        self.claims = []
        for db_claim in db_claims:
            claim_entry = ClaimEntry(
                text=db_claim.claim_text,
                domain=db_claim.domain,
                complexity_score=db_claim.complexity_score,
                ground_truth=db_claim.ground_truth,
                source="database",
                verified=True
            )
            self.claims.append(claim_entry)
        
        logger.success(f"{GREEN}Loaded {len(self.claims)} claims from database{END}")
        return len(self.claims)
    
    def validate_domain_distribution(self) -> Dict[str, Any]:
        """Validate that claims are properly distributed across domains."""
        distribution = {}
        for claim in self.claims:
            domain = claim.domain
            distribution[domain] = distribution.get(domain, 0) + 1
        
        validation_result = {
            "is_valid": True,
            "distribution": distribution,
            "targets": self.domain_targets,
            "total_claims": len(self.claims),
            "issues": []
        }
        
        for domain, target in self.domain_targets.items():
            actual = distribution.get(domain, 0)
            if actual < target * 0.9:  # Allow 10% deviation
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Domain '{domain}' has {actual} claims (target: {target})")
        
        return validation_result
    
    def get_claims_by_domain(self, domain: str) -> List[ClaimEntry]:
        """Get all claims for a specific domain."""
        return [claim for claim in self.claims if claim.domain == domain]
    
    def get_claims_by_complexity(self, min_complexity: float = 0.0, max_complexity: float = 1.0) -> List[ClaimEntry]:
        """Get claims within a specific complexity range."""
        return [
            claim for claim in self.claims 
            if min_complexity <= claim.complexity_score <= max_complexity
        ]
    
    def sample_claims(self, count: int, stratify_by_domain: bool = True) -> List[ClaimEntry]:
        """
        Sample claims from the dataset.
        
        Args:
            count: Number of claims to sample
            stratify_by_domain: Whether to maintain domain distribution
            
        Returns:
            List of sampled claims
        """
        if not self.claims:
            return []
        
        if count >= len(self.claims):
            return self.claims.copy()
        
        if stratify_by_domain:
            # Sample proportionally from each domain
            sampled_claims = []
            domain_counts = {}
            for claim in self.claims:
                domain_counts[claim.domain] = domain_counts.get(claim.domain, 0) + 1
            
            for domain in self.domain_targets.keys():
                domain_claims = self.get_claims_by_domain(domain)
                if domain_claims:
                    # Calculate proportional sample size
                    proportion = len(domain_claims) / len(self.claims)
                    domain_sample_size = int(count * proportion)
                    domain_sample = random.sample(domain_claims, min(domain_sample_size, len(domain_claims)))
                    sampled_claims.extend(domain_sample)
            
            # Fill remaining slots if needed
            remaining = count - len(sampled_claims)
            if remaining > 0:
                available = [c for c in self.claims if c not in sampled_claims]
                if available:
                    additional = random.sample(available, min(remaining, len(available)))
                    sampled_claims.extend(additional)
            
            return sampled_claims[:count]
        else:
            return random.sample(self.claims, count)
    
    def export_claims(self, file_path: str, format: str = "json", include_metadata: bool = True) -> None:
        """Export claims to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = []
        for claim in self.claims:
            claim_dict = {
                "text": claim.text,
                "domain": claim.domain,
            }
            
            if include_metadata:
                claim_dict.update({
                    "complexity_score": claim.complexity_score,
                    "ground_truth": claim.ground_truth,
                    "source": claim.source,
                    "verified": claim.verified
                })
            
            export_data.append(claim_dict)
        
        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.success(f"{GREEN}Exported {len(export_data)} claims to {file_path}{END}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        if not self.claims:
            return {"total_claims": 0, "domains": {}}
        
        stats = {
            "total_claims": len(self.claims),
            "domains": {},
            "complexity": {},
            "verification": {
                "verified": sum(1 for c in self.claims if c.verified),
                "unverified": sum(1 for c in self.claims if not c.verified)
            },
            "sources": {}
        }
        
        # Domain statistics
        for claim in self.claims:
            domain = claim.domain
            if domain not in stats["domains"]:
                stats["domains"][domain] = {
                    "count": 0,
                    "avg_complexity": 0.0,
                    "complexity_scores": []
                }
            
            stats["domains"][domain]["count"] += 1
            stats["domains"][domain]["complexity_scores"].append(claim.complexity_score)
            
            # Source statistics
            source = claim.source or "unknown"
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        # Calculate average complexity per domain
        for domain_data in stats["domains"].values():
            if domain_data["complexity_scores"]:
                domain_data["avg_complexity"] = sum(domain_data["complexity_scores"]) / len(domain_data["complexity_scores"])
        
        # Overall complexity statistics
        all_complexity = [c.complexity_score for c in self.claims]
        if all_complexity:
            stats["complexity"] = {
                "average": sum(all_complexity) / len(all_complexity),
                "min": min(all_complexity),
                "max": max(all_complexity),
                "median": sorted(all_complexity)[len(all_complexity) // 2]
            }
        
        return stats
    
    def __len__(self) -> int:
        """Return the number of claims in the dataset."""
        return len(self.claims)
    
    def __getitem__(self, index: int) -> ClaimEntry:
        """Get claim by index."""
        return self.claims[index]
    
    def __iter__(self):
        """Iterate over claims."""
        return iter(self.claims)