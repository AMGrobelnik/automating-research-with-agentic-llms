"""Configuration management for the Cite-and-Challenge Protocol system."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


class DatabaseConfig(BaseModel):
    """Database configuration model."""
    type: str = Field(..., description="Database type (sqlite or postgresql)")
    path: Optional[str] = Field(None, description="SQLite database path")
    postgresql: Optional[Dict[str, Any]] = Field(None, description="PostgreSQL connection settings")
    
    @field_validator('type')
    def validate_db_type(cls, v):
        if v not in ['sqlite', 'postgresql']:
            raise ValueError(f"Database type must be 'sqlite' or 'postgresql', got: {v}")
        return v


class APIKeysConfig(BaseModel):
    """API keys configuration model."""
    openai_key: str = Field(default="", description="OpenAI API key")
    anthropic_key: str = Field(default="", description="Anthropic API key")
    google_search_key: str = Field(default="", description="Google Search API key")
    bing_search_key: str = Field(default="", description="Bing Search API key")


class SearchConfig(BaseModel):
    """Search configuration model."""
    providers: List[str] = Field(..., description="Search providers list")
    rate_limits: Dict[str, int] = Field(..., description="Rate limits per provider")
    timeout: int = Field(30, description="Search timeout in seconds")
    max_results: int = Field(10, description="Maximum search results per query")


class AgentsConfig(BaseModel):
    """Agents configuration model."""
    max_tokens: int = Field(4000, description="Maximum tokens per agent")
    temperature: float = Field(0.1, description="LLM temperature")
    model: str = Field("gpt-4", description="Primary LLM model")
    backup_model: str = Field("claude-3-sonnet-20241022", description="Backup LLM model")


class SystemConfig(BaseModel):
    """System configuration model."""
    log_level: str = Field("INFO", description="Logging level")
    log_file: str = Field("logs/system.log", description="Log file path")
    max_claim_length: int = Field(500, description="Maximum claim length in characters")
    max_citation_count: int = Field(20, description="Maximum citations per claim")
    revision_rounds: int = Field(1, description="Number of revision rounds")


class DomainConfig(BaseModel):
    """Domain configuration model."""
    name: str = Field(..., description="Domain name")
    keywords: List[str] = Field(..., description="Domain keywords for classification")
    complexity_weight: float = Field(..., description="Domain complexity weight")


class EvaluationConfig(BaseModel):
    """Evaluation configuration model."""
    hallucination_threshold: float = Field(0.85, description="Hallucination detection threshold")
    citation_precision_target: float = Field(0.85, description="Target citation precision")
    citation_recall_target: float = Field(0.90, description="Target citation recall")
    statistical_significance: float = Field(0.05, description="Statistical significance level")


class Config(BaseModel):
    """Main configuration model."""
    database: DatabaseConfig
    api_keys: APIKeysConfig
    search: SearchConfig
    agents: AgentsConfig
    system: SystemConfig
    domains: List[DomainConfig]
    evaluation: EvaluationConfig


class ConfigManager:
    """Configuration manager for loading and validating system configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ConfigManager with optional config path."""
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[Config] = None
        self._load_config()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            "src/config/config.yaml",
            "config/config.yaml", 
            "config.yaml",
            os.path.expanduser("~/.cite_challenge/config.yaml")
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        # If no config found, create default path
        default_path = "src/config/config.yaml"
        logger.warning(f"{YELLOW}No config file found, using default path: {default_path}{END}")
        return default_path
    
    def _load_config(self) -> None:
        """Load and validate configuration from file."""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            logger.info(f"{BLUE}Loading configuration from: {config_path}{END}")
            
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            self._config = Config(**raw_config)
            logger.success(f"{GREEN}Configuration loaded successfully{END}")
            
        except Exception as e:
            logger.error(f"{RED}Failed to load configuration: {e}{END}")
            raise
    
    def get_config(self) -> Config:
        """Get the validated configuration object."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.get_config().database
    
    def get_api_keys(self) -> APIKeysConfig:
        """Get API keys configuration."""
        return self.get_config().api_keys
    
    def get_search_config(self) -> SearchConfig:
        """Get search configuration."""
        return self.get_config().search
    
    def get_agents_config(self) -> AgentsConfig:
        """Get agents configuration."""
        return self.get_config().agents
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.get_config().system
    
    def get_domains(self) -> List[DomainConfig]:
        """Get domains configuration."""
        return self.get_config().domains
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self.get_config().evaluation
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are configured."""
        api_keys = self.get_api_keys()
        required_keys = []
        
        if not api_keys.openai_key and not api_keys.anthropic_key:
            required_keys.append("At least one LLM API key (OpenAI or Anthropic)")
        
        if not api_keys.google_search_key and not api_keys.bing_search_key:
            logger.warning(f"{YELLOW}No search API keys configured, using DuckDuckGo only{END}")
        
        if required_keys:
            logger.error(f"{RED}Missing required API keys: {required_keys}{END}")
            return False
        
        logger.success(f"{GREEN}API keys validation passed{END}")
        return True
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        config = self.get_config()
        
        # Create log directory
        log_path = Path(config.system.log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create data directory for SQLite
        if config.database.type == "sqlite" and config.database.path:
            db_path = Path(config.database.path).parent
            db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{BLUE}Created necessary directories{END}")
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        logger.info(f"{BLUE}Reloading configuration...{END}")
        self._load_config()


# Global config manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get or create global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """Get the global configuration object."""
    return get_config_manager().get_config()