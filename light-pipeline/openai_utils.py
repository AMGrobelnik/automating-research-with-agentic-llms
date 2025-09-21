"""
OpenAI API utilities for AI Scientist Lite pipeline.
Provides interface for OpenAI API with reasoning capabilities.
"""

import yaml
import os
from pathlib import Path
from openai import OpenAI
from loguru import logger
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from xml_escape_utils import escape_line_for_logging

# Color constants for logging
BLUE, END = "\033[94m", "\033[0m"

# Logger configuration is handled by log.py

class OpenAIUtils:
    """OpenAI API utility class with reasoning capabilities."""
    
    def __init__(self, config_path: str = "pipeline_config.yaml"):
        """Initialize OpenAI client with configuration from YAML file."""
        self.config = self._load_config(config_path)
        # Use environment variable for API key, fallback to config if provided
        api_key = os.getenv('OPENAI_API_KEY') or self.config['openai_settings'].get('api_key', '')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.model = self.config['openai_settings']['model']
        self.reasoning_effort = self.config['openai_settings']['reasoning_effort']
        self.verbosity = self.config['openai_settings'].get('verbosity', 'medium')
        self.service_tier = self.config['openai_settings'].get('service_tier', 'default')
        self.last_response_metadata = None
        
        logger.info(f"{BLUE}OpenAI client initialized{END} with model: {self.model}, service tier: {self.service_tier}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'openai_settings' not in config:
            raise ValueError("OpenAI settings not found in config file")
        
        return config
    
    @logger.catch(reraise=True)
    def _save_full_output_to_file(self, text: str, filepath: str, response_metadata: dict = None):
        """Save full output text to specified log file with metadata."""
        log_path = Path(filepath)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model}\n")
            
            # Add response metadata if provided
            if response_metadata:
                f.write(f"Actual Model: {response_metadata.get('model_used', 'unknown')}\n")
                f.write(f"Response ID: {response_metadata.get('response_id', 'unknown')}\n")
                f.write(f"Response Time: {response_metadata.get('duration', 'unknown')}s\n")
                f.write(f"Total Tokens: {response_metadata.get('total_tokens', 'unknown')}\n")
                f.write(f"Input Tokens: {response_metadata.get('input_tokens', 'unknown')}\n")
                f.write(f"Output Tokens: {response_metadata.get('output_tokens', 'unknown')}\n")
                f.write(f"Reasoning Tokens: {response_metadata.get('reasoning_tokens', 'unknown')}\n")
            
            f.write(f"{'='*80}\n\n")
            f.write(text)
            f.write(f"\n\n{'='*80}\n")
        
        logger.debug(f"Full output saved to {filepath}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, Exception)),
        before_sleep=lambda retry_state: logger.warning(f"Connection error, retrying... (attempt {retry_state.attempt_number})")
    )
    def create_response(self, input_text: str, reasoning_effort: str = None, verbosity: str = None, service_tier: str = None) -> object:
        """
        Create a response using OpenAI's reasoning API.
        
        Args:
            input_text (str): The input prompt/question
            reasoning_effort (str, optional): Override default reasoning effort
            verbosity (str, optional): Override default verbosity (low, medium, high)
            service_tier (str, optional): Override default service tier (default, priority)
        
        Returns:
            OpenAI response object
        """
        effort = reasoning_effort or self.reasoning_effort
        verb = verbosity or self.verbosity
        tier = service_tier or self.service_tier
        
        
        # Log request details
        logger.log("REQUEST", f"Model: {self.model}, Effort: {effort}, Verbosity: {verb}, Service Tier: {tier}")
        
        # Log the full input prompt being sent to OpenAI
        prompt_lines = input_text.split('\n')
        logger.info("📝 Full Prompt Being Sent to OpenAI:")
        logger.info("=" * 80)
        for i, line in enumerate(prompt_lines):
            # Escape special characters for loguru
            escaped_line = escape_line_for_logging(line)
            logger.info(f"{i+1:4d} | {escaped_line}")
        logger.info("=" * 80)
        logger.info(f"📊 Total prompt length: {len(input_text)} characters")
        logger.info(f"📊 Total prompt lines: {len(prompt_lines)} lines")
        logger.info("=" * 80)
        
        try:
            start_time = time.time()
            response = self.client.responses.create(
                model=self.model,
                input=input_text,
                reasoning={
                    "effort": effort
                },
                text={
                    "verbosity": verb
                },
                service_tier=tier
            )
            end_time = time.time()
            duration = end_time - start_time
            
            # Log comprehensive response info
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                total_tokens = getattr(usage, 'total_tokens', 'unknown')
                input_tokens = getattr(usage, 'input_tokens', 'unknown')
                output_tokens = getattr(usage, 'output_tokens', 'unknown')
                
                # Get reasoning tokens if available
                reasoning_tokens = 0
                if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                    reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)
                
                # Log token usage
                logger.log("TOKENS", f"Total: {total_tokens}, Input: {input_tokens}, Output: {output_tokens}, Reasoning: {reasoning_tokens}")
            
            # Log response metadata
            response_id = getattr(response, 'id', 'unknown')
            model_used = getattr(response, 'model', 'unknown') 
            status = getattr(response, 'status', 'unknown')
            
            logger.log("METADATA", f"Response ID: {response_id}, Model: {model_used}, Status: {status}")
            
            # Log timing information
            logger.log("TIMING", f"Response time: {duration:.2f}s")
            
            # Log created_at timestamp if available
            if hasattr(response, 'created_at'):
                created_at = datetime.fromtimestamp(response.created_at).isoformat()
                logger.log("TIMING", f"Created at: {created_at}")
            
            if not hasattr(response, 'usage'):
                logger.log("METADATA", "Response created successfully (no usage info available)")
            
            # Store metadata for later use
            self.last_response_metadata = {
                'response_id': response_id,
                'model_used': model_used,
                'status': status,
                'duration': f"{duration:.2f}",
                'total_tokens': total_tokens if 'total_tokens' in locals() else 'unknown',
                'input_tokens': input_tokens if 'input_tokens' in locals() else 'unknown',
                'output_tokens': output_tokens if 'output_tokens' in locals() else 'unknown',
                'reasoning_tokens': reasoning_tokens if 'reasoning_tokens' in locals() else 'unknown'
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating OpenAI response: {e}")
            raise
    
    @logger.catch(default="", reraise=True)
    def extract_text_from_response(self, response) -> str:
        """
        Extract clean text from OpenAI response object.
        
        Args:
            response: OpenAI response object
        
        Returns:
            str: Extracted text content
        """
        if hasattr(response, 'output') and response.output is not None:
            # For responses API, extract from output
            for item in response.output:
                if item is not None and hasattr(item, 'content') and item.content is not None:
                    for content_item in item.content:
                        if content_item is not None and hasattr(content_item, 'text') and content_item.text is not None:
                            text_result = content_item.text
                            
                            
                            # Log the full GPT-5 output content
                            logger.log("OUTPUT", f"GPT-5 Output ({len(text_result)} chars):")
                            # Log the full content without truncation - escape curly braces to avoid format errors
                            logger.log("OUTPUT", text_result.replace("{", "{{").replace("}", "}}"))
                            return text_result
        elif hasattr(response, 'choices') and response.choices:
            # Configure OpenAI logger if not done yet
            _ensure_openai_logger_configured()
                
            # For chat completions API
            text_result = response.choices[0].message.content
            logger.log("OUTPUT", f"Chat Completion Output ({len(text_result)} chars):")
            logger.log("OUTPUT", text_result.replace("{", "{{").replace("}", "}}"))
            return text_result
        elif hasattr(response, 'content'):
            # Configure OpenAI logger if not done yet
            _ensure_openai_logger_configured()
                
            text_result = response.content
            logger.log("OUTPUT", f"Direct Content ({len(text_result)} chars):")
            logger.log("OUTPUT", text_result.replace("{", "{{").replace("}", "}}"))
            return text_result
        else:
            # Configure OpenAI logger if not done yet
            _ensure_openai_logger_configured()
                
            logger.warning(f"Unexpected response structure: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")
            return str(response)
    
    def save_response_with_metadata(self, text: str, filepath: str):
        """Save response text with all metadata to specified file."""
        self._save_full_output_to_file(text, filepath, self.last_response_metadata)
    
    def generate_idea_with_reasoning(self, prompt: str) -> str:
        """
        Generate an idea using OpenAI with high reasoning effort.
        
        Args:
            prompt (str): The idea generation prompt
        
        Returns:
            str: Generated idea response
        """
        
        logger.log("REQUEST", "Generating idea with high reasoning effort")
        
        response = self.create_response(prompt, "high")
        result = self.extract_text_from_response(response)
        
        logger.log("METADATA", f"Idea generation complete - {len(result)} characters")
        return result


# Removed main() function (dead code)