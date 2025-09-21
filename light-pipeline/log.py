#!/usr/bin/env python3
"""
AI Scientist Lite - Logging and Output Utilities
Provides colored logging setup and file output functions.
"""

from loguru import logger
import sys
from xml_escape_utils import escape_xml_tags_for_loguru

# Remove default handler
logger.remove()

# Set up all custom levels for OpenAI and Claude
# OpenAI API levels
logger.level("REQUEST", no=40)    # Request details
logger.level("TOKENS", no=41)     # Token usage
logger.level("METADATA", no=42)   # Response metadata
logger.level("TIMING", no=43)     # Timing information
logger.level("OUTPUT", no=44)     # Output content

# Claude SDK levels  
logger.level("THINKING", no=5)
logger.level("SRCH_IN", no=10)
logger.level("SRCH_OUT", no=11)
logger.level("FTCH_IN", no=12)
logger.level("FTCH_OUT", no=13)
logger.level("READ_IN", no=14)
logger.level("READ_OUT", no=15)
logger.level("WRIT_IN", no=16)
logger.level("WRIT_OUT", no=17)
logger.level("EDIT_IN", no=18)
logger.level("EDIT_OUT", no=19)
logger.level("MULT_IN", no=20)
logger.level("MULT_OUT", no=21)
logger.level("BASH_IN", no=22)
logger.level("BASH_OUT", no=23)
logger.level("GREP_IN", no=24)
logger.level("GREP_OUT", no=25)
logger.level("GLOB_IN", no=26)
logger.level("GLOB_OUT", no=27)
logger.level("LS_IN", no=28)
logger.level("LS_OUT", no=29)
logger.level("TASK_IN", no=30)
logger.level("TASK_OUT", no=31)
logger.level("TODO_IN", no=32)
logger.level("TODO_OUT", no=33)
logger.level("USER", no=34)
logger.level("SYSTEM", no=35)
logger.level("CLAUDE", no=36)

def format_record(record):
    """Custom formatter with ANSI colors."""
    level = record["level"].name
    message = record["message"]
    
    # Dynamically escape XML tags to prevent loguru from interpreting them as color directives
    message = escape_xml_tags_for_loguru(message)
    
    colors = {
        # Standard levels
        "SUCCESS": "\033[32m",      # Green
        "INFO": "\033[34m",         # Blue
        "WARNING": "\033[33m",      # Yellow
        "ERROR": "\033[31m",        # Red
        "DEBUG": "\033[36m",        # Cyan
        
        # OpenAI custom levels
        "REQUEST": "\033[38;5;14m",    # Bright Cyan - Request
        "TOKENS": "\033[38;5;208m",    # Orange - Token usage  
        "METADATA": "\033[38;5;33m",   # Blue - Metadata
        "TIMING": "\033[38;5;10m",     # Bright Green - Timing
        "OUTPUT": "\033[38;5;213m",    # Pink - Output content
        
        # Claude SDK custom levels
        "THINKING": "\033[38;5;201m",  # Hot Magenta (256-color) - Very distinct
        
        # BASH - Completely different colors
        "BASH_IN": "\033[38;5;39m",    # Bright Blue (256-color)
        "BASH_OUT": "\033[38;5;202m",  # Bright Orange (completely different) (256-color)
        
        # READ - Completely different colors
        "READ_IN": "\033[38;5;226m",   # Bright Yellow (256-color)
        "READ_OUT": "\033[38;5;93m",   # Purple (completely different) (256-color)
        
        # WRITE - Completely different colors
        "WRIT_IN": "\033[38;5;46m",    # Bright Green (256-color)
        "WRIT_OUT": "\033[38;5;197m",  # Hot Pink (completely different) (256-color)
        
        # EDIT - Completely different colors
        "EDIT_IN": "\033[38;5;207m",   # Light Pink (256-color)
        "EDIT_OUT": "\033[38;5;22m",   # Dark Green (completely different) (256-color)
        
        # MULTIEDIT - Completely different colors
        "MULT_IN": "\033[38;5;183m",   # Light Lavender (256-color)
        "MULT_OUT": "\033[38;5;75m",   # Steel Blue (completely different) (256-color)
        
        # GREP - Completely different colors
        "GREP_IN": "\033[38;5;178m",   # Light Amber (256-color)
        "GREP_OUT": "\033[38;5;33m",   # Blue (completely different) (256-color)
        
        # GLOB - Completely different colors
        "GLOB_IN": "\033[38;5;51m",    # Bright Cyan (256-color)
        "GLOB_OUT": "\033[38;5;166m",  # Dark Orange (completely different) (256-color)
        
        # LS - Completely different colors
        "LS_IN": "\033[38;5;210m",     # Light Coral (256-color)
        "LS_OUT": "\033[38;5;55m",     # Purple-Blue (completely different) (256-color)
        
        # TASK - Completely different colors
        "TASK_IN": "\033[38;5;208m",   # Bright Orange (256-color)
        "TASK_OUT": "\033[38;5;28m",   # Dark Green (completely different) (256-color)
        
        # TODO - Mixed colors
        "TODO_IN": "\033[38;5;46m",    # Bright Green (256-color)
        "TODO_OUT": "\033[38;5;220m",  # Gold/Yellow (256-color)
        
        # Teal family - SEARCH
        "SRCH_IN": "\033[38;5;50m",    # Light Teal (256-color)
        "SRCH_OUT": "\033[38;5;30m",   # Darker Teal (256-color)
        
        # Orange/Peach family - FETCH (changed from brown since GREP is now brown)
        "FTCH_IN": "\033[38;5;215m",   # Light Peach (256-color)
        "FTCH_OUT": "\033[38;5;166m",  # Darker Orange (256-color)
        
        # Other message types
        "USER": "\033[38;5;250m",      # Light Gray (256-color)
        "SYSTEM": "\033[38;5;240m",    # Dark Gray (256-color)
        "CLAUDE": "\033[38;5;214m",    # Orange (256-color) - UNCHANGED
    }
    reset = "\033[0m"
    
    color = colors.get(level, "\033[37m")  # Default to white
    return f"{color}{level: <8}{reset}|{color}{message}{reset}\n"

# Configure logger
logger.add(
    sys.stderr,
    format=format_record,
    colorize=True,
    level=0  # Show all levels
)

def log_config_loaded(config_path):
    """Log successful config loading."""
    logger.info(f"📋 Config loaded from: {config_path}")

def log_config_error(error):
    """Log config loading error."""
    logger.error(f"❌ Failed to load config: {error}")

def log_model_check():
    """Log model checking."""
    logger.info("🔍 Checking Claude model...")

def log_working_dir_check():
    """Log working directory checking."""
    logger.info("🔍 Checking working directory...")

def log_pipeline_start():
    """Log pipeline start."""
    logger.success("🚀 AI Scientist Lite - Pipeline Runner")
    logger.info("=" * 50)

def log_pipeline_complete():
    """Log successful pipeline completion."""
    logger.success("🎉 Pipeline completed successfully!")

def log_pipeline_failed():
    """Log pipeline failure."""
    logger.error("💥 Pipeline failed!")

def log_idea_gen_start():
    """Log start of idea generation."""
    logger.info("🧠 Step 1: Generating ideas with OpenAI")

def log_idea_filter_start():
    """Log start of idea filtering."""
    logger.info("🔍 Step 2: Filtering ideas with Claude")

def log_idea_gen_success(char_count):
    """Log successful idea generation."""
    logger.success(f"✅ Idea generation completed - {char_count} characters")

# Removed old OpenAI-specific functions (dead code)

def log_claude_filtering_complete(cost):
    """Log successful Claude filtering."""
    logger.success(f"✅ Claude filtering completed - Cost: ${cost:.4f}")

# Removed unused log functions (dead code)