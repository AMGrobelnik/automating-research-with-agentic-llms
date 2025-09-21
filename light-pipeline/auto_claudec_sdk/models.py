#!/usr/bin/env python3
"""
Claude SDK - Model Identification and Management
Handles model identification prompts and model checking functionality.
"""

from loguru import logger


def get_model_identification_prompt() -> str:
    """Get the model identification prompt to include in requests."""
    return "FIRST: Please tell me which Claude model you are."


def get_cwd_identification_prompt() -> str:
    """Get the CWD identification prompt to check Claude's working directory."""
    return "Please run the command 'pwd' to show your current working directory."


async def check_claude_model_and_cwd(query_func, workspace_dir=None, config=None, quiet_mode=False):
    """
    Check Claude model and working directory using the provided query function.
    
    Args:
        query_func: The query function to use for checking (async)
        workspace_dir: Working directory path
        config: Configuration dictionary  
        quiet_mode: If True, suppress tool usage logs and only show Claude responses
        
    Returns:
        Tuple of (model_name, total_cost)
    """
    from log import log_model_check, log_working_dir_check
    
    total_cost = 0.0
    
    # Check model
    log_model_check()
    prompt = get_model_identification_prompt()
    # Use show_tool_usage=False in quiet mode to suppress tool logs
    result, cost = await query_func(
        prompt, 
        workspace_dir=workspace_dir, 
        config=config,
        show_tool_usage=not quiet_mode
    )
    model_name = result['output']
    logger.info(f"💰 Model check cost: ${cost:.6f}")
    total_cost += cost
    
    # Check working directory
    if workspace_dir:
        log_working_dir_check()
        cwd_prompt = get_cwd_identification_prompt()
        # Use show_tool_usage=False in quiet mode to suppress tool logs
        result, cwd_cost = await query_func(
            cwd_prompt, 
            workspace_dir=workspace_dir, 
            config=config,
            show_tool_usage=not quiet_mode
        )
        logger.info(f"💰 Working directory check cost: ${cwd_cost:.4f}")
        total_cost += cwd_cost
    
    return model_name, total_cost


# Export all main functions
__all__ = [
    'get_model_identification_prompt',
    'get_cwd_identification_prompt', 
    'check_claude_model_and_cwd'
]