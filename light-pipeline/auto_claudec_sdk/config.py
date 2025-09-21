#!/usr/bin/env python3
"""
Claude SDK - Configuration Management
Handles configuration validation and parsing for Claude Code SDK.
"""


def validate_claude_settings(config: dict) -> dict:
    """
    Validate and extract Claude SDK settings from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with validated Claude settings
        
    Raises:
        ValueError: If required settings are missing or invalid
    """
    if not config:
        raise ValueError("Configuration is required for Claude SDK settings")
    
    # Get Claude settings from config
    claude_settings = config.get('claude_settings', {})
    if not claude_settings:
        raise ValueError("claude_settings section is required in configuration")
    
    # Extract and validate required settings
    max_turns = claude_settings.get('max_turns')
    allowed_tools = claude_settings.get('allowed_tools')
    permission_mode = claude_settings.get('permission_mode')
    max_thinking_tokens = claude_settings.get('max_thinking_tokens')
    timeout_seconds = claude_settings.get('timeout_seconds')
    max_retries = claude_settings.get('max_retries', 3)  # Default to 3 retries
    
    # Validate required settings
    if max_turns is None:
        raise ValueError("max_turns is required in claude_settings")
    if allowed_tools is None:
        raise ValueError("allowed_tools is required in claude_settings")
    if permission_mode is None:
        raise ValueError("permission_mode is required in claude_settings")
    
    return {
        'max_turns': max_turns,
        'allowed_tools': allowed_tools,
        'permission_mode': permission_mode,
        'max_thinking_tokens': max_thinking_tokens,
        'timeout_seconds': timeout_seconds,
        'max_retries': max_retries
    }


def validate_tool_usage_settings(config: dict) -> dict:
    """
    Validate and extract tool usage settings from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with validated tool usage settings
        
    Raises:
        ValueError: If required settings are missing
    """
    # Get tool usage settings from config
    tool_usage_settings = config.get('tool_usage', {})
    if not tool_usage_settings:
        raise ValueError("tool_usage section is required in configuration")
    
    # Get required settings
    truncate_logs = tool_usage_settings.get('truncate_logs')
    show_all_tool_usage = tool_usage_settings.get('show_all_tool_usage')
    
    # Validate required settings
    if truncate_logs is None:
        raise ValueError("truncate_logs is required in tool_usage settings")
    if show_all_tool_usage is None:
        raise ValueError("show_all_tool_usage is required in tool_usage settings")
    
    return {
        'truncate_logs': truncate_logs,
        'show_all_tool_usage': show_all_tool_usage
    }


def get_timeout_settings(config: dict) -> tuple[float, int]:
    """
    Extract timeout settings from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (timeout_minutes, timeout_seconds) - timeout_minutes for backward compatibility
    """
    claude_settings = validate_claude_settings(config)
    timeout_seconds = claude_settings.get('timeout_seconds', 300)  # Default 5 minutes = 300 seconds
    timeout_minutes = timeout_seconds / 60.0 if timeout_seconds else None
    
    return timeout_minutes, timeout_seconds


# Export all main functions
__all__ = [
    'validate_claude_settings',
    'validate_tool_usage_settings',
    'get_timeout_settings'
]