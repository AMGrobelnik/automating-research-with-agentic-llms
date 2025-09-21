#!/usr/bin/env python3
"""
Auto Claude Code SDK - Simplified toolkit for Claude Code SDK interactions

This package provides:
- Streaming query functionality with timeout support
- Model identification and checking utilities
"""

# Core streaming functionality
from .streaming import query_claudesdk_streaming

# Model management
from .models import check_claude_model_and_cwd

# Package metadata
__version__ = "1.0.0"
__author__ = "AI Scientist Lite"
__description__ = "Auto Claude Code SDK utilities with timeout and restart support"

# Main exports - these are the primary functions users should import
__all__ = [
    # Primary streaming API
    'query_claudesdk_streaming',

    # Model management
    'check_claude_model_and_cwd'
]