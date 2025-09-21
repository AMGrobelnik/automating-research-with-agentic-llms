#!/usr/bin/env python3
"""
Claude SDK - Streaming Query Functions
Core streaming functionality for Claude Code SDK queries with timeout support.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from loguru import logger

from claude_code_sdk import (
    ClaudeCodeOptions,
    query,
)

from .config import (
    validate_claude_settings,
    validate_tool_usage_settings,
    get_timeout_settings,
)
from .logging import show_progress_indicator, complete_progress_indicator, reset_task_counts
from .retry import query_with_retry
from .msg_parse import parse_message


async def query_claudesdk_streaming(
    prompt: str,
    system_prompt: str = None,
    show_tool_usage: bool = None,
    config: dict = None,
    workspace_dir: Path = None,
):
    """
    Claude query function with streaming output and automatic timeout/restart functionality.

    This function wraps the core streaming logic with timeout detection and session restart.
    If no tool activity is detected within the configured timeout period, it will restart
    the session with a continuation prompt.

    Args:
        prompt: The prompt to send to Claude
        system_prompt: System prompt to use (optional, replaces default)
        show_tool_usage: Whether to display all blocks in concise format (default: False)
        config: Configuration dict (required for timeout settings)
        workspace_dir: Claude's working directory for the current run (optional)

    Returns:
        Tuple of (structured_result_dict, cost_usd) where structured_result_dict contains:
        - output: Final Claude responses (what user sees at the end)
        - intermediate: Tool interactions and thinking blocks
        - input: Original user prompt
        - full: Complete conversation with all messages
    """
    # Reset task counts at the start of any new orchestrator session
    # This ensures clean state when transitioning between orchestrators
    reset_task_counts()
    
    # Extract timeout settings from config
    if config:
        _, timeout_seconds = get_timeout_settings(config)
        claude_settings = validate_claude_settings(config)
        max_retries = claude_settings.get("max_retries", 3)
    else:
        timeout_seconds = None
        max_retries = 3  # Default

    # Use the retry wrapper if timeout is configured
    if timeout_seconds:
        return await query_with_retry(
            query_claudesdk_streaming_core,
            prompt=prompt,
            max_retries=max_retries,
            system_prompt=system_prompt,
            show_tool_usage=show_tool_usage,
            config=config,
            workspace_dir=workspace_dir,
            timeout_seconds=timeout_seconds,
        )
    else:
        # No timeout configured, use core function directly
        return await query_claudesdk_streaming_core(
            prompt=prompt,
            system_prompt=system_prompt,
            show_tool_usage=show_tool_usage,
            config=config,
            workspace_dir=workspace_dir,
            timeout_seconds=0,  # No timeout when wrapper not used
        )


async def query_claudesdk_streaming_core(
    prompt: str,
    system_prompt: str = None,
    show_tool_usage: bool = None,
    config: dict = None,
    workspace_dir: Path = None,
    timeout_seconds: int = None,
):
    """
    Core Claude query function with streaming output that returns structured conversation data.

    Args:
        prompt: The prompt to send to Claude
        system_prompt: System prompt to use (optional, replaces default)
        show_tool_usage: Whether to display all blocks in concise format (default: False)
        config: Configuration dict (required)
        workspace_dir: Claude's working directory for the current run (optional)
        timeout_seconds: Timeout per message in seconds (optional, defaults to config or 30s)

    Returns:
        Tuple of (structured_result_dict, cost_usd) where structured_result_dict contains:
        - output: Final Claude responses (what user sees at the end)
        - intermediate: Tool interactions and thinking blocks
        - input: Original user prompt
        - full: Complete conversation with all messages
    """
    # Validate configuration
    claude_settings = validate_claude_settings(config)
    tool_usage_settings = validate_tool_usage_settings(config)

    # Global TODO tracking is handled automatically by logging.py

    # Determine show_tool_usage setting
    if show_tool_usage is None:
        show_tool_usage = tool_usage_settings["show_all_tool_usage"]

    truncate_logs = tool_usage_settings["truncate_logs"]

    # Options for pipeline
    options = ClaudeCodeOptions(
        max_turns=claude_settings["max_turns"],
        allowed_tools=claude_settings["allowed_tools"],
        permission_mode=claude_settings["permission_mode"],
        system_prompt=system_prompt,
        cwd=str(workspace_dir) if workspace_dir else None,
    )

    # Add max_thinking_tokens if specified
    if claude_settings.get("max_thinking_tokens"):
        options.max_thinking_tokens = claude_settings["max_thinking_tokens"]

    # Storage for different message types
    all_messages = []
    assistant_text_blocks = []
    tool_interactions = []
    thinking_blocks = []
    response_text = ""
    cost_usd = 0.0
    tool_usage_count = 0
    last_tool_id = None
    last_tool_name = None
    agent_stack = []  # Stack to track nested agent contexts
    is_first_subagent_msg = False  # Track first message from subagent

    # Track the sequence of final Claude responses (for output section)
    final_claude_responses = []
    collecting_final_responses = False

    # Stream output in real-time with optional per-message timeout
    if timeout_seconds is None:
        timeout_seconds = (
            get_timeout_settings(config)[1] if config else 30
        )  # Default 30s timeout

    # Always use iterator approach to avoid code duplication
    iterator = aiter(query(prompt=prompt, options=options))

    try:
        while True:
            try:
                if timeout_seconds and timeout_seconds > 0:
                    # With timeout
                    message = await asyncio.wait_for(
                        anext(iterator), timeout=timeout_seconds
                    )
                else:
                    # Without timeout
                    message = await anext(iterator)
            except asyncio.TimeoutError:
                logger.error(
                    f"⏰ No message received for {timeout_seconds}s - session timed out"
                )
                raise  # This triggers retry in retry wrapper
            except StopAsyncIteration:
                break

            # DEBUG (DO NOT DELETE): Print full message details as they arrive (no truncation)
            # print(f"\n{'='*80}")
            # print(f"[MESSAGE RECEIVED] Type: {type(message).__name__}")
            # print(f"Full content: {message}")
            # print(f"{'='*80}\n")

            # Track all messages for full conversation
            all_messages.append(
                {
                    "type": type(message).__name__,
                    "content": str(message),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Create state dictionary for the parser
            state = {
                "response_text": response_text,
                "tool_usage_count": tool_usage_count,
                "last_tool_id": last_tool_id,
                "last_tool_name": last_tool_name,
                "assistant_text_blocks": assistant_text_blocks,
                "tool_interactions": tool_interactions,
                "thinking_blocks": thinking_blocks,
                "final_claude_responses": final_claude_responses,
                "collecting_final_responses": collecting_final_responses,
                "cost_usd": cost_usd,
                "agent_stack": agent_stack,
                "is_first_subagent_msg": is_first_subagent_msg,
            }

            # Parse the message using the new parser
            parse_message(message, show_tool_usage, truncate_logs, state)

            # Update local variables from state
            response_text = state["response_text"]
            tool_usage_count = state["tool_usage_count"]
            last_tool_id = state["last_tool_id"]
            last_tool_name = state["last_tool_name"]
            assistant_text_blocks = state["assistant_text_blocks"]
            tool_interactions = state["tool_interactions"]
            thinking_blocks = state["thinking_blocks"]
            final_claude_responses = state["final_claude_responses"]
            collecting_final_responses = state["collecting_final_responses"]
            cost_usd = state["cost_usd"]
            agent_stack = state["agent_stack"]
            is_first_subagent_msg = state["is_first_subagent_msg"]

    except RuntimeError as e:
        # Catch cancel scope error that happens when SDK iterator completes
        # This occurs AFTER all messages have been processed and logged
        if "cancel scope" in str(e).lower():
            logger.debug("Session ended (SDK cancel scope cleanup) - all messages processed successfully")
            # Don't re-raise, just continue to return results
        else:
            # Re-raise other RuntimeErrors
            raise

    # Complete progress indicator if not showing tool usage
    if not show_tool_usage:
        complete_progress_indicator(tool_usage_count)

    # Build structured result
    # Output: Final sequence of Claude responses (what the user sees at the end)
    output_text = "".join(final_claude_responses) if final_claude_responses else ""

    # Intermediate: Tool interactions and thinking
    intermediate_content = {
        "tool_interactions": tool_interactions,
        "thinking_blocks": thinking_blocks,
    }

    # Input: Original user prompt
    input_content = prompt

    # Full: Complete conversation with all messages preserved
    full_conversation = {
        "messages": all_messages,
        "assistant_text_blocks": assistant_text_blocks,
        "cost_usd": cost_usd,
    }

    # Structure the result in the requested order: output, intermediate, input, full
    structured_result = {
        "output": output_text,
        "intermediate": intermediate_content,
        "input": input_content,
        "full": full_conversation,
    }

    return structured_result, cost_usd


# Export all main functions
__all__ = ["query_claudesdk_streaming", "query_claudesdk_streaming_core"]
