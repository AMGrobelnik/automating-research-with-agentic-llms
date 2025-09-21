#!/usr/bin/env python3
"""
Claude SDK - Timeout Management using native asyncio
Handles session timeouts and automatic restarts with continuation prompts.
"""

import asyncio
from loguru import logger

from .logging import get_last_todo_status, reset_task_counts
import re


def create_timeout_continuation_prompt(
    original_prompt: str, timeout_seconds: int, last_todo_status: str = None, error_type: str = "timeout"
) -> str:
    """Create a continuation prompt for when a session fails."""

    # Base continuation prompt
    if error_type == "timeout":
        error_msg = f"it hung after one of its tool calls took more than {timeout_seconds} seconds to complete"
    elif error_type == "incomplete":
        error_msg = "it ended without completing all TODO tasks. ALL tasks must be completed"
    else:
        error_msg = "it encountered an error and had to restart"
    
    continuation_prompt = f"""IMPORTANT: Another Claude Code session was previously working on this task but {error_msg}. You need to continue from where your predecessor left off and complete all the instructions.

YOUR TASK IS:
{original_prompt}"""

    # Add TODO status if available
    if last_todo_status:
        continuation_prompt += f"""

CURRENT PROGRESS STATUS:
The last recorded TODO status before the timeout was:
{last_todo_status}

This shows exactly what has been completed ([completed]), what was in progress ([in_progress]), and what remains to be done ([pending]). Use this information to understand the current state and continue from where the previous session left off."""

    continuation_prompt += """

Please:
1. First, assess the current state of the workspace to understand what work has already been completed
2. Continue from where the previous session left off based on the TODO status above
3. Complete all remaining instructions from the original task
4. Do not restart work that has already been completed successfully
5. Update the TODO list as you progress to track your work

IMPORTANT: ALL TODO TASKS MUST BE COMPLETED. The session will not be considered successful until every single task is marked as [completed]. Any tasks marked as [pending] or [in_progress] must be finished.

Begin by examining the workspace to understand the current progress, then continue with the implementation."""

    return continuation_prompt


async def query_with_retry(query_func, prompt: str, max_retries: int = 3, **kwargs):
    """
    Wrapper function that handles automatic retry when sessions timeout.

    The query function handles per-message timeout detection internally.
    This wrapper catches timeout errors and retries with continuation prompts.

    Args:
        query_func: The query function to wrap (async)
        prompt: The original prompt
        max_retries: Maximum number of retries on timeout
        **kwargs: Additional arguments for the query function

    Returns:
        Tuple of (result, total_cost)
    """
    retry_count = 0
    current_prompt = prompt
    total_cost = 0.0

    # Remove timeout_manager from kwargs if present
    kwargs.pop("timeout_manager", None)

    while retry_count <= max_retries:
        try:
            logger.info(
                f"🚀 Starting Claude session (retry {retry_count}/{max_retries})"
            )
            # Reset task counts for new session
            reset_task_counts()
            
            if retry_count > 0:
                logger.warning(
                    f"⏰ Previous session timed out - creating new session..."
                )

            # Run the query function - it handles per-message timeout internally
            result, cost = await query_func(current_prompt, **kwargs)
            total_cost += cost

            # Check if there are incomplete todos before marking as successful
            last_todo_status = get_last_todo_status()
            if last_todo_status and has_incomplete_todos(last_todo_status):
                logger.warning("⚠️ Session finished but found incomplete todos")
                logger.info("📋 Current TODO status:\n" + str(last_todo_status))
                
                retry_count += 1
                if retry_count <= max_retries:
                    # Create continuation prompt to complete remaining tasks
                    current_prompt = create_timeout_continuation_prompt(
                        prompt,
                        30,  # Default timeout for prompt text
                        last_todo_status=last_todo_status,
                        error_type="incomplete"
                    )
                    logger.warning(
                        f"🔄 Creating new session to complete remaining tasks (retry {retry_count}/{max_retries})"
                    )
                    logger.warning(f"⚠️ IMPORTANT: All todo tasks MUST be completed before the session can finish successfully")
                    continue
                else:
                    logger.error(
                        f"❌ Maximum retries ({max_retries}) exceeded with incomplete todos. Task failed."
                    )
                    raise Exception(
                        f"Session ended with incomplete todos after {max_retries + 1} attempts. All tasks must be completed."
                    )
            
            logger.success(f"✅ Session completed successfully (cost: ${cost:.4f})")
            return result, total_cost

        except Exception as e:
            # Escape curly braces for loguru to prevent formatting errors
            error_msg = str(e).replace("{", "{{").replace("}", "}}")
            
            if isinstance(e, asyncio.TimeoutError):
                logger.error(f"⏰ Session timed out (retry {retry_count}/{max_retries})")
            else:
                logger.error(f"❌ Session failed with error: {error_msg} (retry {retry_count}/{max_retries})")

            retry_count += 1

            if retry_count <= max_retries:
                # Get the last TODO status from global tracking
                last_todo_status = get_last_todo_status()

                # Create continuation prompt for next retry with TODO status
                error_type = "timeout" if isinstance(e, asyncio.TimeoutError) else "error"
                current_prompt = create_timeout_continuation_prompt(
                    prompt,
                    30,  # Default timeout for prompt text
                    last_todo_status=last_todo_status,
                    error_type=error_type
                )
                logger.warning(
                    f"🔄 Creating new session for retry {retry_count}/{max_retries} with continuation prompt"
                )
                if last_todo_status:
                    logger.info(f"📋 Including last TODO status in continuation prompt")
                
                # Add exponential backoff for non-timeout errors
                if not isinstance(e, asyncio.TimeoutError):
                    wait_time = min(2 ** (retry_count - 1), 60)  # Max 60 seconds
                    logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"❌ Maximum retries ({max_retries}) exceeded. Task failed."
                )
                if isinstance(e, asyncio.TimeoutError):
                    raise Exception(
                        f"Session timed out {max_retries + 1} times - giving up"
                    )
                else:
                    raise Exception(
                        f"Failed after {max_retries + 1} attempts. Last error: {error_msg}"
                    )

    # Should never reach here, but just in case
    raise Exception("Unexpected error in timeout wrapper")


def has_incomplete_todos(todo_status: str) -> bool:
    """
    Check if there are any pending or in_progress todos in the status.
    
    Args:
        todo_status: The TODO status string containing items like:
                    "1. [completed] Task 1
                     2. [in_progress] Task 2
                     3. [pending] Task 3"
    
    Returns:
        True if there are any [in_progress] or [pending] tasks, False otherwise
    """
    if not todo_status:
        return False
    
    # Check for any [in_progress] or [pending] status markers
    incomplete_pattern = r'\[(in_progress|pending)\]'
    return bool(re.search(incomplete_pattern, todo_status))


# Export all main functions
__all__ = ["create_timeout_continuation_prompt", "query_with_retry", "has_incomplete_todos"]
