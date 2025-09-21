#!/usr/bin/env python3
"""
Test Claude Code SDK streaming capability with a simple prompt.
Tests the streaming functionality using the auto_claudec_sdk module.
"""

import asyncio
import yaml
from datetime import datetime
from pathlib import Path

# Import the pre-configured logger with all custom levels
from log import logger

# Import utilities from auto_claudec_sdk
from auto_claudec_sdk import query_claudesdk_streaming

# Define color constants for output
BLUE, GREEN, YELLOW, CYAN, END = (
    "\033[94m",
    "\033[92m",
    "\033[93m",
    "\033[96m",
    "\033[0m",
)


async def test_claude_streaming():
    """
    Test Claude streaming with a researcher agent task.
    """
    logger.info(f"{BLUE}Starting Claude Code SDK streaming test{END}")

    # Load pipeline config
    config_path = Path("pipeline_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"📋 Loaded config from: {config_path}")

    # Prompt to use Task tool with researcher agent
    prompt = """Use the Task tool to launch two SEQUENTIAL research analyst agents to research using parallel tool calls and create a detailed report on the latest Apple news and product announcements. 

The agent should:
1. Search for the most recent Apple news (last 30 days)
2. Focus on product announcements, updates, and major company news
3. Create a comprehensive report with at least 5-7 key points
4. Include details about any new products, services, or significant updates

Make sure to use the Task tool with subagent_type="research-analyst" to handle this research task."""

    logger.info(f"{YELLOW}Sending prompt to Claude...{END}")
    logger.info(f"Prompt: {prompt[:100]}...")  # Show first 100 chars

    try:
        # Use query_claudesdk_streaming from auto_claudec_sdk
        structured_result, cost_usd = await query_claudesdk_streaming(
            prompt=prompt,
            show_tool_usage=True,  # Show all tool usage details
            config=config,
            workspace_dir=None,  # No workspace needed for this test
        )

        # Extract the response
        response_text = structured_result.get("output", "")

        # Print the response
        logger.info(f"\n{CYAN}Claude Response:{END}")
        print("-" * 60)
        print(response_text)
        print("-" * 60)

        # Log summary
        logger.success(f"{GREEN}Streaming completed successfully!{END}")
        logger.info(f"Response length: {len(response_text)} characters")
        logger.info(f"Cost: ${cost_usd:.6f} USD")

        # Log intermediate details if present
        intermediate = structured_result.get("intermediate", {})
        tool_interactions = intermediate.get("tool_interactions", [])
        thinking_blocks = intermediate.get("thinking_blocks", [])

        if tool_interactions:
            logger.info(f"Tool interactions: {len(tool_interactions)}")
            # Show tool names used
            tool_names = [t.get("tool_name", "unknown") for t in tool_interactions]
            logger.info(f"Tools used: {', '.join(set(tool_names))}")

        if thinking_blocks:
            logger.info(f"Thinking blocks: {len(thinking_blocks)}")

        return {
            "success": True,
            "response": response_text,
            "cost_usd": cost_usd,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def main():
    """
    Main entry point for the test.
    """
    logger.info(f"{CYAN}=" * 60 + f"{END}")
    logger.info(f"{CYAN}Claude Code SDK Streaming Test - Researcher Agent{END}")
    logger.info(f"{CYAN}=" * 60 + f"{END}")

    result = await test_claude_streaming()

    if result["success"]:
        logger.info(f"\n{GREEN}Test completed successfully!{END}")
        logger.info(f"Result summary:")
        logger.info(f"  - Response received: ✓")
        logger.info(f"  - Cost: ${result['cost_usd']:.6f}")
    else:
        logger.error(f"\n{YELLOW}Test failed!{END}")
        logger.error(f"Error: {result.get('error', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    # Run the async test
    exit_code = asyncio.run(main())
    import sys

    sys.exit(exit_code)
