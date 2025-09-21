#!/usr/bin/env python3
"""
_3_data_exec.py - Data Execution Module
Executes data acquisition and preprocessing stage using Claude.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

# Import utilities
from auto_claudec_sdk import (
    query_claudesdk_streaming
)
from xml_escape_utils import escape_line_for_logging
from truncate_json import process_json_file, process_mini_json_file


@logger.catch(reraise=True)
async def run_data_exec_module(config, input_text, run_dir=None, workspace_dir=None):
    """Run the data execution module with provided input."""
    log_handler_id = None
    try:
        # Import prompt functions based on domain
        domain = config.get('domain', 'MAS')

        if domain == 'MAS':
            from prompts._3_MAS_DataExec_Prompt import get_exec_prompt
            from prompts._3_MAS_DataExec_SysPrompt import get_exec_sysprompt
        else:
            logger.error(f"❌ Unknown domain: {domain}")
            return None

        logger.info(f"✅ Loaded data exec prompts for domain: {domain}")
        # Create output directory within the run directory
        if run_dir:
            output_dir = run_dir / "data_exec"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = config.get('output', {}).get('directory', 'runs')
            output_dir = Path(f"{output_base}/{timestamp}_data_exec")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for logging
        log_file = output_dir / "full_run.log"
        log_handler_id = logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            colorize=False
        )

        logger.info("✅ Loaded data exec prompt function")

        # Get the complete prompt with input and workspace_dir (as absolute path)
        workspace_dir_abs = str(Path(workspace_dir).resolve())
        claude_prompt = get_exec_prompt(input_text, workspace_dir_abs)

        # Log the full Claude prompt
        claude_prompt_lines = claude_prompt.split('\n')
        logger.info("📝 DataExec Claude Prompt (full):")
        logger.info("=" * 80)
        for i, line in enumerate(claude_prompt_lines):
            # Escape special characters for loguru
            escaped_line = escape_line_for_logging(line)
            logger.info(f"{i+1:3d} | {escaped_line}")
        logger.info("=" * 80)

        # Use Claude with streaming (tool usage auto-detected from global config)
        # Get the system prompt for data execution
        system_prompt = get_exec_sysprompt()

        _, total_cost = await query_claudesdk_streaming(
            prompt=claude_prompt,
            system_prompt=system_prompt,  # Use data exec system prompt
            config=config,
            workspace_dir=workspace_dir
        )

        logger.success(f"✅ DataExec execution complete (cost: ${total_cost:.4f})")

        # Skip Claude log file generation for module 3

        # Claude writes the file directly via prompt
        final_output_path = workspace_dir / "data_out.json" if workspace_dir else output_dir / "data_out.json"

        # Create truncated version of data_out.json
        if final_output_path.exists():
            logger.info("📄 Creating truncated version of data_out.json...")
            truncated_path = final_output_path.parent / "data_out_trunc.json"
            success = process_json_file(final_output_path, truncated_path)
            if not success:
                logger.warning("⚠️ Could not create truncated version")

            # Create mini version with just 3 full examples
            logger.info("📄 Creating mini version of data_out.json with 3 full examples...")
            mini_path = final_output_path.parent / "data_out_mini.json"
            success = process_mini_json_file(final_output_path, mini_path, max_examples=3)
            if not success:
                logger.warning("⚠️ Could not create mini version")

        # Create simplified result with only metadata (only thing used by pipeline)
        result = {
            'final_out_fpath': str(final_output_path),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'claude_cost_usd': total_cost,
                'module': 'data_exec',
                'output_dir': str(output_dir)
            }
        }

        # Save result
        with open(output_dir / "module_result.json", 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Skip final_output.txt generation for module 3

        logger.info(f"✅ Module outputs saved to: {output_dir}")
        return result

    finally:
        if log_handler_id is not None:
            logger.remove(log_handler_id)


async def main():
    """Main function for standalone execution."""
    import yaml

    if len(sys.argv) < 2:
        logger.error("Usage: python _3_data_exec.py <input_file_or_text>")
        return 1

    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get input text
    input_arg = sys.argv[1]
    if Path(input_arg).exists():
        # Input is a file path
        input_file = Path(input_arg)
        if input_file.suffix == '.json':
            # Load from module_result.json
            with open(input_file, 'r') as f:
                data = json.load(f)
                input_text = data.get('output', '')
        else:
            # Load from text file
            with open(input_file, 'r') as f:
                input_text = f.read()
    else:
        # Input is direct text
        input_text = input_arg

    if not input_text.strip():
        logger.error("No input text provided")
        return 1

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_data_exec")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create model workspace directory
    workspace_dir = run_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    result = await run_data_exec_module(config, input_text, run_dir=run_dir, workspace_dir=workspace_dir)
    if result:
        logger.success("✅ Data execution completed successfully")
        return 0
    else:
        logger.error("❌ Data execution failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)