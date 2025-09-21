#!/usr/bin/env python3
"""
_6_result_present.py - Result Presentation Module
Generates a LaTeX PDF paper from evaluation results using Claude.
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
# Truncation not needed for result presentation module


@logger.catch(reraise=True)
async def run_result_present_module(config, input_text, run_dir=None, workspace_dir=None):
    """Run the result presentation module with provided input."""
    log_handler_id = None
    try:
        # Import prompt functions based on domain
        domain = config.get('domain', 'MAS')

        if domain == 'MAS':
            from prompts._6_MAS_ResultPresent_Prompt import get_exec_prompt
            from prompts._6_MAS_ResultPresent_SysPrompt import get_exec_sysprompt
        else:
            logger.error(f"❌ Unknown domain: {domain}")
            return None

        logger.info(f"✅ Loaded result presentation prompts for domain: {domain}")

        # Copy checkpoint files if resuming from eval_out.json checkpoint
        start_checkpoint = config.get('pipeline', {}).get('start_checkpoint', None)

        if start_checkpoint and Path(start_checkpoint).exists():
            checkpoint_path = Path(start_checkpoint)

            # If resuming from eval_out.json, copy it to workspace
            if checkpoint_path.name == 'eval_out.json' and workspace_dir:
                import shutil

                checkpoint_dir = checkpoint_path.parent

                # Copy all JSON files and their variations
                json_files_to_copy = [
                    # Main output files (required)
                    'eval_out.json', 'method_out.json', 'data_out.json',
                    # Truncated versions for Claude to read
                    'eval_out_trunc.json', 'method_out_trunc.json', 'data_out_trunc.json',
                    # Mini versions with just a few examples
                    'eval_out_mini.json', 'method_out_mini.json', 'data_out_mini.json',
                    # Format files showing expected structure
                    'eval_out_format.json', 'method_out_format.json', 'data_out_format.json'
                ]

                for json_file in json_files_to_copy:
                    json_path = checkpoint_dir / json_file
                    if json_path.exists():
                        dest_json_path = workspace_dir / json_file
                        shutil.copy2(json_path, dest_json_path)
                        logger.info(f"📄 Copied {json_file} to workspace")

                # Copy code files from the same directory as eval_out.json
                # Copy method.py, data.py, and eval.py files from checkpoint directory
                for py_file in ['method.py', 'data.py', 'eval.py']:
                    py_path = checkpoint_dir / py_file
                    if py_path.exists():
                        dest_py_path = workspace_dir / py_file
                        shutil.copy2(py_path, dest_py_path)
                        logger.info(f"📄 Copied {py_file} from checkpoint directory to workspace")
                    else:
                        logger.warning(f"⚠️ {py_file} not found in checkpoint directory: {checkpoint_dir}")

        # Create output directory within the run directory
        if run_dir:
            output_dir = run_dir / "result_present"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = config.get('output', {}).get('directory', 'runs')
            output_dir = Path(f"{output_base}/{timestamp}_result_present")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for logging
        log_file = output_dir / "full_run.log"
        log_handler_id = logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            colorize=False
        )

        logger.info("✅ Loaded result presentation prompt function")

        # Get the complete prompt with input and workspace_dir (as absolute path)
        workspace_dir_abs = str(Path(workspace_dir).resolve()) if workspace_dir else ""
        claude_prompt = get_exec_prompt(input_text, workspace_dir_abs)

        # Log the full Claude prompt
        claude_prompt_lines = claude_prompt.split('\n')
        logger.info("📝 ResultPresent Claude Prompt (full):")
        logger.info("=" * 80)
        for i, line in enumerate(claude_prompt_lines):
            # Escape special characters for loguru
            escaped_line = escape_line_for_logging(line)
            logger.info(f"{i+1:3d} | {escaped_line}")
        logger.info("=" * 80)

        # Use Claude with streaming (tool usage auto-detected from global config)
        # Get the system prompt for result presentation
        system_prompt = get_exec_sysprompt()

        _, total_cost = await query_claudesdk_streaming(
            prompt=claude_prompt,
            system_prompt=system_prompt,  # Use result presentation system prompt
            config=config,
            workspace_dir=workspace_dir
        )

        logger.success(f"✅ ResultPresent execution complete (cost: ${total_cost:.4f})")

        # Skip Claude log file generation for module 6

        # Claude generates the paper.pdf directly via prompt
        final_output_path = workspace_dir / "paper.pdf" if workspace_dir else output_dir / "paper.pdf"

        # Create simplified result with only metadata (only thing used by pipeline)
        result = {
            'final_out_fpath': str(final_output_path),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'claude_cost_usd': total_cost,
                'module': 'result_present',
                'output_dir': str(output_dir)
            }
        }

        # Save result
        with open(output_dir / "module_result.json", 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Skip final_output.txt generation for module 6

        logger.info(f"✅ Module outputs saved to: {output_dir}")
        return result

    finally:
        if log_handler_id is not None:
            logger.remove(log_handler_id)


async def main():
    """Main function for standalone execution."""
    import yaml

    if len(sys.argv) < 2:
        logger.error("Usage: python _6_result_present.py <input_file_or_text>")
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
    run_dir = Path(f"runs/{timestamp}_result_present")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create model workspace directory
    workspace_dir = run_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    result = await run_result_present_module(config, input_text, run_dir=run_dir, workspace_dir=workspace_dir)
    if result:
        logger.success("✅ Result presentation completed successfully")
        return 0
    else:
        logger.error("❌ Result presentation failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)