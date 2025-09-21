#!/usr/bin/env python3
"""
_2_idea_filter.py - Idea Filter Module
Filters and analyzes research ideas using Claude.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

# Import prompt functions
from prompts._2_IdeaFilter_Prompt import get_idea_filter_prompt
from prompts._2_IdeaFilter_SysPrompt import get_ideafilter_sysprompt

# Import utilities
from auto_claudec_sdk import (
    query_claudesdk_streaming
)
from log import (
    log_idea_filter_start,
    log_claude_filtering_complete
)
from xml_escape_utils import escape_line_for_logging


@logger.catch(reraise=True)
async def run_idea_filter_module(config, input_text, run_dir=None, workspace_dir=None):
    """Run just the idea filter module with provided input."""
    log_handler_id = None
    try:
        # Create output directory within the run directory
        if run_dir:
            output_dir = run_dir / "idea_filter"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = config.get('output', {}).get('directory', 'runs')
            output_dir = Path(f"{output_base}/{timestamp}_idea_filter")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler for logging
        log_file = output_dir / "full_run.log"
        log_handler_id = logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            colorize=False
        )
        
        logger.info("✅ Loaded idea filter prompt function")
        
        
        log_idea_filter_start()
        
        # Get the complete prompt with input concatenated
        claude_prompt = get_idea_filter_prompt(input_text)
        
        # Log the full Claude prompt
        claude_prompt_lines = claude_prompt.split('\n')
        logger.info("📝 Claude Prompt (full):")
        logger.info("=" * 80)
        for i, line in enumerate(claude_prompt_lines):
            # Dynamically escape XML tags and loguru format characters
            escaped_line = escape_line_for_logging(line)
            logger.info(f"{i+1:3d} | {escaped_line}")
        logger.info("=" * 80)
        
        # Use Claude with streaming (tool usage auto-detected from global config)
        # Get orchestrator system prompt (which now includes critical override instructions)
        system_prompt = get_ideafilter_sysprompt()
        
        structured_result, total_cost = await query_claudesdk_streaming(
            prompt=claude_prompt,
            system_prompt=system_prompt,  # Use orchestrator prompt with all instructions
            config=config,
            workspace_dir=workspace_dir
        )
        
        
        log_claude_filtering_complete(total_cost)
        
        # Skip Claude log file generation for module 2
        
        # Claude writes the file directly via prompt
        final_output_path = workspace_dir / "IdeaFilter_final_out.txt" if workspace_dir else output_dir / "IdeaFilter_final_out.txt"
        
        # Create simplified result with only metadata (only thing used by pipeline)
        result = {
            'final_out_fpath': str(final_output_path),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'claude_cost_usd': total_cost,
                'module': 'idea_filter',
                'output_dir': str(output_dir)
            }
        }
        
        # Save result
        with open(output_dir / "module_result.json", 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Skip final_output.txt generation for module 2
        
        logger.info(f"✅ Module outputs saved to: {output_dir}")
        return result
        
    finally:
        if log_handler_id is not None:
            logger.remove(log_handler_id)


async def main():
    """Main function for standalone execution."""
    import yaml
    
    if len(sys.argv) < 2:
        logger.error("Usage: python _2_idea_filter.py <input_file_or_text>")
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
    run_dir = Path(f"runs/{timestamp}_idea_filter")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model workspace directory
    workspace_dir = run_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    result = await run_idea_filter_module(config, input_text, run_dir=run_dir, workspace_dir=workspace_dir)
    if result:
        logger.success("✅ Idea filtering completed successfully")
        return 0
    else:
        logger.error("❌ Idea filtering failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)