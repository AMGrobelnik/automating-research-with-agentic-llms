#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

# Import prompt function
from prompts._1_IdeaGen_Prompt import get_idea_gen_prompt

# Import utilities
from openai_utils import OpenAIUtils
from log import (
    log_idea_gen_start,
    log_idea_gen_success
)


@logger.catch(reraise=True)
async def run_idea_gen_module(config, prep_input=None, run_dir=None, workspace_dir=None):
    """Run the idea generation module using OpenAI."""
    log_handler_id = None
    
    @logger.catch(reraise=True)
    def _cleanup():
        """Cleanup function to ensure log handler is removed."""
        nonlocal log_handler_id
        if log_handler_id is not None:
            logger.remove(log_handler_id)
    
    # Create output directory within the run directory
    if run_dir:
        output_dir = run_dir / "idea_gen"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = config.get('output', {}).get('directory', 'runs')
        output_dir = Path(f"{output_base}/{timestamp}_idea_gen")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for logging
    log_file = output_dir / "full_run.log"
    log_handler_id = logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        colorize=False
    )
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAIUtils()
        
        # Get research domain from config
        research_domain = config.get('research_area', 'Multi LLM Agent Systems')
        
        # Get the complete prompt with prep input handled
        prep_input = prep_input or ""  # Ensure prep_input is never None
        ideagen_prompt = get_idea_gen_prompt(prep_input, research_domain)
        
        if prep_input.strip():
            logger.info("✅ Using research context from idea preparation module")
        else:
            logger.info("⚠️ No prep input provided - generating ideas without research context")
        logger.info(f"✅ Loaded idea gen prompt function with research domain: {research_domain}")
        
        log_idea_gen_start()
        
        
        # Get reasoning effort and verbosity from config
        reasoning_effort = config.get('openai_settings', {}).get('reasoning_effort', 'high')
        verbosity = config.get('openai_settings', {}).get('verbosity', 'medium')
        
        openai_response = openai_client.create_response(
            ideagen_prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity
        )
        
        # Extract the text from OpenAI response
        openai_output = openai_client.extract_text_from_response(openai_response)
        
        if not openai_output:
            logger.error("No output received from OpenAI")
            return None
        
        log_idea_gen_success(len(openai_output))
        
        # Skip IdeaGen.log generation
        
        # Create simplified result with only metadata (only thing used by pipeline)
        final_output_path = workspace_dir / "IdeaGen_final_out.txt" if workspace_dir else output_dir / "IdeaGen_final_out.txt"
        
        # Save final clean output to final path before creating result
        with logger.catch(reraise=True):
            with open(final_output_path, 'w', encoding='utf-8') as f:
                f.write(openai_output)
            logger.info(f"✅ Final clean output saved to {final_output_path}")
        
        result = {
            'final_out_fpath': str(final_output_path),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'openai_model': str(openai_client.model),  # Ensure string conversion
                'module': 'idea_gen',
                'output_dir': str(output_dir)
            }
        }
        
        # Save result
        with open(output_dir / "module_result.json", 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # File already saved above before creating result
        
        logger.info(f"✅ Module outputs saved to: {output_dir}")
        return result
    finally:
        _cleanup()


async def main():
    """Main function for standalone execution."""
    import yaml
    
    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_idea_gen")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create workspace directory
    workspace_dir = run_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    result = await run_idea_gen_module(config, prep_input=None, run_dir=run_dir, workspace_dir=workspace_dir)
    if result:
        logger.success("✅ Idea generation completed successfully")
        return 0
    else:
        logger.error("❌ Idea generation failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)