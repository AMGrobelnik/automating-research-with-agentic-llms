#!/usr/bin/env python3
"""
AI Scientist Lite - Pipeline Runner
Executes research prompts in order to generate innovative ideas.
"""

import asyncio
import yaml
import shutil
from datetime import datetime
from pathlib import Path


# Import logging functions
from log import (
    logger,
    log_config_loaded,
    log_config_error,
    log_pipeline_start,
    log_pipeline_complete,
    log_pipeline_failed
)

# Import module runners
from _1_idea_gen import run_idea_gen_module
from _2_idea_filter import run_idea_filter_module
from _3_data_exec import run_data_exec_module
from _4_method_exec import run_method_exec_module
from _5_eval_exec import run_eval_exec_module
from _6_result_present import run_result_present_module


def load_config(config_path='pipeline_config.yaml'):
    """Load pipeline configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        log_config_error(config_path)
        return None
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing config file: {e}")
        return None

    log_config_loaded(config_path)
    return config


async def run_pipeline(config, run_dir=None, workspace_dir=None):
    """Run the pipeline starting from a given checkpoint or from beginning, respecting end_module."""
    pipeline_config = config.get('pipeline', {})
    start_checkpoint = pipeline_config.get('start_checkpoint', '').strip()
    idea_checkpoint = pipeline_config.get('idea_checkpoint', '').strip()

    # Define the complete pipeline sequence - now with 6 modules
    pipeline_sequence = ['idea_gen', 'idea_filter', 'data_exec', 'method_exec', 'eval_exec', 'result_present']

    # Determine starting point
    if start_checkpoint and Path(start_checkpoint).exists():
        checkpoint_path = Path(start_checkpoint)

        # Handle .txt file checkpoint
        if start_checkpoint.endswith('.txt'):
            logger.info(f"🚀 Running pipeline from checkpoint: {start_checkpoint}")

            # Load checkpoint content from text file
            with open(start_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_content = f.read()

            # Determine module from filename pattern
            filename = Path(start_checkpoint).name
            if 'IdeaGen' in filename:
                module_name = 'idea_gen'
            elif 'IdeaFilter' in filename:
                module_name = 'idea_filter'
            else:
                logger.error(f"Cannot determine module from checkpoint filename: {filename}")
                logger.error("Filename must contain one of: IdeaGen, IdeaFilter")
                return None

        # Handle .json file checkpoint (for three-stage execution)
        elif start_checkpoint.endswith('.json'):
            logger.info(f"🚀 Running pipeline from JSON checkpoint: {start_checkpoint}")

            filename = Path(start_checkpoint).name

            # Map JSON checkpoints to module names
            if filename == 'data_out.json':
                module_name = 'data_exec'
                # Load the idea from idea_checkpoint (required for JSON checkpoints)
                if not idea_checkpoint:
                    logger.error("❌ idea_checkpoint is required when starting from data_out.json")
                    logger.error("Please specify idea_checkpoint in pipeline_config.yaml")
                    return None
                if not Path(idea_checkpoint).exists():
                    logger.error(f"❌ idea_checkpoint file not found: {idea_checkpoint}")
                    return None
                with open(idea_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_content = f.read()
                logger.info(f"📄 Loaded idea from idea_checkpoint: {idea_checkpoint}")

            elif filename == 'method_out.json':
                module_name = 'method_exec'
                # Load the idea from idea_checkpoint (required for JSON checkpoints)
                if not idea_checkpoint:
                    logger.error("❌ idea_checkpoint is required when starting from method_out.json")
                    logger.error("Please specify idea_checkpoint in pipeline_config.yaml")
                    return None
                if not Path(idea_checkpoint).exists():
                    logger.error(f"❌ idea_checkpoint file not found: {idea_checkpoint}")
                    return None
                with open(idea_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_content = f.read()
                logger.info(f"📄 Loaded idea from idea_checkpoint: {idea_checkpoint}")

            elif filename == 'eval_out.json':
                module_name = 'eval_exec'
                # Load the idea from idea_checkpoint (required for JSON checkpoints)
                if not idea_checkpoint:
                    logger.error("❌ idea_checkpoint is required when starting from eval_out.json")
                    logger.error("Please specify idea_checkpoint in pipeline_config.yaml")
                    return None
                if not Path(idea_checkpoint).exists():
                    logger.error(f"❌ idea_checkpoint file not found: {idea_checkpoint}")
                    return None
                with open(idea_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_content = f.read()
                logger.info(f"📄 Loaded idea from idea_checkpoint: {idea_checkpoint}")
            elif filename == 'paper.pdf':
                module_name = 'result_present'
                logger.info("📄 Paper already generated, pipeline complete")
                return None
            else:
                logger.error(f"Unknown JSON checkpoint file: {filename}")
                logger.error("Valid JSON checkpoints: data_out.json, method_out.json, eval_out.json")
                return None

        else:
            logger.error(f"Checkpoint must be a .txt or .json file, got: {start_checkpoint}")
            return None

        # Log checkpoint detection with appropriate description
        checkpoint_descriptions = {
            'idea_gen': 'IdeaGen (idea generation)',
            'idea_filter': 'IdeaFilter (idea filtering/selection)',
            'data_exec': 'DataExec (data acquisition/preprocessing)',
            'method_exec': 'MethodExec (method implementation)',
            'eval_exec': 'EvalExec (evaluation)',
            'result_present': 'ResultPresent (paper generation)'
        }
        module_desc = checkpoint_descriptions.get(module_name, module_name)
        logger.info(f"📄 Detected checkpoint from module: {module_desc}")

        # Find starting point
        try:
            start_index = pipeline_sequence.index(module_name) + 1  # Start from next module
        except ValueError:
            logger.error(f"Unknown module in checkpoint: {module_name}")
            return None

        if start_index >= len(pipeline_sequence):
            logger.error(f"Cannot continue from {module_desc} - it's the last module in the pipeline")
            logger.info("Tip: If you want to re-run the pipeline, use an earlier checkpoint or start from beginning")
            return None

        current_input = checkpoint_content
    else:
        # No checkpoint - start from beginning
        logger.info("🚀 Running pipeline from beginning (no checkpoint provided)")
        start_index = 0
        current_input = None

    # Determine end point
    end_module = pipeline_config.get('end_module', None)
    if end_module:
        try:
            end_index = pipeline_sequence.index(end_module)
        except ValueError:
            logger.error(f"Invalid end_module: {end_module}")
            logger.error("end_module must be one of: idea_gen, idea_filter, data_exec, method_exec, eval_exec, result_present")
            return None

        if end_index < start_index:
            logger.error(f"Invalid configuration: end_module '{end_module}' comes before the starting module '{pipeline_sequence[start_index]}'")
            return None

        modules_to_run = pipeline_sequence[start_index:end_index+1]
        logger.info(f"🎯 Will run modules: {' → '.join(modules_to_run)}")
    else:
        end_index = len(pipeline_sequence) - 1
        modules_to_run = pipeline_sequence[start_index:]
        logger.info(f"🎯 Will run modules: {' → '.join(modules_to_run)}")

    # Track results
    results = {}

    # Run modules in sequence from start to end
    for i in range(start_index, end_index + 1):
        current_module = pipeline_sequence[i]
        # Use same descriptions for consistency
        checkpoint_descriptions = {
            'idea_gen': 'IdeaGen (idea generation)',
            'idea_filter': 'IdeaFilter (idea filtering/selection)',
            'data_exec': 'DataExec (data acquisition/preprocessing)',
            'method_exec': 'MethodExec (method implementation)',
            'eval_exec': 'EvalExec (evaluation)',
            'result_present': 'ResultPresent (paper generation)'
        }
        module_desc = checkpoint_descriptions.get(current_module, current_module)
        logger.info(f"📦 Running module: {module_desc}")

        if current_module == "idea_gen":
            # For idea_gen, current_input will be None when starting from beginning
            result = await run_idea_gen_module(config, prep_input=current_input, run_dir=run_dir, workspace_dir=workspace_dir)
            if not result:
                logger.error(f"Module {module_desc} failed")
                return None
            results[current_module] = result
            # Read output for next module
            output_path = Path(result['final_out_fpath'])
            with open(output_path, 'r', encoding='utf-8') as f:
                current_input = f.read()

        elif current_module == "idea_filter":
            result = await run_idea_filter_module(config, input_text=current_input, run_dir=run_dir, workspace_dir=workspace_dir)
            if not result:
                logger.error(f"Module {module_desc} failed")
                return None
            results[current_module] = result
            # Read output for next module
            output_path = Path(result['final_out_fpath'])
            with open(output_path, 'r', encoding='utf-8') as f:
                current_input = f.read()

        elif current_module == "data_exec":
            result = await run_data_exec_module(config, input_text=current_input, run_dir=run_dir, workspace_dir=workspace_dir)
            if not result:
                logger.error(f"Module {module_desc} failed")
                return None
            results[current_module] = result
            # For next modules, still pass the original idea
            # The data_out.json is available in workspace for the next module to reference

        elif current_module == "method_exec":
            result = await run_method_exec_module(config, input_text=current_input, run_dir=run_dir, workspace_dir=workspace_dir)
            if not result:
                logger.error(f"Module {module_desc} failed")
                return None
            results[current_module] = result
            # For next modules, still pass the original idea
            # The method_out.json is available in workspace for the next module to reference

        elif current_module == "eval_exec":
            result = await run_eval_exec_module(config, input_text=current_input, run_dir=run_dir, workspace_dir=workspace_dir)
            if not result:
                logger.error(f"Module {module_desc} failed")
                return None
            results[current_module] = result
            # For next module, still pass the original idea
            # The eval_out.json is available in workspace for the next module to reference

        elif current_module == "result_present":
            result = await run_result_present_module(config, input_text=current_input, run_dir=run_dir, workspace_dir=workspace_dir)
            if not result:
                logger.error(f"Module {module_desc} failed")
                return None
            results[current_module] = result
            # Final module - paper.pdf is generated

        else:
            logger.error(f"Module {current_module} not implemented")
            return None

    logger.info("✅ Pipeline completed")
    return results


async def main():
    """Main orchestrator entry point."""
    # Load configuration
    config = load_config()
    if not config:
        return 1

    log_pipeline_start()

    # Create shared run directory with project subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = config.get('output', {}).get('directory', 'runs')
    run_dir = Path(f"{output_base}/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create workspace directory for AI model's working space
    workspace_dir = run_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created run directory: {run_dir}")
    logger.info(f"📂 Created model workspace directory: {workspace_dir}")

    # Copy agent files to workspace for Claude Code access
    agents_source = Path("/mnt/c/Users/adria/Downloads/AIScientist-Lite/.claude/agents")
    if agents_source.exists():
        agents_dest = workspace_dir / ".claude" / "agents"
        agents_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(agents_source, agents_dest, dirs_exist_ok=True)
        logger.info(f"📋 Copied agent files to workspace: {agents_dest}")

    # Copy domain-specific files to workspace
    domain = config.get('domain')
    if domain:
        # Look for domain files in the domain subfolder
        prompts_dir = Path("prompts") / domain
        if prompts_dir.exists():
            # Copy ALL files from the domain subfolder (not just those with prefix)
            domain_files = list(prompts_dir.glob("*"))
            # Filter out directories, only copy files
            domain_files = [f for f in domain_files if f.is_file()]

            if domain_files:
                logger.info(f"📚 Copying {len(domain_files)} files from {domain} folder to workspace")
                for source_file in domain_files:
                    # Keep the original filename exactly as is
                    dest_path = workspace_dir / source_file.name
                    shutil.copy2(source_file, dest_path)
                    logger.info(f"  ✓ Copied {source_file.name}")
            else:
                logger.warning(f"⚠️ No files found in {prompts_dir}")
        else:
            logger.warning(f"⚠️ Domain directory '{prompts_dir}' does not exist")
    else:
        logger.info("📋 No domain specified, skipping domain file copying")

    # Run the pipeline
    result = await run_pipeline(config, run_dir=run_dir, workspace_dir=workspace_dir)

    if result:
        log_pipeline_complete()
        return 0
    else:
        log_pipeline_failed()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)