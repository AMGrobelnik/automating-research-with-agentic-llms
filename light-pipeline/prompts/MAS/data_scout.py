#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets==3.6.0",
#   "huggingface-hub"
# ]
# ///

import json
import signal
import os
from datetime import datetime
from datasets import load_dataset
from huggingface_hub import login

def timeout_handler(signum, frame):
    raise TimeoutError("Loading timeout")

signal.signal(signal.SIGALRM, timeout_handler)
# login(token=os.getenv("HF_TOKEN"))  # TODO: Set HF_TOKEN environment variable and uncomment

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Standard format: YYYYMMDD_HHMMSS
output_dir = f"data_scout_out/{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

DATASETS = {  
    # Add your dataset configs here, examples:
    # "humaneval": ("openai/openai_humaneval", None, "test"),
    # "mbpp": ("google-research-datasets/mbpp", None, "test"),
    # "humaneval_plus": ("evalplus/humanevalplus", None, "test"),
    # "livecodebench_lite": ("livecodebench/code_generation_lite", None, "test"),
    # "mbpp_plus": ("evalplus/mbppplus", None, "test"),
    # "gsm8k": ("openai/gsm8k", "main", "test"),
    # "gsm_hard": ("reasoning-machines/gsm-hard", None, "train"),
    # "bug_fix_small": ("semeru/code-code-BugFixingSmall", None, "test"),
    # "sql_context": ("b-mc2/sql-create-context", None, "train"),
    # "synthetic_sql": ("gretelai/synthetic_text_to_sql", None, "test"),
}

for choice, (name, config, split) in DATASETS.items():  # Load ALL datasets
    print(f"\n{'='*60}\nLoading: {choice} -> {name}\n{'='*60}")
    try:
        signal.alarm(120)  # Set 60 second timeout
        dataset = load_dataset(name, config, split=split, trust_remote_code=True) if config else load_dataset(name, split=split, trust_remote_code=True)
        signal.alarm(0)  # Cancel timeout
        print(f"Fields: {list(dataset.features.keys())}\nTotal samples: {len(dataset)}\n")
        data = [dict(item) for item in dataset.select(range(min(200, len(dataset))))]  # Limit to 200 samples
        json.dump({"data": data}, open(f"{output_dir}/data_{choice}.json", "w"), indent=2, ensure_ascii=False, default=str)  # default=str handles datetime
        print(f"--- Sample 1 (of {len(data)}) ---\n{json.dumps(data[0], indent=2, default=str)[:400]}...\n\n✓ Success! Saved {len(data)} samples to {output_dir}/data_{choice}.json")
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"✗ Failed: {choice} - {str(e)}")