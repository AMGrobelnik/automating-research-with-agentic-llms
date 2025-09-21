#!/usr/bin/env python3
"""
truncate_json.py - JSON file processing utilities
- Truncate long fields in JSON files (keeps first 500 and last 100 characters)
- Limit 'examples' field to max 3 examples per dataset type
- Create mini versions with just first 3 full examples
"""

import json
import sys
from pathlib import Path
from loguru import logger


def truncate_middle(text, max_length=600, prefix_length=500):
    """Truncate text in the middle, keeping prefix and suffix."""
    if not isinstance(text, str):
        return text

    if len(text) <= max_length:
        return text

    suffix_length = 100  # Fixed suffix length
    prefix = text[:prefix_length]
    suffix = text[-suffix_length:]

    # Add indicator of truncation
    truncation_msg = f"\n\n[... {len(text) - (prefix_length + suffix_length)} characters truncated ...]\n\n"

    return prefix + truncation_msg + suffix


def truncate_examples_by_dataset(examples, max_per_dataset=3):
    """Truncate examples list to max N examples per dataset type."""
    if not isinstance(examples, list):
        return examples

    # Group examples by dataset
    dataset_groups = {}
    for example in examples:
        if isinstance(example, dict) and 'dataset' in example:
            dataset = example['dataset']
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(example)
        else:
            # If no dataset field, treat as special 'unknown' dataset
            if 'unknown' not in dataset_groups:
                dataset_groups['unknown'] = []
            dataset_groups['unknown'].append(example)

    # Build truncated list with max N examples per dataset
    truncated = []

    for dataset, dataset_examples in sorted(dataset_groups.items()):
        if len(dataset_examples) > max_per_dataset:
            truncated.extend(dataset_examples[:max_per_dataset])
        else:
            truncated.extend(dataset_examples)

    return truncated


def truncate_json_fields(obj, max_length=600, trunc_num_examples=True):
    """Recursively truncate string fields in a JSON object.

    Args:
        obj: The JSON object to truncate
        max_length: Maximum length for string fields
        trunc_num_examples: If True, limit examples to 3 per dataset. If False, keep all examples.
                           Defaults to True for backward compatibility.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Special handling for 'examples' field
            if k == 'examples' and isinstance(v, list) and trunc_num_examples:
                result[k] = truncate_examples_by_dataset(v)
                # Still apply string truncation to the examples
                result[k] = truncate_json_fields(result[k], max_length, trunc_num_examples)
            else:
                result[k] = truncate_json_fields(v, max_length, trunc_num_examples)
        return result
    elif isinstance(obj, list):
        return [truncate_json_fields(item, max_length, trunc_num_examples) for item in obj]
    elif isinstance(obj, str):
        return truncate_middle(obj, max_length)
    else:
        return obj


def create_mini_json(obj, max_examples=3):
    """Create a mini version with just first N full examples (no truncation)."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Special handling for 'examples' field
            if k == 'examples' and isinstance(v, list):
                # Just take first N examples, no truncation
                result[k] = v[:max_examples] if len(v) > max_examples else v
            else:
                result[k] = create_mini_json(v, max_examples)
        return result
    elif isinstance(obj, list):
        # For non-examples lists, keep as is
        return [create_mini_json(item, max_examples) for item in obj]
    else:
        # Keep all other data types unchanged
        return obj


def process_mini_json_file(input_path, output_path=None, max_examples=3):
    """Process a JSON file and create mini version with first N full examples."""
    input_path = Path(input_path)

    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}")
        return False

    # Default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_mini{input_path.suffix}"
    else:
        output_path = Path(output_path)

    try:
        # Read the original JSON
        logger.info(f"📖 Reading: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create mini version
        logger.info(f"📦 Creating mini version with {max_examples} examples...")
        mini_data = create_mini_json(data, max_examples)

        # Save mini version
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mini_data, f, indent=2, ensure_ascii=False)

        logger.success(f"✅ Created mini version: {output_path}")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in {input_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}")
        return False


def process_json_file(input_path, output_path=None, trunc_num_examples=True):
    """Process a JSON file and create truncated version.

    Args:
        input_path: Path to input JSON file
        output_path: Optional output path (defaults to input_trunc.json)
        trunc_num_examples: If True, limit examples to 3 per dataset. If False, keep all examples.
                           Defaults to True for backward compatibility.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}")
        return False

    # Default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_trunc{input_path.suffix}"
    else:
        output_path = Path(output_path)

    try:
        # Read the original JSON
        logger.info(f"📖 Reading: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create truncated version
        if trunc_num_examples:
            logger.info("✂️ Truncating long fields and limiting examples...")
        else:
            logger.info("✂️ Truncating long fields (keeping all examples)...")
        truncated_data = truncate_json_fields(data, trunc_num_examples=trunc_num_examples)

        # Save truncated version
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(truncated_data, f, indent=2, ensure_ascii=False)

        logger.success(f"✅ Created truncated version: {output_path}")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in {input_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}")
        return False


def main():
    """Main entry point for standalone usage."""
    if len(sys.argv) < 2:
        print("Usage: python truncate_json.py <input_json> [output_json]")
        print("  If output_json is not provided, will create <input>_trunc.json")
        return 1

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = process_json_file(input_file, output_file)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())