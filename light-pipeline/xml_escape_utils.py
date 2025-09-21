#!/usr/bin/env python3
"""
XML Escape Utilities
Dynamic XML tag detection and escaping for loguru logging.
"""

import re
from pathlib import Path
from typing import Set
from functools import lru_cache


@lru_cache(maxsize=1)
def get_xml_tags_from_prompts() -> Set[str]:
    """
    Dynamically scan all prompt files and extract XML tags.
    Cached for performance since prompt files don't change during execution.
    
    Returns:
        Set of XML tags found in prompt files
    """
    xml_tags = set()
    prompts_dir = Path(__file__).parent / "prompts"
    
    if not prompts_dir.exists():
        return xml_tags
    
    # Pattern to match XML tags: <tag> or </tag>
    xml_pattern = re.compile(r'<[^>]+>')
    
    for prompt_file in prompts_dir.glob("*.py"):
        try:
            content = prompt_file.read_text(encoding='utf-8')
            matches = xml_pattern.findall(content)
            xml_tags.update(matches)
        except Exception:
            # Silently continue if we can't read a file
            continue
    
    return xml_tags


def escape_xml_tags_for_loguru(message: str) -> str:
    """
    Escape XML tags in a message to prevent loguru from interpreting them as color directives.
    
    Args:
        message: The message containing potential XML tags
        
    Returns:
        Message with XML tags properly escaped for loguru
    """
    xml_tags = get_xml_tags_from_prompts()
    
    # Escape each known XML tag
    for tag in xml_tags:
        escaped_tag = tag.replace('<', '\\<').replace('>', '\\>')
        message = message.replace(tag, escaped_tag)
    
    # Escape any remaining angle brackets that might not be XML tags
    # This is a fallback for any missed patterns
    remaining_brackets = re.findall(r'<[^\\][^>]*[^\\]>', message)
    for bracket in remaining_brackets:
        escaped_bracket = bracket.replace('<', '\\<').replace('>', '\\>')
        message = message.replace(bracket, escaped_bracket)
    
    return message


def escape_line_for_logging(line: str) -> str:
    """
    Escape a line for safe logging with loguru.
    Handles both XML tags and loguru format characters.
    
    Args:
        line: The line to escape
        
    Returns:
        Escaped line safe for loguru logging
    """
    # Escape loguru format characters first
    escaped_line = line.replace('{', '{{').replace('}', '}}')
    
    # Then escape XML tags
    escaped_line = escape_xml_tags_for_loguru(escaped_line)
    
    return escaped_line


if __name__ == "__main__":
    # Test the function
    tags = get_xml_tags_from_prompts()
    print("Found XML tags:")
    for tag in sorted(tags):
        print(f"  {tag}")
        
    # Test escaping
    test_message = "This is a <system_reminder> test </system_reminder> message"
    escaped = escape_xml_tags_for_loguru(test_message)
    print(f"\nOriginal: {test_message}")
    print(f"Escaped:  {escaped}")