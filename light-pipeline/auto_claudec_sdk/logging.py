#!/usr/bin/env python3
"""
Claude SDK - Logging Utilities  
Handles all tool logging and formatting for Claude Code SDK outputs.
"""

# Global TODO status tracking for continuation prompts
_last_todo_status = None

# Track Task tool calls to only update TODO state when orchestrator has control
# When task_in_count == task_out_count, all subagents have finished
_task_in_count = 0
_task_out_count = 0

# Color constants for Claude SDK logging
RED = "\033[31m"       # Red color for agent names
YELLOW = "\033[33m"    # Yellow color for tool names and IDs
RESET = "\033[0m"      # Reset color

def format_claude_message(level: str, message: str, agent_context: str = "") -> str:
    """Format Claude SDK messages with special agent context and color handling."""
    colors = {
        # Standard levels
        "SUCCESS": "\033[32m",      # Green
        "INFO": "\033[34m",         # Blue
        "WARNING": "\033[33m",      # Yellow
        "ERROR": "\033[31m",        # Red
        "DEBUG": "\033[36m",        # Cyan
        
        # OpenAI custom levels
        "REQUEST": "\033[38;5;14m",    # Bright Cyan - Request
        "TOKENS": "\033[38;5;208m",    # Orange - Token usage  
        "METADATA": "\033[38;5;33m",   # Blue - Metadata
        "TIMING": "\033[38;5;10m",     # Bright Green - Timing
        "OUTPUT": "\033[38;5;213m",    # Pink - Output content
        
        # Claude SDK custom levels
        "THINKING": "\033[38;5;201m",  # Hot Magenta (256-color) - Very distinct
        
        # BASH - Completely different colors
        "BASH_IN": "\033[38;5;39m",    # Bright Blue (256-color)
        "BASH_OUT": "\033[38;5;202m",  # Bright Orange (completely different) (256-color)
        
        # READ - Completely different colors
        "READ_IN": "\033[38;5;226m",   # Bright Yellow (256-color)
        "READ_OUT": "\033[38;5;93m",   # Purple (completely different) (256-color)
        
        # WRITE - Completely different colors
        "WRIT_IN": "\033[38;5;46m",    # Bright Green (256-color)
        "WRIT_OUT": "\033[38;5;197m",  # Hot Pink (completely different) (256-color)
        
        # EDIT - Completely different colors
        "EDIT_IN": "\033[38;5;207m",   # Light Pink (256-color)
        "EDIT_OUT": "\033[38;5;22m",   # Dark Green (completely different) (256-color)
        
        # MULTIEDIT - Completely different colors
        "MULT_IN": "\033[38;5;183m",   # Light Lavender (256-color)
        "MULT_OUT": "\033[38;5;75m",   # Steel Blue (completely different) (256-color)
        
        # GREP - Completely different colors
        "GREP_IN": "\033[38;5;178m",   # Light Amber (256-color)
        "GREP_OUT": "\033[38;5;33m",   # Blue (completely different) (256-color)
        
        # GLOB - Completely different colors
        "GLOB_IN": "\033[38;5;51m",    # Bright Cyan (256-color)
        "GLOB_OUT": "\033[38;5;166m",  # Dark Orange (completely different) (256-color)
        
        # LS - Completely different colors
        "LS_IN": "\033[38;5;210m",     # Light Coral (256-color)
        "LS_OUT": "\033[38;5;55m",     # Purple-Blue (completely different) (256-color)
        
        # TASK - Completely different colors
        "TASK_IN": "\033[38;5;208m",   # Bright Orange (256-color)
        "TASK_OUT": "\033[38;5;28m",   # Dark Green (completely different) (256-color)
        
        # TODO - Mixed colors
        "TODO_IN": "\033[38;5;46m",    # Bright Green (256-color)
        "TODO_OUT": "\033[38;5;220m",  # Gold/Yellow (256-color)
        
        # Teal family - SEARCH
        "SRCH_IN": "\033[38;5;50m",    # Light Teal (256-color)
        "SRCH_OUT": "\033[38;5;30m",   # Darker Teal (256-color)
        
        # Orange/Peach family - FETCH (changed from brown since GREP is now brown)
        "FTCH_IN": "\033[38;5;215m",   # Light Peach (256-color)
        "FTCH_OUT": "\033[38;5;166m",  # Darker Orange (256-color)
        
        # Other message types
        "USER": "\033[38;5;250m",      # Light Gray (256-color)
        "SYSTEM": "\033[38;5;240m",    # Dark Gray (256-color)
        "CLAUDE": "\033[38;5;214m",    # Orange (256-color) - UNCHANGED
    }
    
    content_color = colors.get(level, "\033[37m")  # Default to white
    
    # Check if message starts with agent context format (agentname|XX|content)
    if "|" in message:
        parts = message.split("|", 2)
        if len(parts) == 3:
            # Format: agent|ID|content or |ID|content
            agent_part = parts[0]
            tool_id = parts[1]
            content = parts[2]
            
            # Special handling for Task content with TASK_HEADER
            if content.startswith("TASK_HEADER:"):
                # Remove the prefix and split header from prompt
                clean_content = content[12:]  # Remove "TASK_HEADER:"
                if "\n" in clean_content:
                    header, prompt = clean_content.split("\n", 1)
                    formatted_content = f"{YELLOW}{header}{RESET}\n{content_color}{prompt}{RESET}"
                else:
                    formatted_content = f"{YELLOW}{clean_content}{RESET}"
            else:
                formatted_content = f"{content_color}{content}{RESET}"
            
            if agent_part:
                # Has agent context - agent in red, tool name and ID match content color
                return f"{RED}{agent_part}{RESET}|{content_color}{level: <8}{RESET}|{content_color}{tool_id}{RESET}| {formatted_content}\n"
            else:
                # No agent context - tool name and ID match content color
                return f"|{content_color}{level: <8}{RESET}|{content_color}{tool_id}{RESET}| {formatted_content}\n"
    
    # Fallback for old format
    if message and (message[:2].isalnum() or message[:2] == "__") and len(message) > 2 and message[2] == "|":
        # Split ID and content
        tool_id = message[:2]
        content = message[3:]  # Skip the pipe
        
        # Special handling for Task content with TASK_HEADER
        if content.startswith("TASK_HEADER:"):
            clean_content = content[12:]  # Remove "TASK_HEADER:"
            if "\n" in clean_content:
                header, prompt = clean_content.split("\n", 1)
                formatted_content = f"{YELLOW}{header}{RESET}\n{content_color}{prompt}{RESET}"
            else:
                formatted_content = f"{YELLOW}{clean_content}{RESET}"
        else:
            formatted_content = f"{content_color}{content}{RESET}"
            
        return f"|{content_color}{level: <8}{RESET}|{content_color}{tool_id}{RESET}| {formatted_content}\n"
    else:
        # Special handling for Task content with TASK_HEADER
        if message.startswith("TASK_HEADER:"):
            clean_message = message[12:]  # Remove "TASK_HEADER:"
            if "\n" in clean_message:
                header, prompt = clean_message.split("\n", 1)
                formatted_message = f"{YELLOW}{header}{RESET}\n{content_color}{prompt}{RESET}"
            else:
                formatted_message = f"{YELLOW}{clean_message}{RESET}"
        else:
            formatted_message = f"{content_color}{message}{RESET}"
            
        return f"|{content_color}{level: <8}{RESET}|{formatted_message}\n"


def get_last_todo_status():
    """Get the last TODO status for continuation prompts."""
    return _last_todo_status


def _update_todo_status(todo_content: str):
    """Update the last TODO status."""
    global _last_todo_status
    _last_todo_status = todo_content


def log_thinking_block(thinking_text: str, truncate_logs: bool = True, agent_context: str = ""):
    """Log a thinking block with appropriate formatting."""
    
    # Get first line, or full text if truncate_logs is False
    if truncate_logs:
        first_line = thinking_text.split('\n')[0].strip()
        if len(first_line) > 100:
            first_line = first_line[:100] + "..."
        display_text = first_line
    else:
        display_text = thinking_text.strip()
    
    # Build message in format expected by format_claude_message
    message = f"{agent_context}|__|{display_text}"
    
    # Use custom formatting and output directly
    formatted_output = format_claude_message("THINKING", message, agent_context)
    print(formatted_output, end="")


def log_claude_text(text: str, agent_context: str = ""):
    """Log Claude's regular text output."""
    
    # Process each line separately
    lines = text.strip().split('\n')
    for line in lines:
        if line.strip():
            # Build message in format expected by format_claude_message
            message = f"{agent_context}|__|{line}"
            
            # Use custom formatting and output directly
            formatted_output = format_claude_message("CLAUDE", message, agent_context)
            print(formatted_output, end="")


def log_tool_input(tool_name: str, tool_input: dict, truncate_logs: bool = True, tool_id: str = None, agent_context: str = ""):
    """Log tool input with appropriate formatting and level."""
    
    # Declare globals at the beginning of the function
    global _task_in_count, _task_out_count
    
    # Extract just the essential info and determine level
    match tool_name:
        case "Task":
            # Increment task_in_count when a subagent is called
            _task_in_count += 1
            
            description = tool_input.get('description', '')
            subagent_type = tool_input.get('subagent_type', '')
            prompt = tool_input.get('prompt', '')
            # Format: description and subagent type (both in yellow), newline, prompt (normal color)
            # We'll format this specially to handle colors
            content = f"TASK_HEADER:{description} [{subagent_type}]\n{prompt}"
            level_name = "TASK_IN"
        case "WebSearch":
            content = tool_input.get('query', '')
            level_name = "SRCH_IN"
        case "WebFetch":
            content = tool_input.get('url', '')
            level_name = "FTCH_IN"
        case "Read":
            content = tool_input.get('file_path', '')
            level_name = "READ_IN"
        case "Write":
            content = tool_input.get('file_path', '')
            level_name = "WRIT_IN"
        case "Edit":
            file_path = tool_input.get('file_path', '')
            old_str = tool_input.get('old_string', '')
            if truncate_logs and len(old_str) > 20:
                old_str = old_str[:20] + "..."
            content = f"{file_path} [{old_str}]"
            level_name = "EDIT_IN"
        case "MultiEdit":
            file_path = tool_input.get('file_path', '')
            edits = tool_input.get('edits', [])
            content = f"{file_path} [{len(edits)} edits]"
            level_name = "MULT_IN"
        case "Bash":
            content = tool_input.get('command', '')
            level_name = "BASH_IN"
        case "Grep":
            content = tool_input.get('pattern', '')
            level_name = "GREP_IN"
        case "Glob":
            content = tool_input.get('pattern', '')
            level_name = "GLOB_IN"
        case "LS":
            content = tool_input.get('path', '')
            level_name = "LS_IN"
        case "TodoWrite":
            # Show the actual todos content
            todos = tool_input.get('todos', [])
            todo_lines = []
            for i, todo in enumerate(todos, 1):
                status = todo.get('status', 'pending')
                priority = todo.get('priority', 'medium')
                content_text = todo.get('content', '')
                # Add extra newline before todos (except first one) for better readability
                if i > 1:
                    todo_lines.append(f"\n{i}. [{status}] [{priority}] {content_text}")
                else:
                    todo_lines.append(f"{i}. [{status}] [{priority}] {content_text}")
            content = "\n".join(todo_lines) if todo_lines else "No todos"
            level_name = "TODO_IN"
            
            # Only update global TODO status if orchestrator has control
            # (when all subagents have completed: task_in_count == task_out_count)
            if todo_lines:
                # Debug: Uncomment to see task counts
                # print(f"\n[DEBUG] TodoWrite: task_in={_task_in_count}, task_out={_task_out_count}, updating={_task_in_count == _task_out_count}\n", flush=True)
                if _task_in_count == _task_out_count:
                    _update_todo_status(content)
        case _:
            content = str(tool_input)
            if truncate_logs and len(content) > 50:
                content = content[:50] + "..."
            # Create a custom level on the fly for unknown tools
            tool_prefix = (tool_name[:4] if tool_name and len(tool_name) >= 4 else tool_name or "UNKN").upper()
            custom_level = f"{tool_prefix}_IN"
            level_name = custom_level
    
    # Truncate content if too long (only if truncate_logs is True)
    if truncate_logs and len(content) > 150:
        content = content[:150] + "..."
    
    # Build message in format expected by format_claude_message
    tool_id_display = tool_id[-2:] if tool_id else "__"
    message = f"{agent_context}|{tool_id_display}|{content}"
    
    # Use custom formatting and output directly
    formatted_output = format_claude_message(level_name, message, agent_context)
    print(formatted_output, end="")


def log_tool_output(tool_name: str, content: str, is_error: bool = False, truncate_logs: bool = True, tool_id: str = None, agent_context: str = ""):
    """Log tool output with appropriate formatting and level."""
    
    # Declare global at the beginning of the function
    global _task_out_count
    
    # Determine the output level based on tool name
    level_map = {
        "WebSearch": "SRCH_OUT",
        "WebFetch": "FTCH_OUT",
        "Read": "READ_OUT",
        "Write": "WRIT_OUT",
        "Edit": "EDIT_OUT",
        "MultiEdit": "MULT_OUT",
        "Bash": "BASH_OUT",
        "Grep": "GREP_OUT",
        "Glob": "GLOB_OUT",
        "LS": "LS_OUT",
        "Task": "TASK_OUT",
        "TodoWrite": "TODO_OUT"
    }
    
    tool_prefix = (tool_name[:4] if tool_name and len(tool_name) >= 4 else tool_name or "UNKN").upper()
    level_name = level_map.get(tool_name, f"{tool_prefix}_OUT")
    
    # Extract just the essential output
    if is_error:
        content = f"Error: {content}"
    
    # Special handling for TASK_OUT to extract clean text content like SRCH_OUT
    if tool_name == "Task":
        # Increment task_out_count when a subagent completes
        _task_out_count += 1
        
    if tool_name == "Task" and not is_error:
        
        try:
            import json
            import ast
            
            # First try to parse as string representation of Python list/dict
            try:
                # Handle cases like "[{'type': 'text', 'text': '...'}]"
                parsed_content = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                # Fall back to JSON parsing
                if content.strip().startswith('{') or content.strip().startswith('['):
                    parsed_content = json.loads(content)
                else:
                    parsed_content = content
            
            # Extract text from various structures
            if isinstance(parsed_content, list):
                # Handle list of dicts with 'text' field: [{'type': 'text', 'text': '...'}]
                text_parts = []
                for item in parsed_content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, dict) and 'content' in item:
                        text_parts.append(item['content'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    content = '\n'.join(text_parts)
                    
            elif isinstance(parsed_content, dict):
                # Look for common content fields
                text_content = (parsed_content.get('content') or 
                              parsed_content.get('text') or 
                              parsed_content.get('message') or 
                              parsed_content.get('output') or
                              str(parsed_content))
                if isinstance(text_content, str):
                    content = text_content
                    
        except (json.JSONDecodeError, KeyError, AttributeError, ValueError):
            # If parsing fails, use content as-is
            pass
    
    # Truncate if too long (only if truncate_logs is True)
    if truncate_logs and len(content) > 150:
        content = content[:150] + "..."
    
    # Build message in format expected by format_claude_message
    tool_id_display = tool_id[-2:] if tool_id else "__"
    message = f"{agent_context}|{tool_id_display}|{content}"
    
    # Use custom formatting and output directly
    formatted_output = format_claude_message(level_name, message, agent_context)
    print(formatted_output, end="")


def log_system_message(subtype: str, data: any = None, agent_context: str = ""):
    """Log a system message with appropriate formatting."""
    
    # Show system messages - only show important ones
    if subtype == "init":
        # Check if we have rich formatted data from msg_parse
        if isinstance(data, dict) and 'model' in data and 'mode' in data:
            cwd = data.get('cwd', 'unknown')
            content = f"Session initialized | Model: {data['model']} | CWD: {cwd} | Mode: {data['mode']} | MCP: {data.get('mcp', 'none')}"
        else:
            content = "Session initialized"
    else:
        content = f"{subtype}"
        if data and len(str(data)) < 40:
            content += f": {str(data)}"
    
    # Build message in format expected by format_claude_message
    message = f"{agent_context}|__|{content}"
    
    # Use custom formatting and output directly
    formatted_output = format_claude_message("SYSTEM", message, agent_context)
    print(formatted_output, end="")


def show_progress_indicator(tool_usage_count: int):
    """Show minimal progress indicator when not showing tool usage."""
    if tool_usage_count == 1:
        print("\n🔄 Claude is researching... ", end="", flush=True)
    else:
        print(".", end="", flush=True)


def complete_progress_indicator(tool_usage_count: int):
    """Complete the progress indicator."""
    if tool_usage_count > 0:
        print(" ✓")


def reset_task_counts():
    """Reset task counts - useful for new sessions."""
    global _task_in_count, _task_out_count
    _task_in_count = 0
    _task_out_count = 0


# Export all main functions
__all__ = [
    'get_last_todo_status',
    'log_thinking_block',
    'log_claude_text',
    'log_tool_input',
    'log_tool_output',
    'log_system_message',
    'show_progress_indicator',
    'complete_progress_indicator',
    'reset_task_counts'
]