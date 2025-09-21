#!/usr/bin/env python3
"""
Message handler for Claude Code SDK streaming responses.
Processes already-parsed Message objects and updates state accordingly.
"""

from datetime import datetime
from loguru import logger

# Import type definitions from claude_code_sdk
try:
    from claude_code_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
        ToolResultBlock,
        UserMessage,
        SystemMessage,
    )
except ImportError:
    # Fallback for older SDK versions without ThinkingBlock
    from claude_code_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        ToolResultBlock,
        UserMessage,
        SystemMessage,
    )
    ThinkingBlock = None

from .logging import (
    log_thinking_block,
    log_claude_text,
    log_tool_input,
    log_tool_output,
    log_system_message,
    show_progress_indicator,
)


def get_agent_context(state: dict) -> str:
    """Get the current agent context from the stack."""
    agent_stack = state.get('agent_stack', [])
    if agent_stack:
        # Return the most recent agent name
        return agent_stack[-1]
    return ""


def handle_text_block(block: TextBlock, state: dict, show_tool_usage: bool) -> None:
    """Handle a text block from assistant message."""
    text = block.text
    state['response_text'] += text
    
    # Track assistant text blocks
    state['assistant_text_blocks'].append({
        'type': 'text',
        'content': text,
        'timestamp': datetime.now().isoformat()
    })
    
    # Check if this is part of final responses (no tool use after this)
    if text.strip():
        if not state['collecting_final_responses']:
            state['collecting_final_responses'] = True
        state['final_claude_responses'].append(text)
        
    agent_context = get_agent_context(state)
    if text.strip() and show_tool_usage:
        log_claude_text(text, agent_context=agent_context)
    elif text.strip():
        print(text)  # Stream normally when tool usage display is off


def handle_thinking_block(block: ThinkingBlock, state: dict, show_tool_usage: bool, truncate_logs: bool) -> None:
    """Handle a thinking block from assistant message."""
    thinking_text = block.thinking
    if thinking_text:
        state['thinking_blocks'].append({
            'type': 'thinking',
            'content': thinking_text,
            'timestamp': datetime.now().isoformat()
        })
        # Reset final response collection when we see thinking
        state['collecting_final_responses'] = False
    
    if show_tool_usage and thinking_text:
        agent_context = get_agent_context(state)
        log_thinking_block(thinking_text, truncate_logs, agent_context=agent_context)


def handle_tool_use_block(block: ToolUseBlock, state: dict, show_tool_usage: bool, truncate_logs: bool) -> None:
    """Handle a tool use block from assistant message."""
    state['tool_usage_count'] += 1
    
    # Save tool info for matching with results
    state['last_tool_id'] = block.id
    state['last_tool_name'] = block.name
    
    # Track tool interactions
    state['tool_interactions'].append({
        'type': 'tool_use',
        'tool_name': block.name,
        'tool_id': block.id,
        'input': block.input,
        'timestamp': datetime.now().isoformat()
    })
    
    # Reset final response collection when we see tool use
    state['collecting_final_responses'] = False
    state['final_claude_responses'] = []
    
    # Special handling for Task tool to manage agent stack
    if block.name == "Task":
        # Extract agent type from input
        agent_type = block.input.get('subagent_type', 'agent')
        # Push agent context onto stack
        state['agent_stack'].append(agent_type)
        # Set flag for first subagent message
        state['is_first_subagent_msg'] = True
    
    if show_tool_usage:
        agent_context = get_agent_context(state)
        log_tool_input(block.name, block.input, truncate_logs, tool_id=block.id, agent_context=agent_context)
    else:
        show_progress_indicator(state['tool_usage_count'])


def handle_tool_result_block(block: ToolResultBlock, state: dict, show_tool_usage: bool, truncate_logs: bool) -> None:
    """Handle a tool result block from assistant message."""
    # Use the tool_use_id if available on the block
    tool_use_id = block.tool_use_id if hasattr(block, 'tool_use_id') else state['last_tool_id']
    
    # Find the matching tool use and add result
    matching_tool = None
    tool_name = state['last_tool_name']  # Default
    for tool_interaction in reversed(state['tool_interactions']):
        if tool_interaction.get('tool_id') == tool_use_id:
            matching_tool = tool_interaction
            tool_name = tool_interaction.get('tool_name', tool_name)
            break
    
    if matching_tool:
        matching_tool['result'] = {
            'content': str(block.content) if block.content else "",
            'is_error': block.is_error,
            'timestamp': datetime.now().isoformat()
        }
    
    if show_tool_usage:
        agent_context = get_agent_context(state)
        content = str(block.content) if block.content else ""
        log_tool_output(tool_name, content, block.is_error, truncate_logs, tool_id=tool_use_id, agent_context=agent_context)


def handle_user_message_tool_result(message: UserMessage, state: dict, show_tool_usage: bool, truncate_logs: bool) -> None:
    """Handle tool results in user messages."""
    if not show_tool_usage:
        return
    
    # Check if this is a tool result
    handled_tool_result = False
    content = str(message.content)
    
    if isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, ToolResultBlock):
                result_content = str(item.content) if item.content else ''
                # Use the tool_use_id from the ToolResultBlock itself
                tool_use_id = item.tool_use_id if hasattr(item, 'tool_use_id') else state['last_tool_id']
                # Find the matching tool name from tool_interactions
                tool_name = state['last_tool_name']  # Default fallback
                for tool_interaction in state['tool_interactions']:
                    if tool_interaction.get('tool_id') == tool_use_id:
                        tool_name = tool_interaction.get('tool_name', tool_name)
                        break
                agent_context = get_agent_context(state)
                log_tool_output(tool_name, result_content, False, truncate_logs, tool_id=tool_use_id, agent_context=agent_context)
                
                # Special handling for Task tool results to pop agent stack
                if tool_name == "Task":
                    if state.get('agent_stack'):
                        state['agent_stack'].pop()
                handled_tool_result = True
                break
    
    # Fallback to string checking only if we haven't handled it yet
    if not handled_tool_result and "tool_result" in content:
        try:
            if isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and item.get('type') == 'tool_result':
                        result_content = item.get('content', '')
                        if state['last_tool_name']:
                            agent_context = get_agent_context(state)
                            log_tool_output(state['last_tool_name'], result_content, False, truncate_logs, tool_id=state['last_tool_id'], agent_context=agent_context)
        except Exception:
            pass  # Ignore parsing errors for legacy format


def handle_system_message(message: SystemMessage, show_tool_usage: bool, state: dict) -> None:
    """Handle a system message."""
    if show_tool_usage:
        data = message.data
        agent_context = get_agent_context(state)
        
        # Format key system information for init messages
        if message.subtype == "init":
            # Extract key details for init messages
            model = data.get('model', 'N/A')
            permission_mode = data.get('permissionMode', 'N/A')
            
            # Format MCP servers status
            mcp_servers = data.get('mcp_servers', [])
            mcp_status = []
            for server in mcp_servers:
                name = server.get('name', 'unknown')
                status = server.get('status', 'unknown')
                mcp_status.append(f"{name}:{status}")
            mcp_info = ", ".join(mcp_status) if mcp_status else "none"
            
            # Create formatted data with rich info
            formatted_data = {
                'model': model,
                'cwd': data.get('cwd', 'unknown'),
                'mode': permission_mode,
                'mcp': mcp_info
            }
            log_system_message("init", formatted_data, agent_context=agent_context)
        else:
            # Use existing log_system_message for other subtypes
            log_system_message(message.subtype, message.data, agent_context=agent_context)


def handle_result_message(message: ResultMessage, state: dict) -> None:
    """Handle a result message."""
    state['cost_usd'] = message.total_cost_usd
    if message.is_error:
        # Parse error message for better handling
        error_str = str(message.result)
        if "500" in error_str and "Internal server error" in error_str:
            raise Exception(f"Claude API internal server error (500). This is a temporary issue with Claude's servers. Please try again in a few moments.")
        elif "429" in error_str:
            raise Exception(f"Claude API rate limit exceeded (429). Please wait before retrying.")
        elif "503" in error_str:
            raise Exception(f"Claude API service unavailable (503). The service is temporarily down.")
        else:
            raise Exception(f"Claude SDK error: {message.result}")


def parse_message(message, show_tool_usage: bool, truncate_logs: bool, state: dict) -> None:
    """
    Process a parsed message from Claude Code SDK and update state accordingly.
    
    Args:
        message: The parsed Message object to process
        show_tool_usage: Whether to display tool usage
        truncate_logs: Whether to truncate long logs
        state: Dictionary containing current state (will be modified)
            Required keys:
            - response_text
            - tool_usage_count
            - last_tool_id
            - last_tool_name
            - assistant_text_blocks
            - tool_interactions
            - thinking_blocks
            - final_claude_responses
            - collecting_final_responses
            - cost_usd
    """
    # Main message type matching
    match message:
        case AssistantMessage():
            # Check if this is the first message from a subagent
            if show_tool_usage and state.get('is_first_subagent_msg', False) and state.get('agent_stack'):
                agent_context = get_agent_context(state)
                if agent_context:
                    # Extract model from the AssistantMessage
                    model = getattr(message, 'model', 'unknown')
                    
                    # Log agent initialization in SYSTEM format
                    init_message = f"{agent_context}|__|Agent initialized | Model: {model}"
                    from .logging import format_claude_message
                    formatted_init = format_claude_message("SYSTEM", init_message, agent_context)
                    print(formatted_init, end="")
                    
                    # Reset the flag
                    state['is_first_subagent_msg'] = False
            
            # Process the single content block (only 1 block per message)
            if message.content:
                block = message.content[0]
                match block:
                    case TextBlock():
                        handle_text_block(block, state, show_tool_usage)
                    case ThinkingBlock() if ThinkingBlock is not None:
                        handle_thinking_block(block, state, show_tool_usage, truncate_logs)
                    case ToolUseBlock():
                        handle_tool_use_block(block, state, show_tool_usage, truncate_logs)
                    case ToolResultBlock():
                        handle_tool_result_block(block, state, show_tool_usage, truncate_logs)
                    case _:
                        logger.debug(f"Unhandled block type in AssistantMessage: {type(block).__name__}")
                        
        case UserMessage():
            handle_user_message_tool_result(message, state, show_tool_usage, truncate_logs)
            
        case SystemMessage():
            handle_system_message(message, show_tool_usage, state)
            
        case ResultMessage():
            handle_result_message(message, state)
            
        case _:
            logger.debug(f"Unhandled message type: {type(message).__name__}")


# Export main function
__all__ = [
    'parse_message'
]