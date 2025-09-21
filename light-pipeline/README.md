# AI Scientist Lite - Idea Generation Module

🧠 **Novel LLM Agent Research Idea Generator using Claude SDK with UltraThink Approach**

## Overview

This implements the Idea Generation module from the AIS-Spec-Lite specification. It uses the Claude Python SDK with an "ultrathink" approach to generate genuinely novel research ideas in LLM Agent Systems.

## Features

- **UltraThink Mode**: Deep thinking prompts that encourage Claude to thoroughly analyze the problem space
- **Pure LLM Usage**: No external tools - relies entirely on Claude's reasoning capabilities  
- **Structured Output**: Generates ideas in the standardized JSON format from AIS-Spec-Lite
- **Research Focus**: Specifically targets "llm_agent_systems" research area
- **Cost Tracking**: Monitors API costs and generation metadata
- **🎨 Colored Logging**: Beautiful loguru logging with colored output and timestamps
- **🛡️ Error Handling**: @logger.catch decorators with automatic exception logging
- **🧹 JSON Parsing**: Robust JSON cleaning to handle control characters and formatting issues
- **📁 File Logging**: Automatic log files with rotation (logs/idea_gen.log)

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd light-pipeline
   ```

2. **Create and activate virtual environment:**
   ```bash
   uv venv .venv --python=3.10
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -e .
   ```

## Usage

### Quick Start

```bash
# Run idea generation (default command) - includes live streaming & colors
python orchestrator.py

# Or explicitly run idea generation  
python orchestrator.py idea
```

### Configuration

Edit `configs/pipeline_config.yaml` to customize:

```yaml
research_area: "llm_agent_systems"  # Main research area
sub_focus: ""                       # Optional sub-area focus
idea_gen:
  max_iterations: 5                 # Max iterations for refinement
  ultrathink_mode: true             # Enable deep thinking
```

### Expected Output

The system will:

1. Load configuration from `configs/pipeline_config.yaml`
2. Generate a novel research idea using Claude's ultrathink approach
3. Save results to timestamped directory in `outputs/`
4. Display summary and costs

**Example output:**
```
🔬 AI Scientist Lite - Idea Generation Only
🚀 AI Scientist Lite - Idea Generation Module  
📊 Research Area: llm_agent_systems
🧠 Generating novel LLM agent research idea with ultrathink approach...
✅ Idea generation completed successfully!
💾 Saved to: outputs/20250718_012912_idea_gen/generated_idea.json
💰 Cost: $0.6102
```

## Sample Generated Idea

The system recently generated:

**"Predictive Intervention Networks: Causal World Modeling for LLM Agents"**

A novel approach combining Pearl's causal inference framework with LLM reasoning to create agents that build structural causal models and use counterfactual reasoning for planning.

## Project Structure

```
light-pipeline/
├── modules/
│   └── idea_gen/
│       └── generate.py          # Single idea generation module with streaming & colors
├── configs/
│   └── pipeline_config.yaml     # Configuration file
├── outputs/                     # Generated ideas (timestamped)
├── logs/                        # Loguru log files (auto-created)
├── orchestrator.py              # Main entry point
├── pyproject.toml               # Python dependencies and project config
└── README.md                    # This file
```

## Configuration Options

### Research Areas
- `llm_agent_systems` (current focus)
- Future: expandable to other AI research areas

### Claude SDK Settings
- Model: `claude-sonnet-4-20250514`
- Permission mode: `bypassPermissions`
- No tools enabled (pure LLM reasoning)

## Live Streaming Features

### Real-Time Message Tracking

All idea generation now includes live streaming by default:

**What you'll see:**
- 🔧 **SystemMessage** - Session setup (ID, model, 51 tools available)
- 🤖 **AssistantMessage** - Live content blocks with FULL text (no truncation)
- 🛠️ **Tool Usage** - Complete tool inputs and parameters  
- 📊 **ResultMessage** - Final costs, duration, success status
- 🎨 **Colored Logs** - Beautiful colored output with timestamps and function names
- 🛡️ **Error Handling** - Automatic exception catching and detailed error logs

**Example clean streaming output:**
```
[GREEN]SUCCESS [/] | [GREEN]🚀 AI Scientist Lite - Idea Generation[/]
INFO     | 📊 Research Area: llm_agent_systems
[GREEN]SUCCESS [/] | [GREEN]🧠 Generating novel LLM agent research idea...[/]
INFO     | 💭 Watch Claude think in real-time...

Looking at the current landscape of LLM agent systems, I need to identify fundamental gaps...

After extensive consideration of current limitations, here's my novel research idea:

```json
{
  "idea_title": "Hypothesis-Driven Exploration Agents with Emergent Tool Discovery",
  "idea_description": "This agent architecture fundamentally reimagines...",
  ...
}
```

[GREEN]SUCCESS [/] | [GREEN]🎯 Detected complete JSON response![/]
[GREEN]SUCCESS [/] | [GREEN]✅ Complete - Cost: $1.31[/]
[GREEN]SUCCESS [/] | [GREEN]📈 Streaming complete - 8 messages, 5378 characters[/]
INFO     | 🔍 Parsing JSON response...
[GREEN]SUCCESS [/] | [GREEN]✅ JSON parsed successfully![/]
[GREEN]SUCCESS [/] | [GREEN]🎉 Idea saved to: outputs/20250718_idea_gen/generated_idea.json[/]
INFO     | 💰 Cost: $1.3099 | 📨 Messages: 8
[GREEN]SUCCESS [/] | [GREEN]📝 Hypothesis-Driven Exploration Agents with Emergent Tool Discovery (architecture)[/]
```

**Clean Features:**
- 🎨 **Minimal Logs**: Just message type and content - no timestamps/functions
- 📺 **Raw Text**: Claude's thinking and responses shown directly
- ✨ **Clean Flow**: Less noise, more signal

### Available Commands

| Command | Description |
|---------|-------------|
| `python orchestrator.py` | Default: Live streaming with colored logs and full content |
| `python orchestrator.py idea` | Same as default: Live streaming with colored logs |
| `python orchestrator.py full` | Full pipeline (coming soon) |

## Advanced Usage

### Direct Module Usage

```python
from modules.idea_gen.generate import main as generate_idea

# Run idea generation with live streaming & colors
idea_data = await generate_idea()
```

### Custom Research Focus

```yaml
# In configs/pipeline_config.yaml
research_area: "llm_agent_systems"
sub_focus: "multi-agent coordination"  # Specify sub-area
```

## Requirements

- Python 3.10+
- Claude Code subscription/access
- Internet connection for Claude SDK

## Future Enhancements

- Literature verification module
- Novelty scoring
- Iterative refinement
- Integration with implementation and writing modules

## Cost Information

Typical costs per idea generation:
- Simple ideas: $0.20-$0.40
- Complex ideas (ultrathink): $0.40-$0.80
- Costs depend on idea complexity and thinking depth

## Troubleshooting

### Common Issues

1. **Claude SDK not found**: Install with `npm install -g @anthropic-ai/claude-code`
2. **Authentication error**: Ensure Claude subscription is active
3. **JSON parsing error**: Now automatically fixed with robust JSON cleaning
4. **Missing colors**: Ensure terminal supports ANSI colors (most modern terminals do)
5. **Log files not created**: Check write permissions in project directory

### Fixed Issues ✅

- **Invalid control character JSON errors**: Automatically cleaned with regex
- **Truncated output**: Now shows complete messages with no cuts
- **Poor error messages**: Rich colored logging with full stack traces
- **No persistent logs**: Automatic file logging with rotation
- **Color tags not rendering**: Fixed loguru color formatting in format string

### Debug Mode

Add debug prints by modifying `generate.py`:
```python
print(f"Raw response: {generated_text}")  # Before JSON parsing
```

## Contributing

This is part of the AI Scientist Lite pipeline. To extend:

1. Add literature verification in `verify_novelty.py`
2. Implement iteration logic for idea refinement  
3. Add evaluation metrics for idea quality