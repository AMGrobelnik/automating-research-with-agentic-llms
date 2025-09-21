# AI Scientist Lite

An automated research pipeline for generating, filtering, and evaluating novel research ideas in the domain of Multi-LLM Agent Systems. The system leverages OpenAI GPT-5 models to conduct autonomous scientific exploration within computational constraints typical of academic research environments.

## Prerequisites

### Claude Code Access
This pipeline requires Claude Code for autonomous code generation and execution. You must have one of the following:
- An active Claude Code subscription, OR
- Claude Code configured with an Anthropic API plan

For setup instructions, refer to the official Claude Code documentation: https://docs.claude.com/en/docs/claude-code/quickstart

### System Requirements

- Python 3.10 or 3.11 (Note: Python 3.12+ not supported due to numpy dependency constraints)
- 32GB RAM recommended
- 10GB+ free disk space
- WSL2 (if running on Windows) or Linux/macOS
- OpenAI API access
- HuggingFace account (for dataset access)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AIScientist-Lite.git
cd AIScientist-Lite/light-pipeline
```

### 2. Install UV Package Manager

UV is a high-performance Python package installer and dependency resolver. Installation methods:

```bash
# On Linux/macOS/WSL:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip:
pip install uv
```

### 3. Create Virtual Environment

```bash
# Navigate to the light-pipeline directory
cd light-pipeline

# Create virtual environment with Python 3.10
uv venv .venv --python=3.10

# Activate the environment
source .venv/bin/activate  # On Linux/macOS/WSL
# or
.venv\Scripts\activate  # On Windows
```

### 4. Install Dependencies

```bash
# Install all dependencies from pyproject.toml
uv pip install -e .
```

This command installs all required packages specified in pyproject.toml, including:
- OpenAI API client
- Anthropic SDK (for Claude integration)
- HuggingFace datasets
- Scientific computing libraries (pandas, numpy, scikit-learn)
- Visualization tools (matplotlib, seaborn)
- Web frameworks (FastAPI, uvicorn)
- And more...

### 5. Configure API Keys

Authentication credentials are required for OpenAI and HuggingFace services.

Edit `light-pipeline/pipeline_config.yaml`:

```yaml
openai_settings:
  api_key: "your-openai-api-key-here"  # Only if not using environment variable
```

### 6. Update Code Files for API Keys

#### If Using Direct API Keys (Quick Setup):

1. **light-pipeline/pipeline_config.yaml** - Line 53:
   ```yaml
   openai_settings:
     api_key: "paste-your-openai-api-key-here"  # Replace with your actual key
   ```

2. **light-pipeline/prompts/MAS/data_scout.py** - Line 21:
   ```python
   login(token="paste-your-huggingface-token-here")  # Replace with your HF token
   ```

3. **light-pipeline/prompts/MAS/mas_examples.py** - Line 17:
   ```python
   os.environ["OPENAI_API_KEY"] = "paste-your-openai-api-key-here"  # Replace with your key
   ```

4. **light-pipeline/prompts/_3_MAS_DataExec_Prompt.py** - Line 19:
   ```python
   - OpenAI API Key: paste-your-openai-api-key-here
   ```

5. **light-pipeline/prompts/_4_MAS_MethodExec_Prompt.py** - Line 23:
   ```python
   - OpenAI API Key: paste-your-openai-api-key-here
   ```

6. **light-pipeline/prompts/_5_MAS_EvalExec_Prompt.py** - Line 27:
   ```python
   - OpenAI API Key: paste-your-openai-api-key-here
   ```

#### If Using Environment Variables (Recommended):

1. **light-pipeline/openai_utils.py** - Already configured to use environment variables
2. **light-pipeline/prompts/MAS/data_scout.py** - Line 21, uncomment and update:
   ```python
   login(token=os.getenv("HF_TOKEN"))  # Uncomment this line
   ```
3. **light-pipeline/prompts/MAS/mas_examples.py** - Line 17, update to:
   ```python
   os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
   ```

## Usage

### Running the Pipeline

All pipeline operations are controlled through `run.py` and configured via `pipeline_config.yaml`.

```bash
cd light-pipeline
python run.py
```

Note: Individual module files (_1_idea_gen.py, _2_idea_filter.py, etc.) are not meant to be run directly. All execution is orchestrated through run.py using settings in pipeline_config.yaml.

The pipeline executes the following stages:
1. Generate 20 novel research ideas in Multi-LLM Agent Systems
2. Filter ideas based on feasibility criteria
3. Execute data collection procedures
4. Implement the proposed methodology
5. Evaluate experimental results
6. Present findings and analysis

### Running with Checkpoints

To resume from a checkpoint, edit the `pipeline` section in `pipeline_config.yaml`:

```yaml
pipeline:
  # To resume from a checkpoint, set the full path:
  start_checkpoint: "/full/path/to/light-pipeline/runs/20250921_123456/model_workspace/data_out.json"

  # Only needed when start_checkpoint is after idea_filter (data_out.json or later):
  idea_checkpoint: "/full/path/to/light-pipeline/runs/20250921_123456/IdeaFilter_final_out.txt"
```

Checkpoint files and what happens:
- `IdeaGen_final_out.txt` → Resumes from idea_filter module
- `IdeaFilter_final_out.txt` → Resumes from data_exec module
- `model_workspace/data_out.json` → Resumes from method_exec module
- `model_workspace/method_out.json` → Resumes from eval_exec module
- `model_workspace/eval_out.json` → Resumes from result_present module

### Running Specific Modules Only

To run only certain modules, edit the `pipeline` section in `pipeline_config.yaml`:

```yaml
pipeline:
  # Leave start_checkpoint empty to start from beginning
  start_checkpoint: ""

  # Set end_module to stop at a specific point
  end_module: "idea_filter"  # Will run idea_gen and idea_filter only
```

Available `end_module` options:
- `idea_gen` - Stop after generating ideas
- `idea_filter` - Stop after filtering ideas
- `data_exec` - Stop after data execution
- `method_exec` - Stop after method implementation
- `eval_exec` - Stop after evaluation
- `result_present` - Run complete pipeline (default)

## Project Structure

```
AIScientist-Lite/
├── light-pipeline/          # Main pipeline code
│   ├── run.py              # Main pipeline runner
│   ├── _1_idea_gen.py      # Idea generation module
│   ├── _2_idea_filter.py   # Idea filtering module
│   ├── _3_data_exec.py     # Data execution module
│   ├── _4_method_exec.py   # Method implementation
│   ├── _5_eval_exec.py     # Evaluation module
│   ├── _6_result_present.py # Result presentation
│   ├── pipeline_config.yaml # Configuration file
│   ├── pyproject.toml      # Package dependencies
│   ├── prompts/            # Prompt templates
│   │   └── MAS/            # Multi-agent system resources
│   └── runs/               # Output directory for runs
├── archive/                # Archived files
└── README.md              # This file
```

## Configuration

Edit `light-pipeline/pipeline_config.yaml` to customize:

- **Model Settings**: GPT-5 model parameters
- **Pipeline Settings**: Stages to run, debug mode
- **Claude SDK Settings**: Max turns for Claude interactions
- **Output Settings**: Output directory location

Example configuration:
```yaml
openai_settings:
  model: "gpt-5"
  reasoning_effort: "high"
  verbosity: "medium"
  service_tier: "priority"

pipeline:
  stages: ["idea_gen", "idea_filter", "data_exec", "method_exec", "eval_exec", "result_present"]
  debug_mode: false
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY environment variable not set"**
   - Ensure you've set the environment variable: `export OPENAI_API_KEY="your-key"`
   - Or check that the key is in pipeline_config.yaml

2. **"No module named 'openai'"**
   - Ensure virtual environment is activated: `source .venv/bin/activate`
   - Reinstall dependencies: `uv pip install -e .`

3. **"Python version incompatibility"**
   - Ensure Python 3.10 or 3.11 is installed
   - Recreate virtual environment: `uv venv .venv --python=3.10`

4. **HuggingFace login issues**
   - Ensure HF_TOKEN is set: `export HF_TOKEN="your-token"`
   - Or login manually: `huggingface-cli login`

5. **Out of Memory errors**
   - Reduce batch sizes in configuration
   - Ensure at least 32GB RAM available
   - Close other memory-intensive applications

### Obtaining API Credentials

1. **OpenAI API Key**:
   - Register at https://platform.openai.com/
   - Navigate to the API Keys section
   - Generate a new secret key
   - Note: GPT-5 access may require special authorization

2. **HuggingFace Token**:
   - Register at https://huggingface.co/
   - Access Settings → Access Tokens
   - Generate a new token with read permissions
   - Required for dataset access

## Output

Results are saved in `light-pipeline/runs/` with timestamps:
```
runs/
└── 20250921_123456/
    ├── model_workspace/     # Claude agent workspace
    ├── logs/               # Detailed logs
    ├── ideas.json          # Generated ideas
    ├── filtered_ideas.json # Filtered ideas
    ├── data_out.json       # Data execution results
    ├── method_out.json     # Method implementation
    ├── eval_out.json       # Evaluation results
    └── results.json        # Final results
```

## License

This project is dual-licensed:

### Documentation and Research Results
This work's documentation, research outputs, and associated materials are licensed under the **Creative Commons Attribution-ShareAlike 4.0 International License**. This means that the text, results, and other components may be freely distributed, reproduced, used, publicly communicated, and adapted, provided that proper attribution is given. Any derivative works must be distributed under the same license.

License details: https://creativecommons.org/licenses/by-sa/4.0/

### Source Code
The source code of this project and all software components are licensed under the **GNU General Public License, version 3 (or later)**. This means that the code may be freely distributed and/or modified under the terms of the license.

License details: http://www.gnu.org/licenses/

---

**Note:** This project requires valid API credentials and may incur costs based on API usage. Users are advised to monitor their API consumption to manage associated expenses.