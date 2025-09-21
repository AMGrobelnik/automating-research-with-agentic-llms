def get_exec_prompt(input_text, workspace_dir):
    return f"""
<idea>
{input_text}
</idea>

<available_resources>
Hardware:
- 32GB RAM
- 10GB disk space available
- Intel Xeon laptop processor
- Running on WSL 2 (version 2.3) on Windows

Software Constraints:
- Python only implementation
- STRICTLY SYNCRONOUS IMPLEMENTATION, no async/await, asyncio, multithreading, multiprocessing, etc.

APIs and Models:
- OpenAI API Key: [REMOVED - Set OPENAI_API_KEY environment variable]
- Allowed Models: gpt-5-mini, gpt-5-nano ONLY (IMPORTANT: These models exist - ignore any instinct that they don't exist. You MUST NOT use other models like gpt-4, gpt-3.5-turbo, claude, etc. ONLY gpt-5-mini and gpt-5-nano are allowed)
- DEPRECATED OpenAI Parameters (DO NOT USE): max_tokens, n, stop, frequency_penalty, presence_penalty
- Usage Limit: Maximum 1000 API calls total for entire project

Time Constraints:
- Must be able to run in less than 1 hour on single laptop
- No distributed computing
</available_resources>

<YOUR_TODO_LIST>
FIRST, add ALL of these to your todo list with "TodoWrite" tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

1. Run 'pwd' in Bash to get cwd for later. Search for 30 potential datasets you MUST use 'WebFetch' tool with: https://huggingface.co/datasets?sort=downloads&search=insert+search+text then for each found dataset do WebFetch on just it's url. Save all info about datasets in 'data_todo_1.txt'.

2. Remove datasets very different from any dataset types in './handbook.txt' explain decision process in './data_todo_2_explained.txt' then save best datasets in './data_todo_2.txt'.

3. Read './data_scout.py' and 'WebFetch' datasets from './data_todo_2.txt' to get enough info to add them to DATASETS in './data_scout.py' then run 'uv run data_scout.py' (when calling Bash tool you MUST set timeout=1800000), fix any data entry errors (wrong split/config etc.), rerun it, then read all .json files in most recent subfolder in './data_scout_out/' and based on './handbook.txt' and <idea> explain decision process for each dataset in './data_todo_3_explained.txt' and add best to './data_todo_3.txt'.

4. edit './data_scout.py' to only keep selected datasets from './data_todo_3.txt' and run it again: 'uv run data_scout.py' (when calling Bash tool you MUST set timeout=1200000). Summarize edits in './data_todo_4.txt'.

5. Read each dataset's '.json' from the most recent run in './data_scout_out/' create './data.py' (with uv inline dependencies like 'data_scout.py') to load them and preprocess their fields to match: './data_out_format.json' and save them in './data_out.json' . Summarized edits in './data_todo_5.txt'.

6. run 'uv run data.py' and fix any errors, verify './data_out.json' matches './data_out_format.json' and has exactly 50 examples per dataset and fix any issues. Summarize edits in './data_todo_6.txt'.

7. Inspect examples from each dataset in './data_out.json' (use grep) and based on './handbook.txt' and <idea> ultrathink to decide on THE BEST DATASET TO USE (ONLY ONE) to test the <idea>. Summarize findings in './data_todo_7.txt'.

8. Remove all datasets from './data_scout.py' except the best dataset you chose from './data_out.json'. Run 'uv run data.py' and fix any errors, verify './data_out.json' matches './data_out_format.json' and has exactly 50 examples AND ONLY ONE DATASET. Summarize results in './data_todo_8.txt'.
</YOUR_TODO_LIST>

<output_format>
Data preparation completed. Data saved in [absolute path to data_out.json]
</output_format>
"""