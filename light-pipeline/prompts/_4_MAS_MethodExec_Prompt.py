def get_exec_prompt(input_text, workspace_dir):
    return f"""
<idea>
{input_text}
</idea>

<data_filepath>
{workspace_dir}/data_out.json
</data_filepath>

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

1. Run 'pwd' in Bash to get cwd. Read './data_out_trunc.json' to understand what input the method should be based on. Summarize findings in './method_todo_1.txt'.

2. Read './mas_examples.py' to prepare for the implementation of the <idea> in next todo. Test basic functionality of Mirascope by creating and running a simple example script (with uv run). Summarize edits in './method_todo_2.txt'.

3. Read './handbook.txt' to help you, now fully implement our method AND the baseline (comparison) method for <idea> with Mirascope in './method.py' (use .toml file make uv .venv and activate it. NO INLINE DEPENDENCIES), it's input is './data_out_mini.json' and output must match './method_out_format.json'. IMPORTANT: Add extensive loguru debug logging to './method.py' to help you debug any issues when running the script, your logs MUST NOT be too long (if you print something potentially long, add logic to truncate it). VERY IMPORTANT: LOG EVERY SINGLE LLM CALL INPUT and OUTPUT in the logs. IMPORTANT: Include extensive erorr checking and sanity checks in your code, use loguru @catch decorator to catch any errors and log them. Summarize edits in './method_todo_3.txt'.

4. run './method.py' in .venv (when calling Bash tool you MUST set timeout=1200000) without truncating logs (NO 'tail -50' etc.) and fix any errors, verify './method_out.json' matches './method_out_format.json' and fix any issues. Verify you are NOT using mock scripts, mock data, mock APIs. Verify you have exactly 3 examples in './method_out.json'. Reread './handbook.txt' and verify your code follows all it's method guidelines. Summarize edits in './method_todo_4.txt'.

5. Add exhaustive error checking and logging to './method.py'. Run './method.py' in .venv (when calling Bash tool you MUST set timeout=1200000) without truncating logs (NO 'tail -50' etc.) and fix any errors, verify './method_out.json' matches './method_out_format.json' and fix any issues. Verify you are NOT using mock scripts, mock data, mock APIs. Verify you have exactly 3 examples in './method_out.json'. Reread './handbook.txt' and verify your code follows all it's method guidelines. Summarize edits in './method_todo_5.txt'.

6. Add even more error checking and logging to './method.py'. Run './method.py' in .venv (when calling Bash tool you MUST set timeout=1200000) without truncating logs (NO 'tail -50' etc.) and fix any errors, verify './method_out.json' matches './method_out_format.json' and fix any issues. Verify you are NOT using mock scripts, mock data, mock APIs. Verify you have exactly 3 examples in './method_out.json'. Reread './handbook.txt' and verify your code follows all it's method guidelines. Summarize edits in './method_todo_6.txt'.

7. Add even more error checking and logging to './method.py'. Run './method.py' in .venv (when calling Bash tool you MUST set timeout=1200000) without truncating logs (NO 'tail -50' etc.) and fix any errors, verify './method_out.json' matches './method_out_format.json' and fix any issues. Verify you are NOT using mock scripts, mock data, mock APIs. Verify you have exactly 3 examples in './method_out.json'. Reread './handbook.txt' and verify your code follows all it's method guidelines. Summarize edits in './method_todo_7.txt'.

8. Change input to './data_out.json' in './method.py' and rerun './method.py' in .venv (when calling Bash tool you MUST set timeout=3600000) without truncating logs (NO 'tail -50' etc.) and fix any errors, verify './method_out.json' matches './method_out_format.json' and fix any issues. Verify you have exactly 50 examples in './method_out.json'. Reread './handbook.txt' and verify your code follows all it's method guidelines. Summarize edits in './method_todo_8.txt'.
</YOUR_TODO_LIST>

<output_format>
Method implementation completed. Results saved in [absolute filepath to generated method_out.json]
</output_format>
"""