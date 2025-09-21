def get_exec_prompt(input_text, workspace_dir):
    return f"""
<idea>
{input_text}
</idea>

<data_filepath>
{workspace_dir}/data_out.json
</data_filepath>

<prediction_filepath>
{workspace_dir}/method_out.json
</prediction_filepath>

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

Evaluation Libraries:
- HuggingFace Evaluate: https://huggingface.co/docs/evaluate/
- Must use evaluate library for computing metrics
- Use evaluate.load() to load metrics like accuracy, f1, precision, recall
- Example: `import evaluate; accuracy = evaluate.load("accuracy")`
- Can combine multiple metrics for comprehensive evaluation

Time Constraints:
- Must be able to run in less than 1 hour on single laptop
- No distributed computing
</available_resources>

<YOUR_TODO_LIST>
FIRST, add ALL of these to your todo list with "TodoWrite" tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

1. Run 'pwd' in Bash to get cwd. Read './method_out_trunc.json' to understand what predictions the method produced. Summarize findings in './eval_todo_1.txt'.

2. Ultrathink about the evaluation of the <idea> and what metrics you should use. Read './handbook.txt' to help you. Describe ALL your thoughts and decisions in './eval_todo_2.txt'.

3. Read './eval_examples.py' to prepare for the evaluation of the <idea> in next todo. Test basic functionality of HuggingFace evaluate library by creating and running a simple example script (with uv run). Summarize edits in './eval_todo_3.txt'.

4. Fully implement the <idea> evaluation with HuggingFace evaluate in './eval.py' (use .toml file make uv .venv and activate it. NO INLINE DEPENDENCIES), it's input is './method_out_mini.json' (do NOT read this file it is identical to './method_out_trunc.json' just longer) and output must match './eval_out_format.json'. IMPORTANT: Add extensive loguru debug logging to './method.py' to help you debug any issues when running the script, your logs MUST NOT be too long (if you print something potentially long, add logic to truncate it). IMPORTANT: Include extensive erorr checking and sanity checks in your code, use loguru @catch decorator to catch any errors and log them. Summarize edits in './eval_todo_4.txt'.

5. run './eval.py' (when calling Bash tool you MUST set timeout=1200000) without truncating logs (NO 'tail -50' etc.) and fix any errors, verify './eval_out.json' matches './eval_out_format.json' and fix any issues. Verify './eval_out.json' has exactly 3 examples. MAKE SURE YOU HAVE COMPREHENSIVE METRICS IN './eval_out.json'. Summarize edits in './eval_todo_5.txt'.

6. rerun './eval.py' (when calling Bash tool you MUST set timeout=1200000) without truncating logs (NO 'tail -50' etc.) and if there are errors go back to todo 4. Verify './eval_out.json' has exactly 3 examples. Summarize findings in './eval_todo_6.txt'. If you have any issues you MUST go back and REDO todo 4.

7. Change input to './method_out.json' in './eval.py' and rerun './eval.py' in .venv (when calling Bash tool you MUST set timeout=1800000) without truncating logs (NO 'tail -50' etc.) and fix any errors, verify './eval_out.json' matches './eval_out_format.json' and fix any issues. Verify you have exactly 50 examples in './eval_out.json'. Reread './handbook.txt' and verify your code follows all it's method guidelines. Summarize edits in './eval_todo_7.txt'.
</YOUR_TODO_LIST>

<output_format>
Evaluation implementation completed. Results saved in [absolute filepath to generated eval_out.json]
</output_format>
"""