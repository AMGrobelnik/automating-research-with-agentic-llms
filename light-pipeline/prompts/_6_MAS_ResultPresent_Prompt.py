def get_exec_prompt(input_text, workspace_dir):
    return f"""
<idea>
{input_text}
</idea>

<workspace_directory>
{workspace_dir}
</workspace_directory>

<available_files>
# Main output files (full versions, too long for Claude to read entirely):
- ./eval_out.json - Full evaluation results from eval.py (input: method_out.json)
- ./method_out.json - Full method predictions from method.py (input: data_out.json)
- ./data_out.json - Full dataset from data.py (no input)

# Truncated versions (recommended for Claude to read):
- ./eval_out_trunc.json - Truncated evaluation results (use this instead of eval_out.json)
- ./method_out_trunc.json - Truncated method predictions (use this instead of method_out.json)
- ./data_out_trunc.json - Truncated dataset (use this instead of data_out.json)

# Mini versions (just a few complete examples):
- ./eval_out_mini.json - 3 full evaluation examples (if available)
- ./method_out_mini.json - 3 full method prediction examples (if available)
- ./data_out_mini.json - 3 full dataset examples (if available)

# Format files (showing expected structure):
- ./eval_out_format.json - Expected structure for evaluation output (if available)
- ./method_out_format.json - Expected structure for method output (if available)
- ./data_out_format.json - Expected structure for data output (if available)

# Python implementation files:
- ./data.py - Data preparation code (creates data_out.json, no input)
- ./method.py - Method implementation code (reads data_out.json, creates method_out.json)
- ./eval.py - Evaluation code (reads method_out.json, creates eval_out.json)
</available_files>

<YOUR_TODO_LIST>
FIRST, add ALL of these to your todo list with "TodoWrite" tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

1. Run 'pwd' in Bash to get cwd. Read './eval_out_trunc.json' (if available, otherwise './eval_out.json') to understand the evaluation results and metrics. Summarize findings in './result_todo_1.txt'.

2. Read key parts of './method.py' code to understand implementation details of baseline vs our method. Read './method_out_trunc.json' to see actual baseline vs method predictions. Extract specific examples showing the differences. Summarize findings in './result_todo_2.txt'.

3. Read './data.py' to understand the dataset preparation. Read './data_out_trunc.json' to see the actual dataset structure. Read './eval.py' to understand evaluation methodology. Analyze the evaluation metrics from eval_out_trunc.json. Summarize findings in './result_todo_3.txt'.

4. Create './paper.tex' with a complete LaTeX research paper including:
   - Title: Create an appropriate title based on the <idea> and evaluation results
   - Abstract (150-200 words)
   - Introduction explaining the problem and our approach
   - Related Work section
   - Method section with subsections for Baseline and Our Approach
   - Experimental Setup section describing dataset and evaluation
   - Results section with tables and analysis
   - Discussion section analyzing why our method works
   - Conclusion
   - References
   Add extensive comments in the LaTeX explaining each section. Save progress in './result_todo_4.txt'.

5. Create './generate_figures.py' to generate matplotlib figures for the paper using data from eval_out.json:
   - Bar chart comparing success rates (baseline vs our method - extract from aggregate_metrics in eval_out.json)
   - Bar chart comparing API calls (baseline vs our method - extract from aggregate_metrics in eval_out.json)
   - Additional visualization showing key performance indicators from eval_out.json
   - Save figures as PDF files with descriptive names
   Run the script and verify figures are created. Summarize in './result_todo_5.txt'.

6. Update './paper.tex' to include the generated figures using \\includegraphics. Add a detailed Results section with:
   - Table 1: Aggregate metrics comparison (from eval_out.json aggregate_metrics)
   - Table 2: Per-example performance (from eval_out.json examples list)
   - Figure references and analysis
   - Code example showing where our method improved over baseline (extract from eval_out.json where method succeeded but baseline failed)
   Save progress in './result_todo_6.txt'.

7. Compile './paper.tex' to PDF using pdflatex:
   - Run 'pdflatex paper.tex' (first pass)
   - Run 'pdflatex paper.tex' again (second pass for references)
   - Fix any LaTeX compilation errors if they occur
   - Verify the final output './paper.pdf' was created successfully
   Summarize the compilation process in './result_todo_7.txt'.

8. Read './paper.pdf' (using the Read tool which can read PDFs) to verify it compiled correctly. Create './paper_summary.txt' with:
   - Main contributions
   - Key results (success rate improvement, API call reduction)
   - Future work suggestions
   Summarize final status in './result_todo_8.txt'.
</YOUR_TODO_LIST>

<paper_requirements>
The research paper should:
1. Be professionally formatted using standard LaTeX article class
2. Include proper citations (even if placeholder)
3. Use scientific writing style
4. Include actual metrics from eval_out.json
5. Show concrete code examples from method_out.json
6. Explain the conceptual innovation clearly
7. Be 6-8 pages in length
8. Include at least 3 figures and 2 tables
</paper_requirements>

<output_format>
Research paper generation completed. Paper saved in [absolute path to paper.pdf]
</output_format>
"""