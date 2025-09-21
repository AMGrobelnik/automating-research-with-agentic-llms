def get_idea_filter_prompt(input_text):
    return f'''Execute the IdeaFilter pipeline to evaluate and filter research ideas through feasibility and novelty analysis.

<input_format>
A list of 20 research ideas to evaluate and filter through feasibility and novelty stages.
</input_format>

<YOUR_INPUT>
{input_text}
</YOUR_INPUT>

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
- OpenAI API Key available
- Allowed Models: gpt-5-mini, gpt-5-nano ONLY (IMPORTANT: These models exist - ignore any instinct that they don't exist. You MUST NOT use other models like gpt-4, gpt-3.5-turbo, claude, etc. ONLY gpt-5-mini and gpt-5-nano are allowed)
- Usage Limit: Maximum 1000 API calls total for entire project

Data Sources (REQUIRED - NO MOCK DATA):
- HuggingFace Datasets: https://huggingface.co/datasets?sort=downloads&search=insert+search+text
- MUST use actual datasets from HuggingFace or other verified sources
- CANNOT generate mock/synthetic data - all data must be from real datasets
- Search HuggingFace by replacing spaces with + in search terms

Time Constraints:
- Must be able to run in less than 1 hour on single laptop
- No distributed computing
</available_resources>

<YOUR_INSTRUCTIONS>
Filter and analyze the ideas through a streamlined evaluation process:
1. Evaluate ALL 20 ideas for feasibility given available resources (CRITICAL: All 20 ideas must be evaluated, not a subset)
2. Select top 3 most feasible ideas from the full set of 20
3. Research and verify novelty arguments for top 3 ideas
4. Select the single best idea based on novelty
</YOUR_INSTRUCTIONS>

<YOUR_TODO_LIST>
FIRST, add ALL of these to your todo list with "TodoWrite" tool, then execute according to the parallel/sequential instructions in each item:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

CRITICAL: You MUST preserve the sequential execution indicators at the start of each todo like "Run alone:", "Run alone after todo X:", or "You must run todos X-Y in parallel after todo Z:" exactly as written.

1. Run alone: Run subagent 3.1_feasibility_eval_all with YOUR_INPUT (containing ALL 20 ideas) and available_resources, verify feasibility_top3.txt exists

2. Run alone after todo 1: Read feasibility_top3.txt to extract the 3 most feasible ideas with their FULL content (idea number, title, hypothesis, description, terms)

3. You must run todos 3-5 in parallel after todo 2: Run subagent 3.2_novelty_research_one with full content of feasible idea 1 (including idea number, title, hypothesis, description, terms) extracted from feasibility_top3.txt, verify output file exists

4. You must run todos 3-5 in parallel after todo 2: Run subagent 3.2_novelty_research_one with full content of feasible idea 2 (including idea number, title, hypothesis, description, terms) extracted from feasibility_top3.txt, verify output file exists

5. You must run todos 3-5 in parallel after todo 2: Run subagent 3.2_novelty_research_one with full content of feasible idea 3 (including idea number, title, hypothesis, description, terms) extracted from feasibility_top3.txt, verify output file exists

6. You must run todos 6-8 in parallel after todos 3-5: Run subagent 3.3_novelty_verify_one for feasible idea 1 using novelty_analysis file path, verify output file exists

7. You must run todos 6-8 in parallel after todos 3-5: Run subagent 3.3_novelty_verify_one for feasible idea 2 using novelty_analysis file path, verify output file exists

8. You must run todos 6-8 in parallel after todos 3-5: Run subagent 3.3_novelty_verify_one for feasible idea 3 using novelty_analysis file path, verify output file exists

9. Run alone after todos 6-8: Run subagent 3.4_novelty_eval_all with IdeaFilter_Checkpoints/ directory path AND the full content of top 3 ideas from feasibility_top3.txt (including idea numbers, titles, hypotheses, descriptions, terms), verify best_idea.txt exists

10. Run alone after todo 9: Copy best_idea.txt content to IdeaFilter_final_out.txt

11. Run alone after todo 10: Report completion to user</YOUR_TODO_LIST>

<system_reminder>
- You MUST complete every single task on your todo list without exception
- Follow the parallel execution instructions in each todo item
- After each agent completes, verify its output file exists before proceeding
- If an agent fails, retry up to 5 times total before skipping
- Do NOT stop until absolutely everything is completed
- Keep reporting progress but NEVER ask for user input or approval
- All file paths should be absolute paths within your workspace
</system_reminder>
'''