def get_idea_gen_prompt(prep_input, research_domain):
    return f"""You will later receive instructions to generate 20 novel groundbreaking research ideas in the field of {research_domain}.
--------------------------------

Here is information about the research domain to inform your idea generation:

{prep_input}

--------------------------------

<available_resources>
Hardware:
- 32GB RAM
- 10GB disk space available
- Intel Xeon laptop processor
- Running on WSL 2 (version 2.3) on Windows

Software Constraints:
- Python only implementation
- STRICTLY SYNCRONOUS IMPLEMENTATION, no async/await, asyncio, multithreading, multiprocessing, etc.

Dataset Constraints:
- We can only use ONE dataset. Do not provide the exact dataset name, just general type.

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

YOUR INSTRUCTIONS: Generate 20 novel groundbreaking research ideas in the field of {research_domain} that are feasible with the above constraints.

Prioritize simplicity.
Use concise, approachable language. Do not use advanced terminology if there is an elegant alternative. 
The explanation should be fully self contained.

<idea format>
Title: [Idea Title, <=15-words, self-explanatory]
Hypothesis: [Idea Hypothesis, <=15-words, self-explanatory]
Description: [Idea Description, <=200-words, numbered steps. Must include Input, Method, Output, Evaluation. Every detail must be explained, no assumptions.]
Terms: [Definition of Terms, <=100-words, Define any non-obvious/technical words/advanced concepts used in plain language.]
</idea format>

"""
