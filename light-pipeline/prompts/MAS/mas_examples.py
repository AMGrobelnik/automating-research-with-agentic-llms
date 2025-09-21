#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mirascope[openai]",
#   "pydantic",
# ]
# ///

"""Multi-Agent Patterns with Diverse Configurations"""

import os
from typing import List
from pydantic import BaseModel
from mirascope import llm

# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # TODO: Set OPENAI_API_KEY environment variable before running

TASK = "What is special about the number 42?"

# ====== DIVERSE RESPONSE MODELS ======
class Solution(BaseModel):
    answer: str
    confidence: float = 0.8

class Analysis(BaseModel):
    findings: List[str]
    score: int  # Using int response type

class Judgment(BaseModel):
    winner: str
    reasoning: str
    margin: float  # Using float response type


# ====== BASIC REASONING WITH VARIATIONS ======
@llm.call(provider="openai", model="gpt-5", response_model=Solution)  # GPT-5 default
def reason(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-4o-mini", response_model=Solution, call_params={"temperature": 0.3})  # gpt-4o-mini with low temp
def reason_precise(task: str) -> str:
    return task

@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution, call_params={"reasoning_effort": "low", "verbosity": "high"})  # GPT-5-mini creative
def reason_creative(task: str) -> str:
    return task


# ====== 1. REASONING + VERIFICATION ======
@llm.call(provider="openai", model="gpt-5-nano", response_model=bool)  # gpt-5-nano default
def verify(task: str, solution: Solution) -> str:
    return f"Verify if '{solution.answer}' correctly answers '{task}'"


def reason_with_verification():
    print(f"\n[IN] {TASK}")
    solution = reason(TASK)
    print(f"[OUT] {solution.answer} (conf: {solution.confidence})")

    verify_prompt = f"Verify if '{solution.answer}' correctly answers '{TASK}'"
    print(f"\n[IN] {verify_prompt}")
    is_correct = verify(TASK, solution)
    print(f"[OUT] {is_correct}")

    if not is_correct:
        retry_prompt = f"{TASK} (try again)"
        print(f"\n[IN] {retry_prompt}")
        solution = reason(retry_prompt)
        print(f"[OUT] {solution.answer} (conf: {solution.confidence})")

    return solution

# Run pattern 1
print(f"\n{'='*60}\nExample 1: REASONING + VERIFICATION | Task: {TASK}")
result1 = reason_with_verification()


# ====== 2. MAJORITY VOTING ======
@llm.call(provider="openai", model="gpt-4o-mini", response_model=str, call_params={"temperature": 0.1})  # gpt-4o-mini with low temp
def aggregate(solutions: List[Solution]) -> str:
    answers = [s.answer for s in solutions]
    return f"Choose best answer from: {answers}"


def majority_voting():
    solutions = []
    # Use different reasoning functions for variety
    reasoners = [reason, reason_precise, reason_creative]
    models = ["gpt-5", "gpt-4o-mini", "gpt-5-mini"]
    for i, (reasoner, model) in enumerate(zip(reasoners, models)):
        prompt = f"{TASK} (approach {i+1})"
        prefix = '\n' if i == 0 else '\n'
        print(f"{prefix}[IN] {prompt} [{model}]")
        sol = reasoner(prompt)
        print(f"[OUT] {sol.answer}")
        solutions.append(sol)

    aggregate_prompt = f"Choose best answer from: {[s.answer for s in solutions]}"
    print(f"\n[IN] {aggregate_prompt}")
    best = aggregate(solutions)
    print(f"[OUT] {best}")
    return best

# Run pattern 2
print(f"\n{'='*60}\nExample 2: MAJORITY VOTING | Task: {TASK}")
result2 = majority_voting()


# ====== 3. DEBATE WITH JUDGE ======
@llm.call(provider="openai", model="gpt-5-mini", response_model=Judgment, call_params={"reasoning_effort": "low", "verbosity": "medium"})  # GPT-5-mini judge
def judge(task: str, arg1: Solution, arg2: Solution) -> str:
    return f"Judge which answer is better for '{task}': 1) {arg1.answer} 2) {arg2.answer}"

def debate():
    cultural_prompt = f"{TASK} from cultural perspective"
    print(f"\n[IN] {cultural_prompt}")
    cultural = reason_creative(cultural_prompt)  # Creative for cultural
    print(f"[OUT] {cultural.answer} (conf: {cultural.confidence})")

    math_prompt = f"{TASK} from mathematical perspective"
    print(f"\n[IN] {math_prompt}")
    mathematical = reason_precise(math_prompt)  # Precise for math
    print(f"[OUT] {mathematical.answer} (conf: {mathematical.confidence})")

    judge_prompt = f"Judge which answer is better for '{TASK}': 1) {cultural.answer[:50]}... 2) {mathematical.answer[:50]}..."
    print(f"\n[IN] {judge_prompt}")
    decision = judge(TASK, cultural, mathematical)
    print(f"[OUT] Winner: {decision.winner} (margin: {decision.margin})")
    return cultural if "1" in decision.winner or "cultural" in decision.winner.lower() else mathematical

# Run pattern 3
print(f"\n{'='*60}\nExample 3: DEBATE | Task: {TASK}")
result3 = debate()


# ====== 4. HIERARCHICAL DECOMPOSITION ======
@llm.call(provider="openai", model="gpt-5-nano", response_model=List[str])  # gpt-5-nano default
def decompose(task: str) -> str:
    return f"Break down '{task}' into 3 subtasks"

@llm.call(provider="openai", model="gpt-5-mini", response_model=Analysis, call_params={"reasoning_effort": "medium"})  # GPT-5-mini analysis
def analyze_subtasks(results: List[Solution]) -> str:
    return f"Analyze findings from subtasks: {[s.answer[:30] for s in results]}"


def hierarchical():
    decompose_prompt = f"Break down '{TASK}' into 3 subtasks"
    print(f"\n[IN] {decompose_prompt}")
    subtasks = decompose(TASK)
    print(f"[OUT] {subtasks}")

    solutions = []
    reasoners = [reason, reason_precise, reason_creative]  # Vary reasoning
    for i, (st, reasoner) in enumerate(zip(subtasks, reasoners)):
        print(f"\n[IN] {st}")
        sol = reasoner(st)
        print(f"[OUT] {sol.answer}")
        solutions.append(sol)

    analyze_prompt = f"Analyze findings from subtasks: {[s.answer[:30] + '...' for s in solutions]}"
    print(f"\n[IN] {analyze_prompt}")
    analysis = analyze_subtasks(solutions)
    print(f"[OUT] Findings: {len(analysis.findings)} items, Score: {analysis.score}")
    return analysis

# Run pattern 4
print(f"\n{'='*60}\nExample 4: HIERARCHICAL DECOMPOSITION | Task: {TASK}")
result4 = hierarchical()


# ====== 5. REFLECTION ======
@llm.call(provider="openai", model="gpt-5-mini", response_model=Solution, call_params={"reasoning_effort": "high", "verbosity": "high"})  # GPT-5-mini elaborate
def improve(solution: Solution, feedback: bool) -> str:
    return f"Improve '{solution.answer}' (was {'correct' if feedback else 'incorrect'})"

@llm.call(provider="openai", model="gpt-4o-mini", response_model=int, call_params={"temperature": 0.1})  # gpt-4o-mini returns int
def score_answer(answer: str) -> str:
    return f"Score this answer from 1-10: {answer}"


def reflection():
    print(f"\n[IN] {TASK}")
    initial = reason(TASK)
    print(f"[OUT] {initial.answer}")

    score_prompt = f"Score this answer from 1-10: {initial.answer[:50]}..."
    print(f"\n[IN] {score_prompt}")
    score = score_answer(initial.answer)
    print(f"[OUT] Score: {score}")

    if score < 8:  # Use score instead of boolean
        improve_prompt = f"Improve '{initial.answer[:30]}...' (scored {score}/10)"
        print(f"\n[IN] {improve_prompt}")
        refined = improve(initial, False)
        print(f"[OUT] {refined.answer}")
        return refined
    return initial

# Run pattern 5
print(f"\n{'='*60}\nExample 5: REFLECTION | Task: {TASK}")
result5 = reflection()


# ====== 6. TOOL-AUGMENTED ======
def calculator(expr: str) -> float:
    """Simple calculator tool"""
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except:
        return 0.0


def search(query: str) -> str:
    """Simulated search tool"""
    knowledge = {
        "42": "Answer to Life, Universe, Everything (Hitchhiker's Guide)",
        "math": "6 × 7 = 42, highly composite number",
    }
    return knowledge.get(query.split()[0], "No results")


class ToolPlan(BaseModel):
    tools: List[str]  # List response type
    priority: int  # Int response type
    description: str

@llm.call(provider="openai", model="gpt-5-nano", response_model=ToolPlan, call_params={"verbosity": "low"})  # gpt-5-nano concise
def plan_tool_use(task: str) -> str:
    return f"What tools (calculator/search) and queries needed for: {task}"


def with_tools():
    plan_prompt = f"What tools (calculator/search) and queries needed for: {TASK}"
    print(f"\n[IN] {plan_prompt}")
    plan = plan_tool_use(TASK)
    print(f"[OUT] Tools: {plan.tools}, Priority: {plan.priority}, Desc: {plan.description[:30]}...")

    # Execute planned tool calls
    results = {}
    if "calculator" in plan.tools or "calculator" in plan.description.lower():
        print(f"\n[TOOL] calculator(\"6 * 7\")")
        results["calc"] = calculator("6 * 7")
        print(f"[OUT] {results['calc']}")
    if "search" in plan.tools or "search" in plan.description.lower():
        print(f"\n[TOOL] search(\"42\")")
        results["info"] = search("42")
        print(f"[OUT] {results['info']}")

    # Reason with tool results using creative reasoning
    enriched_task = f"{TASK} Context: {results}"
    print(f"\n[IN] {enriched_task}")
    solution = reason_creative(enriched_task)  # Use creative variant
    print(f"[OUT] {solution.answer}")
    return solution

# Run pattern 6
print(f"\n{'='*60}\nExample 6: TOOL-AUGMENTED | Task: {TASK}")
result6 = with_tools()


# ====== 7. PROMPT TEMPLATES WITH SYSTEM MESSAGES ======
from mirascope import prompt_template, Messages

@llm.call(provider="openai", model="gpt-5", response_model=List[str], call_params={"reasoning_effort": "low", "verbosity": "medium"})  # gpt-5 balanced

@prompt_template()
def analysis_prompt(task: str, context: str = "") -> Messages.Type:
    """Template with system message"""
    return [
        Messages.System("You are an expert analyst. Be concise and precise."),
        Messages.User(f"Analyze this: {task}. Context: {context}" if context else f"Analyze this: {task}")
    ]

def extract_themes(task: str):
    return f"Extract 3 key themes from: {task}"

def templated_analysis():
    # Using template with system message
    prompt_messages = analysis_prompt(TASK, "Focus on cultural significance")
    # Check if it's a list of messages or a string
    if isinstance(prompt_messages, list):
        cultural_input = " ".join([m.content if hasattr(m, 'content') else str(m) for m in prompt_messages])
    else:
        cultural_input = str(prompt_messages)
    print(f"\n[IN] {cultural_input[:100]}...")
    cultural = reason_creative(cultural_input)
    print(f"[OUT] {cultural.answer[:80]}...")

    # Extract themes using List[str] response
    themes_prompt = f"Extract 3 key themes from: {TASK}"
    print(f"\n[IN] {themes_prompt}")
    themes = extract_themes(TASK)
    print(f"[OUT] {themes}")

    # Final synthesis with precise reasoning
    synthesis_prompt = f"Synthesize insights about {TASK} considering themes: {themes}"
    print(f"\n[IN] {synthesis_prompt[:100]}...")
    final = reason_precise(synthesis_prompt)
    print(f"[OUT] {final.answer}")
    return final

# Run pattern 7
print(f"\n{'='*60}\nExample 7: PROMPT TEMPLATES | Task: {TASK}")
result7 = templated_analysis()
