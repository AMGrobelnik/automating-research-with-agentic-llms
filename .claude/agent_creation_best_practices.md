# Best Practices for Creating Agent.md Files

## Core Principles

### 1. Agent Naming and Organization
- Use unique numbering system (e.g., 3.1, 3.2, 3.3) to ensure each agent has distinct identifier
- Name format: `{number}_{function}_{type}.md` (e.g., `3.1_sieggr_research_one.md`)
- Type suffix: `_one` for single item processing, `_all` for batch processing
- Group related agents by number prefix (3.x for evaluation pipeline)

### 2. Description Section
- Use standardized XML structure with three required tags:
  - `<use_case>` - When to use the agent (detailed, multi-line)
  - `<input_format>` - What input the agent expects
  - `<output_format>` - What output the agent creates
- NO examples at all - no `<example>` tags in description
- Use YAML multiline format with `|` for readability
- **CRITICAL**: Output format text MUST exactly match the final output in the last todo step

#### Format Pattern:
```yaml
description: |
  <use_case>
  Use this agent when [specific conditions and scenarios].
  When you need [specific capabilities and requirements].
  </use_case>
  
  <input_format>
  [Exact input specification - see rules below]
  </input_format>
  
  <output_format>
  [MUST EXACTLY MATCH what agent returns in last todo step]
  </output_format>
```

### 3. Input Format Rules
- **For "all" agents**: Just the folder location
  ```
  IdeaFilter_Checkpoints/
  ```
- **For "one" agents**: Specific file pattern
  ```
  IdeaFilter_Checkpoints/analysis_[idea_number]_[idea_name].txt
  ```

### 4. Agent Structure and XML Tags
- Use XML tags to clearly separate sections in this exact order:
  - `<identity>` - IMMEDIATELY after frontmatter
  - `<input_format>` - Input specification (duplicates description for clarity)
  - `<YOUR_INSTRUCTIONS>` - Brief main task description
  - `<YOUR_TODO_LIST>` - Detailed step-by-step tasks
  - `<system_reminder>` - AT THE VERY END, operational guidelines

- **NO** `<YOUR_INPUT>` section (deprecated)
- **NO** `<YOUR_OUTPUT>` section (deprecated)
- **NO** separator lines (`------------`) anywhere in system prompt
- Keep system_reminder blocks EXACTLY as-is

### 5. XML Tag Attributes
- **NEVER** include `file="..."` attributes in XML tags when filename is specified in todo step
- **Bad**:
  ```xml
  <simplicity_findings file="simplicity_findings.txt">
  ```
- **Good**:
  ```xml
  <simplicity_findings>
  ```

### 6. Todo List Requirements

#### General Rules:
- Number every step sequentially without gaps
- Every step must be one todo item (no compound steps)
- Put newlines between each numbered item for readability
- **NO** separator lines between groups of steps
- **NO** return statement as final step (the last action IS the completion)
- Always specify which tool to use: "use [Tool] tool"

#### Tool References:
- Use lowercase "use" when referring to tools
- **Bad**: "Use Write tool"
- **Good**: "use Write tool"

#### Formatting Specifications:
- Be explicit about data formats
- **Bad**: "Include all verified arguments"
- **Good**: "numbered list of all 10 verified arguments, each on new line"

#### No Duplicate Operations:
- Never save to subfolder then copy to main folder
- Save directly to final location
- **Bad**:
  ```
  15. Save to 'agent_subfolder/file.txt'
  16. Copy to 'IdeaFilter_Checkpoints/file.txt'
  ```
- **Good**:
  ```
  15. Use Write tool to save to 'IdeaFilter_Checkpoints/file.txt'
  ```

#### No Compilation Steps:
- Don't create separate compilation steps that duplicate main output
- **Bad**:
  ```
  23. Compile all evaluations into comprehensive file
  24. Use Write tool to save compilation
  25. Use Write tool to save simplified ratings
  ```
- **Good**:
  ```
  23. Use Write tool to save simplified ratings to 'IdeaFilter_Checkpoints/ratings.txt'
  ```

### 7. Verification Agent Patterns

#### Granular Processing:
- Process each argument individually with separate todo steps
- Include justification with every verification decision
- Format: `"Arg X: [VERIFIED/REMOVED] - [justification]"`

#### Example Structure:
```
3. Process SIMPLICITY argument 1: Extract text and source URLs, use WebFetch on each URL, verify claims, mark VERIFIED or REMOVED with justification, use Edit tool to append "Arg 1: [VERIFIED/REMOVED] - [justification]" to log.

4. Process SIMPLICITY argument 2: [same pattern]
...
```

### 8. Evaluation Agent Patterns

#### Single-Pass Evaluation:
- Read each file once and rate all dimensions in sequence
- Handle 20 ideas with individual steps (not loops)
- Include consistency maintenance

#### Example Structure:
```
3. Evaluate idea 1: Use Read tool to read file, rate simplicity (1-10), rate elegance (1-10), rate groundbreaking (1-10), ensure consistency with previous ratings, use Write tool to save complete evaluation to 'path/evaluation.txt'.

4. Evaluate idea 2: [same pattern]
...
22. Evaluate idea 20: [same pattern]
```

#### Rating Scales:
- Use 1-10 scales with clear rubrics
- Include rubrics in YOUR_INSTRUCTIONS section
- Example:
  ```
  SIMPLICITY RATING:
  1-2: Extremely complex - Many interdependent components
  3-4: Complex - Multiple components requiring expertise
  5-6: Moderate complexity - Some moving parts but manageable
  7-8: Simple - Few components, straightforward
  9-10: Extremely simple - Elegant minimalism
  ```

#### Score Calculations:
- Show formulas explicitly in todo steps
- Example:
  ```
  7. Calculate technical feasibility score:
     weighted_issues = (severe * 2) + (moderate * 1) + (low * 0.5)
     score = max(1, 10 - min(9, weighted_issues))
     Justification: [X] severe, [Y] moderate, [Z] low issues
  ```

### 9. Model Selection
- **opus**: For evaluation agents requiring:
  - Holistic assessment
  - Consistency maintenance
  - Complex scoring rubrics
  - Final decision making
- **sonnet**: For verification/research agents requiring:
  - Fact checking
  - WebFetch operations
  - Binary decisions

### 10. Directory Structure
- Each agent has its own directory: `IdeaFilter_Checkpoints/{agent_number}_{agent_name}/`
- Instance-specific subdirectories: `{agent_number}_{agent_name}/{idea_number}_{idea_name}/`
- Save ALL intermediate steps for audit trail
- Don't use `mkdir -p` commands - Write tool auto-creates directories

### 11. Logs and Timestamps
- **NO** timestamps in verification or evaluation logs
- **Bad**:
  ```xml
  <verification_log>
  VERIFICATION LOG
  Started: [timestamp]
  </verification_log>
  ```
- **Good**:
  ```xml
  <verification_log>
  VERIFICATION LOG
  Idea: [idea_number] - [idea_name]
  </verification_log>
  ```

### 12. Placeholder Format
- Use `[placeholder]` format throughout, not `{placeholder}`
- Examples: `[idea_number]`, `[idea_name]`, `[score]`
- Applies to file paths, format specifications, and variable references

### 13. WebSearch/WebFetch Ratio
- Always use 2:1 ratio: 2x WebFetch calls for every WebSearch
- Example: 10 WebSearch → 20 WebFetch
- Run in parallel batches for efficiency

### 14. Special Considerations
- ArXiv URLs: Always include note to convert `/abs/` to `/html/` for full paper access
- Tie-breaking: Specify exact rules when selecting best
- Critical sections: Use `CRITICAL:` prefix for important reminders
- File paths: Always use absolute paths, never relative

### 15. Format Specifications
- Use XML-style format tags WITHOUT file attributes when in todo:
  ```xml
  <format_name>
  Field1: [description, constraints]
  Field2: [description, constraints]
  </format_name>
  ```
- Be extremely specific - zero ambiguity principle
- Use generic format patterns with placeholders
- Specify word limits: `[Field; <=N-words; requirements]`

## Template Structure

```markdown
---
name: [agent_name]
description: |
  <use_case>
  Use this agent when [specific conditions].
  When you need to [specific actions].
  </use_case>
  
  <input_format>
  [For "all" agents: folder path]
  [For "one" agents: file pattern]
  </input_format>
  
  <output_format>
  [EXACTLY what will be output in last todo step]
  </output_format>
model: opus|sonnet
---

<identity>
You are a specialized [type] agent that [specific purpose].
</identity>

<input_format>
[Duplicate of description input format for clarity]
</input_format>

<YOUR_INSTRUCTIONS>
[Brief main instructions]

[Include rating rubrics here if evaluation agent]
</YOUR_INSTRUCTIONS>

<YOUR_TODO_LIST>
FIRST, add ALL of these to your todo list with "TodoWrite" tool:

1. [First task with specific tool usage]

2. [Second task]

3. [Continue numbered tasks]

[For verification agents: individual argument processing]
[For evaluation agents: individual idea evaluation]
[No compilation steps]
[No return statements]
[Last step is the actual final output]
</YOUR_TODO_LIST>

<system_reminder>
Do not ask follow up questions and do not ask the user anything. Execute all steps independently.

Create an extremely detailed todo list for all your tasks. Every step must be 1 todo item on your list.

You are not allowed to read or interact with any contents outside of your model_workspace folder.

Your operational guidelines:

**File Operations**:
- Use Grep/Glob for broad searches, Read for specific file paths
- Start broad and narrow down during analysis - cast a wide net first, then focus
- Be thorough: check multiple locations, consider different naming conventions and patterns
- NEVER create files unless absolutely necessary - always prefer editing existing files
- NEVER proactively create documentation unless explicitly requested
- Always use absolute paths in responses, never relative paths
- When investigating, explore related directories and check for configuration files

**Investigation Methodology**:
- Begin with broad searches to understand the overall structure
- Systematically narrow down to specific implementations
- Look for patterns, conventions, and architectural decisions
- Consider edge cases and alternative implementations
- Cross-reference findings across multiple files to build complete understanding

**Communication Standards**:
- Avoid emojis for clear, professional communication
- Share relevant file names and code snippets in final responses
- Provide absolute file paths for all references
- Structure findings logically with clear sections
- Include context about how different parts connect

**Task Execution Principles**:
- Do exactly what has been asked; nothing more, nothing less
- When requirements are unclear, ask for clarification rather than assume
- Use parallel processing for multiple concurrent searches when appropriate
- Be systematic and methodical in your approach
- Test understanding by explaining connections between components

**Quality Assurance**:
- Verify findings by checking multiple related files
- Look for consistency in patterns and implementations
- Identify any gaps or inconsistencies in the codebase
- Provide comprehensive analysis that covers all relevant aspects
</system_reminder>
```

## Common Pitfalls to Avoid

1. **Output format mismatch**: Description output doesn't match last todo step
2. **Redundant file attributes**: Including `file="..."` when unnecessary
3. **Separator lines**: Using `------------` in system prompt
4. **Duplicate saves**: Saving to subfolder then copying
5. **Wrong input format**: Using file pattern for "all" agents
6. **Capitalization**: Using uppercase "Use" for tools
7. **Timestamps**: Including timestamps in logs
8. **Bulk verification**: Not processing arguments individually
9. **Missing justifications**: Verification without reasoning
10. **Compilation steps**: Creating redundant compilation todos
11. **Return statements**: Adding "Return only:" as last todo
12. **Vague formats**: Not specifying exact data structures
13. **Missing rubrics**: Evaluation agents without rating scales
14. **Wrong model**: Using sonnet for complex evaluation

## Testing Checklist

Before finalizing an agent configuration:
- [ ] Output format in description matches last todo step exactly
- [ ] No `file="..."` attributes in XML tags when unnecessary
- [ ] No separator lines in system prompt
- [ ] No duplicate save operations
- [ ] Input format appropriate for agent type
- [ ] All tool references use lowercase "use"
- [ ] No timestamps in logs
- [ ] Verification steps are granular with justifications
- [ ] Evaluation handles correct number of ideas (20)
- [ ] No compilation or return steps at end
- [ ] Todo steps detailed and numbered correctly
- [ ] Rating rubrics included for eval agents
- [ ] Model selection appropriate (opus vs sonnet)
- [ ] System reminder block unchanged
- [ ] All placeholders use `[bracket]` format