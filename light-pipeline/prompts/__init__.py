"""
Prompts package for AI Scientist Lite pipeline.
Contains prompt functions for each pipeline stage.
"""

from ._1_IdeaGen_Prompt import get_idea_gen_prompt
from ._2_IdeaFilter_Prompt import get_idea_filter_prompt
from ._2_IdeaFilter_SysPrompt import get_ideafilter_sysprompt

__all__ = [
    'get_idea_gen_prompt',
    'get_idea_filter_prompt',
    'get_ideafilter_sysprompt'
]