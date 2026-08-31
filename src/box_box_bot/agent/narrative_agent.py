from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

from box_box_bot.agent.time_context import current_date_context
from box_box_bot.tools.rag_tools import RAG_TOOLS

NARRATIVE_SYSTEM_PROMPT = """You are box-box-bot's narrative specialist.

Answer "why" or "what happened" F1 questions using search_race_recaps.
When you use information from search_race_recaps, mention which race/
season it came from.
"""


@dynamic_prompt
def _narrative_prompt(request) -> str:
    return f"{current_date_context()}\n\n{NARRATIVE_SYSTEM_PROMPT}"


def build_narrative_agent(model):
    return create_agent(
        model,
        RAG_TOOLS,
        middleware=[_narrative_prompt],
        name="narrative_agent",
    )
