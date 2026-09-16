from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

from box_box_bot.agent.time_context import current_date_context
from box_box_bot.tools.fastf1_tools import get_most_recent_race
from box_box_bot.tools.rag_tools import RAG_TOOLS

NARRATIVE_TOOLS = RAG_TOOLS + [get_most_recent_race]

NARRATIVE_SYSTEM_PROMPT = """You are box-box-bot's narrative specialist.

Answer "why" or "what happened" F1 questions using search_race_recaps. When you use information from search_race_recaps, mention which race/season it came from.

If the question refers to "the most recent race," "the latest race," or similar with no race named, call get_most_recent_race first to resolve it to an actual race, then search_race_recaps for that specific race by name - don't guess which race is meant or pass a vague query and hope retrieval finds the right one.
"""


@dynamic_prompt
def _narrative_prompt(request) -> str:
    return f"{current_date_context()}\n\n{NARRATIVE_SYSTEM_PROMPT}"


def build_narrative_agent(model):
    return create_agent(
        model,
        NARRATIVE_TOOLS,
        middleware=[_narrative_prompt],
        name="narrative_agent",
    )
