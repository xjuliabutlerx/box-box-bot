from langchain.agents import create_agent

from box_box_bot.tools.rag_tools import RAG_TOOLS

NARRATIVE_SYSTEM_PROMPT = """You are box-box-bot's narrative specialist.

Answer "why" or "what happened" F1 questions using search_race_recaps.
When you use information from search_race_recaps, mention which race/
season it came from.
"""

def build_narrative_agent(model):
    return create_agent(
        model,
        RAG_TOOLS,
        system_prompt=NARRATIVE_SYSTEM_PROMPT,
        name="narrative_agent",
    )
