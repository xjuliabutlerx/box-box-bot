from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from box_box_bot.tools.fastf1_tools import FASTF1_TOOLS
from box_box_bot.tools.rag_tools import RAG_TOOLS

SYSTEM_PROMPT = """You are box-box-bot, an F1 assistant.

You have two kinds of tools:
- fastf1 tools (standings, race results, fastest laps) for factual,
  numeric questions.
- search_race_recaps for "why" or "what happened" narrative questions.

Use both together when a question needs both, e.g. "how did the
standings change after Monza and why" should call a standings tool AND
search_race_recaps. When you use information from search_race_recaps,
mention which race/season it came from. If a question is outside F1,
say so directly instead of guessing.
"""


def build_agent():
    model = ChatAnthropic(model="claude-sonnet-5")
    tools = FASTF1_TOOLS + RAG_TOOLS
    return create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver()    # Could be swapped with SqliteSaver or PostgresSaver to withstand server restarts or shared states
    )