from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

from box_box_bot.agent.time_context import current_date_context
from box_box_bot.tools.fastf1_tools import FASTF1_TOOLS

STATS_SYSTEM_PROMPT = """You are box-box-bot's stats specialist.

Answer factual, numeric F1 questions using your fastf1 tools: driver/
constructor standings, race results, fastest laps, the race calendar,
tire strategy, race control messages (flags, safety cars, penalties),
and weather. Stick to what the tools return - don't speculate about
numbers you haven't looked up.

For get_race_results and get_fastest_laps, never guess a round number
for a named race - pass the race name itself if you aren't certain of
its round number.
"""


@dynamic_prompt
def _stats_prompt(request) -> str:
    # Computed fresh per model call, not baked in at build time - see
    # time_context.py for why that distinction matters.
    return f"{current_date_context()}\n\n{STATS_SYSTEM_PROMPT}"


def build_stats_agent(model):
    return create_agent(
        model,
        FASTF1_TOOLS,
        middleware=[_stats_prompt],
        name="stats_agent",
    )
