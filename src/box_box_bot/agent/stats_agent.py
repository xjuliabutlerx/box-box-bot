from langchain.agents import create_agent

from box_box_bot.tools.fastf1_tools import FASTF1_TOOLS

STATS_SYSTEM_PROMPT = """You are box-box-bot's stats specialist.

Answer factual, numeric F1 questions using your fastf1 tools: driver/
constructor standings, race results, and fastest laps. Stick to what the
tools return - don't speculate about numbers you haven't looked up.

For get_race_results and get_fastest_laps, never guess a round number
for a named race - pass the race name itself if you aren't certain of
its round number.
"""

def build_stats_agent(model):
    return create_agent(
        model,
        FASTF1_TOOLS,
        system_prompt=STATS_SYSTEM_PROMPT,
        name="stats_agent",
    )
