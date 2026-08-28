import json

from langchain_core.tools import tool

from box_box_bot.data import fastf1_client

# Learnings:
#   parse_docstring=True means that LangChain will read the docstring Args block and attach it to the tool description schema
#   The result of each tool should be a string because tool outputs are inserted into the model's context as text

@tool(parse_docstring=True)
def get_driver_standings(season:int, round:int | None = None) -> str:
    """Get F1 driver championship standings for a season.

    Use this to answer questions about who is leading or how many points a driver has. If round is omitted, returns final/current standings.
    
    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season. Omit for the latest or final standings.
    """
    data = fastf1_client.get_driver_standings(season, round)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_constructor_standings(season:int, round:int | None = None) -> str:
    """Get F1 constructor championship standings for a season.

    Use this to answer questions about who is leading or how many points a constructor has. If round is omitted, returns final/current standings.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season. Omit for the latest or final standings.
    """
    data = fastf1_client.get_constructor_standings(season, round)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_race_results(season: int, round: int) -> str:
    """Get the classified results for a single race: grid/finish position, points, and status.

    Use this to answer questions about the result of a particular race.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season. Omit for the latest or final standings.
    """
    data = fastf1_client.get_race_results(season, round)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_fastest_laps(season:int, round:int, session_type: str = "R", top_n: int = 5) -> str:
    """Get each driver's single fastest lap in a session, sorted quickest first.

    Use this to answer questions about the fastest laps for a particular race.
    
    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
        top_n: The fastest n drivers
    """
    data = fastf1_client.get_fastest_laps(season, round, session_type, top_n)
    return json.dumps(data, default=str)

FASTF1_TOOLS = [
    get_driver_standings,
    get_constructor_standings,
    get_race_results,
    get_fastest_laps
]