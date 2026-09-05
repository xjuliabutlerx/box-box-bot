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
def get_race_results(season: int, round: int | str) -> str:
    """Get the classified results for a single race: grid/finish position, points, and status.

    Use this to answer questions about the result of a particular race.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
    """
    data = fastf1_client.get_race_results(season, round)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_fastest_laps(season:int, round: int | str, session_type: str = "R", top_n: int = 5) -> str:
    """Get each driver's single fastest lap in a session, sorted quickest first.

    Use this to answer questions about the fastest laps for a particular race.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
        top_n: The fastest n drivers
    """
    data = fastf1_client.get_fastest_laps(season, round, session_type, top_n)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_season_schedule(season: int) -> str:
    """Get the race calendar for a season: round number, country, location, event name, date, and format (conventional or sprint weekend).

    Use this to answer questions about which races are on the calendar, when a race takes place, or which race a round number refers to. Excludes pre-season testing.

    Args:
        season: The four-digit F1 season year, e.g. 2026
    """
    data = fastf1_client.get_season_schedule(season)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_tire_strategy(season: int, round: int | str, session_type: str = "R") -> str:
    """Get the tire strategy for every driver for a particular session.

    Use this to answer questions about why a race result occured or when evaluating a driver's performance.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
    """
    data = fastf1_client.get_tire_strategy(season, round, session_type)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_race_control_messages(season: int, round: int | str, session_type: str = "R", category: str = "All") -> str:
    """Get race control messages for a session.

    Use this to answer any questions about flags, safety cars, and penalties during a session.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
        category: The type of messages to receive: 'Other' (general messages, notes, and penalties), 'Flag', 'SafetyCar', or 'All' for every message.
    """
    data = fastf1_client.get_race_control_messages(season, round, session_type, category)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_weather(season: int, round: int | str, session_type: str = "R") -> str:
    """Get the weather data by roughly every minute for a session.

    Use this to answer questions about whether a race was hot or cold, if there was rain fall, or when evaluating tire strategy.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
    """
    data = fastf1_client.get_weather_for_session(season, round, session_type)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_all_time_driver_records(top_n: int = 10) -> str:
    """Get the top F1 drivers of all time by career championships and race wins, aggregated across every season since 1950.

    Use this for career/all-time driver questions - who has won the most championships or races ever, historically the most successful drivers, etc. This is real aggregated data, not an opinion - but it only covers championships and wins; it does not cover poles, podiums, or fastest laps.

    Args:
        top_n: How many drivers to return, ranked by championships then wins.
    """
    data = fastf1_client.get_all_time_driver_records(top_n)
    return json.dumps(data, default=str)

FASTF1_TOOLS = [
    get_driver_standings,
    get_constructor_standings,
    get_race_results,
    get_fastest_laps,
    get_season_schedule,
    get_tire_strategy,
    get_race_control_messages,
    get_weather,
    get_all_time_driver_records,
]