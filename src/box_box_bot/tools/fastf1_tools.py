import json

from langchain_core.tools import tool

from box_box_bot.data import fastf1_client

# Learnings:
#   parse_docstring=True means that LangChain will read the docstring Args block and attach it to the tool description schema
#   The result of each tool should be a string because tool outputs are inserted into the model's context as text

# fastf1/Ergast return every column/field they have, including plenty
# that are never useful in an agent's response (headshot image URLs, hex team
# colors, internal driver/team IDs, Wikipedia links).
_RACE_RESULT_EXTRA_FIELDS = {
    "DriverNumber", "BroadcastName", "DriverId", "TeamColor", "TeamId",
    "FirstName", "LastName", "HeadshotUrl", "CountryCode",
    # Always NaT here - get_race_results only ever loads the Race ('R')
    # session, which has no qualifying times.
    "Q1", "Q2", "Q3",
}
_STANDINGS_EXTRA_FIELDS = {"driverUrl", "constructorUrl", "constructorUrls"}

def _drop_extra_fields(rows: list[dict], junk_fields: set[str]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in junk_fields} for row in rows]

def _summarize_weather(rows: list[dict]) -> dict:
    # A session's weather is sampled roughly every minute - 100+ rows for
    # a full race, meaningfully more detail than almost any conversational
    # question needs ("was it hot," "did it rain") and expensive to carry
    # in full through every weather-touching turn. Summarizing here keeps
    # data/fastf1_client.py's own return value as the honest full-fidelity
    # trace (nothing else needs it, but no reason to degrade the data
    # layer's contract for an agent-only formatting concern).
    if not rows:
        return {"SampleCount": 0}
    air_temps = [r["AirTemp"] for r in rows if r.get("AirTemp") is not None]
    track_temps = [r["TrackTemp"] for r in rows if r.get("TrackTemp") is not None]
    wind_speeds = [r["WindSpeed"] for r in rows if r.get("WindSpeed") is not None]
    return {
        "SampleCount": len(rows),
        "AirTemp": {"min": min(air_temps), "max": max(air_temps), "avg": sum(air_temps) / len(air_temps)} if air_temps else None,
        "TrackTemp": {"min": min(track_temps), "max": max(track_temps), "avg": sum(track_temps) / len(track_temps)} if track_temps else None,
        "AvgWindSpeed": sum(wind_speeds) / len(wind_speeds) if wind_speeds else None,
        "RainfallDuringSession": any(r.get("Rainfall") for r in rows),
    }

@tool(parse_docstring=True)
def get_driver_standings(season:int, round:int | None = None) -> str:
    """Get F1 driver championship standings for a season.

    Use this to answer questions about who is leading or how many points a driver has. If round is omitted, returns final/current standings.
    
    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season. Omit for the latest or final standings.
    """
    data = _drop_extra_fields(fastf1_client.get_driver_standings(season, round), _STANDINGS_EXTRA_FIELDS)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_constructor_standings(season:int, round:int | None = None) -> str:
    """Get F1 constructor championship standings for a season.

    Use this to answer questions about who is leading or how many points a constructor has. If round is omitted, returns final/current standings.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season. Omit for the latest or final standings.
    """
    data = _drop_extra_fields(fastf1_client.get_constructor_standings(season, round), _STANDINGS_EXTRA_FIELDS)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_race_results(season: int, round: int | str) -> str:
    """Get the classified results for a single race: grid/finish position, points, and status.

    Use this to answer questions about the result of a particular race.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
    """
    data = _drop_extra_fields(fastf1_client.get_race_results(season, round), _RACE_RESULT_EXTRA_FIELDS)
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
    """Get a weather summary for a session: temperature range, average wind speed, and whether it rained at any point.

    Use this to answer questions about whether a race was hot or cold, if there was rain fall, or when evaluating tire strategy. This is a summary, not the full per-minute trace - it can't answer "what was the temperature at lap 30," only session-wide questions.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
    """
    data = _summarize_weather(fastf1_client.get_weather_for_session(season, round, session_type))
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_pit_stops(season: int, round: int | str, session_type: str = "R") -> str:
    """Get every pit stop made in a session, with how long each one cost.

    Use this to evaluate strategy - e.g. whether an undercut worked, or how much time a slow stop cost a driver. The returned PitLaneTime is the full pit-lane transit time (entry to exit), NOT the on-camera stationary tire-change time broadcasts usually show - don't confuse the two.

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
    """
    data = fastf1_client.get_pit_stops(season, round, session_type)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_circuit_strategy_history(circuit: str, since_season: int = 2018) -> str:
    """Get how often a Safety Car, Virtual Safety Car, or Red Flag has historically occurred at a given circuit, season by season.

    Use this to set pre-race strategic expectations for a track - e.g. whether teams there typically need to plan around a safety car. Defaults to 2018 onward since that's fastf1's own cutoff for this kind of detailed session data; a circuit that hasn't hosted a race in a given season is silently skipped for that season.

    Args:
        circuit: The circuit or race name to look up, e.g. "Monaco", "Spa-Francorchamps", "Silverstone" - fuzzy-matched the same way race names are elsewhere.
        since_season: The first season to include in the walk. Defaults to 2018.
    """
    data = fastf1_client.get_circuit_strategy_history(circuit, since_season)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
def get_circuit_speed_map(season: int, round: int | str, session_type: str = "R", driver: str | None = None) -> str:
    """Get a lap's track outline colored by speed, oriented to match the circuit's real-world layout - use this whenever the user wants to see or visualize what a track looks like.

    Returns the fastest lap's position data (or a specific driver's fastest lap, if given) as a list of rotated points with their speed, plus corner marker positions. This produces a visual track map for the user - do not try to describe the individual coordinates in your answer, just acknowledge what's shown (circuit, driver, lap time).

    Args:
        season: The four-digit F1 season year, e.g. 2023
        round: Race round number within the season (e.g. 4), or the race name if you're not sure of the round number (e.g. "Bahrain", "Monaco", "Emilia Romagna Grand Prix") - this is fuzzy-matched against each event's country/location/name. Prefer passing the name over guessing a round number you aren't certain of.
        session_type: The F1 session type. One of 'FP1', 'FP2', 'FP3' (practice), 'Q' (qualifying), 'R' (race), 'S' (sprint race). Sprint weekends also have a session that sets the sprint grid: pass 'SS' for 2023 events or 'SQ' for 2024+ events.
        driver: A specific driver's three-letter code (e.g. "VER") to use their fastest lap instead of the session's overall fastest. Omit to use the overall fastest lap.
    """
    data = fastf1_client.get_circuit_speed_map(season, round, session_type, driver)
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
    get_all_time_driver_records,
]

STRATEGY_TOOLS = [
    get_tire_strategy,
    get_race_control_messages,
    get_weather,
    get_pit_stops,
    get_circuit_strategy_history,
    get_circuit_speed_map,
]