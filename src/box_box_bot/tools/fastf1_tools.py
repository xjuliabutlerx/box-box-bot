import datetime
import functools
import json
import logging

from langchain_core.tools import tool

from box_box_bot.data import fastf1_client

_logger = logging.getLogger(__name__)


def _find_future_event(season, round_value):
    """If `round_value` resolves to a scheduled-but-not-yet-run event in
    `season`, return that event's schedule row; otherwise None.

    A tool failure for a not-yet-happened session is indistinguishable
    from any other fastf1 failure once it's an exception. The generic message
    doesn't tell the agent it's a *dated*, checkable condition, so it
    silently gives up instead of either explaining why or trying last
    year. This does a cheap schedule lookup (round-number or name/
    location/country substring match, same lightweight style as
    `_build_circuit_strategy_history`'s validation) only on the failure
    path, not on every call.
    """
    try:
        schedule = fastf1_client.get_season_schedule(season)
    except Exception:
        return None

    round_key = str(round_value).strip().casefold()
    match = None
    for row in schedule:
        if str(row.get("RoundNumber")) == round_key:
            match = row
            break
        haystack = f"{row.get('EventName', '')} {row.get('Location', '')} {row.get('Country', '')}".casefold()
        if round_key and round_key in haystack:
            match = row
            break
    if match is None:
        return None

    event_date = match.get("EventDate")
    if hasattr(event_date, "date"):
        event_date = event_date.date()
    if not isinstance(event_date, datetime.date) or event_date <= datetime.date.today():
        return None
    return match


# Any exception escaping a tool function aborts the entire agent turn
# (LangGraph's ToolNode only catches its own internal error type by
# default), so every fastf1-backed tool needs to catch failures here and
# hand the agent a readable message instead of crashing the whole turn.
def _catch_fastf1_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # the agent only ever sees the sanitized message below - this
            # is the one place the real exception (and which tool/args
            # triggered it) gets recorded anywhere
            _logger.warning(
                "%s failed (args=%s kwargs=%s): %s", func.__name__, args, kwargs, exc, exc_info=True
            )

            season = kwargs.get("season")
            round_value = kwargs.get("round")
            future_event = (
                _find_future_event(season, round_value)
                if season is not None and round_value is not None
                else None
            )

            if future_event is not None:
                # Deterministic, bounded fallback - try exactly one year
                # back ourselves rather than leaving it to the model's
                # judgment whether to retry

                # `season: int` is coerced to int by tool-arg validation 
                # before this function body ever runs, or rejected outright
                # if it can't be parsed as one
                fallback_season = season - 1
                fallback_kwargs = {**kwargs, "season": fallback_season}
                try:
                    fallback_data = json.loads(func(*args, **fallback_kwargs))
                except Exception:
                    fallback_data = None

                if fallback_data and not (isinstance(fallback_data, dict) and "error" in fallback_data):
                    _logger.info(
                        "%s: %s (%s) hasn't happened yet - fell back to season=%s",
                        func.__name__, future_event.get("EventName"), season, fallback_season,
                    )
                    return json.dumps({
                        "fallback_note": (
                            f"The {future_event.get('EventName', 'race')} ({season}) "
                            f"hasn't happened yet - it's scheduled for "
                            f"{future_event['EventDate']}. Showing {fallback_season} "
                            f"data instead. You MUST tell the user this is "
                            f"{fallback_season}'s data, not {season}'s, since this "
                            "year's race hasn't run yet."
                        ),
                        "season_used": fallback_season,
                        "result": fallback_data,
                    })

                error_message = (
                    f"The {future_event.get('EventName', 'race')} ({season}) hasn't "
                    f"happened yet - it's scheduled for {future_event['EventDate']} - "
                    f"and {fallback_season} data wasn't available either. Tell the "
                    "user this data isn't available rather than guessing an answer."
                )
            else:
                error_message = (
                    f"Could not load this data: {exc}. This can happen for a "
                    "session that hasn't happened yet, a season before "
                    f"{fastf1_client.FIRST_DETAILED_TIMING_SEASON} (fastf1's "
                    "full timing data doesn't cover it), or a temporary data "
                    "source issue. Tell the user this specific data isn't "
                    "available rather than guessing an answer."
                )
            return json.dumps({
                "error": error_message
            })
    return wrapper

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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
def get_season_schedule(season: int) -> str:
    """Get the race calendar for a season: round number, country, location, event name, date, and format (conventional or sprint weekend).

    Use this to answer questions about which races are on the calendar, when a race takes place, or which race a round number refers to. Excludes pre-season testing.

    Args:
        season: The four-digit F1 season year, e.g. 2026
    """
    data = fastf1_client.get_season_schedule(season)
    return json.dumps(data, default=str)

@tool(parse_docstring=True)
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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
@_catch_fastf1_errors
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