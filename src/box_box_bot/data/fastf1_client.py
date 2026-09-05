"""Thin wrapper around fastf1 + the Ergast standings API.

This is the only module that talks to fastf1/Ergast directly. Tools in
`box_box_bot.tools` call these functions and format their output for the
agent; nothing else in the app should import fastf1 directly.

Session objects (`fastf1.get_session`) give per-race results and lap data,
but not cumulative standings, so standings go through `fastf1.ergast`
instead.
"""

import datetime
import threading

import fastf1
from fastf1.ergast import Ergast

from box_box_bot.config import FASTF1_CACHE_DIR

_cache_ready = False

FIRST_F1_SEASON = 1950

_all_time_records_cache: list[dict] | None = None
_all_time_records_lock = threading.Lock()


def _ensure_cache() -> None:
    global _cache_ready
    if _cache_ready:
        return
    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    _cache_ready = True


def _normalize_round(round: int | str) -> int | str:
    """A numeric round passed as a string (e.g. "7") must become an int
    before reaching `fastf1.get_session`. fastf1 only treats a round as a
    round *number* when it's an int - a string round is always fuzzy-
    matched against each event's country/location/name instead, and a
    bare digit string doesn't resemble any of those. Rather than raising,
    fastf1 silently falls back to the wrong race (round="7" and round="1"
    both resolved to the season's first race) - a real regression hit in
    production once the model started passing round as a JSON string
    instead of a number. Only pass a genuine race name through to
    fastf1's fuzzy matching.
    """
    if isinstance(round, str) and round.strip().isdigit():
        return int(round)
    return round


def get_driver_standings(season: int, round: int | None = None) -> list[dict]:
    """Driver championship standings for a season.

    If `round` is given, standings are as of that round; otherwise the
    final/current standings for the season are returned.
    """
    _ensure_cache()
    response = Ergast().get_driver_standings(season=season, round=round)
    return response.content[0].to_dict(orient="records")


def get_constructor_standings(season: int, round: int | None = None) -> list[dict]:
    """Constructor championship standings for a season.

    If `round` is given, standings are as of that round; otherwise the
    final/current standings for the season are returned.
    """
    _ensure_cache()
    response = Ergast().get_constructor_standings(season=season, round=round)
    return response.content[0].to_dict(orient="records")


def get_race_results(season: int, round: int | str) -> list[dict]:
    """Classified results for a single race: grid/finish position, points, status.

    `round` can be a round number or a race name - `fastf1.get_session`
    fuzzy-matches a string against each event's country/location/name.
    """
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session.results.to_dict(orient="records")


def get_season_schedule(season: int) -> list[dict]:
    """Race calendar for a season: round number, country, location, event
    name, date, and format (conventional or sprint weekend).

    Excludes pre-season testing. Also used internally by predictor/
    features.py to find the latest completed round and the season's
    total round count for an in-progress season.
    """
    _ensure_cache()
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    columns = ["RoundNumber", "Country", "Location", "EventName", "EventFormat", "EventDate"]
    return schedule[columns].to_dict(orient="records")


def get_fastest_laps(season: int, round: int | str, session_type: str = "R", top_n: int = 5) -> list[dict]:
    """Each driver's single fastest lap in a session, sorted quickest first.

    `session_type` follows fastf1 convention: 'FP1'/'FP2'/'FP3', 'Q', 'R'.
    """
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), session_type)
    session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps.dropna(subset=["LapTime"])
    fastest_idx = laps.groupby("Driver")["LapTime"].idxmin()
    fastest = laps.loc[fastest_idx, ["Driver", "Team", "LapTime", "LapNumber", "Compound"]]
    fastest = fastest.sort_values("LapTime").head(top_n).copy()
    fastest["LapTime"] = fastest["LapTime"].astype(str)
    return fastest.to_dict(orient="records")

def get_tire_strategy(season: int, round: int | str, session_type: str = "R") -> list[dict]:
    """Each driver's tire strategy in a session.

    `session_type` follows fastf1 convention: 'FP1'/'FP2'/'FP3', 'Q', 'R'.
    """
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), session_type)
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps

    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})
    return stints.to_dict(orient="records")

def get_race_control_messages(season: int, round: int | str, session_type: str = "R", category: str = "All") -> list[dict]:
    """Get radio messages about a particular session.

    `session_type` follows fastf1 convention: 'FP1'/'FP2'/'FP3', 'Q', 'R'.
    `category` follows the types of race control messages, following fastf1 convention: 'Other', 'Flag', or 'SafetyCar'
    """
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), session_type)
    session.load(laps=False, telemetry=False, weather=False)
    messages = session.race_control_messages

    if category.casefold() == "other":
        messages = messages[messages["Category"] == "Other"]
    elif category.casefold() == "flag":
        messages = messages[messages["Category"] == "Flag"]
    elif category.casefold() == "safetycar":
        messages = messages[messages["Category"] == "SafetyCar"]

    return messages.to_dict(orient="records")

def get_weather_for_session(season: int, round: int | str, session_type: str = "R") -> list[dict]:
    """Get information about the weather for a session."""
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), session_type)
    session.load(laps=False, telemetry=False, messages=False)
    return session.weather_data.to_dict(orient="records")


def _build_all_time_driver_records() -> list[dict]:
    """Career wins and championships per driver, aggregated across every
    F1 season. Ergast has no career-aggregate endpoint, so this walks
    every season's final standings itself - one call per season (not per
    round), since each row already carries that season's `wins` count and
    `position` (1 = that season's champion) alongside a stable `driverId`.
    Podiums/poles would need per-round data instead and aren't covered.
    """
    totals: dict[str, dict] = {}
    current_year = datetime.date.today().year

    for season in range(FIRST_F1_SEASON, current_year + 1):
        response = Ergast().get_driver_standings(season=season, round=None)
        if not response.content:
            continue
        for row in response.content[0].to_dict(orient="records"):
            driver_id = row["driverId"]
            entry = totals.setdefault(
                driver_id,
                {
                    "driverId": driver_id,
                    "driverName": f"{row['givenName']} {row['familyName']}",
                    "totalWins": 0,
                    "championships": 0,
                },
            )
            entry["totalWins"] += row["wins"]
            if row["position"] == 1:
                entry["championships"] += 1

    return sorted(totals.values(), key=lambda entry: (entry["championships"], entry["totalWins"]), reverse=True)


def get_all_time_driver_records(top_n: int = 10) -> list[dict]:
    """Top drivers by career championships and race wins across every F1
    season (1950-present). Computed once and cached for the life of the
    process - the underlying walk takes ~one Ergast call per season, so
    it's too slow to redo on every request but only needs to happen once
    per server run (mirrors `predictor/features.py`'s feature-table cache).
    """
    global _all_time_records_cache
    _ensure_cache()
    if _all_time_records_cache is None:
        with _all_time_records_lock:
            if _all_time_records_cache is None:
                _all_time_records_cache = _build_all_time_driver_records()
    return _all_time_records_cache[:top_n]
