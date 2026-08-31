"""Thin wrapper around fastf1 + the Ergast standings API.

This is the only module that talks to fastf1/Ergast directly. Tools in
`box_box_bot.tools` call these functions and format their output for the
agent; nothing else in the app should import fastf1 directly.

Session objects (`fastf1.get_session`) give per-race results and lap data,
but not cumulative standings, so standings go through `fastf1.ergast`
instead.
"""

import fastf1
from fastf1.ergast import Ergast

from box_box_bot.config import FASTF1_CACHE_DIR

_cache_ready = False


def _ensure_cache() -> None:
    global _cache_ready
    if _cache_ready:
        return
    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    _cache_ready = True


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
    session = fastf1.get_session(season, round, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session.results.to_dict(orient="records")


def get_season_schedule(season: int) -> list[dict]:
    """Race calendar for a season: round number, event name, and date.

    Excludes pre-season testing. Used to find the latest completed round
    and the season's total round count for an in-progress season.
    """
    _ensure_cache()
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    return schedule[["RoundNumber", "EventName", "EventDate"]].to_dict(orient="records")


def get_fastest_laps(season: int, round: int | str, session_type: str = "R", top_n: int = 5) -> list[dict]:
    """Each driver's single fastest lap in a session, sorted quickest first.

    `session_type` follows fastf1 convention: 'FP1'/'FP2'/'FP3', 'Q', 'R'.
    """
    _ensure_cache()
    session = fastf1.get_session(season, round, session_type)
    session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps.dropna(subset=["LapTime"])
    fastest_idx = laps.groupby("Driver")["LapTime"].idxmin()
    fastest = laps.loc[fastest_idx, ["Driver", "Team", "LapTime", "LapNumber", "Compound"]]
    fastest = fastest.sort_values("LapTime").head(top_n).copy()
    fastest["LapTime"] = fastest["LapTime"].astype(str)
    return fastest.to_dict(orient="records")
