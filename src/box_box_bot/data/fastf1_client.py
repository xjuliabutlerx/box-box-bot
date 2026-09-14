"""Thin wrapper around fastf1 + the Ergast standings API.

This is the only module that talks to fastf1/Ergast directly. Tools in
`box_box_bot.tools` call these functions and format their output for the
agent; nothing else in the app should import fastf1 directly.

Session objects (`fastf1.get_session`) give per-race results and lap data,
but not cumulative standings, so standings go through `fastf1.ergast`
instead.
"""

import datetime
import math
import threading

import fastf1
import pandas as pd
from fastf1.ergast import Ergast

from box_box_bot.config import FASTF1_CACHE_DIR

_cache_ready = False

FIRST_F1_SEASON = 1950
FIRST_DETAILED_TIMING_SEASON = 2018  # fastf1's own documented cutoff for full lap-by-lap data (pit in/out, track status) - earlier seasons fall back to Ergast, which lacks it.

_all_time_records_cache: list[dict] | None = None
_all_time_records_lock = threading.Lock()

_circuit_strategy_cache: dict[str, dict] = {}
_circuit_strategy_lock = threading.Lock()


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


def get_qualifying_results(season: int, round: int | str) -> list[dict]:
    """Classified qualifying results for a single round: each driver's
    final qualifying position and Q1/Q2/Q3 times.

    This is genuine qualifying classification, not race grid position -
    grid position reflects post-penalty slots and can differ from how a
    driver actually qualified. Used internally by predictor/
    driver_features.py, which needs the real qualifying result.
    """
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), "Q")
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


def get_pit_stops(season: int, round: int | str, session_type: str = "R") -> list[dict]:
    """Every pit stop made in a session, with how long each one cost.

    fastf1 has no precomputed stop-duration column - a pit stop is
    represented as a non-null `PitInTime` on the in-lap and a non-null
    `PitOutTime` on the very next lap for that driver, so duration is
    derived here as `PitOutTime - PitInTime`. An in-lap with no matching
    following out-lap (e.g. a DNF right after pitting) is skipped rather
    than raising. `round` can be a round number or a race name, same as
    `get_race_results`.

    The returned `PitLaneTime` is the full pit-lane transit (entry line
    to exit line, commonly ~18-25s depending on the track) - it is NOT
    the on-camera stationary tire-change time (~2-3s) that broadcasts
    usually mean by "pit stop time."
    """
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), session_type)
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps[["Driver", "Team", "LapNumber", "PitInTime", "PitOutTime"]]

    stops = []
    for driver, driver_laps in laps.groupby("Driver"):
        driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
        for i, lap in driver_laps.iterrows():
            if pd.isna(lap["PitInTime"]) or i + 1 >= len(driver_laps):
                continue
            next_lap = driver_laps.iloc[i + 1]
            if pd.isna(next_lap["PitOutTime"]):
                continue
            stops.append({
                "Driver": driver,
                "Team": lap["Team"],
                "LapNumber": lap["LapNumber"],
                "PitLaneTime": next_lap["PitOutTime"] - lap["PitInTime"],
            })

    return pd.DataFrame(stops).to_dict(orient="records")


def _build_circuit_strategy_history(circuit: str, since_season: int) -> dict:
    """Walks every season from `since_season` to the current year,
    attempting to resolve `circuit` to that season's race there (fuzzy
    name matching, same as `get_race_results`) and recording whether a
    Safety Car, Virtual Safety Car, or Red Flag occurred. A season where
    the circuit doesn't resolve (rotating calendar slot, renamed event,
    not yet run) is skipped rather than raising - most circuits won't
    have run every season anyway. `session.load()` succeeding is not
    enough on its own to guarantee `session.track_status` is readable:
    fastf1 only populates it when that session has full API support
    (`session.f1_api_support`), which can be False even for a
    post-2018 session (e.g. incomplete/partial data for that event) -
    reading `track_status` must stay inside the same try/except, since a
    successfully-loaded session can still raise `DataNotLoadedError`
    the moment it's accessed. Also, `fastf1.get_session`'s fuzzy name
    match doesn't fail loudly when a circuit genuinely isn't on a given
    season's calendar (e.g. Monaco's 2020 COVID cancellation) - it just
    returns whatever event scored closest, silently, which would
    otherwise corrupt the aggregate with an unrelated race's SC/VSC/red-
    flag status. So the resolved event's own name/location/country must
    actually contain the requested circuit string before it's trusted;
    a resolved event that doesn't is treated as "not on the calendar
    this season" and skipped, the same as an outright lookup failure.
    """
    current_year = datetime.date.today().year
    by_season = []
    circuit_key = circuit.strip().casefold()

    for season in range(since_season, current_year + 1):
        try:
            session = fastf1.get_session(season, circuit, "R")
            session.load(telemetry=False, weather=False, messages=False)
            resolved = f"{session.event['Location']} {session.event['Country']} {session.event['EventName']}"
            if circuit_key not in resolved.casefold():
                continue
            codes = set(session.track_status["Status"].tolist())
        except Exception:
            continue

        by_season.append({
            "season": season,
            "event_name": session.event["EventName"],
            "safety_car": "4" in codes,
            "vsc": "6" in codes,
            "red_flag": "5" in codes,
        })

    return {
        "circuit": circuit,
        "since_season": since_season,
        "total_races_found": len(by_season),
        "safety_car_races": sum(1 for s in by_season if s["safety_car"]),
        "vsc_races": sum(1 for s in by_season if s["vsc"]),
        "red_flag_races": sum(1 for s in by_season if s["red_flag"]),
        "by_season": by_season,
    }


def get_circuit_strategy_history(circuit: str, since_season: int = FIRST_DETAILED_TIMING_SEASON) -> dict:
    """How often a Safety Car, VSC, or Red Flag has historically occurred
    at a given circuit, walked season-by-season since `since_season`
    (defaults to 2018, fastf1's own cutoff for this kind of detailed
    session data). Cached per circuit for the life of the process - the
    underlying walk loads one race session per season, so it's too slow
    to redo on every request but only needs to happen once per circuit
    per server run (same shape as `get_all_time_driver_records`'s cache).
    """
    _ensure_cache()
    key = circuit.strip().casefold()
    if key not in _circuit_strategy_cache:
        with _circuit_strategy_lock:
            if key not in _circuit_strategy_cache:
                _circuit_strategy_cache[key] = _build_circuit_strategy_history(circuit, since_season)
    return _circuit_strategy_cache[key]


def _rotate(x: float, y: float, angle_radians: float) -> tuple[float, float]:
    return (
        x * math.cos(angle_radians) - y * math.sin(angle_radians),
        x * math.sin(angle_radians) + y * math.cos(angle_radians),
    )


def get_circuit_speed_map(
    season: int,
    round: int | str,
    session_type: str = "R",
    driver: str | None = None,
    max_points: int = 400,
) -> dict:
    """A lap's track outline colored by speed, oriented to match the
    circuit's real-world layout.

    fastf1 has no bundled full track-outline dataset - `get_circuit_info()`
    only gives corner/marshal-post markers, not a continuous shape (its
    own docstring says so) - so the outline here comes from a lap's own
    position telemetry instead (X/Y position, merged with car `Speed`).
    Both the outline points and the corner markers are rotated by the
    circuit's documented `rotation` (degrees) so they match the real
    track orientation and stay aligned with each other.

    Telemetry is downsampled to at most `max_points` (even-interval
    slicing) before returning - a full lap's raw telemetry can be several
    hundred to a few thousand samples, far more detail than a clean plot
    needs and far more than's worth spending on LLM context for data the
    model only needs to acknowledge, not read point-by-point.
    """
    _ensure_cache()
    session = fastf1.get_session(season, _normalize_round(round), session_type)
    session.load(telemetry=True, weather=False, messages=False)

    laps = session.laps.pick_drivers(driver) if driver else session.laps
    lap = laps.pick_fastest()
    telemetry = lap.get_telemetry()[["X", "Y", "Speed"]]

    if len(telemetry) > max_points:
        # Ceiling division - floor division here would compute step=1 (no
        # downsampling at all) for any length under 2x max_points, e.g.
        # 729 // 400 == 1, silently letting the result exceed max_points.
        step = -(-len(telemetry) // max_points)
        telemetry = telemetry.iloc[::step]

    circuit_info = session.get_circuit_info()
    rotation = math.radians(circuit_info.rotation)

    points = []
    for _, row in telemetry.iterrows():
        rx, ry = _rotate(row["X"], row["Y"], rotation)
        points.append({"X": rx, "Y": ry, "Speed": row["Speed"]})

    corners = []
    for _, row in circuit_info.corners.iterrows():
        cx, cy = _rotate(row["X"], row["Y"], rotation)
        corners.append({"Number": row["Number"], "Letter": row["Letter"], "X": cx, "Y": cy})

    return {
        "circuit": session.event["EventName"],
        "driver": lap["Driver"],
        "lap_time": str(lap["LapTime"]),
        "rotation_degrees": circuit_info.rotation,
        "points": points,
        "corners": corners,
    }


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
