"""Pre-fetches fastf1 session data into FASTF1_CACHE_DIR so the deployed
app never needs to make the request that F1's own servers block.

`livetiming.formula1.com` (fastf1's primary session-data source) returns
403 Forbidden to datacenter/cloud IP ranges - confirmed both by direct
testing (a fresh, uncached request from this residential-network dev
environment gets 200; the identical request from Streamlit Cloud gets
403) and by multiple independent reports of the same failure on other
cloud hosts (GitHub Actions runners, VPS deployments). fastf1 falls back
to a community mirror on a 403, but that mirror doesn't carry every
session (confirmed 404 for a session this script warms). This affects
every function that calls `fastf1.get_session(...).load(...)` - race
results, fastest laps, tire strategy, pit stops, weather, race control,
circuit speed maps - regardless of which of those flags is requested,
since session info/results load unconditionally. It does NOT affect
Ergast-sourced data (standings, schedule, all-time records), which
isn't subject to this block.

This is a manual, offline step - like rag/ingest.py - not something run
in CI or at deploy time, because it specifically needs a residential
network path Streamlit Cloud's own container doesn't have. Run it here
(or anywhere with normal home-internet access), then commit the
resulting FASTF1_CACHE_DIR files - fastf1's cache is just HTTP response
files keyed by request, so a request the deployed app makes that's
already cached is served locally and never touches the blocked source
at all.

Two warming passes, both WITHOUT telemetry by default - car_data/
position_data run 40-80MB *per session* (confirmed by measuring an
actual warmed cache: a single round's Race telemetry alone was ~150MB,
and warming all of it for every session ballooned the cache to 2.3GB,
unworkable to commit). Every other tool - race results, fastest laps,
tire strategy, pit stops, weather, race control - only needs
laps/weather/messages, not telemetry, so this keeps the routine warm
cheap:

- Current season, every completed round, BOTH Race and Qualifying
  (Qualifying isn't exposed as an agent tool, but the predictor's
  feature-building reads it directly per round).
- Every (season, round) already named in a race_recaps frontmatter
  block, Race only - these are fixed historical results that will
  never change, so caching once covers them permanently.
- STRATEGY_HISTORY_CIRCUITS: circuits get_circuit_strategy_history is
  likely to be asked about for "general strategy" questions but that
  have no race_recaps entry at all (so warm_recap_races() alone doesn't
  cover them) - warmed across every season since
  fastf1_client.FIRST_DETAILED_TIMING_SEASON (2018), since that tool
  walks that same full range. Live-reported gap: a demo run asked about
  Baku, which isn't referenced by any recap, and failed entirely on the
  deployed app (no cached data, and the IP block prevents fetching it
  live) even though the exact same question worked fine locally, where
  an uncached call just silently live-fetches instead of failing - local
  testing looking correct is not proof the deployed app has the data.

`warm_current_season_telemetry()` is separate: it warms Race-session
telemetry for every completed round of the current season, so
get_circuit_speed_map - and the "always show a circuit visualization
for a fastest-lap question" behavior in SUPERVISOR_PROMPT - work for
any race this season, not just the most recent one. This is the most
expensive pass (~100-150MB per round, ~1.5GB for a full season) - a
deliberate size/coverage tradeoff, not something to extend to the
historical race_recaps races too without reconsidering it.

Re-run this after each new round of the current season completes to
keep the deployed app's coverage current - it's additive (skips
whatever's already cached) and safe to run repeatedly.
"""

import datetime

import yaml
import fastf1

from box_box_bot.config import RACE_RECAPS_DIR
from box_box_bot.data import fastf1_client

# Circuits worth having full get_circuit_strategy_history coverage for
# even though no race_recaps entry references them - add a circuit here
# the moment a "general strategy at X" question is expected to come up
# for it (e.g. via a demo script), rather than discovering the gap only
# when the deployed app fails on it.
STRATEGY_HISTORY_CIRCUITS = ["Baku"]


def _warm_session(season: int, round_value, session_type: str, telemetry: bool) -> bool:
    try:
        fastf1_client._ensure_cache()
        session = fastf1.get_session(season, round_value, session_type)
        session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
        print(f"  OK    {season} round {round_value} {session_type}: {session.event['EventName']}")
        return True
    except Exception as exc:
        print(f"  FAILED {season} round {round_value} {session_type}: {exc}")
        return False


def _completed_rounds(season: int) -> list[int]:
    schedule = fastf1_client.get_season_schedule(season)
    today = datetime.date.today()
    rounds = []
    for row in schedule:
        event_date = row["EventDate"]
        if hasattr(event_date, "date"):
            event_date = event_date.date()
        if isinstance(event_date, datetime.date) and event_date <= today:
            rounds.append(row["RoundNumber"])
    return rounds


def _recap_race_combos() -> list[tuple[int, int]]:
    combos = []
    for path in sorted(RACE_RECAPS_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text()
        _, frontmatter_block, _ = text.split("---", 2)
        metadata = yaml.safe_load(frontmatter_block)
        if "season" in metadata and "round" in metadata:
            combos.append((metadata["season"], metadata["round"]))
    return combos


def warm_current_season(season: int | None = None) -> None:
    season = season or datetime.date.today().year
    rounds = _completed_rounds(season)
    print(f"Warming {len(rounds)} completed round(s) of {season} (Race + Qualifying, no telemetry)...")
    for round_number in rounds:
        _warm_session(season, round_number, "R", telemetry=False)
        _warm_session(season, round_number, "Q", telemetry=False)


def warm_recap_races() -> None:
    combos = _recap_race_combos()
    print(f"Warming {len(combos)} race-recap-referenced race(s) (Race only, no telemetry)...")
    for season, round_number in combos:
        _warm_session(season, round_number, "R", telemetry=False)


def warm_current_season_telemetry(season: int | None = None) -> None:
    season = season or datetime.date.today().year
    rounds = _completed_rounds(season)
    print(f"Warming telemetry for {len(rounds)} completed round(s) of {season} (Race only)...")
    for round_number in rounds:
        _warm_session(season, round_number, "R", telemetry=True)


def warm_strategy_history_circuits() -> None:
    current_year = datetime.date.today().year
    seasons = range(fastf1_client.FIRST_DETAILED_TIMING_SEASON, current_year + 1)
    print(f"Warming {len(STRATEGY_HISTORY_CIRCUITS)} strategy-history circuit(s), "
          f"{fastf1_client.FIRST_DETAILED_TIMING_SEASON}-{current_year} each (Race only, no telemetry)...")
    for circuit in STRATEGY_HISTORY_CIRCUITS:
        for season in seasons:
            _warm_session(season, circuit, "R", telemetry=False)


if __name__ == "__main__":
    warm_current_season()
    print()
    warm_recap_races()
    print()
    warm_strategy_history_circuits()
    print()
    warm_current_season_telemetry()
