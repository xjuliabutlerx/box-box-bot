"""Feature engineering for the constructor-ranking model, ported from
f1-constructors-predictor's src/data/data_pipeline.py.

The model's weights are calibrated to the exact numeric distributions
that pipeline produced, so every formula here is reproduced as-is,
including two known quirks (see FormRatio and RoundsRemaining below) -
"fixing" either would silently feed the model out-of-distribution
numbers rather than improve anything.

Two deliberate deviations from the source, both documented rather than
hidden:
- Only the race ('R') session is used, never sprint ('S') - the source
  pipeline's per-round aggregates accidentally mix sprint + race rows on
  sprint weekends; this port doesn't reproduce that mixing.
- RelativePointsShare/PercentileRankAfterRound are computed from our own
  locally-accumulated TotalPoints (already available since every team's
  race results get pulled anyway) rather than a separate Ergast
  standings call - simpler and self-consistent.
"""

import threading

import numpy as np
import pandas as pd

from box_box_bot.data import fastf1_client

FEATURE_COLUMNS = [
    "Year", "Round", "RoundsRemaining", "PointsEarnedThisRound", "DriverPointsGap",
    "DNFsThisRound", "PointsLast3Rounds", "DNFsLast3Rounds", "DNFRate",
    "AvgGridPosition", "AvgPosition", "AvgPointsPerRace", "TotalPointFinishes",
    "FormRatio", "Consistency", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear",
    "ProjectedSeasonTotalPoints", "RelativePointsShare", "PercentileRankAfterRound",
]

# Columns log1p'd before the model sees them - ported verbatim (same set,
# same place in the pipeline: applied once per built feature table).
SKEWED_FEATURE_COLUMNS = [
    "DNFsThisRound", "DNFsLast3Rounds", "PointsEarnedThisRound", "PointsLast3Rounds",
    "TotalPointFinishes", "TotalPodiums", "TotalPoints", "DriverPointsGap",
]

# Historical team-rebrand mapping, ported verbatim.
_TEAM_ID_RENAMES = {
    "alfa": "sauber",
    "renault": "alpine",
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
}
_TEAM_NAME_RENAMES = {
    "Alfa Romeo Racing": "Alfa Romeo",
    "Sauber": "Kick Sauber",
}

# Point penalties applied to a team's standing - ported verbatim, would
# need a new entry added by hand for any future penalty.
_PENALTIES = {2020: {"aston_martin": 15}}


def _normalize_team_ids(df: pd.DataFrame) -> pd.DataFrame:
    df["TeamId"] = df["TeamId"].replace(_TEAM_ID_RENAMES)
    df["TeamName"] = df["TeamName"].replace(_TEAM_NAME_RENAMES)
    return df


def _completed_rounds_and_total(season: int) -> tuple[list[int], int]:
    schedule = fastf1_client.get_season_schedule(season)
    now = pd.Timestamp.now()
    completed = sorted(
        row["RoundNumber"] for row in schedule if pd.Timestamp(row["EventDate"]) <= now
    )
    return completed, len(schedule)


def _round_team_table(season: int, round_number: int) -> pd.DataFrame:
    results = pd.DataFrame(fastf1_client.get_race_results(season, round_number))
    results = _normalize_team_ids(results)

    results["isDNF"] = results["ClassifiedPosition"].apply(lambda x: 1 if not str(x).isnumeric() else 0)
    results["isPointsFinish"] = (results["Points"] > 0).astype(int)
    results["isPodiumFinish"] = (results["Points"] >= 15).astype(int)

    grouped = results.groupby(["TeamId", "TeamName"], as_index=False).agg(
        DNFsThisRound=("isDNF", "sum"),
        PointsEarnedThisRound=("Points", "sum"),
        MaxDriverPoints=("Points", "max"),
        MinDriverPoints=("Points", "min"),
        RoundAvgGridPosition=("GridPosition", "mean"),
        RoundAvgPosition=("Position", "mean"),
        RoundDNFRate=("isDNF", "mean"),
        RoundAvgPointsPerDriver=("Points", "mean"),
        TotalPointFinishes=("isPointsFinish", "sum"),
        TotalPodiums=("isPodiumFinish", "sum"),
    )
    grouped["DriverPointsGap"] = grouped["MaxDriverPoints"] - grouped["MinDriverPoints"]
    grouped["Round"] = round_number
    return grouped


def build_team_features(season: int) -> pd.DataFrame:
    """Every completed round's per-team feature row for a season.

    One row per (team, round), sorted by team then round - callers
    predicting the current standings should take each team's row for the
    latest round.
    """
    completed_rounds, total_rounds = _completed_rounds_and_total(season)
    if not completed_rounds:
        raise ValueError(f"No completed rounds found for {season} yet.")

    df = pd.concat([_round_team_table(season, r) for r in completed_rounds], ignore_index=True)
    df = df.sort_values(["TeamId", "Round"]).reset_index(drop=True)

    g = df.groupby("TeamId")
    df["AvgGridPosition"] = g["RoundAvgGridPosition"].expanding().mean().reset_index(level=0, drop=True)
    df["AvgPosition"] = g["RoundAvgPosition"].expanding().mean().reset_index(level=0, drop=True)
    df["DNFRate"] = g["RoundDNFRate"].expanding().mean().reset_index(level=0, drop=True)
    df["AvgPointsPerRace"] = g["RoundAvgPointsPerDriver"].expanding().mean().reset_index(level=0, drop=True)

    df["TotalPointFinishes"] = g["TotalPointFinishes"].cumsum()
    df["TotalPodiums"] = g["TotalPodiums"].cumsum()
    df["TotalPoints"] = g["PointsEarnedThisRound"].cumsum()

    df["PointsLast3Rounds"] = g["PointsEarnedThisRound"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["DNFsLast3Rounds"] = g["DNFsThisRound"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    # Numerator is the team's 3-round total; denominator is a *per-driver*
    # average x3 - this makes the ratio run ~2x what the name implies.
    # A real quirk in the source model's training data, reproduced as-is.
    df["FormRatio"] = df["PointsLast3Rounds"] / (df["AvgPointsPerRace"] * 3 + 1e-6)

    rolling_mean_5 = g["PointsEarnedThisRound"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    rolling_std_5 = g["PointsEarnedThisRound"].rolling(5, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    df["Consistency"] = 1 / (1 + rolling_std_5 / (rolling_mean_5 + 1e-6))

    # RoundsCompleted = Round - 1 and RoundsRemaining = total - RoundsCompleted
    # (not total - Round) - a one-off inconsistency in the source between its
    # base formula and its in-progress-season override. Since predicting an
    # in-progress season is exactly our use case, the override is the
    # formula that matters here, reproduced exactly.
    rounds_completed = df["Round"] - 1
    df["RoundsRemaining"] = total_rounds - rounds_completed
    df["ProjectedSeasonTotalPoints"] = df["TotalPoints"] + rolling_mean_5 * df["RoundsRemaining"]

    df["Year"] = season
    df["hadPenaltyThisYear"] = df["TeamId"].apply(lambda tid: 1 if tid in _PENALTIES.get(season, {}) else 0)

    round_totals = df.groupby("Round")["TotalPoints"].transform("sum")
    df["RelativePointsShare"] = df["TotalPoints"] / round_totals
    rank = df.groupby("Round")["TotalPoints"].rank(method="dense", ascending=False)
    n_teams = df.groupby("Round")["TeamId"].transform("nunique")
    df["PercentileRankAfterRound"] = 1.0 - (rank - 1) / (n_teams - 1)

    for col in SKEWED_FEATURE_COLUMNS:
        df[col] = np.log1p(df[col])

    return df[["TeamId", "TeamName"] + FEATURE_COLUMNS]


_feature_cache: dict[int, pd.DataFrame] = {}
_feature_cache_lock = threading.Lock()


def get_team_features(season: int) -> pd.DataFrame:
    """Cached wrapper around build_team_features.

    Building this table means looping every completed round of the
    season through fastf1 - expensive on a cold cache. Cached in-process
    per season so only the first predictor query per season pays that
    cost; every later query (any agent, any session, same server
    process) reuses it.
    """
    if season not in _feature_cache:
        with _feature_cache_lock:
            if season not in _feature_cache:
                _feature_cache[season] = build_team_features(season)
    return _feature_cache[season]
