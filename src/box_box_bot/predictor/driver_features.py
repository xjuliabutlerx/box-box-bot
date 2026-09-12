"""Feature engineering for the drivers-ranking model, ported from
f1-drivers-predictor's src/data/data_pipeline.py.

Adapted the same way predictor/features.py adapted the constructors
pipeline: computed live from fastf1/Ergast per request, not a literal
port of the source's raw-CSV batch functions. The model's weights are
calibrated to the source's exact formulas, so every one below is
reproduced as closely as could be confirmed during porting.

Three categories of fidelity, documented rather than hidden:

1. Confirmed-faithful (verified during exploration of the source):
   TotalPodiums' points>=15 proxy, the fixed sorted TeamId one-hot
   vocabulary, the asymmetric skewed-column log1p list, dropping (not
   imputing) rows with missing TeamId/numeric features, and every
   feature that mirrors a constructors formula this project already
   verified (DNFRate, AvgGridPosition, AvgPointsPerRace, FormRatio,
   Consistency, ProjectedSeasonTotalPoints, RelativePointsShare,
   PercentileRankAfterRound - all reused from predictor/features.py's
   team-level table via a per-round join, not recomputed).
2. Best-effort reconstructions (the exact source formula wasn't directly
   available during exploration - implemented from the feature's name
   and the surrounding pipeline's conventions, flagged here for live
   verification): RoundsRemaining, TeammatePointsGap/QualifyingGapToTeammate's
   sign convention, GridPenaltyPositions, and the DNF-cause
   classification (DriverFaultDNFRate vs MechanicalDNFRate) - the source
   classifies by status-string keyword, but the exact keyword list
   wasn't confirmed, so this uses a reasonable common-F1-status mapping.
3. Disclosed simplifications (true career-spanning computation would be
   prohibitively expensive per live request): CareerRoundsRaced and
   TeamRoundsWithCurrentTeam use within-*this*-season cumulative counts
   rather than a true multi-decade race-by-race career total (which
   would need walking every driver's full result history back to 1950,
   not just season-level standings). CareerSeasonsRaced and
   TeamSeasonsWithCurrentTeam ARE computed as true career values (see
   _get_career_history) since that only needs one Ergast call per season
   (cheap, same pattern as fastf1_client.get_all_time_driver_records),
   not per round.
"""

import threading

import numpy as np
import pandas as pd

from box_box_bot.data import fastf1_client
from box_box_bot.data.fastf1_client import FIRST_F1_SEASON
from box_box_bot.predictor.features import _completed_rounds_and_total, _normalize_team_ids, get_team_features

TEAM_ID_VOCAB = [
    "alpine", "aston_martin", "ferrari", "haas", "mclaren",
    "mercedes", "rb", "red_bull", "sauber", "williams",
]

NUMERIC_FEATURE_COLUMNS = [
    "Year", "Round", "RoundsCompleted", "RoundsRemaining", "CareerSeasonsRaced", "TeamSeasonsWithCurrentTeam",
    "DriverFaultDNFRate", "MechanicalDNFRate", "TeammatePointsGap", "BeatTeammateThisRound", "BeatTeammateRate",
    "PositionsGainedThisRound", "AvgPositionsGained", "HasQualifyingData", "QualifyingPosition", "AvgQualifyingPosition",
    "QualifyingGapToTeammate", "GridPenaltyPositions", "DNFRate", "AvgGridPosition", "AvgPosition", "AvgPointsPerRace",
    "FormRatio", "Consistency", "ProjectedSeasonTotalPoints", "RelativePointsShare", "PercentileRankAfterRound",
    "DriverPointsShareOfTeam", "TeamAvgPointsPerRace", "TeamFormRatio", "TeamConsistency", "TeamDNFRate",
    "TeamProjectedSeasonTotalPoints", "TeamRelativePointsShare", "TeamPercentileRankAfterRound",
]

# Skewed = log1p'd before the model sees them - excludes signed/gap
# columns on purpose (log1p of a negative is invalid).
SKEWED_FEATURE_COLUMNS = [
    "PointsEarnedThisRound", "DNFsThisRound", "DriverFaultDNFsThisRound", "MechanicalDNFsThisRound",
    "PointsLast3Rounds", "DNFsLast3Rounds", "TotalPointFinishes", "TotalPodiums", "TotalPoints",
    "CareerRoundsRaced", "TeamRoundsWithCurrentTeam", "TeamTotalPoints",
]

TEAM_ID_COLUMNS = [f"TeamId_{t}" for t in TEAM_ID_VOCAB]

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + SKEWED_FEATURE_COLUMNS + TEAM_ID_COLUMNS

# Best-effort DNF-cause classification (see module docstring, category 2).
_DRIVER_FAULT_KEYWORDS = ("accident", "collision", "spun off", "disqualified")


def _is_driver_fault_dnf(status: str) -> bool:
    status = status.lower()
    return any(keyword in status for keyword in _DRIVER_FAULT_KEYWORDS)


def _round_driver_table(season: int, round_number: int) -> pd.DataFrame:
    results = pd.DataFrame(fastf1_client.get_race_results(season, round_number))
    results = _normalize_team_ids(results)

    quali = pd.DataFrame(fastf1_client.get_qualifying_results(season, round_number))
    quali = quali[["DriverId", "Position"]].rename(columns={"Position": "QualifyingPosition"})

    results["isDNF"] = results["ClassifiedPosition"].apply(lambda x: 1 if not str(x).isnumeric() else 0)
    results["isDriverFaultDNF"] = results.apply(
        lambda r: 1 if r["isDNF"] and _is_driver_fault_dnf(str(r["Status"])) else 0, axis=1
    )
    results["isMechanicalDNF"] = results["isDNF"] - results["isDriverFaultDNF"]
    results["isPointsFinish"] = (results["Points"] > 0).astype(int)
    results["isPodiumFinish"] = (results["Points"] >= 15).astype(int)
    # A DNF has no finishing Position, so "positions gained" is undefined
    # rather than negative/wrong - default to 0 rather than leaving NaN,
    # which would otherwise drop the entire row later (dropna below).
    results["PositionsGainedThisRound"] = (results["GridPosition"] - results["Position"]).fillna(0)

    df = results.merge(quali, on="DriverId", how="left")
    df["HasQualifyingData"] = df["QualifyingPosition"].notna().astype(int)
    df["GridPenaltyPositions"] = df["GridPosition"] - df["QualifyingPosition"]
    df["Round"] = round_number
    return df[[
        "DriverId", "FullName", "TeamId", "TeamName", "Position", "GridPosition", "Points", "Round",
        "isDNF", "isDriverFaultDNF", "isMechanicalDNF", "isPointsFinish", "isPodiumFinish",
        "PositionsGainedThisRound", "QualifyingPosition", "HasQualifyingData", "GridPenaltyPositions",
    ]]


def _add_teammate_features(df: pd.DataFrame) -> pd.DataFrame:
    # Two cars per team per round is the real-world assumption the source
    # project itself relies on for these features.
    def _teammate_stats(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        if len(group) != 2:
            group["TeammatePointsGap"] = 0.0
            group["BeatTeammateThisRound"] = 0
            group["QualifyingGapToTeammate"] = 0.0
            return group
        a, b = group.index
        group.loc[a, "TeammatePointsGap"] = group.loc[a, "TotalPoints"] - group.loc[b, "TotalPoints"]
        group.loc[b, "TeammatePointsGap"] = group.loc[b, "TotalPoints"] - group.loc[a, "TotalPoints"]
        group.loc[a, "BeatTeammateThisRound"] = int(group.loc[a, "Position"] < group.loc[b, "Position"])
        group.loc[b, "BeatTeammateThisRound"] = int(group.loc[b, "Position"] < group.loc[a, "Position"])
        group.loc[a, "QualifyingGapToTeammate"] = group.loc[a, "QualifyingPosition"] - group.loc[b, "QualifyingPosition"]
        group.loc[b, "QualifyingGapToTeammate"] = group.loc[b, "QualifyingPosition"] - group.loc[a, "QualifyingPosition"]
        return group

    return df.groupby(["Round", "TeamId"], group_keys=False).apply(_teammate_stats)


def _career_history() -> dict:
    """driverId -> {"seasons": {year, ...}, "team_seasons": {teamId: {year, ...}}}.

    Cheap relative to a per-round walk (one Ergast call per season, same
    pattern as fastf1_client.get_all_time_driver_records), so this is
    computed as true career history rather than approximated.
    """
    history: dict = {}
    for year in range(FIRST_F1_SEASON, pd.Timestamp.now().year + 1):
        standings = fastf1_client.get_driver_standings(year)
        for row in standings:
            entry = history.setdefault(row["driverId"], {"seasons": set(), "team_seasons": {}})
            entry["seasons"].add(year)
            for team_id in row.get("constructorIds") or []:
                entry["team_seasons"].setdefault(team_id, set()).add(year)
    return history


_career_history_cache: dict | None = None
_career_history_lock = threading.Lock()


def _get_career_history() -> dict:
    global _career_history_cache
    if _career_history_cache is None:
        with _career_history_lock:
            if _career_history_cache is None:
                _career_history_cache = _career_history()
    return _career_history_cache


def build_driver_features(season: int) -> pd.DataFrame:
    """Every completed round's per-driver feature row for a season.

    One row per (driver, round), sorted by driver then round - callers
    predicting the current standings should take each driver's row for
    the latest round.
    """
    completed_rounds, total_rounds = _completed_rounds_and_total(season)
    if not completed_rounds:
        raise ValueError(f"No completed rounds found for {season} yet.")

    df = pd.concat([_round_driver_table(season, r) for r in completed_rounds], ignore_index=True)
    df = df.sort_values(["DriverId", "Round"]).reset_index(drop=True)

    g = df.groupby("DriverId")
    df["RoundsCompleted"] = g.cumcount() + 1
    df["DNFRate"] = g["isDNF"].expanding().mean().reset_index(level=0, drop=True)
    df["DriverFaultDNFRate"] = g["isDriverFaultDNF"].expanding().mean().reset_index(level=0, drop=True)
    df["MechanicalDNFRate"] = g["isMechanicalDNF"].expanding().mean().reset_index(level=0, drop=True)
    df["AvgGridPosition"] = g["GridPosition"].expanding().mean().reset_index(level=0, drop=True)
    df["AvgPosition"] = g["Position"].expanding().mean().reset_index(level=0, drop=True)
    df["AvgPointsPerRace"] = g["Points"].expanding().mean().reset_index(level=0, drop=True)
    df["AvgPositionsGained"] = g["PositionsGainedThisRound"].expanding().mean().reset_index(level=0, drop=True)
    df["AvgQualifyingPosition"] = g["QualifyingPosition"].expanding().mean().reset_index(level=0, drop=True)

    df["TotalPointFinishes"] = g["isPointsFinish"].cumsum()
    df["TotalPodiums"] = g["isPodiumFinish"].cumsum()
    df["TotalPoints"] = g["Points"].cumsum()
    df["DriverFaultDNFsThisRound"] = df["isDriverFaultDNF"]
    df["MechanicalDNFsThisRound"] = df["isMechanicalDNF"]
    df["DNFsThisRound"] = df["isDNF"]
    df["PointsEarnedThisRound"] = df["Points"]
    df["PointsLast3Rounds"] = g["Points"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["DNFsLast3Rounds"] = g["isDNF"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["FormRatio"] = df["PointsLast3Rounds"] / (df["AvgPointsPerRace"] * 3 + 1e-6)

    # Needs TotalPoints/Position/QualifyingPosition, so it must run after
    # the per-driver cumulative stats above, not before. The groupby-apply
    # inside doesn't preserve (DriverId, Round) row order, so re-sort
    # before any further expanding/rolling computation relies on it.
    df = _add_teammate_features(df)
    df = df.sort_values(["DriverId", "Round"]).reset_index(drop=True)
    df["BeatTeammateRate"] = df.groupby("DriverId")["BeatTeammateThisRound"].expanding().mean().reset_index(level=0, drop=True)

    rolling_mean_5 = g["Points"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    rolling_std_5 = g["Points"].rolling(5, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    df["Consistency"] = 1 / (1 + rolling_std_5 / (rolling_mean_5 + 1e-6))

    # Best-effort (see module docstring, category 2): RoundsRemaining
    # uses true RoundsCompleted (driver-specific) rather than Round-1,
    # since a driver can miss rounds a team never does.
    df["RoundsRemaining"] = total_rounds - df["RoundsCompleted"]
    df["ProjectedSeasonTotalPoints"] = df["TotalPoints"] + rolling_mean_5 * df["RoundsRemaining"]

    df["Year"] = season

    round_totals = df.groupby("Round")["TotalPoints"].transform("sum")
    df["RelativePointsShare"] = df["TotalPoints"] / round_totals
    rank = df.groupby("Round")["TotalPoints"].rank(method="dense", ascending=False)
    n_drivers = df.groupby("Round")["DriverId"].transform("nunique")
    df["PercentileRankAfterRound"] = 1.0 - (rank - 1) / (n_drivers - 1)

    # Disclosed simplification (category 3): within-season, not true
    # multi-decade career round counts.
    df["CareerRoundsRaced"] = df["RoundsCompleted"]

    history = _get_career_history()

    def _career_seasons(driver_id: str) -> int:
        seasons = history.get(driver_id, {}).get("seasons", set())
        return sum(1 for s in seasons if s < season)

    def _team_seasons_with_current_team(row) -> int:
        team_seasons = history.get(row["DriverId"], {}).get("team_seasons", {}).get(row["TeamId"], set())
        return sum(1 for s in team_seasons if s < season)

    df["CareerSeasonsRaced"] = df["DriverId"].apply(_career_seasons)
    df["TeamSeasonsWithCurrentTeam"] = df.apply(_team_seasons_with_current_team, axis=1)
    # Disclosed simplification (category 3): same reasoning as CareerRoundsRaced.
    df["TeamRoundsWithCurrentTeam"] = df["RoundsCompleted"]

    # Team-context columns reuse the already-verified team-level table
    # (predictor/features.py) via a per-round join, rather than
    # recomputing the same formulas a second time.
    team_features = get_team_features(season)[["TeamId", "Round", "AvgPointsPerRace", "FormRatio", "Consistency", "DNFRate", "ProjectedSeasonTotalPoints", "RelativePointsShare", "PercentileRankAfterRound", "TotalPoints"]]
    team_features = team_features.rename(columns={
        "AvgPointsPerRace": "TeamAvgPointsPerRace",
        "FormRatio": "TeamFormRatio",
        "Consistency": "TeamConsistency",
        "DNFRate": "TeamDNFRate",
        "ProjectedSeasonTotalPoints": "TeamProjectedSeasonTotalPoints",
        "RelativePointsShare": "TeamRelativePointsShare",
        "PercentileRankAfterRound": "TeamPercentileRankAfterRound",
        "TotalPoints": "TeamTotalPoints",
    })
    df = df.merge(team_features, on=["TeamId", "Round"], how="left")
    df["DriverPointsShareOfTeam"] = df["TotalPoints"] / df["TeamTotalPoints"].replace(0, np.nan)
    df["DriverPointsShareOfTeam"] = df["DriverPointsShareOfTeam"].fillna(0)

    df = df.dropna(subset=["TeamId"] + [c for c in NUMERIC_FEATURE_COLUMNS if c not in ("HasQualifyingData",)])

    for col in SKEWED_FEATURE_COLUMNS:
        df[col] = np.log1p(df[col].clip(lower=0))

    for team_id in TEAM_ID_VOCAB:
        df[f"TeamId_{team_id}"] = (df["TeamId"] == team_id).astype(int)

    return df[["DriverId", "FullName"] + FEATURE_COLUMNS]


_feature_cache: dict[int, pd.DataFrame] = {}
_feature_cache_lock = threading.Lock()


def get_driver_features(season: int) -> pd.DataFrame:
    """Cached wrapper around build_driver_features - same lazy,
    thread-safe, per-season cache shape as get_team_features."""
    if season not in _feature_cache:
        with _feature_cache_lock:
            if season not in _feature_cache:
                _feature_cache[season] = build_driver_features(season)
    return _feature_cache[season]
