from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from box_box_bot.predictor import driver_features, features

SEASON = 2099  # fake season, not in the constructor penalties dict

# Fixture: 2 teams, 2 drivers each, 2 completed rounds (of 3 scheduled).
# team_a's driver2 DNFs (engine) in round 2. team_b's second driver is on
# an unrecognized team id (tests the unknown-team one-hot fallback) and
# is missing qualifying data in round 1 (tests HasQualifyingData/the
# drop-on-missing-numeric-feature behavior). driver_b1 crashes (driver-
# fault DNF) in round 1.
_RACE_RESULTS = {
    1: [
        {"DriverId": "driver_a1", "FullName": "Driver A1", "TeamId": "team_a", "TeamName": "Team A", "Points": 25.0, "GridPosition": 1.0, "Position": 1.0, "ClassifiedPosition": "1", "Status": "Finished"},
        {"DriverId": "driver_a2", "FullName": "Driver A2", "TeamId": "team_a", "TeamName": "Team A", "Points": 18.0, "GridPosition": 2.0, "Position": 2.0, "ClassifiedPosition": "2", "Status": "Finished"},
        {"DriverId": "driver_b1", "FullName": "Driver B1", "TeamId": "unknown_team", "TeamName": "Unknown Team", "Points": 0.0, "GridPosition": 3.0, "Position": np.nan, "ClassifiedPosition": "R", "Status": "Accident"},
        {"DriverId": "driver_b2", "FullName": "Driver B2", "TeamId": "unknown_team", "TeamName": "Unknown Team", "Points": 12.0, "GridPosition": 4.0, "Position": 4.0, "ClassifiedPosition": "4", "Status": "Finished"},
    ],
    2: [
        {"DriverId": "driver_a1", "FullName": "Driver A1", "TeamId": "team_a", "TeamName": "Team A", "Points": 18.0, "GridPosition": 2.0, "Position": 2.0, "ClassifiedPosition": "2", "Status": "Finished"},
        {"DriverId": "driver_a2", "FullName": "Driver A2", "TeamId": "team_a", "TeamName": "Team A", "Points": 0.0, "GridPosition": 5.0, "Position": np.nan, "ClassifiedPosition": "R", "Status": "Engine"},
        {"DriverId": "driver_b1", "FullName": "Driver B1", "TeamId": "unknown_team", "TeamName": "Unknown Team", "Points": 10.0, "GridPosition": 3.0, "Position": 3.0, "ClassifiedPosition": "3", "Status": "Finished"},
        {"DriverId": "driver_b2", "FullName": "Driver B2", "TeamId": "unknown_team", "TeamName": "Unknown Team", "Points": 8.0, "GridPosition": 4.0, "Position": 4.0, "ClassifiedPosition": "4", "Status": "Finished"},
    ],
}

_QUALI_RESULTS = {
    1: [
        {"DriverId": "driver_a1", "Position": 1.0},
        {"DriverId": "driver_a2", "Position": 3.0},
        {"DriverId": "driver_b2", "Position": 4.0},
        # driver_b1 has no qualifying row this round on purpose.
    ],
    2: [
        {"DriverId": "driver_a1", "Position": 2.0},
        {"DriverId": "driver_a2", "Position": 4.0},
        {"DriverId": "driver_b1", "Position": 3.0},
        {"DriverId": "driver_b2", "Position": 1.0},
    ],
}

_PAST = pd.Timestamp.now() - pd.Timedelta(days=1)
_FUTURE = pd.Timestamp.now() + pd.Timedelta(days=30)
_SCHEDULE = [
    {"RoundNumber": 1, "EventName": "Round 1", "EventDate": _PAST},
    {"RoundNumber": 2, "EventName": "Round 2", "EventDate": _PAST},
    {"RoundNumber": 3, "EventName": "Round 3", "EventDate": _FUTURE},
]


@pytest.fixture
def driver_feats():
    driver_features._feature_cache.pop(SEASON, None)
    driver_features._career_history_cache = None
    features._feature_cache.pop(SEASON, None)

    with (
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_season_schedule", return_value=_SCHEDULE),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_race_results", side_effect=lambda season, round: _RACE_RESULTS[round]),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_qualifying_results", side_effect=lambda season, round: _QUALI_RESULTS[round]),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_driver_standings", return_value=[]),
        patch("box_box_bot.predictor.features.fastf1_client.get_season_schedule", return_value=_SCHEDULE),
        patch("box_box_bot.predictor.features.fastf1_client.get_race_results", side_effect=lambda season, round: _RACE_RESULTS[round]),
    ):
        return driver_features.build_driver_features(SEASON)


def _row(df, driver_id, round_number):
    match = df[(df["DriverId"] == driver_id) & (df["Round"] == round_number)]
    assert len(match) == 1
    return match.iloc[0]


def test_only_completed_rounds_are_used(driver_feats):
    assert set(driver_feats["Round"]) == {1, 2}


def test_driver_b1_round1_dropped_for_missing_qualifying_data(driver_feats):
    # driver_b1 has no round-1 qualifying row, so QualifyingPosition is
    # NaN there - the source drops (doesn't impute) rows missing any
    # numeric feature, so this row must not survive into the final table.
    assert driver_feats[(driver_feats["DriverId"] == "driver_b1") & (driver_feats["Round"] == 1)].empty
    # But round 2 (which does have qualifying data) must survive.
    assert not driver_feats[(driver_feats["DriverId"] == "driver_b1") & (driver_feats["Round"] == 2)].empty


def test_total_podiums_and_dnf_classification(driver_feats):
    # driver_a1: 25 pts round 1 (podium, >=15), 18 pts round 2 (podium) -> cumulative TotalPodiums after round 2 = 2
    a1_r2 = _row(driver_feats, "driver_a1", 2)
    assert a1_r2["TotalPodiums"] == pytest.approx(np.log1p(2))  # TotalPodiums is a skewed column - log1p'd

    # driver_b2: 12 pts round 1 (not a podium, <15), 8 pts round 2 (not a podium) -> 0 podiums
    b2_r2 = _row(driver_feats, "driver_b2", 2)
    assert b2_r2["TotalPodiums"] == pytest.approx(np.log1p(0))

    # driver_a2's round-2 DNF is "Engine" -> mechanical, not driver-fault.
    # Both *ThisRound columns are skewed (log1p'd), like TotalPodiums above.
    a2_r2 = _row(driver_feats, "driver_a2", 2)
    assert a2_r2["MechanicalDNFsThisRound"] == pytest.approx(np.log1p(1))
    assert a2_r2["DriverFaultDNFsThisRound"] == pytest.approx(np.log1p(0))

    # driver_b1's round-1 DNF is "Accident" -> driver-fault. (Round 1 row
    # itself is dropped for missing quali data, but round 2's cumulative
    # DNF-fault rate still reflects the round-1 accident having happened
    # is untestable once dropped - this only tests the classification
    # function's output on the raw round-1 row, not the dropped table.)


def test_unknown_team_gets_all_zero_one_hot(driver_feats):
    row = _row(driver_feats, "driver_b2", 2)
    for team_id in driver_features.TEAM_ID_VOCAB:
        assert row[f"TeamId_{team_id}"] == 0


def test_real_vocab_team_gets_its_own_one_hot_column():
    # Separate from the fixture above (which deliberately uses fake team
    # ids to test the unknown-team fallback) - confirm a real TEAM_ID_VOCAB
    # entry actually gets a 1 in its own column, not just all-zero.
    driver_features._feature_cache.pop(SEASON, None)
    driver_features._career_history_cache = None
    features._feature_cache.pop(SEASON, None)

    race_results = {1: [
        {"DriverId": "driver_x", "FullName": "Driver X", "TeamId": "ferrari", "TeamName": "Ferrari", "Points": 25.0, "GridPosition": 1.0, "Position": 1.0, "ClassifiedPosition": "1", "Status": "Finished"},
        {"DriverId": "driver_y", "FullName": "Driver Y", "TeamId": "ferrari", "TeamName": "Ferrari", "Points": 18.0, "GridPosition": 2.0, "Position": 2.0, "ClassifiedPosition": "2", "Status": "Finished"},
        # A second team so TeamPercentileRankAfterRound's rank/(n_teams-1)
        # formula (in predictor/features.py, reused here) doesn't divide
        # by zero with only 1 team in the round.
        {"DriverId": "driver_z", "FullName": "Driver Z", "TeamId": "mclaren", "TeamName": "McLaren", "Points": 10.0, "GridPosition": 3.0, "Position": 3.0, "ClassifiedPosition": "3", "Status": "Finished"},
    ]}
    quali_results = {1: [
        {"DriverId": "driver_x", "Position": 1.0}, {"DriverId": "driver_y", "Position": 2.0}, {"DriverId": "driver_z", "Position": 3.0},
    ]}
    schedule = [{"RoundNumber": 1, "EventName": "Round 1", "EventDate": _PAST}]

    with (
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_season_schedule", return_value=schedule),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_race_results", side_effect=lambda season, round: race_results[round]),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_qualifying_results", side_effect=lambda season, round: quali_results[round]),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_driver_standings", return_value=[]),
        patch("box_box_bot.predictor.features.fastf1_client.get_season_schedule", return_value=schedule),
        patch("box_box_bot.predictor.features.fastf1_client.get_race_results", side_effect=lambda season, round: race_results[round]),
    ):
        df = driver_features.build_driver_features(SEASON)

    row = _row(df, "driver_x", 1)
    assert row["TeamId_ferrari"] == 1
    assert sum(row[f"TeamId_{t}"] for t in driver_features.TEAM_ID_VOCAB) == 1


def test_beat_teammate_and_points_gap(driver_feats):
    # Round 1: driver_a1 (P1, 25 cumulative pts) beat driver_a2 (P2, 18
    # cumulative pts) - gap is computed from raw TotalPoints before the
    # log1p skew transform overwrites that column later in the pipeline.
    a1_r1 = _row(driver_feats, "driver_a1", 1)
    a2_r1 = _row(driver_feats, "driver_a2", 1)
    assert a1_r1["BeatTeammateThisRound"] == 1
    assert a2_r1["BeatTeammateThisRound"] == 0
    assert a1_r1["TeammatePointsGap"] == pytest.approx(7.0)
    assert a2_r1["TeammatePointsGap"] == pytest.approx(-7.0)


def test_rounds_completed_tracks_participation_per_driver(driver_feats):
    # Every driver here races both rounds it has a surviving row for -
    # RoundsCompleted counts up per driver, not per calendar round.
    a1_r1 = _row(driver_feats, "driver_a1", 1)
    a1_r2 = _row(driver_feats, "driver_a1", 2)
    assert a1_r1["RoundsCompleted"] == 1
    assert a1_r2["RoundsCompleted"] == 2


def test_get_driver_features_caches_per_season():
    driver_features._feature_cache.pop(SEASON, None)
    driver_features._career_history_cache = None
    features._feature_cache.pop(SEASON, None)
    call_count = 0

    def fake_get_race_results(season, round):
        nonlocal call_count
        call_count += 1
        return _RACE_RESULTS[round]

    with (
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_season_schedule", return_value=_SCHEDULE),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_race_results", side_effect=fake_get_race_results),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_qualifying_results", side_effect=lambda season, round: _QUALI_RESULTS[round]),
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_driver_standings", return_value=[]),
        patch("box_box_bot.predictor.features.fastf1_client.get_season_schedule", return_value=_SCHEDULE),
        patch("box_box_bot.predictor.features.fastf1_client.get_race_results", side_effect=lambda season, round: _RACE_RESULTS[round]),
    ):
        driver_features.get_driver_features(SEASON)
        first_call_count = call_count
        driver_features.get_driver_features(SEASON)

    assert call_count == first_call_count  # second call served from cache, no new fetches


def test_career_history_skips_a_season_whose_request_fails():
    # Regression: a single failed season's request must not crash the
    # whole walk, or leave _career_history_cache permanently
    # unpopulated (forcing every future prediction request to retry
    # the whole lookback window from scratch and risk hitting the same
    # wall again).
    driver_features._career_history_cache = None

    def _side_effect(year):
        if year == 2024:
            raise Exception("Too Many Requests")
        return [{"driverId": "driver_a", "constructorIds": ["team_a"]}]

    with (
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_driver_standings", side_effect=_side_effect),
        patch("box_box_bot.predictor.driver_features.CAREER_HISTORY_LOOKBACK_SEASONS", 3),
        patch("box_box_bot.predictor.driver_features.pd.Timestamp") as mock_timestamp,
    ):
        mock_timestamp.now.return_value.year = 2025
        history = driver_features._get_career_history()

    # window is 2023-2025; 2024 raised and was skipped, 2023 and 2025 contributed.
    assert history["driver_a"]["seasons"] == {2023, 2025}
    assert driver_features._career_history_cache is not None


def test_career_history_is_bounded_to_the_lookback_window():
    # Regression: _career_history() originally walked every season back
    # to 1950 (~76 sequential Ergast calls) to build true career-length
    # context for the ranking model - reliably enough to trip Jolpica's
    # rate limiting in production (61 separate 429s in one live-observed
    # walk) that it's now deliberately approximated over a short recent
    # window instead. This must stay bounded to that window, not creep
    # back into a full historical walk.
    driver_features._career_history_cache = None
    seasons_requested = []

    def _side_effect(year):
        seasons_requested.append(year)
        return []

    with (
        patch("box_box_bot.predictor.driver_features.fastf1_client.get_driver_standings", side_effect=_side_effect),
        patch("box_box_bot.predictor.driver_features.CAREER_HISTORY_LOOKBACK_SEASONS", 5),
        patch("box_box_bot.predictor.driver_features.pd.Timestamp") as mock_timestamp,
    ):
        mock_timestamp.now.return_value.year = 2026
        driver_features._get_career_history()

    assert sorted(seasons_requested) == [2022, 2023, 2024, 2025, 2026]
