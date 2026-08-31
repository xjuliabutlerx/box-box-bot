from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from box_box_bot.predictor import features

SEASON = 2099  # fake season, not in the penalties dict

# Fixture: 2 teams, 3 completed rounds (of 5 scheduled), 2 drivers each.
# team_a's driver2 DNFs in round 2. Hand-computed expected values are in
# each assertion below.
_ROUND_RESULTS = {
    1: [
        {"TeamId": "team_a", "TeamName": "Team A", "Points": 25.0, "GridPosition": 1.0, "Position": 1.0, "ClassifiedPosition": "1"},
        {"TeamId": "team_a", "TeamName": "Team A", "Points": 18.0, "GridPosition": 2.0, "Position": 2.0, "ClassifiedPosition": "2"},
        {"TeamId": "team_b", "TeamName": "Team B", "Points": 15.0, "GridPosition": 3.0, "Position": 3.0, "ClassifiedPosition": "3"},
        {"TeamId": "team_b", "TeamName": "Team B", "Points": 12.0, "GridPosition": 4.0, "Position": 4.0, "ClassifiedPosition": "4"},
    ],
    2: [
        {"TeamId": "team_a", "TeamName": "Team A", "Points": 18.0, "GridPosition": 2.0, "Position": 2.0, "ClassifiedPosition": "2"},
        {"TeamId": "team_a", "TeamName": "Team A", "Points": 0.0, "GridPosition": 5.0, "Position": np.nan, "ClassifiedPosition": "R"},
        {"TeamId": "team_b", "TeamName": "Team B", "Points": 10.0, "GridPosition": 3.0, "Position": 3.0, "ClassifiedPosition": "3"},
        {"TeamId": "team_b", "TeamName": "Team B", "Points": 8.0, "GridPosition": 4.0, "Position": 4.0, "ClassifiedPosition": "4"},
    ],
    3: [
        {"TeamId": "team_a", "TeamName": "Team A", "Points": 25.0, "GridPosition": 1.0, "Position": 1.0, "ClassifiedPosition": "1"},
        {"TeamId": "team_a", "TeamName": "Team A", "Points": 15.0, "GridPosition": 3.0, "Position": 3.0, "ClassifiedPosition": "3"},
        {"TeamId": "team_b", "TeamName": "Team B", "Points": 8.0, "GridPosition": 5.0, "Position": 5.0, "ClassifiedPosition": "5"},
        {"TeamId": "team_b", "TeamName": "Team B", "Points": 6.0, "GridPosition": 6.0, "Position": 6.0, "ClassifiedPosition": "6"},
    ],
}

_PAST = pd.Timestamp.now() - pd.Timedelta(days=1)
_FUTURE = pd.Timestamp.now() + pd.Timedelta(days=30)
_SCHEDULE = [
    {"RoundNumber": 1, "EventName": "Round 1", "EventDate": _PAST},
    {"RoundNumber": 2, "EventName": "Round 2", "EventDate": _PAST},
    {"RoundNumber": 3, "EventName": "Round 3", "EventDate": _PAST},
    {"RoundNumber": 4, "EventName": "Round 4", "EventDate": _FUTURE},
    {"RoundNumber": 5, "EventName": "Round 5", "EventDate": _FUTURE},
]


@pytest.fixture
def team_features():
    with (
        patch("box_box_bot.predictor.features.fastf1_client.get_season_schedule", return_value=_SCHEDULE),
        patch(
            "box_box_bot.predictor.features.fastf1_client.get_race_results",
            side_effect=lambda season, round: _ROUND_RESULTS[round],
        ),
    ):
        return features.build_team_features(SEASON)


def _row(df, team_id, round_number):
    match = df[(df["TeamId"] == team_id) & (df["Round"] == round_number)]
    assert len(match) == 1
    return match.iloc[0]


def test_only_completed_rounds_are_used(team_features):
    # 5 rounds scheduled, only 3 have passed - rounds 4/5 must never be
    # fetched (side_effect would KeyError if they were).
    assert set(team_features["Round"]) == {1, 2, 3}


def test_avg_grid_position_expands_across_rounds(team_features):
    # team_a round-avg grid positions: R1=1.5, R2=3.5, R3=2.0
    # expanding mean: R1=1.5, R2=2.5, R3=7/3
    row = _row(team_features, "team_a", 3)
    assert row["AvgGridPosition"] == pytest.approx(7 / 3)


def test_dnf_rate_expands_across_rounds(team_features):
    # team_a: R1 DNF rate 0, R2 DNF rate 0.5 (one driver retired), R3 rate 0
    # expanding mean at R3: (0 + 0.5 + 0) / 3
    row = _row(team_features, "team_a", 3)
    assert row["DNFRate"] == pytest.approx(0.5 / 3)


def test_rounds_remaining_uses_rounds_completed_off_by_one(team_features):
    # RoundsCompleted = Round - 1 (not Round), and RoundsRemaining = total
    # (5) - RoundsCompleted - a quirk preserved on purpose because the
    # model was trained on this exact off-by-one.
    row = _row(team_features, "team_a", 1)
    assert row["RoundsRemaining"] == 5  # 5 - (1 - 1)
    row = _row(team_features, "team_a", 3)
    assert row["RoundsRemaining"] == 3  # 5 - (3 - 1)


def test_form_ratio_reproduces_known_unit_mismatch_quirk(team_features):
    # FormRatio = team's 3-round point total / (per-driver average x3) -
    # a deliberate quirk (team total vs per-driver average), not a bug to
    # fix. At round 1: PointsLast3Rounds=43 (raw, pre-log1p... but the
    # ratio itself is computed on raw values before the log1p step, so we
    # recompute the expected ratio from the known raw inputs).
    row = _row(team_features, "team_a", 1)
    assert row["FormRatio"] == pytest.approx(43 / (21.5 * 3 + 1e-6))


def test_consistency_uses_inverse_coefficient_of_variation(team_features):
    row = _row(team_features, "team_a", 1)
    assert row["Consistency"] == pytest.approx(1.0)  # single data point -> std 0


def test_relative_points_share_and_percentile_rank_at_latest_round(team_features):
    # At round 3: team_a TotalPoints (raw) = 43+18+40=101, team_b = 27+18+14=59
    row_a = _row(team_features, "team_a", 3)
    row_b = _row(team_features, "team_b", 3)
    assert row_a["RelativePointsShare"] == pytest.approx(101 / 160)
    assert row_a["PercentileRankAfterRound"] == pytest.approx(1.0)  # rank 1 of 2
    assert row_b["PercentileRankAfterRound"] == pytest.approx(0.0)  # rank 2 of 2


def test_skewed_columns_are_log1p_transformed(team_features):
    # team_a TotalPoints at round 1 is raw 43 before log1p.
    row = _row(team_features, "team_a", 1)
    assert row["TotalPoints"] == pytest.approx(np.log1p(43))


def test_normalize_team_ids_applies_historical_rebrands():
    df = pd.DataFrame(
        {
            "TeamId": ["alfa", "renault", "toro_rosso", "alphatauri", "force_india", "racing_point"],
            "TeamName": ["Alfa Romeo Racing", "Renault", "Toro Rosso", "AlphaTauri", "Force India", "Racing Point"],
        }
    )
    result = features._normalize_team_ids(df)
    assert list(result["TeamId"]) == ["sauber", "alpine", "rb", "rb", "aston_martin", "aston_martin"]
    assert result.loc[0, "TeamName"] == "Alfa Romeo"


def test_penalty_flag_only_set_for_penalized_team_and_year():
    with (
        patch("box_box_bot.predictor.features.fastf1_client.get_season_schedule", return_value=_SCHEDULE),
        patch(
            "box_box_bot.predictor.features.fastf1_client.get_race_results",
            side_effect=lambda season, round: [
                {"TeamId": "racing_point", "TeamName": "Racing Point", "Points": 10.0, "GridPosition": 1.0, "Position": 1.0, "ClassifiedPosition": "1"},
                {"TeamId": "racing_point", "TeamName": "Racing Point", "Points": 8.0, "GridPosition": 2.0, "Position": 2.0, "ClassifiedPosition": "2"},
                {"TeamId": "team_b", "TeamName": "Team B", "Points": 6.0, "GridPosition": 3.0, "Position": 3.0, "ClassifiedPosition": "3"},
                {"TeamId": "team_b", "TeamName": "Team B", "Points": 4.0, "GridPosition": 4.0, "Position": 4.0, "ClassifiedPosition": "4"},
            ],
        ),
    ):
        # racing_point normalizes to aston_martin, which has a 2020 penalty.
        df = features.build_team_features(2020)

    aston = df[df["TeamId"] == "aston_martin"].iloc[0]
    other = df[df["TeamId"] == "team_b"].iloc[0]
    assert aston["hadPenaltyThisYear"] == 1
    assert other["hadPenaltyThisYear"] == 0


def test_get_team_features_caches_per_season():
    call_count = 0

    def fake_get_race_results(season, round):
        nonlocal call_count
        call_count += 1
        return _ROUND_RESULTS[round]

    with (
        patch("box_box_bot.predictor.features.fastf1_client.get_season_schedule", return_value=_SCHEDULE),
        patch("box_box_bot.predictor.features.fastf1_client.get_race_results", side_effect=fake_get_race_results),
    ):
        features._feature_cache.pop(SEASON, None)
        features.get_team_features(SEASON)
        first_call_count = call_count
        features.get_team_features(SEASON)

    assert call_count == first_call_count  # second call served from cache, no new fetches
