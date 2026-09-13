from unittest.mock import MagicMock, patch

import fastf1
import pandas as pd
import pytest

from box_box_bot.data import fastf1_client


def test_get_driver_standings_calls_ergast_with_season_and_round():
    fake_df = pd.DataFrame([{"position": 1, "driverCode": "NOR", "points": 423.0}])
    fake_response = MagicMock(content=[fake_df])

    with patch("box_box_bot.data.fastf1_client.Ergast") as mock_ergast_cls, \
         patch("box_box_bot.data.fastf1_client.fastf1"):
        mock_ergast_cls.return_value.get_driver_standings.return_value = fake_response
        result = fastf1_client.get_driver_standings(2025, round=4)

    mock_ergast_cls.return_value.get_driver_standings.assert_called_once_with(season=2025, round=4)
    assert result == [{"position": 1, "driverCode": "NOR", "points": 423.0}]


def test_get_driver_standings_round_defaults_to_none():
    fake_response = MagicMock(content=[pd.DataFrame([{"position": 1}])])
    with patch("box_box_bot.data.fastf1_client.Ergast") as mock_ergast_cls, \
         patch("box_box_bot.data.fastf1_client.fastf1"):
        mock_ergast_cls.return_value.get_driver_standings.return_value = fake_response
        fastf1_client.get_driver_standings(2025)

    mock_ergast_cls.return_value.get_driver_standings.assert_called_once_with(season=2025, round=None)


def test_get_constructor_standings_calls_ergast_with_season_and_round():
    fake_df = pd.DataFrame([{"position": 1, "constructorName": "McLaren", "points": 640.0}])
    fake_response = MagicMock(content=[fake_df])

    with patch("box_box_bot.data.fastf1_client.Ergast") as mock_ergast_cls, \
         patch("box_box_bot.data.fastf1_client.fastf1"):
        mock_ergast_cls.return_value.get_constructor_standings.return_value = fake_response
        result = fastf1_client.get_constructor_standings(2025, round=16)

    mock_ergast_cls.return_value.get_constructor_standings.assert_called_once_with(season=2025, round=16)
    assert result == [{"position": 1, "constructorName": "McLaren", "points": 640.0}]


_ALL_TIME_SEASON_STANDINGS = {
    2023: [
        {"position": 1, "wins": 10, "driverId": "driver_a", "givenName": "Driver", "familyName": "A"},
        {"position": 2, "wins": 5, "driverId": "driver_b", "givenName": "Driver", "familyName": "B"},
    ],
    2024: [
        {"position": 1, "wins": 8, "driverId": "driver_a", "givenName": "Driver", "familyName": "A"},
        {"position": 2, "wins": 3, "driverId": "driver_b", "givenName": "Driver", "familyName": "B"},
        {"position": 3, "wins": 1, "driverId": "driver_c", "givenName": "Driver", "familyName": "C"},
    ],
    2025: [
        {"position": 1, "wins": 12, "driverId": "driver_b", "givenName": "Driver", "familyName": "B"},
        {"position": 2, "wins": 2, "driverId": "driver_a", "givenName": "Driver", "familyName": "A"},
    ],
}


def _fake_all_time_standings_response(season, round=None):
    return MagicMock(content=[pd.DataFrame(_ALL_TIME_SEASON_STANDINGS[season])])


def test_get_all_time_driver_records_sums_wins_and_counts_championships():
    fastf1_client._all_time_records_cache = None
    with (
        patch("box_box_bot.data.fastf1_client.datetime") as mock_datetime,
        patch("box_box_bot.data.fastf1_client.FIRST_F1_SEASON", 2023),
        patch("box_box_bot.data.fastf1_client.Ergast") as mock_ergast_cls,
        patch("box_box_bot.data.fastf1_client.fastf1"),
    ):
        mock_datetime.date.today.return_value.year = 2025
        mock_ergast_cls.return_value.get_driver_standings.side_effect = _fake_all_time_standings_response
        result = fastf1_client.get_all_time_driver_records()

    by_id = {row["driverId"]: row for row in result}
    assert by_id["driver_a"]["championships"] == 2
    assert by_id["driver_a"]["totalWins"] == 20
    assert by_id["driver_b"]["championships"] == 1
    assert by_id["driver_b"]["totalWins"] == 20
    assert by_id["driver_c"]["championships"] == 0
    assert by_id["driver_c"]["totalWins"] == 1
    # driver_a ranks first: same total wins as driver_b, but more championships
    assert [row["driverId"] for row in result] == ["driver_a", "driver_b", "driver_c"]


def test_get_all_time_driver_records_respects_top_n():
    fastf1_client._all_time_records_cache = None
    with (
        patch("box_box_bot.data.fastf1_client.datetime") as mock_datetime,
        patch("box_box_bot.data.fastf1_client.FIRST_F1_SEASON", 2023),
        patch("box_box_bot.data.fastf1_client.Ergast") as mock_ergast_cls,
        patch("box_box_bot.data.fastf1_client.fastf1"),
    ):
        mock_datetime.date.today.return_value.year = 2025
        mock_ergast_cls.return_value.get_driver_standings.side_effect = _fake_all_time_standings_response
        result = fastf1_client.get_all_time_driver_records(top_n=1)

    assert len(result) == 1
    assert result[0]["driverId"] == "driver_a"


def test_get_all_time_driver_records_caches_across_calls():
    fastf1_client._all_time_records_cache = None
    with (
        patch("box_box_bot.data.fastf1_client.datetime") as mock_datetime,
        patch("box_box_bot.data.fastf1_client.FIRST_F1_SEASON", 2023),
        patch("box_box_bot.data.fastf1_client.Ergast") as mock_ergast_cls,
        patch("box_box_bot.data.fastf1_client.fastf1"),
    ):
        mock_datetime.date.today.return_value.year = 2025
        mock_ergast_cls.return_value.get_driver_standings.side_effect = _fake_all_time_standings_response

        fastf1_client.get_all_time_driver_records()
        call_count_after_first = mock_ergast_cls.return_value.get_driver_standings.call_count
        fastf1_client.get_all_time_driver_records()

    assert mock_ergast_cls.return_value.get_driver_standings.call_count == call_count_after_first


def test_normalize_round_converts_numeric_string_to_int():
    assert fastf1_client._normalize_round("7") == 7
    assert fastf1_client._normalize_round(" 7 ") == 7


def test_normalize_round_leaves_int_and_race_name_untouched():
    assert fastf1_client._normalize_round(7) == 7
    assert fastf1_client._normalize_round("Bahrain") == "Bahrain"


def test_get_race_results_loads_session_without_laps_or_telemetry():
    fake_session = MagicMock()
    fake_session.results = pd.DataFrame([{"Position": 1.0, "Abbreviation": "PIA"}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_race_results(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    fake_session.load.assert_called_once_with(laps=False, telemetry=False, weather=False, messages=False)
    assert result == [{"Position": 1.0, "Abbreviation": "PIA"}]


def test_get_qualifying_results_loads_q_session():
    fake_session = MagicMock()
    fake_session.results = pd.DataFrame([{"Position": 1.0, "Abbreviation": "PIA", "Q1": pd.Timedelta(seconds=90)}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_qualifying_results(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "Q")
    fake_session.load.assert_called_once_with(laps=False, telemetry=False, weather=False, messages=False)
    assert result[0]["Position"] == 1.0


def test_get_qualifying_results_normalizes_numeric_string_round_to_int():
    fake_session = MagicMock()
    fake_session.results = pd.DataFrame([{"Position": 1.0, "Abbreviation": "PIA"}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_qualifying_results(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "Q")


def test_get_qualifying_results_accepts_race_name_instead_of_round():
    fake_session = MagicMock()
    fake_session.results = pd.DataFrame([{"Position": 1.0, "Abbreviation": "PIA"}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_qualifying_results(2025, "Bahrain")

    mock_fastf1.get_session.assert_called_once_with(2025, "Bahrain", "Q")


def test_get_season_schedule_returns_calendar_columns():
    fake_schedule = pd.DataFrame(
        [
            {
                "RoundNumber": 4,
                "Country": "Bahrain",
                "Location": "Sakhir",
                "EventName": "Bahrain Grand Prix",
                "EventFormat": "conventional",
                "EventDate": pd.Timestamp("2025-04-13"),
                "OfficialEventName": "FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2025",
            }
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_event_schedule.return_value = fake_schedule
        result = fastf1_client.get_season_schedule(2025)

    mock_fastf1.get_event_schedule.assert_called_once_with(2025, include_testing=False)
    assert result == [
        {
            "RoundNumber": 4,
            "Country": "Bahrain",
            "Location": "Sakhir",
            "EventName": "Bahrain Grand Prix",
            "EventFormat": "conventional",
            "EventDate": pd.Timestamp("2025-04-13"),
        }
    ]


def test_get_race_results_normalizes_numeric_string_round_to_int():
    # Regression test: fastf1.get_session only treats `round` as a round
    # number when it's an int - a string round is always fuzzy-matched
    # against event country/location/name instead, and a bare digit
    # string like "7" doesn't resemble any of those. Rather than raising,
    # fastf1 silently falls back to the wrong race - round="7" and
    # round="1" both resolved to the season's first race in production,
    # after the model passed round as a JSON string. Numeric strings must
    # be converted to int before reaching fastf1.get_session.
    fake_session = MagicMock()
    fake_session.results = pd.DataFrame([{"Position": 1.0, "Abbreviation": "HAM"}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_race_results(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "R")


def test_get_race_results_accepts_race_name_instead_of_round():
    # Regression test: round used to be int-only, forcing the model to
    # recall/guess a round number for a named race - a real hallucination
    # that once pulled the wrong race entirely. fastf1.get_session already
    # fuzzy-matches a string round against event country/location/name, so
    # this should just pass the name straight through untouched.
    fake_session = MagicMock()
    fake_session.results = pd.DataFrame([{"Position": 1.0, "Abbreviation": "VER"}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_race_results(2025, "Bahrain")

    mock_fastf1.get_session.assert_called_once_with(2025, "Bahrain", "R")


def test_get_fastest_laps_normalizes_numeric_string_round_to_int():
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [{"Driver": "VER", "Team": "Red Bull", "LapTime": pd.Timedelta(seconds=90), "LapNumber": 1, "Compound": "SOFT"}]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_fastest_laps(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "R")


def test_get_fastest_laps_accepts_race_name_instead_of_round():
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [{"Driver": "VER", "Team": "Red Bull", "LapTime": pd.Timedelta(seconds=90), "LapNumber": 1, "Compound": "SOFT"}]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_fastest_laps(2025, "Bahrain")

    mock_fastf1.get_session.assert_called_once_with(2025, "Bahrain", "R")


def test_get_fastest_laps_picks_each_drivers_single_fastest_lap():
    # Two drivers, each with multiple laps (including a NaN LapTime that
    # should be dropped) - only the fastest lap per driver should survive,
    # sorted quickest first.
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [
            {"Driver": "VER", "Team": "Red Bull", "LapTime": pd.Timedelta(seconds=90), "LapNumber": 1, "Compound": "SOFT"},
            {"Driver": "VER", "Team": "Red Bull", "LapTime": pd.Timedelta(seconds=88), "LapNumber": 2, "Compound": "SOFT"},
            {"Driver": "VER", "Team": "Red Bull", "LapTime": pd.NaT, "LapNumber": 3, "Compound": "SOFT"},
            {"Driver": "NOR", "Team": "McLaren", "LapTime": pd.Timedelta(seconds=89), "LapNumber": 1, "Compound": "MEDIUM"},
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_fastest_laps(2025, 4, session_type="R", top_n=5)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    assert [row["Driver"] for row in result] == ["VER", "NOR"]
    assert result[0]["LapTime"] == str(pd.Timedelta(seconds=88))


def test_get_fastest_laps_respects_top_n():
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [
            {"Driver": d, "Team": "Team", "LapTime": pd.Timedelta(seconds=90 - i), "LapNumber": 1, "Compound": "SOFT"}
            for i, d in enumerate(["A", "B", "C", "D"])
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_fastest_laps(2025, 4, top_n=2)

    assert len(result) == 2
    # fastest (largest i -> smallest LapTime) should come first
    assert result[0]["Driver"] == "D"
    assert result[1]["Driver"] == "C"


def test_get_tire_strategy_groups_laps_into_stints():
    # VER: 3 laps on SOFT (stint 1), then 2 laps on MEDIUM (stint 2).
    # NOR: 4 laps on HARD (stint 1) only.
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [
            {"Driver": "VER", "Stint": 1, "Compound": "SOFT", "LapNumber": 1},
            {"Driver": "VER", "Stint": 1, "Compound": "SOFT", "LapNumber": 2},
            {"Driver": "VER", "Stint": 1, "Compound": "SOFT", "LapNumber": 3},
            {"Driver": "VER", "Stint": 2, "Compound": "MEDIUM", "LapNumber": 4},
            {"Driver": "VER", "Stint": 2, "Compound": "MEDIUM", "LapNumber": 5},
            {"Driver": "NOR", "Stint": 1, "Compound": "HARD", "LapNumber": 1},
            {"Driver": "NOR", "Stint": 1, "Compound": "HARD", "LapNumber": 2},
            {"Driver": "NOR", "Stint": 1, "Compound": "HARD", "LapNumber": 3},
            {"Driver": "NOR", "Stint": 1, "Compound": "HARD", "LapNumber": 4},
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_tire_strategy(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    fake_session.load.assert_called_once_with(telemetry=False, weather=False, messages=False)

    by_driver_stint = {(row["Driver"], row["Stint"]): row for row in result}
    assert by_driver_stint[("VER", 1)]["Compound"] == "SOFT"
    assert by_driver_stint[("VER", 1)]["StintLength"] == 3
    assert by_driver_stint[("VER", 2)]["Compound"] == "MEDIUM"
    assert by_driver_stint[("VER", 2)]["StintLength"] == 2
    assert by_driver_stint[("NOR", 1)]["Compound"] == "HARD"
    assert by_driver_stint[("NOR", 1)]["StintLength"] == 4


def test_get_tire_strategy_normalizes_numeric_string_round_to_int():
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "LapNumber": 1}]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_tire_strategy(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "R")


def test_get_tire_strategy_accepts_race_name_and_session_type():
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "LapNumber": 1}]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_tire_strategy(2025, "Bahrain", session_type="Q")

    mock_fastf1.get_session.assert_called_once_with(2025, "Bahrain", "Q")


def _fake_race_control_session():
    fake_session = MagicMock()
    fake_session.race_control_messages = pd.DataFrame(
        [
            {"Category": "Flag", "Message": "YELLOW FLAG"},
            {"Category": "SafetyCar", "Message": "SAFETY CAR DEPLOYED"},
            {"Category": "Other", "Message": "CAR 44 TIME PENALTY"},
        ]
    )
    return fake_session


def test_get_race_control_messages_normalizes_numeric_string_round_to_int():
    fake_session = _fake_race_control_session()
    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_race_control_messages(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "R")


def test_get_race_control_messages_defaults_to_all_categories():
    fake_session = _fake_race_control_session()
    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_race_control_messages(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    fake_session.load.assert_called_once_with(laps=False, telemetry=False, weather=False)
    assert len(result) == 3


def test_get_race_control_messages_filters_by_flag_category():
    fake_session = _fake_race_control_session()
    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_race_control_messages(2025, 4, category="Flag")

    assert [row["Category"] for row in result] == ["Flag"]


def test_get_race_control_messages_filters_by_safety_car_category():
    # Regression test: this branch used to filter on "Flag" instead of
    # "SafetyCar" (copy-paste bug), so a SafetyCar-only query silently
    # returned flag messages instead.
    fake_session = _fake_race_control_session()
    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_race_control_messages(2025, 4, category="SafetyCar")

    assert [row["Category"] for row in result] == ["SafetyCar"]


def test_get_race_control_messages_filters_by_other_category():
    fake_session = _fake_race_control_session()
    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_race_control_messages(2025, 4, category="Other")

    assert [row["Category"] for row in result] == ["Other"]


def test_get_race_control_messages_category_is_case_insensitive():
    # Regression test: the filter used to compare "Other".casefold() (a
    # constant) against the raw category argument instead of the other
    # way around, so a capitalized category like "Flag" (as documented
    # and as the tool would actually pass it) never matched anything.
    fake_session = _fake_race_control_session()
    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_race_control_messages(2025, 4, category="flag")

    assert [row["Category"] for row in result] == ["Flag"]


def test_get_weather_for_session_returns_weather_data():
    fake_session = MagicMock()
    fake_session.weather_data = pd.DataFrame(
        [
            {"Time": pd.Timedelta(minutes=1), "AirTemp": 24.5, "TrackTemp": 32.1, "Rainfall": False},
            {"Time": pd.Timedelta(minutes=2), "AirTemp": 24.6, "TrackTemp": 32.3, "Rainfall": False},
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_weather_for_session(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    fake_session.load.assert_called_once_with(laps=False, telemetry=False, messages=False)
    assert len(result) == 2
    assert result[0]["AirTemp"] == 24.5


def test_get_weather_for_session_normalizes_numeric_string_round_to_int():
    fake_session = MagicMock()
    fake_session.weather_data = pd.DataFrame([{"AirTemp": 24.5}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_weather_for_session(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "R")


def test_get_weather_for_session_accepts_race_name_and_session_type():
    fake_session = MagicMock()
    fake_session.weather_data = pd.DataFrame([{"AirTemp": 20.0}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_weather_for_session(2025, "Monaco", session_type="Q")

    mock_fastf1.get_session.assert_called_once_with(2025, "Monaco", "Q")


def test_get_pit_stops_pairs_in_lap_and_out_lap_and_computes_duration():
    # VER pits on lap 2 (PitInTime set) and exits on lap 3 (PitOutTime set) -
    # duration is the gap between those two timestamps. NOR never pits.
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [
            {"Driver": "VER", "Team": "Red Bull Racing", "LapNumber": 1, "PitInTime": pd.NaT, "PitOutTime": pd.NaT},
            {
                "Driver": "VER",
                "Team": "Red Bull Racing",
                "LapNumber": 2,
                "PitInTime": pd.Timedelta(minutes=30),
                "PitOutTime": pd.NaT,
            },
            {
                "Driver": "VER",
                "Team": "Red Bull Racing",
                "LapNumber": 3,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.Timedelta(minutes=30, seconds=22),
            },
            {"Driver": "NOR", "Team": "McLaren", "LapNumber": 1, "PitInTime": pd.NaT, "PitOutTime": pd.NaT},
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_pit_stops(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    assert len(result) == 1
    stop = result[0]
    assert stop["Driver"] == "VER"
    assert stop["Team"] == "Red Bull Racing"
    assert stop["LapNumber"] == 2
    assert stop["PitLaneTime"] == pd.Timedelta(seconds=22)


def test_get_pit_stops_skips_in_lap_with_no_following_out_lap():
    # A pit-in with no next lap at all (e.g. the driver retired in the pits)
    # must be skipped, not raise an IndexError.
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [
            {
                "Driver": "VER",
                "Team": "Red Bull Racing",
                "LapNumber": 10,
                "PitInTime": pd.Timedelta(minutes=40),
                "PitOutTime": pd.NaT,
            },
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_pit_stops(2025, 4)

    assert result == []


def test_get_pit_stops_skips_in_lap_when_next_lap_has_no_out_time():
    # The next lap exists but never registered a PitOutTime (e.g. a crash
    # in the pit lane before rejoining) - still not a valid stop.
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [
            {
                "Driver": "VER",
                "Team": "Red Bull Racing",
                "LapNumber": 10,
                "PitInTime": pd.Timedelta(minutes=40),
                "PitOutTime": pd.NaT,
            },
            {
                "Driver": "VER",
                "Team": "Red Bull Racing",
                "LapNumber": 11,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
            },
        ]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_pit_stops(2025, 4)

    assert result == []


def test_get_pit_stops_normalizes_numeric_string_round_to_int():
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [{"Driver": "VER", "Team": "Red Bull Racing", "LapNumber": 1, "PitInTime": pd.NaT, "PitOutTime": pd.NaT}]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_pit_stops(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "R")


def test_get_pit_stops_accepts_race_name_and_session_type():
    fake_session = MagicMock()
    fake_session.laps = pd.DataFrame(
        [{"Driver": "VER", "Team": "Red Bull Racing", "LapNumber": 1, "PitInTime": pd.NaT, "PitOutTime": pd.NaT}]
    )

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_pit_stops(2025, "Bahrain", session_type="Q")

    mock_fastf1.get_session.assert_called_once_with(2025, "Bahrain", "Q")


def _fake_circuit_session(event_name, status_codes, location="Monaco", country="Monaco"):
    fake_session = MagicMock()
    fake_session.event = {"EventName": event_name, "Location": location, "Country": country}
    fake_session.track_status = pd.DataFrame({"Status": status_codes})
    return fake_session


def test_get_circuit_strategy_history_aggregates_status_codes_across_seasons():
    fastf1_client._circuit_strategy_cache = {}

    def fake_get_session(season, circuit, session_type):
        if season == 2019:
            return _fake_circuit_session("Monaco Grand Prix", ["1", "2", "4"])  # a Safety Car
        if season == 2020:
            return _fake_circuit_session("Monaco Grand Prix", ["1", "6", "7"])  # a VSC only
        raise ValueError("no event found matching that name")  # circuit not on this season's calendar

    with (
        patch("box_box_bot.data.fastf1_client.datetime") as mock_datetime,
        patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1,
    ):
        mock_datetime.date.today.return_value.year = 2020
        mock_fastf1.get_session.side_effect = fake_get_session
        result = fastf1_client.get_circuit_strategy_history("Monaco", since_season=2018)

    assert result["circuit"] == "Monaco"
    assert result["since_season"] == 2018
    assert result["total_races_found"] == 2
    assert result["safety_car_races"] == 1
    assert result["vsc_races"] == 1
    assert result["red_flag_races"] == 0
    assert [s["season"] for s in result["by_season"]] == [2019, 2020]
    assert result["by_season"][0]["safety_car"] is True
    assert result["by_season"][1]["vsc"] is True


def test_get_circuit_strategy_history_skips_season_with_mismatched_event():
    # Regression test: fastf1's fuzzy name match doesn't fail loudly when
    # a circuit genuinely isn't on a season's calendar (e.g. Monaco's
    # 2020 COVID cancellation, where "Monaco" fuzzy-resolved to that
    # year's Italian Grand Prix at Monza) - it silently returns whatever
    # event scored closest. A resolved event whose name/location/country
    # doesn't actually contain the requested circuit must be treated as
    # "not on the calendar" and skipped, not counted as a real match.
    fastf1_client._circuit_strategy_cache = {}

    def fake_get_session(season, circuit, session_type):
        if season == 2019:
            return _fake_circuit_session("Monaco Grand Prix", ["1", "4"], location="Monaco", country="Monaco")
        return _fake_circuit_session("Italian Grand Prix", ["1", "5"], location="Monza", country="Italy")

    with (
        patch("box_box_bot.data.fastf1_client.datetime") as mock_datetime,
        patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1,
    ):
        mock_datetime.date.today.return_value.year = 2020
        mock_fastf1.get_session.side_effect = fake_get_session
        result = fastf1_client.get_circuit_strategy_history("Monaco", since_season=2019)

    assert result["total_races_found"] == 1
    assert [s["season"] for s in result["by_season"]] == [2019]
    assert result["red_flag_races"] == 0  # the mismatched 2020 "Italian GP" red flag must not be counted


def test_get_circuit_strategy_history_skips_season_when_track_status_not_loaded():
    # Regression test: session.load() succeeding does not guarantee
    # session.track_status is readable - fastf1 only populates it when
    # that specific session has full API support, which can be False
    # even for a post-2018 event (partial/incomplete data). Accessing it
    # then raises fastf1's own DataNotLoadedError, which must be caught
    # and treated as "skip this season," not crash the whole tool call.
    fastf1_client._circuit_strategy_cache = {}

    class _UnloadedTrackStatusSession(MagicMock):
        @property
        def track_status(self):
            raise fastf1.exceptions.DataNotLoadedError("track_status not loaded")

    def fake_get_session(season, circuit, session_type):
        if season == 2018:
            unloaded = _UnloadedTrackStatusSession()
            unloaded.event = {"EventName": "Monaco Grand Prix", "Location": "Monaco", "Country": "Monaco"}
            return unloaded
        return _fake_circuit_session("Monaco Grand Prix", ["1", "4"])

    with (
        patch("box_box_bot.data.fastf1_client.datetime") as mock_datetime,
        patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1,
    ):
        mock_datetime.date.today.return_value.year = 2019
        mock_fastf1.get_session.side_effect = fake_get_session
        result = fastf1_client.get_circuit_strategy_history("Monaco", since_season=2018)

    assert result["total_races_found"] == 1
    assert [s["season"] for s in result["by_season"]] == [2019]
    assert result["safety_car_races"] == 1


def test_get_circuit_strategy_history_caches_per_circuit():
    fastf1_client._circuit_strategy_cache = {}

    with (
        patch("box_box_bot.data.fastf1_client.datetime") as mock_datetime,
        patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1,
    ):
        mock_datetime.date.today.return_value.year = 2018
        mock_fastf1.get_session.return_value = _fake_circuit_session("Monaco Grand Prix", ["1"])

        fastf1_client.get_circuit_strategy_history("Monaco", since_season=2018)
        call_count_after_first = mock_fastf1.get_session.call_count
        fastf1_client.get_circuit_strategy_history("Monaco", since_season=2018)

    assert mock_fastf1.get_session.call_count == call_count_after_first


def _fake_speed_map_session(telemetry_df, corners_df, rotation=0.0, driver_code="VER", lap_time="0 days 00:01:30"):
    fake_lap = MagicMock()
    fake_lap.get_telemetry.return_value = telemetry_df
    fake_lap.__getitem__.side_effect = lambda k: {"Driver": driver_code, "LapTime": lap_time}[k]

    fake_laps = MagicMock()
    fake_laps.pick_fastest.return_value = fake_lap
    fake_laps.pick_drivers.return_value = fake_laps  # .pick_drivers(x).pick_fastest() chains back here

    fake_circuit_info = MagicMock()
    fake_circuit_info.rotation = rotation
    fake_circuit_info.corners = corners_df

    fake_session = MagicMock()
    fake_session.laps = fake_laps
    fake_session.event = {"EventName": "Monaco Grand Prix"}
    fake_session.get_circuit_info.return_value = fake_circuit_info
    return fake_session


def test_get_circuit_speed_map_returns_points_and_corners_with_zero_rotation():
    telemetry = pd.DataFrame([{"X": 10.0, "Y": 20.0, "Speed": 250.0}, {"X": 15.0, "Y": 25.0, "Speed": 260.0}])
    corners = pd.DataFrame([{"Number": 1, "Letter": "", "X": 10.0, "Y": 20.0}])
    fake_session = _fake_speed_map_session(telemetry, corners, rotation=0.0)

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_circuit_speed_map(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    fake_session.load.assert_called_once_with(telemetry=True, weather=False, messages=False)
    assert result["circuit"] == "Monaco Grand Prix"
    assert result["driver"] == "VER"
    assert len(result["points"]) == 2
    assert result["points"][0]["X"] == pytest.approx(10.0)
    assert result["points"][0]["Y"] == pytest.approx(20.0)
    assert result["points"][0]["Speed"] == 250.0
    assert len(result["corners"]) == 1
    assert result["corners"][0]["Number"] == 1


def test_get_circuit_speed_map_rotates_points_and_corners_together():
    # A 90-degree rotation should send (1, 0) -> (0, 1) - confirms the
    # same rotation is applied consistently to both points and corners
    # so they stay aligned with each other after rotating.
    telemetry = pd.DataFrame([{"X": 1.0, "Y": 0.0, "Speed": 300.0}])
    corners = pd.DataFrame([{"Number": 1, "Letter": "", "X": 1.0, "Y": 0.0}])
    fake_session = _fake_speed_map_session(telemetry, corners, rotation=90.0)

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_circuit_speed_map(2025, 4)

    assert result["points"][0]["X"] == pytest.approx(0.0, abs=1e-9)
    assert result["points"][0]["Y"] == pytest.approx(1.0, abs=1e-9)
    assert result["corners"][0]["X"] == pytest.approx(0.0, abs=1e-9)
    assert result["corners"][0]["Y"] == pytest.approx(1.0, abs=1e-9)


def test_get_circuit_speed_map_downsamples_to_max_points():
    telemetry = pd.DataFrame([{"X": float(i), "Y": float(i), "Speed": 200.0} for i in range(1000)])
    corners = pd.DataFrame([])
    fake_session = _fake_speed_map_session(telemetry, corners)

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_circuit_speed_map(2025, 4, max_points=100)

    assert len(result["points"]) <= 100


def test_get_circuit_speed_map_downsamples_when_length_is_less_than_double_max_points():
    # Regression test: floor division for the downsample step (len // max_points)
    # computes step=1 (i.e. no downsampling at all) for any length under
    # 2x max_points - e.g. 729 // 400 == 1 - silently letting the result
    # exceed max_points. Ceiling division must be used instead.
    telemetry = pd.DataFrame([{"X": float(i), "Y": float(i), "Speed": 200.0} for i in range(729)])
    fake_session = _fake_speed_map_session(telemetry, pd.DataFrame([]))

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_circuit_speed_map(2025, 4, max_points=400)

    assert len(result["points"]) <= 400


def test_get_circuit_speed_map_filters_by_driver_when_given():
    telemetry = pd.DataFrame([{"X": 1.0, "Y": 1.0, "Speed": 200.0}])
    corners = pd.DataFrame([])
    fake_session = _fake_speed_map_session(telemetry, corners, driver_code="HAM")

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_circuit_speed_map(2025, 4, driver="HAM")

    fake_session.laps.pick_drivers.assert_called_once_with("HAM")
    assert result["driver"] == "HAM"


def test_get_circuit_speed_map_normalizes_numeric_string_round_to_int():
    telemetry = pd.DataFrame([{"X": 1.0, "Y": 1.0, "Speed": 200.0}])
    fake_session = _fake_speed_map_session(telemetry, pd.DataFrame([]))

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_circuit_speed_map(2026, "7")

    mock_fastf1.get_session.assert_called_once_with(2026, 7, "R")


def test_get_circuit_speed_map_accepts_race_name_and_session_type():
    telemetry = pd.DataFrame([{"X": 1.0, "Y": 1.0, "Speed": 200.0}])
    fake_session = _fake_speed_map_session(telemetry, pd.DataFrame([]))

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        fastf1_client.get_circuit_speed_map(2025, "Bahrain", session_type="Q")

    mock_fastf1.get_session.assert_called_once_with(2025, "Bahrain", "Q")
