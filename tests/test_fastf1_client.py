from unittest.mock import MagicMock, patch

import pandas as pd

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
