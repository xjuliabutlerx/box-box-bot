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


def test_get_race_results_loads_session_without_laps_or_telemetry():
    fake_session = MagicMock()
    fake_session.results = pd.DataFrame([{"Position": 1.0, "Abbreviation": "PIA"}])

    with patch("box_box_bot.data.fastf1_client.fastf1") as mock_fastf1:
        mock_fastf1.get_session.return_value = fake_session
        result = fastf1_client.get_race_results(2025, 4)

    mock_fastf1.get_session.assert_called_once_with(2025, 4, "R")
    fake_session.load.assert_called_once_with(laps=False, telemetry=False, weather=False, messages=False)
    assert result == [{"Position": 1.0, "Abbreviation": "PIA"}]


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
