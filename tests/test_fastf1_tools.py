import json
from unittest.mock import patch

import pandas as pd

from box_box_bot.tools.fastf1_tools import (
    FASTF1_TOOLS,
    get_constructor_standings,
    get_driver_standings,
    get_fastest_laps,
    get_race_results,
)


def test_all_four_tools_are_registered():
    assert {t.name for t in FASTF1_TOOLS} == {
        "get_driver_standings",
        "get_constructor_standings",
        "get_race_results",
        "get_fastest_laps",
    }


def test_get_driver_standings_calls_data_layer_and_returns_json():
    fake_data = [{"position": 1, "driverCode": "NOR", "points": 423.0}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_driver_standings", return_value=fake_data) as mock_fn:
        result = get_driver_standings.invoke({"season": 2025, "round": 4})

    mock_fn.assert_called_once_with(2025, 4)
    assert json.loads(result) == fake_data


def test_get_constructor_standings_round_is_optional():
    fake_data = [{"position": 1, "constructorName": "McLaren"}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_constructor_standings", return_value=fake_data) as mock_fn:
        result = get_constructor_standings.invoke({"season": 2025})

    mock_fn.assert_called_once_with(2025, None)
    assert json.loads(result) == fake_data


def test_get_race_results_has_a_body_and_returns_data():
    # Regression test: an early draft of this tool had a docstring but no
    # function body, so it silently returned None on every call.
    fake_data = [{"Position": 1.0, "Abbreviation": "PIA"}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_race_results", return_value=fake_data) as mock_fn:
        result = get_race_results.invoke({"season": 2025, "round": 4})

    mock_fn.assert_called_once_with(2025, 4)
    assert result is not None
    assert json.loads(result) == fake_data


def test_get_race_results_serializes_pandas_nat_and_timedelta():
    # Regression test: json.dumps(data) (without default=str) crashes on
    # NaT/Timedelta values that show up in real fastf1 output (e.g. Q1/Q2/Q3
    # for a driver who didn't set a time, or the race Time column).
    fake_data = [
        {
            "Abbreviation": "PIA",
            "Q1": pd.NaT,
            "Time": pd.Timedelta(hours=1, minutes=27, seconds=38),
        }
    ]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_race_results", return_value=fake_data):
        result = get_race_results.invoke({"season": 2025, "round": 4})

    parsed = json.loads(result)  # would raise if serialization crashed
    assert parsed[0]["Abbreviation"] == "PIA"
    assert "NaT" in parsed[0]["Q1"]


def test_get_race_results_accepts_race_name_for_round():
    # Regression test: round used to be int-only, which meant the model
    # had to guess a round number for a named race rather than pass a name
    # it already knew for certain - see test_fastf1_client.py for the fix.
    fake_data = [{"Position": 1.0, "Abbreviation": "VER"}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_race_results", return_value=fake_data) as mock_fn:
        result = get_race_results.invoke({"season": 2025, "round": "Bahrain"})

    mock_fn.assert_called_once_with(2025, "Bahrain")
    assert json.loads(result) == fake_data


def test_get_fastest_laps_passes_all_args_through():
    fake_data = [{"Driver": "VER", "LapTime": "0 days 00:01:29.708000"}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_fastest_laps", return_value=fake_data) as mock_fn:
        result = get_fastest_laps.invoke({"season": 2026, "round": 4, "session_type": "S", "top_n": 3})

    mock_fn.assert_called_once_with(2026, 4, "S", 3)
    assert json.loads(result) == fake_data
