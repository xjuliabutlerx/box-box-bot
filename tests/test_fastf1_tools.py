import datetime
import json
from unittest.mock import patch

import pandas as pd
import pytest
from fastf1.exceptions import DataNotLoadedError

from box_box_bot.tools.fastf1_tools import (
    FASTF1_TOOLS,
    STRATEGY_TOOLS,
    get_all_time_driver_records,
    get_circuit_speed_map,
    get_circuit_strategy_history,
    get_constructor_standings,
    get_driver_standings,
    get_fastest_laps,
    get_pit_stops,
    get_race_control_messages,
    get_race_results,
    get_season_schedule,
    get_tire_strategy,
    get_weather,
)


def test_stats_tools_are_registered():
    assert {t.name for t in FASTF1_TOOLS} == {
        "get_driver_standings",
        "get_constructor_standings",
        "get_race_results",
        "get_fastest_laps",
        "get_season_schedule",
        "get_all_time_driver_records",
    }


def test_strategy_tools_are_registered():
    assert {t.name for t in STRATEGY_TOOLS} == {
        "get_tire_strategy",
        "get_race_control_messages",
        "get_weather",
        "get_pit_stops",
        "get_circuit_strategy_history",
        "get_circuit_speed_map",
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


def test_get_driver_standings_strips_wikipedia_urls():
    fake_data = [
        {
            "position": 1,
            "driverCode": "VER",
            "points": 437.0,
            "driverUrl": "http://en.wikipedia.org/wiki/Max_Verstappen",
            "constructorUrls": ["https://en.wikipedia.org/wiki/Red_Bull_Racing"],
        }
    ]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_driver_standings", return_value=fake_data):
        result = get_driver_standings.invoke({"season": 2025})

    parsed = json.loads(result)[0]
    assert "driverUrl" not in parsed
    assert "constructorUrls" not in parsed
    assert parsed["driverCode"] == "VER"
    assert parsed["points"] == 437.0


def test_get_constructor_standings_strips_wikipedia_url():
    fake_data = [{"position": 1, "constructorName": "McLaren", "constructorUrl": "https://en.wikipedia.org/wiki/McLaren"}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_constructor_standings", return_value=fake_data):
        result = get_constructor_standings.invoke({"season": 2025})

    parsed = json.loads(result)[0]
    assert "constructorUrl" not in parsed
    assert parsed["constructorName"] == "McLaren"


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
    # NaT/Timedelta values that show up in real fastf1 output (e.g. a
    # DNF's Time column, which is NaT rather than a finish time).
    fake_data = [
        {
            "Abbreviation": "PIA",
            "Status": "Retired",
            "Time": pd.NaT,
        }
    ]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_race_results", return_value=fake_data):
        result = get_race_results.invoke({"season": 2025, "round": 4})

    parsed = json.loads(result)  # would raise if serialization crashed
    assert parsed[0]["Abbreviation"] == "PIA"
    assert "NaT" in parsed[0]["Time"]


def test_get_race_results_strips_noisy_fields():
    # Regression test found via live use: get_race_results' underlying
    # fastf1 data carries headshot image URLs, hex team colors, internal
    # driver/team IDs, and always-empty Q1/Q2/Q3 (this tool only ever
    # loads the Race session) - none of it belongs in a chat answer, and
    # dumping it all in verbatim was flooding both the model's context
    # and the auto-rendered table with useless columns.
    fake_data = [
        {
            "DriverNumber": "1",
            "BroadcastName": "M VERSTAPPEN",
            "Abbreviation": "VER",
            "DriverId": "max_verstappen",
            "TeamName": "Red Bull Racing",
            "TeamColor": "3671C6",
            "TeamId": "red_bull",
            "FirstName": "Max",
            "LastName": "Verstappen",
            "FullName": "Max Verstappen",
            "HeadshotUrl": "https://example.com/headshot.png",
            "CountryCode": "NED",
            "Position": 1.0,
            "ClassifiedPosition": "1",
            "GridPosition": 2.0,
            "Q1": pd.NaT,
            "Q2": pd.NaT,
            "Q3": pd.NaT,
            "Status": "Finished",
            "Points": 25.0,
            "Laps": 66.0,
        }
    ]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_race_results", return_value=fake_data):
        result = get_race_results.invoke({"season": 2025, "round": 4})

    parsed = json.loads(result)[0]
    assert "HeadshotUrl" not in parsed
    assert "TeamColor" not in parsed
    assert "BroadcastName" not in parsed
    assert "DriverId" not in parsed
    assert "TeamId" not in parsed
    assert "DriverNumber" not in parsed
    assert "Q1" not in parsed
    assert "Q2" not in parsed
    assert "Q3" not in parsed
    # what's actually useful for a chat answer survives
    assert parsed["Abbreviation"] == "VER"
    assert parsed["FullName"] == "Max Verstappen"
    assert parsed["TeamName"] == "Red Bull Racing"
    assert parsed["Position"] == 1.0
    assert parsed["ClassifiedPosition"] == "1"
    assert parsed["GridPosition"] == 2.0
    assert parsed["Status"] == "Finished"
    assert parsed["Points"] == 25.0
    assert parsed["Laps"] == 66.0


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


def test_get_season_schedule_calls_data_layer_and_returns_json():
    fake_data = [
        {
            "RoundNumber": 4,
            "Country": "Bahrain",
            "Location": "Sakhir",
            "EventName": "Bahrain Grand Prix",
            "EventFormat": "conventional",
            "EventDate": "2025-04-13",
        }
    ]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule", return_value=fake_data) as mock_fn:
        result = get_season_schedule.invoke({"season": 2025})

    mock_fn.assert_called_once_with(2025)
    assert json.loads(result) == fake_data


def test_get_tire_strategy_calls_data_layer_and_returns_json():
    fake_data = [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 20}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy", return_value=fake_data) as mock_fn:
        result = get_tire_strategy.invoke({"season": 2025, "round": 4})

    mock_fn.assert_called_once_with(2025, 4, "R")
    assert json.loads(result) == fake_data


def test_get_tire_strategy_accepts_race_name_and_session_type():
    fake_data = [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 10}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy", return_value=fake_data) as mock_fn:
        get_tire_strategy.invoke({"season": 2025, "round": "Bahrain", "session_type": "Q"})

    mock_fn.assert_called_once_with(2025, "Bahrain", "Q")


def test_get_race_control_messages_calls_data_layer_and_returns_json():
    fake_data = [{"Category": "Flag", "Message": "YELLOW FLAG"}]
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_race_control_messages", return_value=fake_data
    ) as mock_fn:
        result = get_race_control_messages.invoke({"season": 2025, "round": 4})

    mock_fn.assert_called_once_with(2025, 4, "R", "All")
    assert json.loads(result) == fake_data


def test_get_race_control_messages_passes_category_through():
    fake_data = [{"Category": "SafetyCar", "Message": "SAFETY CAR DEPLOYED"}]
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_race_control_messages", return_value=fake_data
    ) as mock_fn:
        get_race_control_messages.invoke(
            {"season": 2025, "round": 4, "session_type": "R", "category": "SafetyCar"}
        )

    mock_fn.assert_called_once_with(2025, 4, "R", "SafetyCar")


def test_get_weather_calls_data_layer_and_returns_summary():
    # get_weather summarizes the raw per-minute samples rather than
    # passing them through - a full session's weather trace is 100+
    # rows, far more than almost any conversational question needs.
    fake_data = [
        {"Time": "0 days 00:01:00", "AirTemp": 20.0, "TrackTemp": 30.0, "WindSpeed": 2.0, "Rainfall": False},
        {"Time": "0 days 00:02:00", "AirTemp": 24.0, "TrackTemp": 34.0, "WindSpeed": 4.0, "Rainfall": True},
    ]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_weather_for_session", return_value=fake_data) as mock_fn:
        result = get_weather.invoke({"season": 2025, "round": 4})

    mock_fn.assert_called_once_with(2025, 4, "R")
    parsed = json.loads(result)
    assert parsed["SampleCount"] == 2
    assert parsed["AirTemp"] == {"min": 20.0, "max": 24.0, "avg": 22.0}
    assert parsed["TrackTemp"] == {"min": 30.0, "max": 34.0, "avg": 32.0}
    assert parsed["AvgWindSpeed"] == 3.0
    assert parsed["RainfallDuringSession"] is True


def test_get_weather_accepts_race_name_and_session_type():
    fake_data = [{"AirTemp": 20.0}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_weather_for_session", return_value=fake_data) as mock_fn:
        get_weather.invoke({"season": 2025, "round": "Monaco", "session_type": "Q"})

    mock_fn.assert_called_once_with(2025, "Monaco", "Q")


def test_get_all_time_driver_records_calls_data_layer_and_returns_json():
    fake_data = [{"driverId": "hamilton", "driverName": "Lewis Hamilton", "totalWins": 105, "championships": 7}]
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_all_time_driver_records", return_value=fake_data
    ) as mock_fn:
        result = get_all_time_driver_records.invoke({})

    mock_fn.assert_called_once_with(10)
    assert json.loads(result) == fake_data


def test_get_all_time_driver_records_passes_top_n_through():
    fake_data = [{"driverId": "hamilton", "driverName": "Lewis Hamilton", "totalWins": 105, "championships": 7}]
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_all_time_driver_records", return_value=fake_data
    ) as mock_fn:
        get_all_time_driver_records.invoke({"top_n": 3})

    mock_fn.assert_called_once_with(3)


def test_get_pit_stops_calls_data_layer_and_returns_json():
    fake_data = [{"Driver": "VER", "Team": "Red Bull Racing", "LapNumber": 20.0, "PitLaneTime": "0 days 00:00:22"}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_pit_stops", return_value=fake_data) as mock_fn:
        result = get_pit_stops.invoke({"season": 2025, "round": 4})

    mock_fn.assert_called_once_with(2025, 4, "R")
    assert json.loads(result) == fake_data


def test_get_pit_stops_accepts_race_name_and_session_type():
    fake_data = [{"Driver": "VER", "Team": "Red Bull Racing", "LapNumber": 5.0, "PitLaneTime": "0 days 00:00:23"}]
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_pit_stops", return_value=fake_data) as mock_fn:
        get_pit_stops.invoke({"season": 2025, "round": "Bahrain", "session_type": "Q"})

    mock_fn.assert_called_once_with(2025, "Bahrain", "Q")


def test_get_circuit_strategy_history_calls_data_layer_and_returns_json():
    fake_data = {
        "circuit": "Monaco",
        "since_season": 2018,
        "total_races_found": 6,
        "safety_car_races": 3,
        "vsc_races": 2,
        "red_flag_races": 0,
        "by_season": [],
    }
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_circuit_strategy_history", return_value=fake_data
    ) as mock_fn:
        result = get_circuit_strategy_history.invoke({"circuit": "Monaco"})

    mock_fn.assert_called_once_with("Monaco", 2018)
    assert json.loads(result) == fake_data


def test_get_circuit_strategy_history_passes_since_season_through():
    fake_data = {"circuit": "Spa-Francorchamps", "since_season": 2020, "total_races_found": 0}
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_circuit_strategy_history", return_value=fake_data
    ) as mock_fn:
        get_circuit_strategy_history.invoke({"circuit": "Spa-Francorchamps", "since_season": 2020})

    mock_fn.assert_called_once_with("Spa-Francorchamps", 2020)


def test_get_circuit_speed_map_calls_data_layer_and_returns_json():
    fake_data = {
        "circuit": "Monaco Grand Prix",
        "driver": "VER",
        "lap_time": "0 days 00:01:12.909000",
        "rotation_degrees": 42.0,
        "points": [{"X": 1.0, "Y": 2.0, "Speed": 250.0}],
        "corners": [{"Number": 1, "Letter": "", "X": 1.0, "Y": 2.0}],
    }
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_circuit_speed_map", return_value=fake_data) as mock_fn:
        result = get_circuit_speed_map.invoke({"season": 2025, "round": 4})

    mock_fn.assert_called_once_with(2025, 4, "R", None)
    assert json.loads(result) == fake_data


def test_get_circuit_speed_map_passes_driver_and_session_type_through():
    fake_data = {"circuit": "Bahrain", "driver": "HAM", "points": [], "corners": []}
    with patch("box_box_bot.tools.fastf1_tools.fastf1_client.get_circuit_speed_map", return_value=fake_data) as mock_fn:
        get_circuit_speed_map.invoke({"season": 2025, "round": "Bahrain", "session_type": "Q", "driver": "HAM"})

    mock_fn.assert_called_once_with(2025, "Bahrain", "Q", "HAM")


def test_get_tire_strategy_returns_error_json_instead_of_raising():
    # Regression: a pre-2018 season has f1_api_support=False, so fastf1's
    # session.load() silently skips loading laps and session.laps raises
    # DataNotLoadedError the moment it's touched - this must not crash
    # the whole agent turn.
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy",
        side_effect=DataNotLoadedError("The data you are trying to access has not been loaded yet."),
    ), patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule", return_value=[]
    ):
        result = get_tire_strategy.invoke({"season": 2016, "round": 5})

    parsed = json.loads(result)
    assert "error" in parsed
    assert "not been loaded yet" in parsed["error"]


def test_get_race_results_returns_error_json_instead_of_raising():
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_race_results",
        side_effect=DataNotLoadedError("boom"),
    ), patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule", return_value=[]
    ):
        result = get_race_results.invoke({"season": 2016, "round": 5})

    parsed = json.loads(result)
    assert "error" in parsed


_FUTURE_BAKU_SCHEDULE = [
    {
        "RoundNumber": 15,
        "Country": "Azerbaijan",
        "Location": "Baku",
        "EventName": "Azerbaijan Grand Prix",
        "EventFormat": "conventional",
        "EventDate": datetime.date(2026, 9, 26),
    },
]


def _tire_strategy_2025_only(season, round, session_type="R"):
    # Simulates 2026 having no data yet (scheduled but not run) while
    # 2025's real data is available - the exact shape of the live bug.
    if season == 2026:
        raise DataNotLoadedError("The data you are trying to access has not been loaded yet.")
    return [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 20}]


def test_future_session_automatically_falls_back_to_last_year():
    # Regression: live-observed bug - asking about a session that's on
    # the calendar but hasn't run yet (e.g. 2026 Baku, scheduled after
    # today) failed with the same generic "could not load" message as
    # any other failure, so the agent silently gave up on the tool
    # entirely instead of explaining why or trying last year. The retry
    # must happen automatically here, not be left to the model's
    # judgment whether to retry (LLM tool-routing is probabilistic) -
    # but the result must still be clearly, machine-checkably labeled
    # as a fallback, not silently presented as this year's data.
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy",
        side_effect=_tire_strategy_2025_only,
    ), patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule",
        return_value=_FUTURE_BAKU_SCHEDULE,
    ):
        result = get_tire_strategy.invoke({"season": 2026, "round": "Baku"})

    parsed = json.loads(result)
    assert "error" not in parsed
    assert parsed["season_used"] == 2025
    assert parsed["result"] == [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 20}]
    assert "Azerbaijan Grand Prix" in parsed["fallback_note"]
    assert "hasn't happened yet" in parsed["fallback_note"]
    assert "2025" in parsed["fallback_note"]


def test_future_session_falls_back_when_round_is_a_number_too():
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy",
        side_effect=_tire_strategy_2025_only,
    ), patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule",
        return_value=_FUTURE_BAKU_SCHEDULE,
    ):
        result = get_tire_strategy.invoke({"season": 2026, "round": 15})

    parsed = json.loads(result)
    assert parsed["season_used"] == 2025


def test_future_session_reports_unavailable_when_fallback_also_fails():
    # Both 2026 (not yet happened) and the 2025 fallback fail here - the
    # agent must be told plainly that neither year has data, not left
    # thinking a retry might still help.
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy",
        side_effect=DataNotLoadedError("boom"),
    ), patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule",
        return_value=_FUTURE_BAKU_SCHEDULE,
    ):
        result = get_tire_strategy.invoke({"season": 2026, "round": "Baku"})

    parsed = json.loads(result)
    assert "Azerbaijan Grand Prix" in parsed["error"]
    assert "2025" in parsed["error"]


def test_error_stays_generic_when_the_session_already_happened():
    past_schedule = [{**_FUTURE_BAKU_SCHEDULE[0], "EventDate": datetime.date(2020, 1, 1)}]
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy",
        side_effect=DataNotLoadedError("boom"),
    ), patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule",
        return_value=past_schedule,
    ):
        result = get_tire_strategy.invoke({"season": 2026, "round": "Baku"})

    parsed = json.loads(result)
    assert "Azerbaijan Grand Prix" not in parsed["error"]
    assert "Could not load this data" in parsed["error"]


def test_error_stays_generic_when_schedule_lookup_itself_fails():
    with patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_tire_strategy",
        side_effect=DataNotLoadedError("boom"),
    ), patch(
        "box_box_bot.tools.fastf1_tools.fastf1_client.get_season_schedule",
        side_effect=Exception("network error"),
    ):
        result = get_tire_strategy.invoke({"season": 2026, "round": "Baku"})

    parsed = json.loads(result)
    assert "Azerbaijan Grand Prix" not in parsed["error"]
    assert "Could not load this data" in parsed["error"]


@pytest.mark.parametrize("tool", FASTF1_TOOLS + STRATEGY_TOOLS)
def test_every_fastf1_tool_is_wrapped_against_unhandled_exceptions(tool):
    # Regression guard: a new fastf1-backed tool added without
    # @_catch_fastf1_errors would reintroduce the whole-turn-crashing bug.
    assert hasattr(tool.func, "__wrapped__")
