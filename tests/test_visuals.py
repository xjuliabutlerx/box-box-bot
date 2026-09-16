import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from box_box_bot.agent.visuals import extract_visuals


def _tool_message(data, name: str, tool_call_id: str = "call_1") -> ToolMessage:
    content = data if isinstance(data, str) else json.dumps(data)
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)


def _ai_call(name: str, args: dict, tool_call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": tool_call_id}])


def test_extract_visuals_generic_table_from_list_of_dicts():
    messages = [
        HumanMessage(content="Who won the 2025 Bahrain GP?"),
        _tool_message([{"Position": 1, "Abbreviation": "PIA"}], name="get_race_results"),
        AIMessage(content="Piastri won."),
    ]
    visuals = extract_visuals(messages)
    assert visuals == [
        {
            "type": "table",
            "tool": "get_race_results",
            "data": [{"Position": 1, "Abbreviation": "PIA"}],
            "label": "get race results",
        }
    ]


def test_extract_visuals_tire_strategy_becomes_a_chart_not_a_table():
    data = [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 20}]
    messages = [
        HumanMessage(content="What was the tire strategy?"),
        _tool_message(data, name="get_tire_strategy"),
    ]
    assert extract_visuals(messages) == [
        {"type": "tire_strategy_chart", "tool": "get_tire_strategy", "data": data, "label": "get tire strategy"}
    ]


def test_extract_visuals_circuit_speed_map_becomes_a_track_map():
    data = {"circuit": "Monaco", "driver": "VER", "points": [{"X": 1, "Y": 2, "Speed": 300}], "corners": []}
    messages = [
        HumanMessage(content="Show me the track"),
        _tool_message(data, name="get_circuit_speed_map"),
    ]
    assert extract_visuals(messages) == [
        {"type": "track_map", "tool": "get_circuit_speed_map", "data": data, "label": "get circuit speed map"}
    ]


def test_extract_visuals_pulls_nested_by_season_list_for_circuit_strategy_history():
    data = {
        "circuit": "Monaco",
        "since_season": 2018,
        "by_season": [{"season": 2019, "safety_car": True}],
    }
    messages = [
        HumanMessage(content="How often is there a safety car at Monaco?"),
        _tool_message(data, name="get_circuit_strategy_history"),
    ]
    assert extract_visuals(messages) == [
        {
            "type": "table",
            "tool": "get_circuit_strategy_history",
            "data": [{"season": 2019, "safety_car": True}],
            "label": "get circuit strategy history",
        }
    ]


def test_extract_visuals_ignores_prose_tool_results():
    messages = [
        HumanMessage(content="Why did Bahrain matter?"),
        _tool_message("[Source: Bahrain Grand Prix (2025)]\nPiastri won.", name="search_race_recaps"),
    ]
    assert extract_visuals(messages) == []


def test_extract_visuals_reshapes_constructor_predictions_into_a_table():
    # Predictor tools return {"predicted_orders": {model_name: [...]}, ...}
    # - a dict of per-model orders, not a flat list of records - reshaped
    # into one row per finishing position, one column per model, so all
    # 5 (or 3) independently-trained models compare side by side.
    data = {
        "predicted_orders": {
            "Monaco": ["Team A", "Team B"],
            "Silverstone": ["Team B", "Team A"],
        },
        "as_of_round": 5,
    }
    messages = [
        HumanMessage(content="Who will win the constructors championship?"),
        _tool_message(data, name="predict_constructor_championship"),
    ]
    assert extract_visuals(messages) == [
        {
            "type": "table",
            "tool": "predict_constructor_championship",
            "data": [
                {"Position": 1, "Monaco": "Team A", "Silverstone": "Team B"},
                {"Position": 2, "Monaco": "Team B", "Silverstone": "Team A"},
            ],
            "label": "predict constructor championship",
        }
    ]


def test_extract_visuals_reshapes_driver_predictions_into_a_table():
    data = {"predicted_orders": {"Prost": ["Driver A", "Driver B", "Driver C"]}, "as_of_round": 10}
    messages = [
        HumanMessage(content="Who will win the drivers championship?"),
        _tool_message(data, name="predict_drivers_championship"),
    ]
    visuals = extract_visuals(messages)
    assert visuals[0]["type"] == "table"
    assert visuals[0]["data"] == [
        {"Position": 1, "Prost": "Driver A"},
        {"Position": 2, "Prost": "Driver B"},
        {"Position": 3, "Prost": "Driver C"},
    ]


def test_extract_visuals_predictor_table_pads_ragged_model_orders():
    data = {"predicted_orders": {"A": ["X", "Y"], "B": ["X"]}, "as_of_round": 1}
    messages = [
        HumanMessage(content="Predict it"),
        _tool_message(data, name="predict_constructor_championship"),
    ]
    visuals = extract_visuals(messages)
    assert visuals[0]["data"] == [
        {"Position": 1, "A": "X", "B": "X"},
        {"Position": 2, "A": "Y", "B": None},
    ]


def test_extract_visuals_ignores_empty_predictor_orders():
    data = {"predicted_orders": {}, "as_of_round": 1}
    messages = [
        HumanMessage(content="Predict it"),
        _tool_message(data, name="predict_constructor_championship"),
    ]
    assert extract_visuals(messages) == []


def test_extract_visuals_ignores_empty_tool_result():
    messages = [
        HumanMessage(content="What was the tire strategy for round 99?"),
        _tool_message([], name="get_tire_strategy"),
    ]
    assert extract_visuals(messages) == []


def test_extract_visuals_ignores_a_failed_tire_strategy_call():
    # Regression: a failed call (tools/fastf1_tools.py's
    # _catch_fastf1_errors) returns {"error": "..."} - a truthy JSON
    # value that `if data:` alone can't tell apart from real chart data.
    # Handing that to build_tire_strategy_figure crashes
    # (pd.DataFrame({"error": "..."}) raises ValueError) - live-reproduced
    # after the future-session retry made a failed call followed by a
    # successful one common within a single turn.
    messages = [
        HumanMessage(content="What's the tire strategy at Baku?"),
        _tool_message({"error": "This race hasn't happened yet."}, name="get_tire_strategy"),
    ]
    assert extract_visuals(messages) == []


def test_extract_visuals_ignores_a_failed_circuit_speed_map_call():
    messages = [
        HumanMessage(content="Show me the Baku track"),
        _tool_message({"error": "This race hasn't happened yet."}, name="get_circuit_speed_map"),
    ]
    assert extract_visuals(messages) == []


def test_extract_visuals_only_looks_at_current_turn():
    messages = [
        HumanMessage(content="Tell me the results for Bahrain"),
        _tool_message([{"Position": 1}], name="get_race_results"),
        AIMessage(content="..."),
        HumanMessage(content="What about Monza?"),
        _tool_message([{"Position": 2}], name="get_race_results"),
    ]
    assert extract_visuals(messages) == [
        {"type": "table", "tool": "get_race_results", "data": [{"Position": 2}], "label": "get race results"}
    ]


def test_extract_visuals_no_tool_messages():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert extract_visuals(messages) == []


def test_extract_visuals_labels_table_from_the_originating_tool_calls_args():
    # Regression: without correlating back to the AIMessage's tool_calls,
    # several tables from the same tool (e.g. get_race_results pulled
    # for several different seasons/circuits) render as an unlabeled
    # stack of identical-looking tables - a live-reported bug.
    messages = [
        HumanMessage(content="Who's won most at Baku?"),
        _ai_call("get_race_results", {"season": 2019, "round": "Baku"}),
        _tool_message([{"Position": 1}], name="get_race_results"),
    ]
    assert extract_visuals(messages)[0]["label"] == "Baku 2019 — get race results"


def test_extract_visuals_labels_distinguish_multiple_calls_to_the_same_tool():
    messages = [
        HumanMessage(content="Who's won most at Baku?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_race_results", "args": {"season": 2019, "round": "Baku"}, "id": "call_1"},
                {"name": "get_race_results", "args": {"season": 2020, "round": "Baku"}, "id": "call_2"},
            ],
        ),
        _tool_message([{"Position": 1}], name="get_race_results", tool_call_id="call_1"),
        _tool_message([{"Position": 2}], name="get_race_results", tool_call_id="call_2"),
    ]
    visuals = extract_visuals(messages)
    labels = [v["label"] for v in visuals]
    assert labels == ["Baku 2019 — get race results", "Baku 2020 — get race results"]


def test_extract_visuals_label_includes_session_type_when_not_race():
    messages = [
        HumanMessage(content="Quali pace at Monza?"),
        _ai_call("get_fastest_laps", {"season": 2025, "round": "Monza", "session_type": "Q"}),
        _tool_message([{"Driver": "VER"}], name="get_fastest_laps"),
    ]
    assert extract_visuals(messages)[0]["label"] == "Monza 2025 Q — get fastest laps"


def test_extract_visuals_label_falls_back_to_tool_name_with_no_useful_args():
    messages = [
        HumanMessage(content="Top drivers of all time?"),
        _ai_call("get_all_time_driver_records", {"top_n": 10}),
        _tool_message([{"driverId": "hamilton"}], name="get_all_time_driver_records"),
    ]
    assert extract_visuals(messages)[0]["label"] == "get all time driver records"


def test_extract_visuals_unwraps_a_fallback_tire_strategy_result():
    # tools/fastf1_tools.py's _catch_fastf1_errors wraps a successful
    # fallback (this season's race hasn't happened yet, last year's
    # data was used instead) as {"fallback_note", "season_used",
    # "result"} rather than the tool's normal shape. This must still
    # become a real chart, and its label must reflect the season
    # actually shown (2025) rather than the one originally asked for
    # (2026) - a stale label would misrepresent a fallback as current.
    fallback_data = [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 20}]
    messages = [
        HumanMessage(content="Tire strategy at Baku?"),
        _ai_call("get_tire_strategy", {"season": 2026, "round": "Baku"}),
        _tool_message(
            {
                "fallback_note": "The Azerbaijan Grand Prix (2026) hasn't happened yet - showing 2025 data instead.",
                "season_used": 2025,
                "result": fallback_data,
            },
            name="get_tire_strategy",
        ),
    ]
    visuals = extract_visuals(messages)
    assert visuals == [
        {
            "type": "tire_strategy_chart",
            "tool": "get_tire_strategy",
            "data": fallback_data,
            "label": "Baku 2025 — get tire strategy",
        }
    ]


def test_extract_visuals_unwraps_a_fallback_table_result():
    fallback_rows = [{"Position": 1, "Driver": "PER"}]
    messages = [
        HumanMessage(content="Results at Baku?"),
        _ai_call("get_race_results", {"season": 2026, "round": "Baku"}),
        _tool_message(
            {
                "fallback_note": "hasn't happened yet - showing 2025 data instead.",
                "season_used": 2025,
                "result": fallback_rows,
            },
            name="get_race_results",
        ),
    ]
    visuals = extract_visuals(messages)
    assert visuals == [
        {"type": "table", "tool": "get_race_results", "data": fallback_rows, "label": "Baku 2025 — get race results"}
    ]
