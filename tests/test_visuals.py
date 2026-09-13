import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from box_box_bot.agent.visuals import extract_visuals


def _tool_message(data, name: str) -> ToolMessage:
    content = data if isinstance(data, str) else json.dumps(data)
    return ToolMessage(content=content, tool_call_id="call_1", name=name)


def test_extract_visuals_generic_table_from_list_of_dicts():
    messages = [
        HumanMessage(content="Who won the 2025 Bahrain GP?"),
        _tool_message([{"Position": 1, "Abbreviation": "PIA"}], name="get_race_results"),
        AIMessage(content="Piastri won."),
    ]
    visuals = extract_visuals(messages)
    assert visuals == [
        {"type": "table", "tool": "get_race_results", "data": [{"Position": 1, "Abbreviation": "PIA"}]}
    ]


def test_extract_visuals_tire_strategy_becomes_a_chart_not_a_table():
    data = [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 20}]
    messages = [
        HumanMessage(content="What was the tire strategy?"),
        _tool_message(data, name="get_tire_strategy"),
    ]
    assert extract_visuals(messages) == [{"type": "tire_strategy_chart", "tool": "get_tire_strategy", "data": data}]


def test_extract_visuals_circuit_speed_map_becomes_a_track_map():
    data = {"circuit": "Monaco", "driver": "VER", "points": [{"X": 1, "Y": 2, "Speed": 300}], "corners": []}
    messages = [
        HumanMessage(content="Show me the track"),
        _tool_message(data, name="get_circuit_speed_map"),
    ]
    assert extract_visuals(messages) == [{"type": "track_map", "tool": "get_circuit_speed_map", "data": data}]


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
        {"type": "table", "tool": "get_circuit_strategy_history", "data": [{"season": 2019, "safety_car": True}]}
    ]


def test_extract_visuals_ignores_prose_tool_results():
    messages = [
        HumanMessage(content="Why did Bahrain matter?"),
        _tool_message("[Source: Bahrain Grand Prix (2025)]\nPiastri won.", name="search_race_recaps"),
    ]
    assert extract_visuals(messages) == []


def test_extract_visuals_ignores_non_row_shaped_dict_results():
    # Predictor tools return {"predicted_orders": {model_name: [...]}, ...}
    # - a dict of dicts/lists, not a flat list of records, so it isn't a
    # natural table and must not be shown as one.
    data = {"predicted_orders": {"Monaco": ["Team A", "Team B"]}, "as_of_round": 5}
    messages = [
        HumanMessage(content="Who will win the constructors championship?"),
        _tool_message(data, name="predict_constructor_championship"),
    ]
    assert extract_visuals(messages) == []


def test_extract_visuals_ignores_empty_tool_result():
    messages = [
        HumanMessage(content="What was the tire strategy for round 99?"),
        _tool_message([], name="get_tire_strategy"),
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
        {"type": "table", "tool": "get_race_results", "data": [{"Position": 2}]}
    ]


def test_extract_visuals_no_tool_messages():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert extract_visuals(messages) == []
