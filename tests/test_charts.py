from box_box_bot.app import charts


def test_build_table_returns_dataframe_with_expected_columns():
    data = [{"Position": 1, "Abbreviation": "PIA"}, {"Position": 2, "Abbreviation": "NOR"}]
    df = charts.build_table(data)
    assert list(df.columns) == ["Position", "Abbreviation"]
    assert len(df) == 2


def test_build_tire_strategy_figure_one_trace_per_compound():
    data = [
        {"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 10},
        {"Driver": "VER", "Stint": 2, "Compound": "MEDIUM", "StintLength": 15},
        {"Driver": "NOR", "Stint": 1, "Compound": "SOFT", "StintLength": 8},
    ]
    fig = charts.build_tire_strategy_figure(data)

    trace_names = {trace.name for trace in fig.data}
    assert trace_names == {"SOFT", "MEDIUM"}


def test_build_tire_strategy_figure_stint_lengths_sum_correctly_per_driver():
    data = [
        {"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 10},
        {"Driver": "VER", "Stint": 2, "Compound": "MEDIUM", "StintLength": 15},
    ]
    fig = charts.build_tire_strategy_figure(data)

    soft_trace = next(t for t in fig.data if t.name == "SOFT")
    medium_trace = next(t for t in fig.data if t.name == "MEDIUM")
    assert list(soft_trace.x) == [10]
    assert list(soft_trace.base) == [0]
    assert list(medium_trace.x) == [15]
    assert list(medium_trace.base) == [10]  # starts right after the SOFT stint ends


def test_build_tire_strategy_figure_uses_known_compound_colors():
    data = [{"Driver": "VER", "Stint": 1, "Compound": "SOFT", "StintLength": 10}]
    fig = charts.build_tire_strategy_figure(data)
    assert fig.data[0].marker.color == charts.COMPOUND_COLORS["SOFT"]


def test_build_tire_strategy_figure_unknown_compound_gets_fallback_color():
    data = [{"Driver": "VER", "Stint": 1, "Compound": "UNKNOWN_COMPOUND", "StintLength": 10}]
    fig = charts.build_tire_strategy_figure(data)
    assert fig.data[0].marker.color == charts.UNKNOWN_COMPOUND_COLOR


def test_build_track_map_figure_point_count_matches_input():
    data = {
        "circuit": "Monaco",
        "driver": "VER",
        "lap_time": "0 days 00:01:12",
        "points": [{"X": 1.0, "Y": 2.0, "Speed": 250.0}, {"X": 3.0, "Y": 4.0, "Speed": 260.0}],
        "corners": [{"Number": 1, "Letter": "", "X": 1.0, "Y": 2.0}],
    }
    fig = charts.build_track_map_figure(data)

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [1.0, 3.0]
    assert list(fig.data[0].y) == [2.0, 4.0]
    assert list(fig.data[0].marker.color) == [250.0, 260.0]
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "1"


def test_build_track_map_figure_handles_no_corners():
    data = {"circuit": "Monaco", "driver": "VER", "lap_time": "0:01:12", "points": [{"X": 1.0, "Y": 2.0, "Speed": 100.0}]}
    fig = charts.build_track_map_figure(data)
    assert len(fig.layout.annotations) == 0
