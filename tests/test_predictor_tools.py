import json
from unittest.mock import patch

from box_box_bot.tools.predictor_tools import PREDICTOR_TOOLS, predict_constructor_championship, predict_drivers_championship


def test_both_predictor_tools_are_registered():
    assert {t.name for t in PREDICTOR_TOOLS} == {"predict_constructor_championship", "predict_drivers_championship"}


def test_predict_constructor_championship_calls_predictor_and_returns_json():
    fake_data = {"predicted_orders": {"Monaco": ["Team A", "Team B"]}, "as_of_round": 5}
    with patch("box_box_bot.tools.predictor_tools.predictor.predict_constructor_championship", return_value=fake_data) as mock_fn:
        result = predict_constructor_championship.invoke({"season": 2026})

    mock_fn.assert_called_once_with(2026)
    assert json.loads(result) == fake_data


def test_predict_drivers_championship_calls_predictor_and_returns_json():
    fake_data = {"predicted_orders": {"Prost": ["Driver A", "Driver B"]}, "as_of_round": 5}
    with patch("box_box_bot.tools.predictor_tools.driver_predict.predict_drivers_championship", return_value=fake_data) as mock_fn:
        result = predict_drivers_championship.invoke({"season": 2026})

    mock_fn.assert_called_once_with(2026)
    assert json.loads(result) == fake_data
