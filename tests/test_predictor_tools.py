import json
from unittest.mock import patch

from box_box_bot.tools.predictor_tools import PREDICTOR_TOOLS, predict_constructor_championship


def test_predictor_tool_is_registered():
    assert {t.name for t in PREDICTOR_TOOLS} == {"predict_constructor_championship"}


def test_predict_constructor_championship_calls_predictor_and_returns_json():
    fake_data = {"predicted_order": ["Team A", "Team B"], "as_of_round": 5, "model": "monaco_v3"}
    with patch("box_box_bot.tools.predictor_tools.predictor.predict_constructor_championship", return_value=fake_data) as mock_fn:
        result = predict_constructor_championship.invoke({"season": 2026})

    mock_fn.assert_called_once_with(2026)
    assert json.loads(result) == fake_data
