from unittest.mock import patch

import pandas as pd
import torch

from box_box_bot.predictor import driver_predict


def _fake_model(scores):
    def forward(x):
        return torch.tensor(scores, dtype=torch.float32)
    return forward


_FAKE_DRIVER_FEATURES = pd.DataFrame([
    {"DriverId": "driver_a", "FullName": "Driver A", "Round": 5},
    {"DriverId": "driver_b", "FullName": "Driver B", "Round": 5},
])
for col in driver_predict.FEATURE_COLUMNS:
    if col not in _FAKE_DRIVER_FEATURES.columns:
        _FAKE_DRIVER_FEATURES[col] = 0.0


def test_predict_drivers_championship_runs_all_three_models_independently():
    driver_predict._models = None  # reset the module-level cache before this test

    fake_models = {
        "Prost": _fake_model([1.0, 2.0]),  # Driver B ranked first
        "Schumacher": _fake_model([2.0, 1.0]),  # Driver A ranked first
    }

    with (
        patch("box_box_bot.predictor.driver_predict.get_driver_features", return_value=_FAKE_DRIVER_FEATURES),
        patch("box_box_bot.predictor.driver_predict._get_models", return_value=fake_models),
    ):
        result = driver_predict.predict_drivers_championship(2026)

    assert result["as_of_round"] == 5
    assert set(result["predicted_orders"].keys()) == {"Prost", "Schumacher"}
    assert result["predicted_orders"]["Prost"] == ["Driver B", "Driver A"]
    assert result["predicted_orders"]["Schumacher"] == ["Driver A", "Driver B"]


def test_get_models_loads_and_caches_all_three_checkpoints():
    driver_predict._models = None
    with patch("box_box_bot.predictor.driver_predict.load_model") as mock_load:
        mock_load.side_effect = lambda name: name  # cheap sentinel, avoids real torch load

        first = driver_predict._get_models()
        call_count_after_first = mock_load.call_count
        second = driver_predict._get_models()

    assert set(first.keys()) == set(driver_predict.DRIVER_MODEL_FILES.keys())
    assert mock_load.call_count == call_count_after_first  # second call served from cache
    assert second is first
