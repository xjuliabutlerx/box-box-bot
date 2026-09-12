from unittest.mock import patch

import pandas as pd
import torch

from box_box_bot.predictor import predict


def _fake_model(scores):
    def forward(x):
        return torch.tensor(scores, dtype=torch.float32)
    return forward


_FAKE_TEAM_FEATURES = pd.DataFrame([
    {"TeamId": "team_a", "TeamName": "Team A", "Round": 5, "Year": 2026, "hadPenaltyThisYear": 0},
    {"TeamId": "team_b", "TeamName": "Team B", "Round": 5, "Year": 2026, "hadPenaltyThisYear": 0},
])
for col in predict.FEATURE_COLUMNS:
    if col not in _FAKE_TEAM_FEATURES.columns:
        _FAKE_TEAM_FEATURES[col] = 0.0


def test_predict_constructor_championship_runs_all_five_models_independently():
    predict._models = None  # reset the module-level cache before this test

    fake_models = {
        "Monaco": _fake_model([1.0, 2.0]),  # Team B ranked first
        "Silverstone": _fake_model([2.0, 1.0]),  # Team A ranked first
    }

    with (
        patch("box_box_bot.predictor.predict.get_team_features", return_value=_FAKE_TEAM_FEATURES),
        patch("box_box_bot.predictor.predict._get_models", return_value=fake_models),
    ):
        result = predict.predict_constructor_championship(2026)

    assert result["as_of_round"] == 5
    assert set(result["predicted_orders"].keys()) == {"Monaco", "Silverstone"}
    assert result["predicted_orders"]["Monaco"] == ["Team B", "Team A"]
    assert result["predicted_orders"]["Silverstone"] == ["Team A", "Team B"]


def test_get_models_loads_and_caches_all_five_checkpoints():
    predict._models = None
    with patch("box_box_bot.predictor.predict.load_model") as mock_load:
        mock_load.side_effect = lambda filename: filename  # cheap sentinel, avoids real torch load

        first = predict._get_models()
        call_count_after_first = mock_load.call_count
        second = predict._get_models()

    assert set(first.keys()) == set(predict.CONSTRUCTOR_MODEL_FILES.keys())
    assert mock_load.call_count == call_count_after_first  # second call served from cache
    assert second is first
