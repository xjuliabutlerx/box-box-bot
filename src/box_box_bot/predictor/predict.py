import threading

import pandas as pd
import torch

from box_box_bot.predictor.features import FEATURE_COLUMNS, get_team_features
from box_box_bot.predictor.model import load_model

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_model()
    return _model


def predict_constructor_championship(season: int) -> dict:
    """Predicted constructor championship order for a season, as of its
    latest completed round.

    Returns {"predicted_order": [team names, best to worst],
    "as_of_round": round number, "model": model identifier}.
    """
    features = get_team_features(season)
    latest_round = int(features["Round"].max())
    latest = features[features["Round"] == latest_round].reset_index(drop=True)

    X = torch.tensor(latest[FEATURE_COLUMNS].values, dtype=torch.float32)
    with torch.no_grad():
        scores = _get_model()(X).numpy()

    latest = latest.copy()
    latest["PredictedRank"] = pd.Series(scores).rank(method="first", ascending=False)
    ordered = latest.sort_values("PredictedRank")["TeamName"].tolist()

    return {
        "predicted_order": ordered,
        "as_of_round": latest_round,
        "model": "monaco_v3",
    }
