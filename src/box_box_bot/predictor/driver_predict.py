import threading

import pandas as pd
import torch
from torch import nn

from box_box_bot.predictor.driver_features import FEATURE_COLUMNS, get_driver_features
from box_box_bot.predictor.driver_model import DRIVER_MODEL_FILES, load_model

_models: dict[str, nn.Module] | None = None
_models_lock = threading.Lock()


def _get_models():
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                _models = {name: load_model(name) for name in DRIVER_MODEL_FILES}
    return _models


def _rank_by_model(model: nn.Module, features_df: pd.DataFrame) -> list[str]:
    X = torch.tensor(features_df[FEATURE_COLUMNS].values, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X).numpy()
    ranked = features_df.copy()
    ranked["PredictedRank"] = pd.Series(scores).rank(method="first", ascending=False)
    return ranked.sort_values("PredictedRank")["FullName"].tolist()


def predict_drivers_championship(season: int) -> dict:
    """Predicted drivers' championship order for a season, as of its
    latest completed round.

    Runs all 3 independently-trained models and returns each one's own
    predicted order - they are NOT averaged into one answer, matching how
    predict_constructor_championship handles its 5 models.

    Returns {"predicted_orders": {model_name: [driver names, best to
    worst], ...}, "as_of_round": round number}.
    """
    features = get_driver_features(season)
    latest_round = int(features["Round"].max())
    latest = features[features["Round"] == latest_round].reset_index(drop=True)

    predicted_orders = {name: _rank_by_model(model, latest) for name, model in _get_models().items()}

    return {
        "predicted_orders": predicted_orders,
        "as_of_round": latest_round,
    }
