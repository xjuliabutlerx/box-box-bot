import threading

import pandas as pd
import torch

from box_box_bot.predictor.features import FEATURE_COLUMNS, get_team_features
from box_box_bot.predictor.model import CONSTRUCTOR_MODEL_FILES, F1ConstructorsClassifier, load_model

_models: dict[str, F1ConstructorsClassifier] | None = None
_models_lock = threading.Lock()


def _get_models():
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                _models = {name: load_model(filename) for name, filename in CONSTRUCTOR_MODEL_FILES.items()}
    return _models


def _rank_by_model(model, features_df: pd.DataFrame) -> list[str]:
    X = torch.tensor(features_df[FEATURE_COLUMNS].values, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X).numpy()

    ranked = features_df.copy()
    ranked["PredictedRank"] = pd.Series(scores).rank(method="first", ascending=False)
    return ranked.sort_values("PredictedRank")["TeamName"].tolist()


def predict_constructor_championship(season: int) -> dict:
    """Predicted constructor championship order for a season, as of its
    latest completed round - run through all 5 independently-trained v3
    checkpoints (each from a separate training run), not averaged.

    Returns {"predicted_orders": {model_name: [team names, best to
    worst], ...}, "as_of_round": round number}.
    """
    features = get_team_features(season)
    latest_round = int(features["Round"].max())
    latest = features[features["Round"] == latest_round].reset_index(drop=True)

    predicted_orders = {name: _rank_by_model(model, latest) for name, model in _get_models().items()}

    return {
        "predicted_orders": predicted_orders,
        "as_of_round": latest_round,
    }
