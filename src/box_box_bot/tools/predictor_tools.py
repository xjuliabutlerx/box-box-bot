import json

from langchain_core.tools import tool

from box_box_bot.predictor import driver_predict
from box_box_bot.predictor import predict as predictor


@tool(parse_docstring=True)
def predict_constructor_championship(season: int) -> str:
    """Predict how the constructors' championship will likely finish, using 5 independently-trained ranking models.

    This runs 5 separate models (each from a different training run) and returns each one's own predicted order - they are NOT averaged into one answer. This is a MODEL PREDICTION based on the season's results so far, not a fact - only meaningful for the current, in-progress season. Predicting an already-completed season is pointless since the real result is already known; use the stats tools for that instead.

    Args:
        season: The four-digit F1 season year to predict, e.g. 2026. Should be the current in-progress season.
    """
    data = predictor.predict_constructor_championship(season)
    return json.dumps(data, default=str)


@tool(parse_docstring=True)
def predict_drivers_championship(season: int) -> str:
    """Predict how the drivers' championship will likely finish, using 3 independently-trained ranking models.

    This runs 3 separate models (each from a different training run) and returns each one's own predicted order - they are NOT averaged into one answer. This is a MODEL PREDICTION based on the season's results so far, not a fact - only meaningful for the current, in-progress season. Predicting an already-completed season is pointless since the real result is already known; use the stats tools for that instead.

    Args:
        season: The four-digit F1 season year to predict, e.g. 2026. Should be the current in-progress season.
    """
    data = driver_predict.predict_drivers_championship(season)
    return json.dumps(data, default=str)


PREDICTOR_TOOLS = [predict_constructor_championship, predict_drivers_championship]
