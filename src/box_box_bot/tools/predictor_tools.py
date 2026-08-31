import json

from langchain_core.tools import tool

from box_box_bot.predictor import predict as predictor


@tool(parse_docstring=True)
def predict_constructor_championship(season: int) -> str:
    """Predict how the constructors' championship will likely finish, using a trained ranking model.

    This is a MODEL PREDICTION based on the season's results so far, not a fact - only meaningful for the current, in-progress season. Predicting an already-completed season is pointless since the real result is already known; use the stats tools for that instead.

    Args:
        season: The four-digit F1 season year to predict, e.g. 2026. Should be the current in-progress season.
    """
    data = predictor.predict_constructor_championship(season)
    return json.dumps(data, default=str)


PREDICTOR_TOOLS = [predict_constructor_championship]
