from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

from box_box_bot.agent.time_context import current_date_context
from box_box_bot.tools.predictor_tools import PREDICTOR_TOOLS

PREDICTOR_SYSTEM_PROMPT = """You are box-box-bot's predictor specialist.

Predict how the constructors' championship is likely to finish using
your prediction tool. This tool runs a trained machine learning model -
always present its output as a model's prediction, not a fact or
certainty. State which round the prediction is based on (the tool tells
you this). Never claim to know who will actually win the championship -
only what the model currently projects based on the season so far.
"""


@dynamic_prompt
def _predictor_prompt(request) -> str:
    return f"{current_date_context()}\n\n{PREDICTOR_SYSTEM_PROMPT}"


def build_predictor_agent(model):
    return create_agent(
        model,
        PREDICTOR_TOOLS,
        middleware=[_predictor_prompt],
        name="predictor_agent",
    )
