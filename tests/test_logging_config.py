import logging
import os
from unittest.mock import patch

from box_box_bot.logging_config import configure_logging


def _reset():
    logger = logging.getLogger("box_box_bot")
    logger.handlers.clear()


def test_configure_logging_attaches_a_handler():
    _reset()
    configure_logging()

    logger = logging.getLogger("box_box_bot")
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_configure_logging_is_idempotent_across_streamlit_reruns():
    # Streamlit reruns the whole script on every interaction - a second
    # call (as would happen on every rerun) must not pile up handlers,
    # or every log line would print once per accumulated handler.
    _reset()
    configure_logging()
    configure_logging()
    configure_logging()

    assert len(logging.getLogger("box_box_bot").handlers) == 1


def test_configure_logging_leaves_fastf1_at_default_level_when_not_opted_in():
    # fastf1's DEBUG level is noisy well beyond the swallowed-failure
    # tracebacks it's meant to surface (e.g. one traceback per driver for
    # a single failed lookup) - it must stay off unless explicitly opted
    # into via FASTF1_DEBUG_LOGGING, not always-on.
    _reset()
    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("FASTF1_DEBUG_LOGGING", None)
        with patch("box_box_bot.logging_config.fastf1.set_log_level") as mock_set_level:
            configure_logging()

    mock_set_level.assert_not_called()


def test_configure_logging_raises_fastf1_log_level_to_debug_when_opted_in():
    # Must go through fastf1's own public fastf1.set_log_level(), not by
    # reaching into logging.getLogger("fastf1").handlers directly -
    # fastf1 builds its console handler lazily on first import, so a
    # handler found by name may not exist yet depending on import order.
    _reset()
    with patch.dict("os.environ", {"FASTF1_DEBUG_LOGGING": "true"}):
        with patch("box_box_bot.logging_config.fastf1.set_log_level") as mock_set_level:
            configure_logging()

    mock_set_level.assert_called_once_with(logging.DEBUG)
