import fastf1
import logging
import os

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

def configure_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("box_box_bot")
    if logger.handlers:
        # Streamlit reruns the whole script on every interaction - without
        # this guard, every rerun would add another handler and each log
        # line would print once per accumulated handler.
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    # fastf1's DEBUG level is noisy well beyond the swallowed-failure
    # tracebacks this is meant to surface - e.g. a downstream Ergast
    # lookup failing produces one such traceback per driver in a session,
    # not just once. Only worth the clutter while actively diagnosing a
    # specific issue, so it's opt-in (same off-by-default,
    # flip-via-secrets pattern as REQUIRE_PASSWORD) rather than always on.
    if os.environ.get("FASTF1_DEBUG_LOGGING", "false").strip().lower() == "true":
        fastf1.set_log_level(logging.DEBUG)
