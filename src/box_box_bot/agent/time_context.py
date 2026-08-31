import datetime


def current_date_context() -> str:
    """One line stating today's date, computed fresh on every call.

    Agents are built once and cached for the life of the server process
    (see app/streamlit_app.py), so this must never be baked into a
    static prompt string at build time - it has to be recomputed on
    every model call, or it goes stale the longer the process runs.
    """
    today = datetime.date.today()
    return (
        f"Today's date is {today.isoformat()}. When the user says "
        f'"this year," "the current season," or "now," they mean {today.year}.'
    )
