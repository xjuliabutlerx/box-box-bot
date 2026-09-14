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
        f'"this year," "the current season," or "now," they mean {today.year}. '
        f"If a question doesn't name a season/year at all, assume {today.year} "
        "for any tool call that needs one - a vague strategy/stats question "
        "means the current season, not a scan across several past ones. "
        "Don't call the same kind of tool for multiple different seasons "
        "just to build a 'general' or 'typical' answer. This doesn't apply "
        "to a tool whose whole purpose is looking across many seasons at "
        "once (career/all-time records, a circuit's historical safety-car "
        "pattern) - those are supposed to span years; everything else "
        "defaults to the single current season."
    )
