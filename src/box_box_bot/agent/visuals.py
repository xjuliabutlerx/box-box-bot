import json

CHART_TOOLS = {
    "get_tire_strategy": "tire_strategy_chart",
    "get_circuit_speed_map": "track_map",
}

NESTED_TABLE_KEYS = {
    "get_circuit_strategy_history": "by_season",
}


def _build_label(name: str, call_args: dict) -> str:
    """A short, human-readable caption for one visual, built from the
    tool CALL's arguments (season/round/circuit/...), not its output -
    when a turn produces several visuals from the same tool (e.g. race
    results pulled for several different seasons), this is what lets a
    viewer tell them apart at all, instead of an unlabeled stack of
    identical-looking tables.
    """
    descriptor_parts = []
    for key in ("round", "circuit"):
        if call_args.get(key) is not None:
            descriptor_parts.append(str(call_args[key]))
    if call_args.get("season") is not None:
        descriptor_parts.append(str(call_args["season"]))
    if call_args.get("since_season") is not None:
        descriptor_parts.append(f"since {call_args['since_season']}")
    session_type = call_args.get("session_type")
    if session_type and session_type != "R":
        descriptor_parts.append(str(session_type))

    descriptor = " ".join(descriptor_parts)
    pretty_name = name.replace("_", " ")
    return f"{descriptor} — {pretty_name}" if descriptor else pretty_name


def _unwrap_fallback(data):
    """A not-yet-happened session (tools/fastf1_tools.py's
    _catch_fastf1_errors) automatically retries one season back and
    wraps a successful fallback as {"fallback_note", "season_used",
    "result"} instead of the tool's normal shape, so the agent can see
    and disclose it. Unwrap to the normal `result` for the usual
    chart/table logic below, and return the season actually shown so
    the label reflects reality (2025, say) instead of the season the
    agent originally asked for (2026) - a stale label would otherwise
    misrepresent a fallback as this year's data.
    """
    if isinstance(data, dict) and "fallback_note" in data and "result" in data:
        return data["result"], data.get("season_used")
    return data, None


def _is_valid_chart_data(name: str, data) -> bool:
    """A failed tool call also returns a truthy JSON value -
    `{"error": "..."}` (see tools/fastf1_tools.py's _catch_fastf1_errors)
    - which `if data:` alone can't tell apart from real chart data. Each
    chart tool has a distinct expected shape, so check for that shape
    specifically rather than just non-emptiness; an error dict fails
    both checks below and is correctly dropped instead of being handed
    to a chart builder that assumes real fields exist and crashes.
    """
    if name == "get_tire_strategy":
        return isinstance(data, list) and bool(data) and isinstance(data[0], dict)
    if name == "get_circuit_speed_map":
        return isinstance(data, dict) and isinstance(data.get("points"), list)
    return False


def extract_visuals(messages: list) -> list[dict]:
    """Pull structured, renderable data out of this turn's tool results
    so the UI can show a real table or chart instead of the model
    re-describing numbers in prose.

    Parsed from the tool's own JSON output (ground truth), not the
    model's answer text - same reasoning as extract_citations. Unlike
    citations, this doesn't filter by whether the answer text mentions
    it: every tool call this turn is a deliberate, targeted lookup (not
    RAG's noisier top-k retrieval), so its data is always worth showing
    even if the model's prose only narrates part of it.

    Most tools already return list[dict] JSON, which becomes a generic
    table with no per-tool allowlist to maintain. Two tools override
    that with a specific chart type (CHART_TOOLS); one returns a dict
    whose table content lives under a nested key (NESTED_TABLE_KEYS).
    Anything else (prose like search_race_recaps, or a non-row-shaped
    dict like the predictor tools' per-model order dict) is skipped.
    """
    last_human_idx = max(i for i, m in enumerate(messages) if m.type == "human")
    turn_messages = messages[last_human_idx:]

    # A ToolMessage only carries the tool's OUTPUT, not the arguments it
    # was called with - those live on the AIMessage that requested it,
    # linked by tool_call_id. Needed to label each visual (see
    # _build_label) so multiple visuals from the same tool are
    # distinguishable instead of an unlabeled stack of lookalike tables.
    call_args_by_id = {}
    for m in turn_messages:
        if m.type == "ai":
            for call in getattr(m, "tool_calls", None) or []:
                call_args_by_id[call["id"]] = call.get("args", {})

    visuals = []
    for m in turn_messages:
        if m.type != "tool":
            continue
        name = getattr(m, "name", None)
        call_args = call_args_by_id.get(getattr(m, "tool_call_id", None), {})

        try:
            data = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue

        data, fallback_season = _unwrap_fallback(data)
        if fallback_season is not None:
            call_args = {**call_args, "season": fallback_season}
        label = _build_label(name, call_args) if name else None

        if name in CHART_TOOLS:
            if _is_valid_chart_data(name, data):
                visuals.append({"type": CHART_TOOLS[name], "tool": name, "data": data, "label": label})
            continue

        if name in NESTED_TABLE_KEYS:
            data = data.get(NESTED_TABLE_KEYS[name], []) if isinstance(data, dict) else None

        if isinstance(data, list) and data and isinstance(data[0], dict):
            visuals.append({"type": "table", "tool": name, "data": data, "label": label})

    return visuals
