import json

CHART_TOOLS = {
    "get_tire_strategy": "tire_strategy_chart",
    "get_circuit_speed_map": "track_map",
}

NESTED_TABLE_KEYS = {
    "get_circuit_strategy_history": "by_season",
}


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

    visuals = []
    for m in turn_messages:
        if m.type != "tool":
            continue
        name = getattr(m, "name", None)

        try:
            data = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue

        if name in CHART_TOOLS:
            if data:
                visuals.append({"type": CHART_TOOLS[name], "tool": name, "data": data})
            continue

        if name in NESTED_TABLE_KEYS:
            data = data.get(NESTED_TABLE_KEYS[name], []) if isinstance(data, dict) else None

        if isinstance(data, list) and data and isinstance(data[0], dict):
            visuals.append({"type": "table", "tool": name, "data": data})

    return visuals
