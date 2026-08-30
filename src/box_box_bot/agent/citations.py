import re

_SOURCE_PATTERN = re.compile(r"\[Source: (.+?) \((\d{4})\)\]")

def extract_citations(messages: list) -> list[dict]:
    """Pull structured citations out of the most recent turn's search_race_recaps tool results.

    We parse this out of the tool output itself rather than the model's final answer text, because the model might paraphrase or drop a mention - the tool result is ground truth for what was actually retrieved.
    """
    last_human_idx = max(i for i, m in enumerate(messages) if m.type == "human")
    turn_messages = messages[last_human_idx:]

    citations = []
    seen = set()
    for m in turn_messages:
        if m.type != "tool" or getattr(m, "name", None) != "search_race_recaps":
            continue
        for race_name, season in _SOURCE_PATTERN.findall(m.content):
            key = (race_name, season)
            if key not in seen:
                seen.add(key)
                citations.append({"race_name": race_name, "season": int(season)})

    return citations

def filter_citations_by_answer(citations: list[dict], answer_text: str) -> list[dict]:
    """Keep only citations whose race is actually named in the model's
    answer, so a retrieved-but-unused chunk (see rag/README's retrieval
    precision caveat) doesn't show up as a false citation.

    Matches on the race's short name ("Bahrain") rather than the full
    official name ("Bahrain Grand Prix") - live testing showed the model
    doesn't reliably use the full name, which made citations disappear
    nondeterministically even when the source was clearly used.
    """
    answer_lower = answer_text.lower()
    matched = []
    for c in citations:
        short_name = c["race_name"].lower().replace("grand prix", "").strip()
        if short_name in answer_lower:
            matched.append(c)
    return matched