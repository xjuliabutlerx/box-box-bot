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
    """Keep only citations whose race is actually named in the model's answer, so a retrieved-but-unused chunk doesn't show up as a false citation.s
    """
    answer_lower = answer_text.lower()
    return [c for c in citations if c["race_name"].lower() in answer_lower]