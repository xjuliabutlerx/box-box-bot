import re

_RACE_SOURCE_PATTERN = re.compile(r"\[Source: (.+?) \((\d{4})\)\]")
_TRACK_SOURCE_PATTERN = re.compile(r"\[Source: Track Info - (.+?)\]")

# Generic circuit-naming words stripped before matching a circuit name
# against the model's answer text - same reasoning as the "grand prix"
# strip below, just without a single consistent suffix to strip, since
# circuit names use different formal conventions ("Circuit de Monaco",
# "Autodromo Nazionale Monza", "Marina Bay Street Circuit"...).
_CIRCUIT_STOPWORDS = {
    "circuit", "de", "international", "autodromo", "nazionale", "street",
    "raceway", "park", "national", "track", "grand", "prix",
}


def extract_citations(messages: list) -> list[dict]:
    """Pull structured citations out of the most recent turn's RAG tool
    results (search_race_recaps, search_track_info).

    We parse this out of the tool output itself rather than the model's
    final answer text, because the model might paraphrase or drop a
    mention - the tool result is ground truth for what was actually
    retrieved. The two tools tag their sources differently (race recaps
    carry a season; track info doesn't, since circuits aren't season-
    specific), so each citation carries a "type" field telling the
    caller which shape ("race": race_name/season, "track": circuit) a
    given entry is.
    """
    last_human_idx = max(i for i, m in enumerate(messages) if m.type == "human")
    turn_messages = messages[last_human_idx:]

    citations = []
    seen = set()
    for m in turn_messages:
        if m.type != "tool":
            continue

        if getattr(m, "name", None) == "search_race_recaps":
            for race_name, season in _RACE_SOURCE_PATTERN.findall(m.content):
                key = ("race", race_name, season)
                if key not in seen:
                    seen.add(key)
                    citations.append({"type": "race", "race_name": race_name, "season": int(season)})
        elif getattr(m, "name", None) == "search_track_info":
            for circuit in _TRACK_SOURCE_PATTERN.findall(m.content):
                key = ("track", circuit)
                if key not in seen:
                    seen.add(key)
                    citations.append({"type": "track", "circuit": circuit})

    return citations


def filter_citations_by_answer(citations: list[dict], answer_text: str) -> list[dict]:
    """Keep only citations whose source is actually named in the model's
    answer, so a retrieved-but-unused chunk (see rag/README's retrieval
    precision caveat) doesn't show up as a false citation.

    Races match on their short name ("Bahrain") rather than the full
    official name ("Bahrain Grand Prix") - live testing showed the model
    doesn't reliably use the full name, which made citations disappear
    nondeterministically even when the source was clearly used. Circuits
    get the same treatment via _CIRCUIT_STOPWORDS: a model is far more
    likely to say "Monza" than "Autodromo Nazionale Monza," so any
    significant (non-stopword) word from the circuit name is enough to
    count as a match, not the full formal name.
    """
    answer_lower = answer_text.lower()
    matched = []
    for c in citations:
        if c["type"] == "race":
            short_name = c["race_name"].lower().replace("grand prix", "").strip()
            if short_name in answer_lower:
                matched.append(c)
        elif c["type"] == "track":
            words = [w.strip("()") for w in c["circuit"].lower().split()]
            significant_words = [w for w in words if w not in _CIRCUIT_STOPWORDS and len(w) > 2]
            if any(word in answer_lower for word in significant_words):
                matched.append(c)
    return matched
