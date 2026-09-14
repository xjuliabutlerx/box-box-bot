import threading

from langchain_core.tools import tool

from box_box_bot.rag.ingest import TRACK_INFO_COLLECTION_NAME
from box_box_bot.rag.retriever import get_retriever

_retriever = None
_retriever_lock = threading.Lock()

_track_retriever = None
_track_retriever_lock = threading.Lock()

def _get_retriever():
    # A multi-agent supervisor can issue parallel tool calls (e.g. two
    # search_race_recaps calls in the same turn), so this lazy singleton
    # needs a lock - two threads racing through the first `is None` check
    # both tried to construct the Chroma client for the same path
    # concurrently and hit a KeyError inside chromadb's client registry.
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = get_retriever()
    return _retriever

def _get_track_retriever():
    # Separate lazy singleton (and separate lock) from _get_retriever()
    # above - a different Chroma collection, so racing through it
    # concurrently with a race-recap query hits the same construction
    # race the lock above exists to prevent, just for a different client.
    global _track_retriever
    if _track_retriever is None:
        with _track_retriever_lock:
            if _track_retriever is None:
                _track_retriever = get_retriever(collection_name=TRACK_INFO_COLLECTION_NAME)
    return _track_retriever

@tool(parse_docstring=True)
def search_race_recaps(query: str) -> str:
    """Search narrative race recaps for context on why something happened (e.g. championship swings, controversies, driver/team storylings, comebacks).

    Use this for "why" or "what happened" questions; use the fastf1 tools instead for numeric facts like standings, results, or lap times.

    Args:
        query: A natural-language description of what you're looking for, e.g. "papaya rules controversy" or "Verstappen 2025 comeback"
    """
    docs = _get_retriever().invoke(query)
    if not docs:
        return "No relevant race recaps found."

    formatted = [
        f"[Source: {doc.metadata.get('race_name')} ({doc.metadata.get('season')})]\n"
        f"{doc.page_content}"
        for doc in docs
    ]
    return "\n\n".join(formatted)

@tool(parse_docstring=True)
def search_track_info(query: str) -> str:
    """Search circuit/track profiles for qualitative track characteristics - overtaking difficulty, tire degradation character, elevation, DRS zones, typical race strategy pattern.

    Use this for "what's this track like" / "why is [circuit] hard" / "how should teams approach strategy here" questions. This is static circuit knowledge, not live session data - use the other strategy tools (tire strategy, pit stops, safety car history) for what actually happened at a specific race.

    Args:
        query: A natural-language description of what you're looking for, e.g. "why is Monaco hard to overtake at" or "Monza tire strategy"
    """
    docs = _get_track_retriever().invoke(query)
    if not docs:
        return "No relevant track info found."

    formatted = [
        f"[Source: Track Info - {doc.metadata.get('circuit')}]\n"
        f"{doc.page_content}"
        for doc in docs
    ]
    return "\n\n".join(formatted)

RAG_TOOLS = [search_race_recaps]
TRACK_INFO_TOOLS = [search_track_info]