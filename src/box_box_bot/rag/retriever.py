"""Opens the persisted Chroma vector store and hands back a retriever.

Run `python -m box_box_bot.rag.ingest` at least once before calling this —
it expects `RAG_PERSIST_DIR` to already contain a populated collection.
"""

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from box_box_bot.config import RAG_PERSIST_DIR
from box_box_bot.rag.embeddings import FastEmbedEmbeddings
from box_box_bot.rag.ingest import COLLECTION_NAME


def get_retriever(k: int = 4, collection_name: str = COLLECTION_NAME) -> VectorStoreRetriever:
    """k is how many chunks come back per query. 4 is a starting point:
    enough for the agent to synthesize an answer across 1-2 source races,
    small enough to keep the model's context (and the LangSmith trace)
    readable. Tune it once you see real queries come back too thin or too
    noisy.

    collection_name defaults to the race-recaps collection; pass
    ingest.TRACK_INFO_COLLECTION_NAME for the track-info collection -
    both live in the same persist_directory (Chroma supports multiple
    named collections per directory), kept separate so a query for one
    corpus never pulls in the other's chunks.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=FastEmbedEmbeddings(),
        persist_directory=str(RAG_PERSIST_DIR),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
