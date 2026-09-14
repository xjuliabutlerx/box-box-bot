"""Builds the Chroma vector store from data/race_recaps/*.md.

This is a rebuild script, not something the running app calls on every
request: run it once (`python -m box_box_bot.rag.ingest`) whenever the
recap corpus changes, and `retriever.py` just opens the persisted result.
Separating "build the index" from "query the index" keeps startup fast and
mirrors how you'd run this in a real pipeline (an offline/batch indexing
job, separate from the online serving path).
"""

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from box_box_bot.config import RACE_RECAPS_DIR, RAG_PERSIST_DIR, TRACK_INFO_DIR
from box_box_bot.rag.embeddings import FastEmbedEmbeddings

COLLECTION_NAME = "race_recaps"
TRACK_INFO_COLLECTION_NAME = "track_info"


def _load_documents_from(source_dir) -> list[Document]:
    """Read every .md file in source_dir into a LangChain Document, with
    its YAML frontmatter attached as metadata - whatever fields a given
    corpus's frontmatter happens to have (race recaps: season/round/
    race_name/date; track_info: circuit/location/country - no fixed
    schema is assumed here).

    That metadata is what step 5's citations will point back to: once a
    chunk is retrieved, its metadata tells us which source document an
    answer came from.
    """
    documents = []
    for path in sorted(source_dir.glob("*.md")):
        if path.name == "README.md":
            continue

        text = path.read_text()
        _, frontmatter_block, body = text.split("---", 2)
        metadata = yaml.safe_load(frontmatter_block)
        # PyYAML parses "date: 2025-03-16" into a datetime.date object, but
        # Chroma's metadata store only accepts str/int/float/bool/None -
        # stringify it rather than let Chroma reject the whole document.
        # Not every corpus has a date field (track_info doesn't), so this
        # is conditional rather than assumed.
        if "date" in metadata:
            metadata["date"] = str(metadata["date"])
        metadata["source"] = path.name

        documents.append(Document(page_content=body.strip(), metadata=metadata))
    return documents


def _load_documents() -> list[Document]:
    return _load_documents_from(RACE_RECAPS_DIR)


def _build_collection(documents: list[Document], collection_name: str) -> Chroma:
    # chunk_size is in characters, not tokens; these documents run roughly
    # 1,300-1,600 characters each, so this splits each into a few
    # overlapping chunks rather than embedding whole documents as single
    # vectors. Overlap keeps a sentence that got cut at a chunk boundary
    # readable in both halves.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Chroma.from_documents() ADDS to an existing collection rather than
    # replacing it - re-running this script against a persist_directory
    # that already has data from a prior run (the normal case: this is a
    # rebuild script you run again after editing the corpus) would
    # silently duplicate every chunk on top of whatever was already
    # there, growing unbounded and degrading retrieval (duplicate chunks
    # crowd out k's limited result slots with redundant copies of the
    # same content). Deleting first makes a rebuild idempotent instead of
    # cumulative. Safe to call even when the collection doesn't exist yet
    # (a first-ever run) - it's a no-op in that case, not an error.
    embeddings = FastEmbedEmbeddings()
    Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(RAG_PERSIST_DIR),
    ).delete_collection()

    return Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(RAG_PERSIST_DIR),
    )


def build_vectorstore() -> Chroma:
    return _build_collection(_load_documents(), COLLECTION_NAME)


def build_track_info_vectorstore() -> Chroma:
    return _build_collection(_load_documents_from(TRACK_INFO_DIR), TRACK_INFO_COLLECTION_NAME)


if __name__ == "__main__":
    vectorstore = build_vectorstore()
    count = vectorstore._collection.count()
    print(f"Indexed {count} chunks from {RACE_RECAPS_DIR} into {RAG_PERSIST_DIR} (collection: {COLLECTION_NAME})")

    track_vectorstore = build_track_info_vectorstore()
    track_count = track_vectorstore._collection.count()
    print(f"Indexed {track_count} chunks from {TRACK_INFO_DIR} into {RAG_PERSIST_DIR} (collection: {TRACK_INFO_COLLECTION_NAME})")
