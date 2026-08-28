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

from box_box_bot.config import RACE_RECAPS_DIR, RAG_PERSIST_DIR
from box_box_bot.rag.embeddings import FastEmbedEmbeddings

COLLECTION_NAME = "race_recaps"


def _load_documents() -> list[Document]:
    """Read every recap .md file into a LangChain Document, with the YAML
    frontmatter (season/round/race_name/date) attached as metadata.

    That metadata is what step 5's citations will point back to: once a
    chunk is retrieved, its `.metadata["race_name"]` etc. tells us which
    source document an answer came from.
    """
    documents = []
    for path in sorted(RACE_RECAPS_DIR.glob("*.md")):
        if path.name == "README.md":
            continue

        text = path.read_text()
        _, frontmatter_block, body = text.split("---", 2)
        metadata = yaml.safe_load(frontmatter_block)
        # PyYAML parses "date: 2025-03-16" into a datetime.date object, but
        # Chroma's metadata store only accepts str/int/float/bool/None -
        # stringify it rather than let Chroma reject the whole document.
        metadata["date"] = str(metadata["date"])
        metadata["source"] = path.name

        documents.append(Document(page_content=body.strip(), metadata=metadata))
    return documents


def build_vectorstore() -> Chroma:
    documents = _load_documents()

    # chunk_size is in characters, not tokens; these recaps run ~1,300-1,600
    # characters each, so this splits each into roughly 3 overlapping chunks
    # rather than embedding whole documents as single vectors. Overlap keeps
    # a sentence that got cut at a chunk boundary readable in both halves.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        chunks,
        embedding=FastEmbedEmbeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(RAG_PERSIST_DIR),
    )


if __name__ == "__main__":
    vectorstore = build_vectorstore()
    count = vectorstore._collection.count()
    print(f"Indexed {count} chunks from {RACE_RECAPS_DIR} into {RAG_PERSIST_DIR}")
