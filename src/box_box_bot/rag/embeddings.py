"""Adapts fastembed's local ONNX embedding model to LangChain's `Embeddings`
interface.

We use fastembed instead of `langchain_community.embeddings.FastEmbedEmbeddings`
because langchain-community is being sunset upstream in favor of small,
standalone integration packages. There isn't a standalone one for fastembed
yet, so this ~15-line adapter is the more future-proof option: any vector
store or retriever that accepts a LangChain `Embeddings` object accepts this
one, and nothing outside this file needs to know embeddings are local/ONNX
rather than an API call.
"""

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

# Small (~130MB), runs on CPU, no API key or network calls after first download.
_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # fastembed's .embed() returns a lazy generator of numpy arrays;
        # LangChain's interface expects a fully materialized list of
        # plain-Python-float lists (numpy arrays aren't JSON/pickle friendly
        # the way Chroma expects when it persists vectors to disk).
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
