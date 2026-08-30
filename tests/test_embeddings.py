import pytest

from box_box_bot.rag.embeddings import FastEmbedEmbeddings


@pytest.fixture(scope="module")
def embeddings():
    # Local ONNX model, no API key/network calls after the first download -
    # cheap enough to use directly rather than mock, shared across this
    # file's tests so the model only loads once.
    return FastEmbedEmbeddings()


def test_embed_documents_returns_one_vector_per_input(embeddings):
    vectors = embeddings.embed_documents(["hello world", "formula 1 racing", "a third document"])
    assert len(vectors) == 3


def test_embed_documents_returns_plain_float_lists(embeddings):
    vectors = embeddings.embed_documents(["Bahrain Grand Prix"])
    assert isinstance(vectors[0], list)
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_query_returns_single_vector(embeddings):
    vector = embeddings.embed_query("who won the championship")
    assert isinstance(vector, list)
    assert all(isinstance(x, float) for x in vector)


def test_embed_query_and_embed_documents_share_dimension(embeddings):
    doc_vector = embeddings.embed_documents(["test text"])[0]
    query_vector = embeddings.embed_query("test text")
    assert len(doc_vector) == len(query_vector)
