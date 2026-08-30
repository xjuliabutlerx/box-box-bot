import textwrap

import pytest

from box_box_bot.rag import ingest

FIXTURE_DOC = textwrap.dedent(
    """\
    ---
    season: 2025
    round: 4
    race_name: Bahrain Grand Prix
    date: 2025-04-13
    ---

    # 2025 Bahrain Grand Prix

    Piastri won from pole, taking over the championship lead.
    """
)


@pytest.fixture
def recaps_dir(tmp_path, monkeypatch):
    (tmp_path / "2025_r04_bahrain_gp.md").write_text(FIXTURE_DOC)
    (tmp_path / "README.md").write_text("# corpus readme, should be skipped by ingestion")
    monkeypatch.setattr(ingest, "RACE_RECAPS_DIR", tmp_path)
    return tmp_path


def test_load_documents_parses_frontmatter_metadata(recaps_dir):
    docs = ingest._load_documents()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["season"] == 2025
    assert doc.metadata["round"] == 4
    assert doc.metadata["race_name"] == "Bahrain Grand Prix"
    assert doc.metadata["source"] == "2025_r04_bahrain_gp.md"


def test_load_documents_stringifies_date(recaps_dir):
    # Regression test: PyYAML parses "date: 2025-04-13" into a
    # datetime.date object, and Chroma's metadata store only accepts
    # str/int/float/bool/None - it rejected the whole document outright
    # until this got stringified.
    docs = ingest._load_documents()
    assert isinstance(docs[0].metadata["date"], str)
    assert docs[0].metadata["date"] == "2025-04-13"


def test_load_documents_skips_readme(recaps_dir):
    docs = ingest._load_documents()
    assert all(doc.metadata["source"] != "README.md" for doc in docs)
    assert len(docs) == 1


def test_load_documents_body_excludes_frontmatter(recaps_dir):
    docs = ingest._load_documents()
    assert "season:" not in docs[0].page_content
    assert "date:" not in docs[0].page_content
    assert "Piastri won" in docs[0].page_content
    assert docs[0].page_content.startswith("# 2025 Bahrain Grand Prix")


def test_build_vectorstore_embeds_and_persists(recaps_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAG_PERSIST_DIR", tmp_path / "vectorstore")

    vectorstore = ingest.build_vectorstore()

    assert vectorstore._collection.count() >= 1
    results = vectorstore.similarity_search("Bahrain championship lead", k=1)
    assert results[0].metadata["race_name"] == "Bahrain Grand Prix"
