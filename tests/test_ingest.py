import textwrap
from unittest.mock import patch

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


def test_build_vectorstore_is_idempotent_on_repeat_runs(recaps_dir, tmp_path, monkeypatch):
    # Regression test: Chroma.from_documents() adds to an existing
    # collection rather than replacing it - found live when re-running
    # ingest.py against an already-populated persist_directory (the
    # normal "rebuild after editing the corpus" case) silently doubled
    # the chunk count instead of leaving it unchanged. Re-running the
    # build against the same source files must produce the same count,
    # not grow it.
    monkeypatch.setattr(ingest, "RAG_PERSIST_DIR", tmp_path / "vectorstore")

    first_count = ingest.build_vectorstore()._collection.count()
    second_count = ingest.build_vectorstore()._collection.count()

    assert second_count == first_count


def test_ensure_vectorstore_built_builds_when_persist_dir_missing(recaps_dir, tmp_path, monkeypatch):
    # Regression test: RAG_PERSIST_DIR is gitignored, so a fresh deploy
    # (Streamlit Cloud checks out a clean git tree every time) starts
    # with no vector store at all - Chroma doesn't raise for that, it
    # silently creates an empty one, so every narrative query returned
    # zero results with no error. This is what makes a fresh deploy
    # self-healing instead of silently empty.
    persist_dir = tmp_path / "vectorstore"
    monkeypatch.setattr(ingest, "RAG_PERSIST_DIR", persist_dir)
    monkeypatch.setattr(ingest, "TRACK_INFO_DIR", recaps_dir)
    assert not persist_dir.exists()

    ingest.ensure_vectorstore_built()

    assert persist_dir.exists()


def test_ensure_vectorstore_built_is_a_noop_when_already_present(tmp_path, monkeypatch):
    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()
    monkeypatch.setattr(ingest, "RAG_PERSIST_DIR", persist_dir)

    with patch("box_box_bot.rag.ingest.build_vectorstore") as mock_build, \
         patch("box_box_bot.rag.ingest.build_track_info_vectorstore") as mock_build_track:
        ingest.ensure_vectorstore_built()

    mock_build.assert_not_called()
    mock_build_track.assert_not_called()
