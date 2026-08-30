from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from box_box_bot.tools.rag_tools import RAG_TOOLS, search_race_recaps


def _fake_doc(race_name: str, season: int, content: str = "some recap text") -> Document:
    return Document(page_content=content, metadata={"race_name": race_name, "season": season})


def test_search_race_recaps_is_registered():
    assert RAG_TOOLS == [search_race_recaps]


def test_search_race_recaps_formats_source_tag():
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [_fake_doc("Bahrain Grand Prix", 2025, "Piastri won.")]

    with patch("box_box_bot.tools.rag_tools._get_retriever", return_value=fake_retriever):
        result = search_race_recaps.invoke({"query": "who took the championship lead"})

    fake_retriever.invoke.assert_called_once_with("who took the championship lead")
    assert result == "[Source: Bahrain Grand Prix (2025)]\nPiastri won."


def test_search_race_recaps_formats_multiple_documents():
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [
        _fake_doc("Italian Grand Prix", 2025, "Monza recap."),
        _fake_doc("Singapore Grand Prix", 2025, "Singapore recap."),
    ]

    with patch("box_box_bot.tools.rag_tools._get_retriever", return_value=fake_retriever):
        result = search_race_recaps.invoke({"query": "papaya rules"})

    assert "[Source: Italian Grand Prix (2025)]\nMonza recap." in result
    assert "[Source: Singapore Grand Prix (2025)]\nSingapore recap." in result


def test_search_race_recaps_handles_no_results():
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = []

    with patch("box_box_bot.tools.rag_tools._get_retriever", return_value=fake_retriever):
        result = search_race_recaps.invoke({"query": "something totally unrelated"})

    assert result == "No relevant race recaps found."
