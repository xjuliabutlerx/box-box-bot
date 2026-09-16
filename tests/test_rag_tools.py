from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from box_box_bot.tools.rag_tools import RAG_TOOLS, TRACK_INFO_TOOLS, search_race_recaps, search_track_info


def _fake_doc(race_name: str, season: int, content: str = "some recap text", location: str | None = None) -> Document:
    metadata = {"race_name": race_name, "season": season}
    if location:
        metadata["location"] = location
    return Document(page_content=content, metadata=metadata)


def _fake_track_doc(circuit: str, content: str = "some track info") -> Document:
    return Document(page_content=content, metadata={"circuit": circuit})


def test_search_race_recaps_is_registered():
    assert RAG_TOOLS == [search_race_recaps]


def test_search_track_info_is_registered():
    assert TRACK_INFO_TOOLS == [search_track_info]


def test_search_track_info_formats_source_tag():
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [_fake_track_doc("Circuit de Monaco", "Tightest track on the calendar.")]

    with patch("box_box_bot.tools.rag_tools._get_track_retriever", return_value=fake_retriever):
        result = search_track_info.invoke({"query": "why is Monaco hard to overtake at"})

    fake_retriever.invoke.assert_called_once_with("why is Monaco hard to overtake at")
    assert result == "[Source: Track Info - Circuit de Monaco]\nTightest track on the calendar."


def test_search_track_info_handles_no_results():
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = []

    with patch("box_box_bot.tools.rag_tools._get_track_retriever", return_value=fake_retriever):
        result = search_track_info.invoke({"query": "something totally unrelated"})

    assert result == "No relevant track info found."


def test_search_track_info_uses_a_separate_retriever_from_race_recaps():
    # Regression guard: the two tools must not share a lazy singleton -
    # they're different Chroma collections, and racing through a shared
    # one would reintroduce the exact concurrency bug the retriever lock
    # already exists to prevent, just across the wrong collection.
    fake_race_retriever = MagicMock()
    fake_race_retriever.invoke.return_value = []
    fake_track_retriever = MagicMock()
    fake_track_retriever.invoke.return_value = []

    with (
        patch("box_box_bot.tools.rag_tools._get_retriever", return_value=fake_race_retriever),
        patch("box_box_bot.tools.rag_tools._get_track_retriever", return_value=fake_track_retriever),
    ):
        search_race_recaps.invoke({"query": "test"})
        search_track_info.invoke({"query": "test"})

    fake_race_retriever.invoke.assert_called_once()
    fake_track_retriever.invoke.assert_called_once()


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


def test_search_race_recaps_includes_optional_location_in_source_tag():
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [
        _fake_doc("Spanish Grand Prix", 2026, "Antonelli won.", location="Madrid")
    ]

    with patch("box_box_bot.tools.rag_tools._get_retriever", return_value=fake_retriever):
        result = search_race_recaps.invoke({"query": "who won the Madrid GP"})

    assert result == "[Source: Spanish Grand Prix (2026) - Madrid]\nAntonelli won."


def test_search_race_recaps_handles_no_results():
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = []

    with patch("box_box_bot.tools.rag_tools._get_retriever", return_value=fake_retriever):
        result = search_race_recaps.invoke({"query": "something totally unrelated"})

    assert result == "No relevant race recaps found."
