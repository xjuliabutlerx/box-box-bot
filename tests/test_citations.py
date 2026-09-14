from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from box_box_bot.agent.citations import extract_citations, filter_citations_by_answer


def _tool_message(content: str, name: str = "search_race_recaps") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="call_1", name=name)


def test_extract_citations_single_source():
    messages = [
        HumanMessage(content="Who won the 2025 Bahrain GP?"),
        AIMessage(content=""),
        _tool_message("[Source: Bahrain Grand Prix (2025)]\nPiastri won."),
        AIMessage(content="Piastri won."),
    ]
    assert extract_citations(messages) == [{"type": "race", "race_name": "Bahrain Grand Prix", "season": 2025}]


def test_extract_citations_multiple_sources_deduped():
    messages = [
        HumanMessage(content="What happened at Monza and Singapore?"),
        _tool_message(
            "[Source: Italian Grand Prix (2025)]\nMonza recap.\n\n"
            "[Source: Singapore Grand Prix (2025)]\nSingapore recap.\n\n"
            "[Source: Italian Grand Prix (2025)]\nMonza recap again."
        ),
    ]
    citations = extract_citations(messages)
    assert citations == [
        {"type": "race", "race_name": "Italian Grand Prix", "season": 2025},
        {"type": "race", "race_name": "Singapore Grand Prix", "season": 2025},
    ]


def test_extract_citations_ignores_non_rag_tool_messages():
    messages = [
        HumanMessage(content="Who won the 2025 Bahrain GP?"),
        _tool_message('[{"position": 1, "driver": "PIA"}]', name="get_race_results"),
    ]
    assert extract_citations(messages) == []


def test_extract_citations_only_looks_at_current_turn():
    messages = [
        HumanMessage(content="Tell me about Bahrain"),
        _tool_message("[Source: Bahrain Grand Prix (2025)]\n..."),
        AIMessage(content="..."),
        HumanMessage(content="What about Monza?"),
        _tool_message("[Source: Italian Grand Prix (2025)]\n..."),
    ]
    assert extract_citations(messages) == [{"type": "race", "race_name": "Italian Grand Prix", "season": 2025}]


def test_extract_citations_no_tool_messages():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert extract_citations(messages) == []


def test_extract_citations_track_info_single_source():
    messages = [
        HumanMessage(content="Why is Monaco hard to overtake at?"),
        _tool_message("[Source: Track Info - Circuit de Monaco]\nNarrow streets...", name="search_track_info"),
    ]
    assert extract_citations(messages) == [{"type": "track", "circuit": "Circuit de Monaco"}]


def test_extract_citations_track_info_multiple_sources_deduped():
    messages = [
        HumanMessage(content="Compare Monza and Spa"),
        _tool_message(
            "[Source: Track Info - Autodromo Nazionale Monza]\nLow downforce...\n\n"
            "[Source: Track Info - Circuit de Spa-Francorchamps]\nElevation...\n\n"
            "[Source: Track Info - Autodromo Nazionale Monza]\nLow downforce again...",
            name="search_track_info",
        ),
    ]
    citations = extract_citations(messages)
    assert citations == [
        {"type": "track", "circuit": "Autodromo Nazionale Monza"},
        {"type": "track", "circuit": "Circuit de Spa-Francorchamps"},
    ]


def test_extract_citations_handles_both_race_and_track_tools_in_same_turn():
    messages = [
        HumanMessage(content="Why did strategy matter at Monaco this year?"),
        _tool_message("[Source: Monaco Grand Prix (2026)]\nChaotic race...", name="search_race_recaps"),
        _tool_message("[Source: Track Info - Circuit de Monaco]\nNarrow streets...", name="search_track_info"),
    ]
    citations = extract_citations(messages)
    assert citations == [
        {"type": "race", "race_name": "Monaco Grand Prix", "season": 2026},
        {"type": "track", "circuit": "Circuit de Monaco"},
    ]


def test_filter_citations_by_answer_keeps_named_race():
    citations = [{"type": "race", "race_name": "Bahrain Grand Prix", "season": 2025}]
    answer = "Piastri won, which put him ahead in Bahrain."
    assert filter_citations_by_answer(citations, answer) == citations


def test_filter_citations_by_answer_drops_unmentioned_race():
    citations = [
        {"type": "race", "race_name": "Bahrain Grand Prix", "season": 2025},
        {"type": "race", "race_name": "Dutch Grand Prix", "season": 2026},
    ]
    answer = "Piastri won in Bahrain, taking the championship lead."
    assert filter_citations_by_answer(citations, answer) == [
        {"type": "race", "race_name": "Bahrain Grand Prix", "season": 2025}
    ]


def test_filter_citations_by_answer_matches_short_name_not_full_name():
    # The exact scenario found via live testing: the model writes "Bahrain"
    # rather than the full "Bahrain Grand Prix" - the filter must still match.
    citations = [{"type": "race", "race_name": "Bahrain Grand Prix", "season": 2025}]
    answer = "The win in Bahrain kicked off Piastri's championship lead."
    assert filter_citations_by_answer(citations, answer) == citations


def test_filter_citations_by_answer_case_insensitive():
    citations = [{"type": "race", "race_name": "Italian Grand Prix", "season": 2025}]
    answer = "The ITALIAN result mattered for the title fight."
    assert filter_citations_by_answer(citations, answer) == citations


def test_filter_citations_by_answer_empty_input():
    assert filter_citations_by_answer([], "some answer") == []


def test_filter_citations_by_answer_keeps_track_named_by_distinctive_word():
    # A model is far more likely to say "Monza" than the full formal name
    # "Autodromo Nazionale Monza" - same reasoning as the race short-name
    # match, applied via stopword-stripping instead of a fixed suffix.
    citations = [{"type": "track", "circuit": "Autodromo Nazionale Monza"}]
    answer = "Monza rewards a low-downforce setup thanks to its long straights."
    assert filter_citations_by_answer(citations, answer) == citations


def test_filter_citations_by_answer_drops_unmentioned_track():
    citations = [
        {"type": "track", "circuit": "Autodromo Nazionale Monza"},
        {"type": "track", "circuit": "Circuit de Monaco"},
    ]
    answer = "Monza rewards a low-downforce setup thanks to its long straights."
    assert filter_citations_by_answer(citations, answer) == [
        {"type": "track", "circuit": "Autodromo Nazionale Monza"}
    ]


def test_filter_citations_by_answer_track_case_insensitive():
    citations = [{"type": "track", "circuit": "Circuit de Monaco"}]
    answer = "MONACO is famously difficult to overtake at."
    assert filter_citations_by_answer(citations, answer) == citations


def test_filter_citations_by_answer_handles_mixed_race_and_track_citations():
    citations = [
        {"type": "race", "race_name": "Monaco Grand Prix", "season": 2026},
        {"type": "track", "circuit": "Autodromo Nazionale Monza"},
    ]
    answer = "The Monaco race this year was chaotic."
    assert filter_citations_by_answer(citations, answer) == [
        {"type": "race", "race_name": "Monaco Grand Prix", "season": 2026}
    ]
