from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from box_box_bot.agent.run import OFF_TOPIC_MESSAGE, ask


def _ai_message(content, input_tokens: int = 100, output_tokens: int = 10) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        },
    )


def _fake_agent(invoke_return=None, prior_messages=None):
    agent = MagicMock()
    if invoke_return is not None:
        agent.invoke.return_value = invoke_return
    # agent.get_state(config).values is a real dict, matching what
    # LangGraph's InMemorySaver actually returns ({} on a fresh thread).
    agent.get_state.return_value.values = {"messages": prior_messages} if prior_messages else {}
    return agent


def test_ask_short_circuits_when_off_topic():
    fake_agent = _fake_agent()

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": False, "cost_usd": 0.0002}):
        result = ask(fake_agent, "give me a sorting algorithm", "thread-1")

    assert result["answer"] == OFF_TOPIC_MESSAGE
    assert result["citations"] == []
    assert result["usage"]["cost_usd"] == 0.0002
    fake_agent.invoke.assert_not_called()


def test_ask_invokes_agent_and_extracts_answer_when_on_topic():
    fake_agent = _fake_agent(
        {
            "messages": [
                HumanMessage(content="Who won Bahrain?"),
                _ai_message("Piastri won the Bahrain Grand Prix."),
            ]
        }
    )

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        result = ask(fake_agent, "Who won Bahrain?", "thread-1")

    assert result["answer"] == "Piastri won the Bahrain Grand Prix."
    assert result["citations"] == []
    assert result["usage"]["cost_usd"] > 0
    fake_agent.invoke.assert_called_once()


def test_ask_passes_thread_id_through_config():
    fake_agent = _fake_agent({"messages": [HumanMessage(content="hi"), _ai_message("hello")]})

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        ask(fake_agent, "hi", "my-thread-id")

    _input, config = fake_agent.invoke.call_args[0]
    assert config == {"configurable": {"thread_id": "my-thread-id"}}


def test_ask_extracts_citations_from_tool_results():
    tool_msg = ToolMessage(
        content="[Source: Bahrain Grand Prix (2025)]\nPiastri won.",
        tool_call_id="c1",
        name="search_race_recaps",
    )
    fake_agent = _fake_agent(
        {
            "messages": [
                HumanMessage(content="Why did Bahrain matter?"),
                tool_msg,
                _ai_message("Piastri won in Bahrain, taking the championship lead."),
            ]
        }
    )

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        result = ask(fake_agent, "Why did Bahrain matter?", "thread-1")

    assert result["citations"] == [{"race_name": "Bahrain Grand Prix", "season": 2025}]


def test_ask_extracts_text_from_list_content_with_thinking_blocks():
    # Claude Sonnet 5 returns content as a list of thinking/text blocks
    # when tools were used, not a plain string.
    fake_agent = _fake_agent(
        {
            "messages": [
                HumanMessage(content="Who won?"),
                _ai_message(
                    [
                        {"type": "thinking", "thinking": "internal reasoning..."},
                        {"type": "text", "text": "Piastri won the Bahrain Grand Prix."},
                    ]
                ),
            ]
        }
    )

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        result = ask(fake_agent, "Who won?", "thread-1")

    assert result["answer"] == "Piastri won the Bahrain Grand Prix."


def test_ask_passes_no_context_on_fresh_thread():
    fake_agent = _fake_agent({"messages": [HumanMessage(content="hi"), _ai_message("hello")]})

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}) as mock_check:
        ask(fake_agent, "hi", "brand-new-thread")

    mock_check.assert_called_once_with("hi", recent_context=None)


def test_ask_passes_prior_reply_as_recent_context():
    prior_messages = [
        HumanMessage(content="Tell me about the 2025 title fight"),
        _ai_message("It came down to the last race between Norris and Verstappen."),
    ]
    fake_agent = _fake_agent(
        {"messages": prior_messages + [HumanMessage(content="yes"), _ai_message("Sure, here's more detail...")]},
        prior_messages=prior_messages,
    )

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}) as mock_check:
        ask(fake_agent, "yes", "existing-thread")

    mock_check.assert_called_once_with(
        "yes", recent_context="It came down to the last race between Norris and Verstappen."
    )
