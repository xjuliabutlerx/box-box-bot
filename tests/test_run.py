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


def test_ask_short_circuits_when_off_topic():
    fake_agent = MagicMock()

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": False, "cost_usd": 0.0002}):
        result = ask(fake_agent, "give me a sorting algorithm", "thread-1")

    assert result["answer"] == OFF_TOPIC_MESSAGE
    assert result["citations"] == []
    assert result["usage"]["cost_usd"] == 0.0002
    fake_agent.invoke.assert_not_called()


def test_ask_invokes_agent_and_extracts_answer_when_on_topic():
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [
            HumanMessage(content="Who won Bahrain?"),
            _ai_message("Piastri won the Bahrain Grand Prix."),
        ]
    }

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        result = ask(fake_agent, "Who won Bahrain?", "thread-1")

    assert result["answer"] == "Piastri won the Bahrain Grand Prix."
    assert result["citations"] == []
    assert result["usage"]["cost_usd"] > 0
    fake_agent.invoke.assert_called_once()


def test_ask_passes_thread_id_through_config():
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [HumanMessage(content="hi"), _ai_message("hello")]}

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        ask(fake_agent, "hi", "my-thread-id")

    _input, config = fake_agent.invoke.call_args[0]
    assert config == {"configurable": {"thread_id": "my-thread-id"}}


def test_ask_extracts_citations_from_tool_results():
    fake_agent = MagicMock()
    tool_msg = ToolMessage(
        content="[Source: Bahrain Grand Prix (2025)]\nPiastri won.",
        tool_call_id="c1",
        name="search_race_recaps",
    )
    fake_agent.invoke.return_value = {
        "messages": [
            HumanMessage(content="Why did Bahrain matter?"),
            tool_msg,
            _ai_message("Piastri won in Bahrain, taking the championship lead."),
        ]
    }

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        result = ask(fake_agent, "Why did Bahrain matter?", "thread-1")

    assert result["citations"] == [{"race_name": "Bahrain Grand Prix", "season": 2025}]


def test_ask_extracts_text_from_list_content_with_thinking_blocks():
    # Claude Sonnet 5 returns content as a list of thinking/text blocks
    # when tools were used, not a plain string.
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
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

    with patch("box_box_bot.agent.run.check_topic", return_value={"on_topic": True, "cost_usd": 0.0}):
        result = ask(fake_agent, "Who won?", "thread-1")

    assert result["answer"] == "Piastri won the Bahrain Grand Prix."
