from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from box_box_bot.agent.topic_gate import check_topic


def _fake_response(text: str, input_tokens: int = 50, output_tokens: int = 5) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        },
    )


def test_check_topic_ontopic():
    fake_model = MagicMock()
    fake_model.invoke.return_value = _fake_response("ontopic")

    with patch("box_box_bot.agent.topic_gate._get_gate_model", return_value=fake_model):
        result = check_topic("Who won the 2025 Bahrain Grand Prix?")

    assert result["on_topic"] is True
    assert result["cost_usd"] > 0


def test_check_topic_offtopic():
    fake_model = MagicMock()
    fake_model.invoke.return_value = _fake_response("offtopic")

    with patch("box_box_bot.agent.topic_gate._get_gate_model", return_value=fake_model):
        result = check_topic("Give me a Python sorting algorithm")

    assert result["on_topic"] is False


def test_check_topic_handles_whitespace_and_case():
    fake_model = MagicMock()
    fake_model.invoke.return_value = _fake_response("  OnTopic  ")

    with patch("box_box_bot.agent.topic_gate._get_gate_model", return_value=fake_model):
        result = check_topic("anything")

    assert result["on_topic"] is True


def test_check_topic_sends_system_prompt_and_user_message():
    fake_model = MagicMock()
    fake_model.invoke.return_value = _fake_response("ontopic")

    with patch("box_box_bot.agent.topic_gate._get_gate_model", return_value=fake_model):
        check_topic("Who won Monza?")

    sent_messages = fake_model.invoke.call_args[0][0]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1] == {"role": "user", "content": "Who won Monza?"}


def test_check_topic_cost_uses_haiku_pricing():
    fake_model = MagicMock()
    fake_model.invoke.return_value = _fake_response("ontopic", input_tokens=1_000_000, output_tokens=0)

    with patch("box_box_bot.agent.topic_gate._get_gate_model", return_value=fake_model):
        result = check_topic("anything")

    # 1,000,000 input tokens at Haiku 4.5's $1.00/million input rate
    assert result["cost_usd"] == 1.00
