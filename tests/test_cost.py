from langchain_core.messages import AIMessage, HumanMessage

from box_box_bot.agent.cost import (
    HAIKU_4_5_PRICING,
    SONNET_5_PRICING,
    _price,
    estimate_cost,
    estimate_gate_cost,
)


def _ai_message(input_tokens: int, output_tokens: int, cache_read: int = 0, cache_creation: int = 0) -> AIMessage:
    return AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
        },
    )


def test_price_basic_arithmetic():
    rates = {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20}
    # 1,000,000 input tokens + 1,000,000 output tokens at $2/$10 per million = $12
    assert _price(1_000_000, 1_000_000, 0, 0, rates) == 12.0


def test_price_includes_cache_tokens_additively():
    rates = {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20}
    # cache tokens are additional to input/output, not a subset of them
    cost_without_cache = _price(1000, 0, 0, 0, rates)
    cost_with_cache = _price(1000, 0, 1000, 0, rates)
    assert cost_with_cache > cost_without_cache


def test_estimate_cost_sums_multiple_ai_messages_in_one_turn():
    # A single turn can involve more than one model call (decide to use a
    # tool, then a follow-up call once the tool result comes back).
    messages = [
        HumanMessage(content="Who won Bahrain?"),
        _ai_message(input_tokens=1000, output_tokens=50),  # tool-selection call
        _ai_message(input_tokens=2000, output_tokens=100),  # follow-up call
    ]
    result = estimate_cost(messages)
    assert result["input_tokens"] == 3000
    assert result["output_tokens"] == 150
    expected = _price(3000, 150, 0, 0, SONNET_5_PRICING)
    assert result["cost_usd"] == expected


def test_estimate_cost_only_counts_current_turn():
    messages = [
        HumanMessage(content="Turn 1"),
        _ai_message(input_tokens=5000, output_tokens=500),
        HumanMessage(content="Turn 2"),
        _ai_message(input_tokens=100, output_tokens=10),
    ]
    result = estimate_cost(messages)
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 10


def test_estimate_cost_ignores_messages_without_usage_metadata():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="no usage metadata here"),
    ]
    result = estimate_cost(messages)
    assert result == {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def test_estimate_gate_cost_uses_haiku_pricing_not_sonnet():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "input_token_details": {}}
    gate_cost = estimate_gate_cost(usage)
    assert gate_cost == HAIKU_4_5_PRICING["input"]
    assert gate_cost != SONNET_5_PRICING["input"]


def test_estimate_gate_cost_handles_none():
    assert estimate_gate_cost(None) == 0.0


def test_estimate_gate_cost_handles_missing_details():
    assert estimate_gate_cost({"input_tokens": 100, "output_tokens": 10}) > 0.0
