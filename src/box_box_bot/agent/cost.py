# Anthropic pricing per 1M tokens.
# https://docs.claude.com/en/docs/about-claude/pricing
SONNET_5_PRICING = {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20}
HAIKU_4_5_PRICING = {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10}


def _price(input_tokens: int, output_tokens: int, cache_read_tokens: int, cache_write_tokens: int, rates: dict) -> float:
    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read_tokens * rates["cache_read"]
        + cache_write_tokens * rates["cache_write"]
    ) / 1_000_000


def estimate_cost(messages: list) -> dict:
    """Sum token usage across this turn's model calls and estimate the
    dollar cost, using LangChain's provider-normalized usage_metadata.

    A single agent turn can involve multiple model calls (decide to call a
    tool, then a follow-up call once the tool result comes back), so this
    sums usage_metadata across every AIMessage since the last human turn -
    same slicing approach as citations.py. This is the main agent's model
    (Sonnet 5); the topic gate uses estimate_gate_cost instead, since it
    runs on a cheaper model with its own pricing.
    """
    last_human_idx = max(i for i, m in enumerate(messages) if m.type == "human")
    turn_messages = messages[last_human_idx:]

    input_tokens = output_tokens = cache_read_tokens = cache_write_tokens = 0
    for m in turn_messages:
        usage = getattr(m, "usage_metadata", None)
        if m.type != "ai" or not usage:
            continue
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        details = usage.get("input_token_details") or {}
        cache_read_tokens += details.get("cache_read", 0)
        cache_write_tokens += details.get("cache_creation", 0)

    cost_usd = _price(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, SONNET_5_PRICING)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}


def estimate_gate_cost(usage_metadata: dict | None) -> float:
    """Cost of a single topic-gate call (Haiku 4.5). Folded into the same
    cost tracking as the main agent so a stream of off-topic messages the
    gate catches still counts toward the session/global cost caps, rather
    than looking free just because the main agent never ran.
    """
    if not usage_metadata:
        return 0.0
    details = usage_metadata.get("input_token_details") or {}
    return _price(
        usage_metadata.get("input_tokens", 0),
        usage_metadata.get("output_tokens", 0),
        details.get("cache_read", 0),
        details.get("cache_creation", 0),
        HAIKU_4_5_PRICING,
    )