# Anthropic pricing for claude-sonnet-5, per 1M tokens.
# https://docs.claude.com/en/docs/about-claude/pricing
INPUT_PRICE_PER_MILLION = 2.00
OUTPUT_PRICE_PER_MILLION = 10.00
CACHE_WRITE_PRICE_PER_MILLION = 2.50
CACHE_READ_PRICE_PER_MILLION = 0.20


def estimate_cost(messages: list) -> dict:
    """Sum token usage across this turn's model calls and estimate the
    dollar cost, using LangChain's provider-normalized usage_metadata.

    A single agent turn can involve multiple model calls (decide to call a
    tool, then a follow-up call once the tool result comes back), so this
    sums usage_metadata across every AIMessage since the last human turn -
    same slicing approach as citations.py.
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

    cost_usd = (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
        + cache_read_tokens * CACHE_READ_PRICE_PER_MILLION
        + cache_write_tokens * CACHE_WRITE_PRICE_PER_MILLION
    ) / 1_000_000

    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}