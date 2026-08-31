from box_box_bot.agent.citations import extract_citations, filter_citations_by_answer
from box_box_bot.agent.cost import estimate_cost
from box_box_bot.agent.topic_gate import check_topic

OFF_TOPIC_MESSAGE = (
    "I can only help with Formula 1 questions - standings, race results, "
    "lap times, or the story behind a season. Try asking about a race or "
    "championship instead!"
)

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text")

def ask(agent, message: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    prior_messages = agent.get_state(config).values.get("messages", [])
    recent_context = _extract_text(prior_messages[-1].content) if prior_messages else None

    gate = check_topic(message, recent_context=recent_context)
    if not gate["on_topic"]:
        return {
            "answer": OFF_TOPIC_MESSAGE,
            "citations": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": gate["cost_usd"]},
        }

    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)

    answer = _extract_text(result["messages"][-1].content)
    candidates = extract_citations(result["messages"])

    return {
        "answer": answer,
        "citations": filter_citations_by_answer(candidates, answer),
        "usage": estimate_cost(result["messages"]),
    }