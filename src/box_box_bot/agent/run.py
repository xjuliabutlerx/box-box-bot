import logging

from box_box_bot.agent.citations import extract_citations, filter_citations_by_answer
from box_box_bot.agent.cost import estimate_cost
from box_box_bot.agent.input_guard import check_input_safety
from box_box_bot.agent.topic_gate import check_topic
from box_box_bot.agent.visuals import extract_visuals

_logger = logging.getLogger(__name__)

OFF_TOPIC_MESSAGE = (
    "I can only help with Formula 1 questions - standings, race results, "
    "lap times, or the story behind a season. Try asking about a race or "
    "championship instead!"
)

UNSAFE_INPUT_MESSAGE = (
    "Your message looks like it is too long, contains a shell command, an IP address, a URL, and/or prompt injection "
    "phrasing, none of which I can process here. Try rephrasing your F1 "
    "question in plain text."
)

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text")

def ask(agent, message: str, thread_id: str) -> dict:
    guard = check_input_safety(message)
    if not guard["safe"]:
        _logger.info("Blocked unsafe input on thread %s (reason=%s)", thread_id, guard["reason"])
        return {
            "answer": UNSAFE_INPUT_MESSAGE,
            "citations": [],
            "visuals": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            "blocked_reason": "unsafe_input",
        }

    config = {"configurable": {"thread_id": thread_id}}

    prior_messages = agent.get_state(config).values.get("messages", [])
    recent_context = _extract_text(prior_messages[-1].content) if prior_messages else None

    gate = check_topic(message, recent_context=recent_context)
    if not gate["on_topic"]:
        _logger.info("Blocked off-topic message on thread %s", thread_id)
        return {
            "answer": OFF_TOPIC_MESSAGE,
            "citations": [],
            "visuals": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": gate["cost_usd"]},
            "blocked_reason": "off_topic",
        }

    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)

    answer = _extract_text(result["messages"][-1].content)
    candidates = extract_citations(result["messages"])
    usage = estimate_cost(result["messages"])
    _logger.info("Completed turn on thread %s: cost=$%.4f", thread_id, usage["cost_usd"])

    return {
        "answer": answer,
        "citations": filter_citations_by_answer(candidates, answer),
        "visuals": extract_visuals(result["messages"]),
        "usage": usage,
        "blocked_reason": None,
    }