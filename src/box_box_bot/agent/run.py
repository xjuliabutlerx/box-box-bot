from box_box_bot.agent.citations import extract_citations, filter_citations_by_answer
from box_box_bot.agent.cost import estimate_cost

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text")

def ask(agent, message: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)

    answer = _extract_text(result["messages"][-1].content)
    candidates = extract_citations(result["messages"])

    return {
        "answer": answer,
        "citations": filter_citations_by_answer(candidates, answer),
        "usage": estimate_cost(result["messages"]),
    }