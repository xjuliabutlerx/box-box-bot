import os
import streamlit as st

try:
    for key in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

from box_box_bot.agent.graph import build_agent
from box_box_bot.agent.run import ask

st.set_page_config(page_title="BoxBoxBot", page_icon="🏎️")
st.title("🏎️ BoxBoxBot")
st.caption(
    "An F1 chatbot grounded in live fastf1 data and race-recap RAG. "
    "Ask about standings, race results, lap times, or the story behind a season."
)

@st.cache_resource
def get_agent():
    return build_agent()

import threading

MAX_TOTAL_COST_USD = 5.00  # hard ceiling across every visitor combined
MAX_MESSAGES_PER_SESSION = 5    # per-session max messages

@st.cache_resource
def get_usage_tracker():
    return {"total_cost_usd": 0.0, "lock": threading.Lock()}

import uuid

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0

if "last_turn_cost_usd" not in st.session_state:
    st.session_state.last_turn_cost_usd = 0.0

if "message_count" not in st.session_state:
    st.session_state.message_count = 0

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            sources = ", ".join(f"{c['race_name']} ({c['season']})" for c in message["citations"])
            st.caption(f"_Sources: {sources}_")

tracker = get_usage_tracker()
with tracker["lock"]:  # scoped to this in-memory read only - never held across the slow ask() call below
    over_limit = tracker["total_cost_usd"] >= MAX_TOTAL_COST_USD

session_limit_reached = st.session_state.message_count >= MAX_MESSAGES_PER_SESSION

if over_limit:
    st.error("This demo has hit its usage cap for now. Thanks for trying it out!")
elif session_limit_reached:
    st.warning(f"You've reached this session's {MAX_MESSAGES_PER_SESSION}-message demo limit.")

chat_disabled = over_limit or session_limit_reached

if user_input := st.chat_input(
    "Ask about an F1 season, race, or driver...", disabled=chat_disabled
):
    st.session_state.message_count += 1
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(get_agent(), user_input, st.session_state.thread_id)
            st.session_state.last_turn_cost_usd = result["usage"]["cost_usd"]
            st.session_state.total_cost_usd += result["usage"]["cost_usd"]
            with tracker["lock"]:  # ask() already returned - only the in-memory increment is locked
                tracker["total_cost_usd"] += result["usage"]["cost_usd"]

        st.markdown(result["answer"])
        if result["citations"]:
            sources = ", ".join(f"{c['race_name']} ({c['season']})" for c in result["citations"])
            st.caption(f"_Sources: {sources}_")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"]
    })

st.caption(
    f"This query: \\${st.session_state.last_turn_cost_usd:.4f} · Session total: \\${st.session_state.total_cost_usd:.4f}"
)