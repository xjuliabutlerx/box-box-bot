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

st.set_page_config(page_title="Box Box Bot", page_icon="🏎️")
st.title("🏎️ Box Box Bot")
st.caption(
    "An F1 chatbot grounded in live fastf1 data and race-recap RAG. "
    "Ask about standings, race results, lap times, or the story behind a season."
)

@st.cache_resource
def get_agent():
    return build_agent()

import uuid

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0
if "last_turn_cost_usd" not in st.session_state:
    st.session_state.last_turn_cost_usd = 0.0

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            sources = ", ".join(f"{c['race_name']} ({c['season']})" for c in message["citations"])
            st.caption(f"_Sources: {sources}_")

if user_input := st.chat_input("Ask about an F1 season, race, or driver..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(get_agent(), user_input, st.session_state.thread_id)
            st.session_state.last_turn_cost_usd = result["usage"]["cost_usd"]
            st.session_state.total_cost_usd += result["usage"]["cost_usd"]

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