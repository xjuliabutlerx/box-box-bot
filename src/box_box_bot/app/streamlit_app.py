import logging
import os
import streamlit as st

try:
    for key in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "REQUIRE_PASSWORD", "APP_PASSWORDS", "FASTF1_DEBUG_LOGGING"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

from box_box_bot.agent.graph import build_agent
from box_box_bot.agent.run import ask
from box_box_bot.app import charts
from box_box_bot.logging_config import configure_logging

configure_logging()
_logger = logging.getLogger(__name__)

ASSISTANT_AVATAR = "🏁"

def _render_blocked(text, blocked_reason):
    if blocked_reason == "unsafe_input":
        st.error(text)
    elif blocked_reason == "off_topic":
        st.warning(text)

def _render_visuals(visuals, key_prefix):
    for i, visual in enumerate(visuals):
        key = f"{key_prefix}-visual-{i}"
        if visual["type"] == "table":
            st.dataframe(charts.build_table(visual["data"]), key=key, hide_index=True)
        elif visual["type"] == "tire_strategy_chart":
            st.plotly_chart(charts.build_tire_strategy_figure(visual["data"]), key=key)
        elif visual["type"] == "track_map":
            st.plotly_chart(charts.build_track_map_figure(visual["data"]), key=key)

def _format_citation(citation):
    if citation["type"] == "race":
        return f"{citation['race_name']} ({citation['season']})"
    return f"{citation['circuit']} (track info)"

def _render_citations(citations):
    sources = ", ".join(_format_citation(c) for c in citations)
    st.caption(f"_Sources: {sources}_")

st.set_page_config(page_title="BoxBoxBot", page_icon="🏁", layout="wide")
# "wide" alone stretches to the full viewport; Streamlit's layout config
# only offers "centered" (~730px) or "wide" (no cap) - nothing in
# between - so this caps "wide" mode's container width instead of
# fighting a competing built-in max-width rule.
st.html("""
<style>
[data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""")

st.title("🏎️ BoxBoxBot")
st.caption(
    "Box, box!\n\nI'm your multi-agent F1 pit wall assistant for standings, race results, pit strategy, and the stories behind them powered by live `fastf1` data, retrieval-augmented race recaps, and trained prediction models."
    "\n\nAsk about standings, results, tire strategy and safety cars, or the story behind a season (narrative deep-dives cover select races from 2016-2026)."
)

# Password gate
if os.environ.get("REQUIRE_PASSWORD", "false").strip().lower() == "true":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        entered_password = st.text_input("Enter the password to continue", type="password")
        if entered_password:
            # comma-separated so different people can each get their own password without the
            # app needing to know who's who
            valid_passwords = {
                p.strip() for p in os.environ.get("APP_PASSWORDS", "").split(",") if p.strip()
            }
            if entered_password in valid_passwords:
                st.session_state.authenticated = True
                _logger.info("Successful password auth")
                st.rerun()
            else:
                _logger.info("Failed password attempt")
                st.error("Incorrect password.")
        st.stop()

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

for idx, message in enumerate(st.session_state.messages):
    if message.get("blocked_reason"):
        _render_blocked(message["content"], message["blocked_reason"])
        continue
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("visuals"):
            _render_visuals(message["visuals"], key_prefix=f"history-{idx}")
        if message.get("citations"):
            _render_citations(message["citations"])

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

    with st.spinner("Thinking..."):
        result = ask(get_agent(), user_input, st.session_state.thread_id)
        st.session_state.last_turn_cost_usd = result["usage"]["cost_usd"]
        st.session_state.total_cost_usd += result["usage"]["cost_usd"]
        with tracker["lock"]:  # ask() already returned - only the in-memory increment is locked
            tracker["total_cost_usd"] += result["usage"]["cost_usd"]
            over_limit_now = tracker["total_cost_usd"] >= MAX_TOTAL_COST_USD

    if result["blocked_reason"]:
        # a rejected-input notice (not an agent answer) rendered as a
        # plain Streamlit error/warning
        _render_blocked(result["answer"], result["blocked_reason"])
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(result["answer"])
            if result["visuals"]:
                _render_visuals(result["visuals"], key_prefix=f"live-{st.session_state.message_count}")
            if result["citations"]:
                _render_citations(result["citations"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
        "visuals": result["visuals"],
        "blocked_reason": result["blocked_reason"],
    })

    # the top-of-script over_limit/session_limit_reached check above ran
    # before this turn's message_count increment and cost update, so it
    # didn't reflect a limit this exact turn just hit
    if over_limit_now:
        _logger.warning("Total cost cap hit: $%.4f", tracker["total_cost_usd"])
        st.error("This demo has hit its usage cap for now. Thanks for trying it out!")
    elif st.session_state.message_count >= MAX_MESSAGES_PER_SESSION:
        _logger.info("Session %s hit its message limit", st.session_state.thread_id)
        st.warning(f"You've reached this session's {MAX_MESSAGES_PER_SESSION}-message demo limit.")

st.caption(
    f"This query: \\${st.session_state.last_turn_cost_usd:.4f} · Session total: \\${st.session_state.total_cost_usd:.4f}"
)