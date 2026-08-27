# box-box-bot

A conversational F1 chatbot agent built with LangChain/LangGraph. Answers
natural-language questions using both live/structured data (fastf1 tool
calls) and narrative context (RAG over race recaps), e.g. "How did the
constructor standings change after Monza?" or "Why did [driver] lose the
championship in [year]?"

## Stack

- **langgraph** — agent graph (current prebuilt agent pattern, not the
  deprecated `AgentExecutor`)
- **fastf1** + **fastf1.ergast** — race/session results and championship
  standings
- **chromadb** — vector store for RAG over race summaries
- **langchain-anthropic** — Claude as the agent's model
- **LangSmith** — tracing
- **streamlit** — UI

## Project layout

```
src/box_box_bot/
  config.py        # env loading, LangSmith wiring, fastf1 cache path
  data/
    fastf1_client.py  # only module that talks to fastf1/Ergast directly
  tools/            # LangChain tool wrappers around data/ (step 2)
  rag/              # embedding + retrieval over race recaps (step 3)
  agent/            # LangGraph graph definition (step 4)
  app/              # Streamlit entrypoint (step 6)
data/
  cache/            # fastf1 disk cache (gitignored, rebuilds on first fetch)
```

`data/fastf1_client.py` is the only place fastf1/Ergast get imported.
Everything above it — tools, agent, app — should go through this module
instead of calling fastf1 directly, so the data layer can be swapped or
mocked without touching agent logic.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env  # then fill in ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

## Data access layer

`box_box_bot.data.fastf1_client` currently exposes:

- `get_driver_standings(season, round=None)`
- `get_constructor_standings(season, round=None)`
- `get_race_results(season, round)`
- `get_fastest_laps(season, round, session_type="R", top_n=5)`

Standings come from `fastf1.ergast` (session objects don't carry cumulative
championship state); race results and lap data come from
`fastf1.get_session(...).load()`.
