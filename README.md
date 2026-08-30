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
  tools/
    fastf1_tools.py   # LangChain @tool wrappers around data/ (step 2)
  rag/
    embeddings.py      # local fastembed model adapted to LangChain's Embeddings
    ingest.py           # builds the Chroma vector store from data/race_recaps/
    retriever.py        # opens the persisted store, hands back a retriever
  agent/            # LangGraph graph definition (step 4)
  app/              # Streamlit entrypoint (step 6)
data/
  cache/            # fastf1 disk cache (gitignored, rebuilds on first fetch)
  race_recaps/      # 12 original recap docs (2025 title fight + 2026 so far)
  vectorstore/      # persisted Chroma index (gitignored, rebuild via ingest.py)
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

## RAG layer

`data/race_recaps/` holds 12 original recap documents (see its own README
for the full list and sourcing notes) covering the 2025 title fight and the
2026 season so far. `rag/ingest.py` chunks them (`RecursiveCharacterTextSplitter`,
500 chars / 50 overlap) and embeds each chunk locally via `fastembed`
(`BAAI/bge-small-en-v1.5`, no API key, no network calls after the first
model download) into a persisted Chroma collection. `rag/retriever.py`
opens that collection and returns a standard LangChain retriever.

Rebuild the index after editing the corpus:

```bash
python -m box_box_bot.rag.ingest
```

**Known limitation:** retrieval quality is noticeably better for
specific queries ("2025 Abu Dhabi Grand Prix championship finale") than
vague ones ("why did Verstappen lose the championship?") — the latter
pulled in irrelevant 2026 chunks in testing, since the small local
embedding model doesn't strongly separate the two seasons on vague
phrasing alone. Step 4 (the agent) needs to account for this rather than
assume raw user questions are good retrieval queries as-is.
