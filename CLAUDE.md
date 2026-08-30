# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup (Python >= 3.10, `src/` layout package):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .            # or rely on `-e .` at the end of requirements.txt
cp .env.example .env        # then fill in ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

Rebuild the RAG vector store after editing anything under `data/race_recaps/`
— nothing else does this automatically:

```bash
python -m box_box_bot.rag.ingest
```

Run the app locally:

```bash
streamlit run src/box_box_bot/app/streamlit_app.py
```

Run the test suite (needs `requirements-dev.txt`, not just `requirements.txt`):

```bash
pip install -r requirements-dev.txt
pytest tests/                          # full suite
pytest tests/test_citations.py -v      # single file
pytest tests/test_cost.py::test_estimate_gate_cost_handles_none  # single test
```

Every external call (`fastf1`/`Ergast`, `ChatAnthropic`) is mocked in tests
— the suite runs in ~1s with no network calls, no Anthropic spend, and no
`ANTHROPIC_API_KEY` required. The one exception is `fastembed`'s local
model in `test_embeddings.py`, which needs network access only for its
one-time download if not already cached. There's no configured linter —
UI changes still get verified manually by driving the Streamlit app in a
real browser (`tests/` doesn't cover `app/streamlit_app.py`).

## Architecture

The app is a layered pipeline; each layer only talks to the one below it,
and the intent is that any layer can be swapped without touching the ones
above it:

```
data/fastf1_client.py  -->  tools/*.py  -->  agent/graph.py  -->  app/streamlit_app.py
   (fastf1/Ergast)         (@tool wrappers)   (create_agent)      (chat UI)
                                                     ^
                              rag/{embeddings,ingest,retriever}.py
                                        (Chroma + fastembed)
```

- **`data/fastf1_client.py`** is the *only* module that imports `fastf1`
  directly. It returns plain `list[dict]` (via `.to_dict(orient="records")`
  on pandas output). Championship standings come from `fastf1.ergast`
  (session objects don't carry cumulative state); race results/lap data
  come from `fastf1.get_session(...).load()`.
- **`tools/`** wraps both `data/fastf1_client.py` and `rag/retriever.py` as
  LangChain `@tool(parse_docstring=True)` functions. Every tool output goes
  through `json.dumps(..., default=str)` — pandas output routinely contains
  `Timedelta`/`NaT`/`Timestamp` values that the stdlib `json` module can't
  serialize natively, and `default=str` is the general fix rather than
  patching individual columns as new ones show up. `rag_tools.py` tags each
  retrieved chunk with `[Source: {race_name} ({season})]` directly in the
  tool's text output — this is what makes the citation system in
  `agent/citations.py` possible later; nothing about citations is bolted on
  after the fact.
- **`agent/graph.py`** builds the agent via `langchain.agents.create_agent`
  (not `langgraph.prebuilt.create_react_agent`, which this project migrated
  away from mid-build after it was deprecated in favor of the former —
  same underlying tool-calling loop, but the system-prompt kwarg is
  `system_prompt`, not `prompt`). Conversation memory is an `InMemorySaver`
  checkpointer keyed by `thread_id` — passing the same `thread_id` across
  `agent/run.py`'s `ask()` calls continues that conversation.
- **`agent/run.py`**'s `ask(agent, message, thread_id)` is the single entry
  point everything downstream calls. It invokes the agent, extracts the
  answer text (Claude Sonnet 5 returns `content` as a list of
  thinking/text blocks when tools were used, not a plain string — always
  filter for `type == "text"` blocks rather than assuming a string),
  extracts citations, and estimates cost, returning
  `{"answer", "citations", "usage"}`.
- **`agent/citations.py`** is two separate steps, not one:
  `extract_citations()` pulls every `[Source: ...]` tag out of this turn's
  `search_race_recaps` tool results (sliced from the last human message
  onward, so old turns don't resurface their sources on a later turn), and
  `filter_citations_by_answer()` keeps only the ones the model's answer
  text actually names — matched on each race's *short* name ("Bahrain"),
  not the full official name ("Bahrain Grand Prix"), because the model
  doesn't reliably use the full name and matching on it made citations
  disappear nondeterministically between otherwise-identical runs. This
  means a retrieved-but-unused chunk (retrieval isn't perfectly precise —
  see the RAG note below) never shows up as a false citation.
- **`agent/cost.py`** sums `usage_metadata` (LangChain's provider-normalized
  token counts) across every model call in a turn — a single turn is often
  more than one call (decide to use a tool, then a follow-up call once the
  tool result comes back) — and prices it against Claude Sonnet 5's
  per-token rates.
- **`rag/ingest.py`** (offline/manual) vs. **`rag/retriever.py`** (what the
  running app calls) are deliberately separate: rebuilding the Chroma index
  is a batch step you run after editing `data/race_recaps/*.md`, not
  something that happens on every app startup. Embeddings are local
  (`fastembed`, `BAAI/bge-small-en-v1.5`) since Anthropic has no embeddings
  API — this is the only reason a non-Anthropic model shows up anywhere in
  the stack. **Known limitation:** retrieval is noticeably weaker on vague
  queries ("why did Verstappen lose the championship?") than specific ones,
  sometimes pulling in chunks from the wrong season.
- **`app/streamlit_app.py`**: the agent itself (`get_agent()`) is built
  once per server process via `@st.cache_resource` and shared across every
  visitor; what's *not* shared is conversation state — each browser session
  gets its own `thread_id` in `st.session_state`, so the shared agent's
  checkpointer still keeps every visitor's conversation isolated. Streamlit
  reruns the entire script on every interaction and widgets aren't
  reactive — a widget rendered earlier in the script than the state update
  that's supposed to feed it will show stale, one-turn-late values, so
  ordering within the script matters. Two cost guardrails exist because
  this demo runs on the author's own API key: `MAX_MESSAGES_PER_SESSION`
  (per-`st.session_state`, resets on page refresh) and `MAX_TOTAL_COST_USD`
  (a `@st.cache_resource`-shared dict + `threading.Lock`, survives a
  refresh, only resets if the server process restarts) — neither is a
  perfectly airtight ceiling under concurrent load, so the actual backstop
  is a spend limit set directly in the Anthropic Console.
- **Secrets**: locally via `.env` (loaded by `config.py`); on Streamlit
  Community Cloud via the dashboard's Secrets panel, which `streamlit_app.py`
  mirrors into `os.environ` *before* importing anything from `box_box_bot`
  (env vars are read at module import time, so the mirroring has to happen
  first). `requirements.txt` ends with `-e .` so Streamlit Cloud's
  `pip install -r requirements.txt` alone makes `box_box_bot` importable.

See `README.md` for the full stack list, setup steps, and a "Failure modes
hit" section with more detail on several of the bugs referenced above.
