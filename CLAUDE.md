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
data/fastf1_client.py  -->  tools/*.py  -->  agent/{stats,narrative,predictor}_agent.py  -->  agent/graph.py  -->  app/streamlit_app.py
   (fastf1/Ergast)         (@tool wrappers)         (create_agent specialists)             (supervisor)        (chat UI)
                                                     ^                        ^
                              rag/{embeddings,ingest,retriever}.py   predictor/{model,features,predict}.py
                                        (Chroma + fastembed)                (torch)
```

- **`data/fastf1_client.py`** is the *only* module that imports `fastf1`
  directly. It returns plain `list[dict]` (via `.to_dict(orient="records")`
  on pandas output). Championship standings come from `fastf1.ergast`
  (session objects don't carry cumulative state); race results/lap data
  come from `fastf1.get_session(...).load()`. `get_race_results`,
  `get_fastest_laps`, `get_tire_strategy`, `get_race_control_messages`,
  and `get_weather_for_session` take `round: int | str` on purpose —
  `fastf1.get_session` already fuzzy-matches a string round against each
  event's country/location/name, so the model can pass a race name
  directly instead of recalling/guessing a round number. A stricter
  `int`-only signature once caused the model to hallucinate the wrong
  round for a named race (see README's Failure modes). **Every one of
  those five functions must run `round` through `_normalize_round()`
  before calling `fastf1.get_session`** — fastf1 only parses `round` as a
  round *number* when it's an `int`; a `str` round always goes through
  fuzzy name-matching instead, so a JSON string like `"7"` (which the
  model does sometimes emit even though the tool schema says `int`)
  can't match any race name and silently falls back to the season's
  first race rather than raising. This is the same class of bug as the
  hallucination above but in the opposite direction — also see README's
  Failure modes. **`get_all_time_driver_records(top_n)`** is the one
  function here that doesn't call `fastf1.get_session` at all — it walks
  every season's *final standings* via `Ergast().get_driver_standings`
  since 1950 (one call per season; each row already carries that season's
  `wins` and `position`, so career totals never need per-round data) and
  caches the result process-lifetime behind a `threading.Lock` — same
  lazy-singleton shape as `predictor/features.py`'s feature cache, reused
  here because the walk is the same kind of "expensive once, cheap after"
  problem. `STATS_SYSTEM_PROMPT` also permits answering *other* all-time
  F1 trivia this tool doesn't cover (poles, podiums, GOAT debates) from
  general knowledge, explicitly labeled as such — see README's Failure
  modes for why this exists (without it, the supervisor had nothing
  concrete to say and looped through disclaimers instead of answering).
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
- **`agent/graph.py`** builds a `langgraph-supervisor` root agent
  (`create_supervisor`) that delegates to three specialist sub-agents, each
  built via `langchain.agents.create_agent` (not
  `langgraph.prebuilt.create_react_agent`, which this project migrated
  away from mid-build after it was deprecated in favor of the former):
  `agent/stats_agent.py` wraps `FASTF1_TOOLS` for factual/numeric
  questions, `agent/narrative_agent.py` wraps `RAG_TOOLS` for "why"/
  "what happened" questions, `agent/predictor_agent.py` wraps
  `PREDICTOR_TOOLS` for forward-looking "who will win" model
  predictions. Each specialist needs a unique `name=` on
  `create_agent(...)` — `create_supervisor` uses it to build a
  `transfer_to_<name>` handoff tool per agent and raises if any two
  agents share a name. **`create_supervisor` must be called with
  `output_mode="full_history"`** — the default (`"last_message"`) only
  passes each specialist's final summarized answer back up, not its tool
  calls, which silently breaks `agent/citations.py` and `agent/cost.py`
  (both scan `result["messages"]` for specific tool/AI message shapes,
  and both degrade to empty/zero rather than erroring on a miss — this is
  the kind of failure you won't notice without checking `citations`
  actually comes back non-empty on a live query). Conversation memory is
  an `InMemorySaver` checkpointer attached at `.compile(checkpointer=...)`
  on the *supervisor* graph, not the individual specialists — passing the
  same `thread_id` across `agent/run.py`'s `ask()` calls continues that
  conversation.
- **Supervisor routing isn't guaranteed by tool availability alone.**
  Early testing showed the supervisor would sometimes answer a "why did X
  matter" sub-question directly from `stats_agent`'s numbers instead of
  calling `narrative_agent`, even with an explicit routing example in the
  prompt — LLM tool-routing is probabilistic, and rich enough intermediate
  data makes the model more confident it doesn't need another hop. Fixed
  by making the supervisor prompt explicitly forbid answering "why"/"what
  happened" content from `stats_agent`'s data alone.
- **`tools/rag_tools.py`'s lazy retriever singleton needed a lock.** A
  supervisor can issue parallel tool calls within one turn (e.g. two
  `search_race_recaps` calls for two sub-queries), and two threads racing
  through `_get_retriever()`'s first `is None` check both tried to
  construct the Chroma client for the same path at once, raising a
  `KeyError` inside chromadb's client registry. This never surfaced with
  the single-agent version because it never had two threads hitting that
  first-call path simultaneously. Fixed with a `threading.Lock` double-
  checked lock, the same pattern `app/streamlit_app.py` already uses for
  its shared cost tracker.
- **`SUPERVISOR_PROMPT` must state that its final message is the only
  thing the user sees.** Without that, the supervisor sometimes ended a
  turn with something like "let me know if you'd like a deeper dive..."
  instead of restating a specialist's actual answer — `run.py`'s `ask()`
  only returns `result["messages"][-1]`, so if the supervisor doesn't
  repeat the substance, the user gets a non-answer even though the
  correct data is sitting earlier in the (fully preserved,
  `full_history`) message trace. Fixed by adding an explicit instruction
  to the prompt: restate the specialist's actual content, don't just
  reference that it answered.
- **`predictor/features.py` reproduces the ported model's training
  pipeline exactly, quirks included** (e.g. `FormRatio`'s
  per-driver-vs-team-total unit mismatch, `RoundsRemaining`'s
  `total_rounds - (Round - 1)` off-by-one for an in-progress season) —
  the model's weights are calibrated to those exact numeric
  distributions, so "fixing" a formula during the port would silently
  miscalibrate predictions rather than improve them. `predict.py` keeps
  a lazy, thread-safe, in-process cache of the built feature table keyed
  by season (same lock pattern as `rag_tools.py` above) since building it
  means walking every completed round of a season through fastf1.
- **`agent/time_context.py::current_date_context()`** states today's
  date to every agent — LLMs have no innate sense of "now," and without
  this, "this year"/"the current season" silently resolved to whatever
  year the model leaned toward from training, not the real one. It must
  be recomputed on every model call, never baked into a static prompt
  string at `build_agent()` time, since the compiled agent is cached for
  the life of the server process (`@st.cache_resource` in
  `app/streamlit_app.py`) and a baked-in date would just go stale on a
  different schedule than the bug it fixes. Wired in two different ways
  because the three `create_agent`-based specialists and the
  `create_supervisor`-based supervisor don't share a prompt mechanism:
  specialists use `langchain.agents.middleware.dynamic_prompt` (a
  decorated function returning the system prompt *string*, passed via
  `middleware=[...]`); the supervisor's `create_supervisor(prompt=...)`
  goes through `langgraph.prebuilt.create_react_agent` underneath, whose
  callable `prompt` contract is different and easy to get wrong: it must
  return the *entire* message list (`[SystemMessage(...)] +
  state["messages"]`), not just the system prompt text. Returning a bare
  string once replaced the model's entire input with just the system
  prompt, silently dropping the user's actual message — the supervisor
  responded as if introducing itself rather than answering, caught only
  by asking it a real question and reading the answer.
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
