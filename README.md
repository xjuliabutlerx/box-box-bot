# box-box-bot

A conversational F1 chatbot agent built with LangChain/LangGraph. Answers
natural-language questions using both live/structured data (fastf1 tool
calls) and narrative context (RAG over race recaps), e.g. "How did the
constructor standings change after Monza?" or "Why did [driver] lose the
championship in [year]?"

## Stack

- **langchain** (`langchain.agents.create_agent`) — builds each specialist
  agent's tool-calling loop; this superseded
  `langgraph.prebuilt.create_react_agent` mid-build (see Failure modes
  below)
- **langgraph-supervisor** (`create_supervisor`) — the root agent that
  routes between specialists and composes the final answer
- **langgraph** (`checkpoint.memory.InMemorySaver`) — conversation memory
- **fastf1** + **fastf1.ergast** — race/session results and championship
  standings
- **chromadb** + **fastembed** — local vector store + local embeddings for
  RAG over race summaries (no embeddings API key required)
- **torch** — runs the constructor-championship prediction model
  (ported from a separate project, see Predictor below); CPU-only, no
  GPU required
- **langchain-anthropic** — Claude Sonnet 5 as the agent's model
- **LangSmith** — tracing
- **streamlit** — chat UI, deployed on Streamlit Community Cloud

## Project layout

```
src/box_box_bot/
  config.py        # env loading, LangSmith wiring, fastf1 cache path
  data/
    fastf1_client.py  # only module that talks to fastf1/Ergast directly
  rag/
    embeddings.py      # local fastembed model adapted to LangChain's Embeddings
    ingest.py           # builds the Chroma vector store from data/race_recaps/
    retriever.py        # opens the persisted store, hands back a retriever
  predictor/
    model.py            # the ported PyTorch ranking network + checkpoint loader
    features.py          # feature engineering, ported from f1-constructors-predictor
    predict.py            # orchestrates features -> model -> ranked team list
    weights/
      monaco_model_v3.pt  # bundled checkpoint (63KB), checked into git
  tools/
    fastf1_tools.py   # LangChain @tool wrappers around data/
    rag_tools.py       # @tool wrapper around rag/retriever.py
    predictor_tools.py  # @tool wrapper around predictor/predict.py
  agent/
    graph.py            # builds the supervisor: specialists + routing prompt + checkpointer
    stats_agent.py       # specialist: fastf1 tools, for factual/numeric questions
    narrative_agent.py    # specialist: the RAG tool, for "why"/"what happened" questions
    predictor_agent.py     # specialist: the prediction tool, for "who will win" questions
    run.py               # ask() - the interface the app calls: invoke + extract answer/citations/cost
    citations.py         # pulls structured citations out of tool results, verified against the answer text
    cost.py               # per-turn token usage -> estimated dollar cost
  app/
    streamlit_app.py    # chat UI, session memory wiring, cost guardrails
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

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

53 tests covering every layer except the Streamlit app itself (which is
verified manually, by driving it in a real browser — see Failure modes
below for why that mattered). Every `fastf1`/`Ergast` and `ChatAnthropic`
call is mocked, so the suite runs in about a second with no network calls,
no Anthropic spend, and no `ANTHROPIC_API_KEY` required — the one
exception is `fastembed`'s local model in `test_embeddings.py`, which
needs network access only for its one-time download if it isn't already
cached.

## Data access layer

`box_box_bot.data.fastf1_client` currently exposes:

- `get_driver_standings(season, round=None)`
- `get_constructor_standings(season, round=None)`
- `get_race_results(season, round)`
- `get_fastest_laps(season, round, session_type="R", top_n=5)`
- `get_season_schedule(season)`

Standings come from `fastf1.ergast` (session objects don't carry cumulative
championship state); race results, lap data, and the race calendar come
from `fastf1.get_session(...).load()` and `fastf1.get_event_schedule(...)`.

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

## Tools layer

`tools/fastf1_tools.py` wraps each `data/fastf1_client.py` function as a
`@tool(parse_docstring=True)`-decorated function: the docstring's `Args:`
block becomes per-parameter descriptions in the tool's JSON schema, and
`json.dumps(data, default=str)` turns pandas output (including
`Timedelta`/`NaT`/`Timestamp` values that plain `json.dumps` can't
serialize) into text for the model. `tools/rag_tools.py` wraps the
retriever the same way, as a single `search_race_recaps` tool, and tags
every retrieved chunk with `[Source: {race_name} ({season})]` in the
tool's own output — that tag is what makes citations possible later,
since the model reads it as part of the tool result rather than us
reverse-engineering sources after the fact.

## Agent

box-box-bot is a multi-agent system built with `langgraph-supervisor`:
`agent/graph.py`'s `build_agent()` builds a supervisor that delegates to
two specialist sub-agents, each its own `langchain.agents.create_agent`
instance with its own tools and system prompt —
`agent/stats_agent.py` (fastf1 tools, for factual/numeric questions) and
`agent/narrative_agent.py` (the RAG tool, for "why"/"what happened"
questions). The supervisor's own prompt decides which specialist(s) a
question needs and composes the final answer from what they return; a
question like "how did the standings change after Monza and why" hits
both in one turn. `agent/run.py`'s `ask(agent, message, thread_id)` is
still the interface everything else calls, unchanged by this — invoke,
extract the answer text (Claude Sonnet 5 returns content as a list of
thinking/text blocks when tools are involved, not a plain string),
extract citations, and estimate cost — returning
`{"answer", "citations", "usage"}`. This is a deliberate property of the
layered design: `run.py` only depends on `build_agent()` returning
something with `.invoke()`, so swapping a single agent for a supervisor
graph underneath it required zero changes above this layer.

`create_supervisor` must be called with `output_mode="full_history"`
(the default, `"last_message"`, only passes each specialist's final
answer up — not its tool calls — which would silently zero out citations
and per-call cost tracking below, since both scan the message list for
specific tool/AI message shapes and degrade to empty rather than
erroring). See Failure modes below for two more things live testing
turned up when this went from one agent to three.

### Memory

Passing the same `thread_id` across calls to `ask()` continues that
conversation — the checkpointer persists the full message state keyed by
thread ID, so follow-ups like "what about the Italian Grand Prix that
same year?" resolve correctly without the caller re-supplying context.
`InMemorySaver` is process-local (fine for a single Streamlit process);
swapping to `SqliteSaver`/`PostgresSaver` would be a one-line change if
conversations ever needed to survive a server restart.

### Citations

`agent/citations.py` does this in two steps rather than one:

1. `extract_citations(messages)` parses every `search_race_recaps` tool
   result from the current turn (sliced from the last human message
   onward, so a multi-turn conversation doesn't re-surface old turns'
   sources) and pulls out every `[Source: ...]` tag as a candidate.
2. `filter_citations_by_answer(citations, answer_text)` keeps only the
   candidates the model's answer actually names, matching on each race's
   short name ("Bahrain") rather than the full official name ("Bahrain
   Grand Prix") — see Failure modes below for why.

Splitting it this way means citations are true to what the model actually
used, not just what got retrieved — a retrieved-but-unused chunk (see the
RAG retrieval-precision limitation above) doesn't show up as a false
citation.

### Cost estimation

`agent/cost.py` sums `usage_metadata` (LangChain's provider-normalized
token counts) across every model call in a turn — a single turn can
involve multiple calls (decide to use a tool, then a follow-up call once
the tool result comes back) — and prices it against Claude Sonnet 5's
published per-token rates. Observed cost in testing: roughly
$0.03-$0.15 per query depending on how many tool calls it took.

## Predictor

`predictor_agent` wraps a constructor-championship ranking model ported
from a separate project (`f1-constructors-predictor`) rather than
trained here — a small PyTorch network (`predictor/model.py`) trained
with a pairwise-ranking loss, so its output is a per-team score to sort,
not a probability or a points prediction. Scoped down from that
project's full 3-model-version x 5-circuit ensemble to one version, one
checkpoint (`monaco_model_v3.pt`, its own dashboard's pick as "the most
accurate").

**The governing rule for the ported feature pipeline
(`predictor/features.py`): reproduce every formula exactly as it was
during training, quirks included.** The model's weights are calibrated
to the exact numeric distributions the source pipeline produced — so a
formula that looks slightly odd is *supposed* to look that way, and
"fixing" it would silently feed the model out-of-distribution numbers
rather than improve anything. Two quirks preserved on purpose:
`FormRatio` divides the team's 3-round point total by a *per-driver*
average x3, making the ratio run ~2x what its name implies; and
`RoundsRemaining` for an in-progress season is `total_rounds -
(Round - 1)`, not `total_rounds - Round` — a one-off inconsistency in
the source between its base formula and its in-progress-season
override, replicated exactly since predicting an in-progress season is
the only thing this tool is for.

Two accepted, documented deviations from the source (not bugs to chase):
box-box-bot only ever fetches the race (`'R'`) session, so it doesn't
reproduce the source's accidental sprint+race row mixing on sprint
weekends; and the cross-team features (`RelativePointsShare`,
`PercentileRankAfterRound`) are computed from points already
accumulated locally while building the feature table, rather than a
separate Ergast standings call.

Building the full feature table means walking every completed round of
a season through fastf1 — expensive on a cold cache. `predict.py` keeps
a lazy, thread-safe, in-process cache keyed by season (same
double-checked-lock shape as `rag_tools.py`'s retriever singleton, see
Failure modes below): the first predictor query for a season pays the
full cost, every later one is near-instant.

## App (Streamlit) & cost guardrails

`app/streamlit_app.py` is a standard `st.chat_input`/`st.chat_message`
loop, with two things worth calling out:

- **Shared vs. per-session state.** The agent itself (`get_agent()`) is
  built once per server process via `@st.cache_resource` and shared
  across every visitor — rebuilding it (and its `ChatAnthropic` client)
  on every message would be wasteful. What's *not* shared is
  conversation state: each browser session gets its own `thread_id`
  (`st.session_state`), so the checkpointer keeps every visitor's
  conversation isolated even though they're all served by the same
  cached agent.
- **Two-layer cost guardrail**, since this is a public demo funded by
  the author's own API key:
  - `MAX_MESSAGES_PER_SESSION` (per-`st.session_state`, so a page refresh
    resets it) caps how far any single conversation can run.
  - `MAX_TOTAL_COST_USD` (a `@st.cache_resource`-shared dict + a
    `threading.Lock`) caps total spend across *every* visitor combined,
    and survives a page refresh since it isn't session-scoped — it only
    resets if the server process itself restarts. This is the one that
    actually matters if a link gets shared widely; the per-session cap is
    just pacing for a single visitor.
  - Neither is a perfectly airtight ceiling (see Failure modes below) —
    the real backstop is a spend limit set directly in the Anthropic
    Console, which holds regardless of anything the app does.

### Deployment

Deployed on Streamlit Community Cloud from `master`, main file path
`src/box_box_bot/app/streamlit_app.py`. Secrets (`ANTHROPIC_API_KEY`,
`LANGSMITH_API_KEY`, etc.) are set via the Cloud dashboard's Secrets
panel rather than committing `.env` — `streamlit_app.py` mirrors
`st.secrets` into `os.environ` before importing anything from
`box_box_bot`, so `config.py`'s `os.environ.get(...)` calls work
unchanged whether running locally (`.env`) or deployed (`st.secrets`).
`requirements.txt` ends with `-e .` so Streamlit Cloud's
`pip install -r requirements.txt` alone makes `box_box_bot` importable,
without a separate editable-install step.

## Failure modes hit

Real bugs found while building this, mostly via actually running the
code rather than reading it — a reminder that "the code looks right" and
"the code works" are different claims:

- **Silent no-op tool.** An early draft of `get_race_results` had a
  docstring but no function body — it silently returned `None` on every
  call. Caught by invoking the tool directly, not by review.
- **`json.dumps` crashing on pandas types.** Race-results output includes
  `NaT` (missing qualifying times) and `Timedelta` (race time), neither
  of which the stdlib `json` module can serialize. Fixed with
  `json.dumps(data, default=str)` across all four tools rather than
  patching individual columns — a more general fix for a class of
  problem that recurs with any pandas-derived tool output.
- **`create_react_agent` deprecated mid-build.** `langgraph.prebuilt`
  flagged it in favor of `langchain.agents.create_agent` — same
  underlying tool-calling loop, but the system-prompt parameter was
  renamed (`prompt` -> `system_prompt`).
- **`temperature` rejected outright.** Claude Sonnet 5 returns a 400 for
  `temperature` — the parameter is deprecated for this model rather than
  merely optional.
- **Citations disappeared nondeterministically.** Matching citations
  against the full official race name ("Bahrain Grand Prix") worked only
  when the model happened to use that exact phrase; testing the same
  question twice showed a citation appear in one run and vanish in
  another because the model wrote "winning in Bahrain" instead. Fixed by
  matching on the race's short name instead.
- **Streamlit's stale module cache.** Editing `agent/run.py` and
  `agent/cost.py` while a `streamlit run` dev server was already running
  produced a `KeyError` that had nothing to do with the (correct) code —
  the running process hadn't picked up the edit to an editable-installed
  package under `src/`. Restarting the dev server resolved it; worth
  knowing this can happen with `pip install -e .` packages specifically.
- **`st.metric` captured stale state.** A sidebar cost widget was called
  *before* the code that updates the cost for the current turn ran,
  so it always displayed the previous turn's numbers, one turn late.
  Streamlit widgets aren't reactive — they render with whatever state
  existed at the moment they're called in that script run, not whatever
  state exists by the end of it. Fixed by moving the widget to run after
  the state update.
- **Markdown's LaTeX math mode ate a cost display.** `st.caption(f"...
  ${x} ... ${y}")` rendered the text *between* the two `$` signs in math
  font, since Markdown/KaTeX treats a `$...$` pair as inline math
  delimiters — not a bug in the numbers, just in the string. Fixed by
  escaping to `\$`.
- **Cost caps aren't perfectly airtight.** The global cost cap checks the
  running total, then only updates it after the (multi-second) API call
  returns — concurrent requests arriving in that window could all pass
  the check before any of them are counted, letting the total briefly
  overshoot. Acceptable for expected traffic on a portfolio demo; the
  actual hard backstop is the spend limit set in the Anthropic Console,
  not this in-app counter.
- **A racing lazy-singleton crashed the RAG tool.** Going from one agent
  to a supervisor made LangGraph issue parallel tool calls far more often
  (e.g. two `search_race_recaps` calls for two sub-queries in the same
  turn). `tools/rag_tools.py`'s retriever was a lazy singleton with no
  lock — two threads racing through its first `is None` check both tried
  to construct the Chroma client for the same path at once, and one of
  them hit a raw `KeyError` inside chromadb's own client registry. Fixed
  with a `threading.Lock` double-checked lock, the same pattern already
  used for the Streamlit cost tracker below. This bug existed before the
  supervisor refactor too — it just never had two threads hit that
  first-call path at the same moment until multi-agent tool calls made it
  common instead of theoretical.
- **The supervisor doesn't reliably delegate just because a specialist
  exists.** Early testing showed it would sometimes answer a "why did X
  matter" sub-question directly from `stats_agent`'s numbers instead of
  calling `narrative_agent`, even with an explicit routing example in the
  prompt — a "why" question with rich enough stats already sitting in
  context is enough for the model to feel it doesn't need another hop.
  Fixed by making the supervisor prompt explicitly forbid answering
  "why"/"what happened" content from `stats_agent`'s data alone. Worth
  remembering for the phase 2 predictor agent too: tool/agent
  *availability* isn't the same as tool/agent *use*, and that gap only
  shows up by actually running real questions through it.
- **`stats_agent` hallucinated a round number for a named race.**
  `get_race_results(season, round)` required an integer round, so a
  question about "the 2025 Bahrain Grand Prix" forced the model to
  recall which round number Bahrain was that year rather than just using
  the name it already knew for certain — and it got Round 1 instead of
  Round 4. `stats_agent`'s wrong results then visibly contradicted
  `narrative_agent`'s (correct) RAG answer in the same turn. Root cause:
  `fastf1.get_session(year, gp, ...)` already accepts `gp` as a string
  and fuzzy-matches it against each event's country/location/name — the
  tool wrapper was more restrictive than the library it wraps, for no
  reason. Fixed by widening `round` to `int | str` on `get_race_results`
  and `get_fastest_laps` (the two tools that hit `get_session`) and
  telling the model in both the tool docstring and `STATS_SYSTEM_PROMPT`
  to pass the race name instead of guessing a round number it isn't sure
  of. This bug predates the supervisor split — the single-agent version
  had the identical vulnerability, it just never had a second data
  source in the same turn to visibly contradict.
- **The supervisor's final answer sometimes dropped the actual result.**
  Adding `predictor_agent` surfaced this: `predictor_agent` returned a
  fully correct, well-formatted prediction, but the supervisor's own
  final message to the user was just *"let me know if you'd like a
  deeper dive into why the model favors Mercedes..."* — no restatement
  of the prediction itself. `run.py`'s `ask()` only returns the last
  message in `result["messages"]`, which for a supervisor graph is
  whatever the supervisor says last, not necessarily a specialist's
  content — and nothing in the prompt told the supervisor its final
  reply was the *only* thing the user would ever see. Fixed by adding
  that constraint explicitly to `SUPERVISOR_PROMPT`: restate a
  specialist's actual substantive content, don't just reference that it
  answered. Same lesson as the routing gap above - a well-designed
  message pipeline (`output_mode="full_history"`, citation/cost
  extraction) doesn't guarantee a *complete* final answer; that's a
  prompt-behavior question you only catch by reading what the model
  actually says back to a real question, not by inspecting the message
  trace.
- **The topic gate rejected legitimate follow-ups and unfamiliar
  names.** `agent/topic_gate.py::check_topic()` originally classified
  each message in total isolation — no conversation history at all — so
  a bare "yes" replying to the agent's own follow-up question had
  nothing to anchor it to the F1 conversation it was actually part of,
  and got bounced as topic-less. Separately, the gate's strict prompt
  gave it no room to say "I don't recognize this driver's name, but the
  message is still shaped like an F1 question" — a real gap, since
  Haiku's training data doesn't cover this project's live/current-season
  content, which is exactly what the main agent's tools exist to look
  up. Fixed two ways: `run.py::ask()` now reads the prior turn's last
  message via `agent.get_state(config)` (a free local checkpointer read,
  no API call) and passes it to `check_topic()` as `recent_context`, so
  a short reply can be judged as a continuation instead of standalone;
  and the gate's prompt got two narrow, specific carve-outs (short
  replies given context, unrecognized proper nouns) rather than a
  blanket "when unsure, allow it" policy — the gate should stay just as
  strict everywhere else, confirmed by re-running the original
  prompt-injection and jailbreak examples after the fix.
- **Agents had no sense of "now."** LLMs don't know today's date unless
  it's in their context, and nothing in any system prompt stated it — so
  "this year" silently resolved to whatever year the model's training
  data leaned toward, not the real current year, quietly passing the
  wrong `season` to tools rather than erroring. Fixed by computing the
  date fresh on every model call via `agent/time_context.py`, injected
  through `langchain.agents.middleware.dynamic_prompt` for the three
  `create_agent`-based specialists and a callable `prompt` for the
  `create_supervisor`-based supervisor — deliberately *not* baked into a
  static string at `build_agent()` time, since the compiled agent is
  cached for the life of the server process (`@st.cache_resource`) and a
  baked-in date would just go stale on a different schedule than the bug
  being fixed. **A real bug turned up wiring the supervisor's half of
  this**: `create_react_agent` (which `create_supervisor` builds its
  root agent on) expects a callable `prompt` to return the *entire*
  message list (system message + existing conversation) — not just the
  system prompt text, unlike the `dynamic_prompt` middleware used for
  the specialists. Returning a bare string replaced the model's *entire*
  input with just the system prompt, so the user's actual question never
  reached the model at all — it just introduced itself instead of
  answering. Live-testing caught this immediately (the answer to a real
  standings question was "I'm ready to help, please go ahead and ask");
  fixed by returning `[SystemMessage(...)] + state["messages"]`,
  matching exactly what `create_react_agent` itself does internally for
  a plain string prompt.
- **A numeric string round silently resolved to the wrong race.**
  Widening `round` to `int | str` fixed the hallucination bug above, but
  introduced a narrower regression: `fastf1.get_session` only treats
  `round` as a round *number* when it's an `int` — a `str` round is
  *always* fuzzy-matched against each event's country/location/name
  instead, never parsed as a number. When the model passed `round` as a
  JSON string (e.g. `"7"` instead of `7`), fastf1 couldn't fuzzy-match a
  bare digit against any race name, and rather than raising, it silently
  fell back to the season's first race — `round="7"` and `round="1"`
  both resolved to the same (wrong) event. Caught when a Barcelona
  (round 7) question returned Australian Grand Prix (round 1) results;
  LangSmith showed the *tool call* used the correct round, so the bug had
  to be in how that round value was resolved downstream, not in routing
  or prompting. Fixed with `_normalize_round()` in `fastf1_client.py`:
  any string round that's purely digits is converted to `int` before
  reaching `fastf1.get_session`, so only genuine race names still go
  through fuzzy matching. Applied at every `fastf1.get_session` call site
  (`get_race_results`, `get_fastest_laps`, `get_tire_strategy`,
  `get_race_control_messages`, `get_weather_for_session`), not just the
  one that surfaced it.
- **Follow-up questions looked like memory loss but weren't.** Asking
  "who is the best driver" got a clarifying question back; answering
  "historically best" got a vague non-answer that just restated the same
  ambiguity — looking, from the outside, exactly like the agent forgot
  the conversation. Live tracing the full message history proved
  otherwise: `langgraph_supervisor`'s handoff tool forwards the *entire*
  conversation to whichever specialist it calls, and every prior turn was
  present in the state. The real problem was capability, not memory —
  `stats_agent`'s tools are all `(season, round)`-scoped, `narrative_agent`
  is single-race RAG, and `predictor_agent` only forecasts the *current*
  constructors' championship, so nothing in the system could answer a
  career-spanning question at all. `stats_agent`'s own prompt forbidding
  it from "speculating about numbers you haven't looked up" meant it
  couldn't fall back on its own knowledge either, so the supervisor just
  cycled through rephrased disclaimers forever. Fixed two ways: added
  `get_all_time_driver_records()` (`fastf1_client.py`) — real aggregated
  data, not a guess — by walking every season's *final standings* since
  1950 via Ergast (confirmed live that each row already carries that
  season's `wins` count, a stable `driverId`, and `position` where `1`
  means champion, so career totals only cost one Ergast call per season,
  no per-round data needed) cached process-lifetime behind a
  `threading.Lock`, the same shape as `predictor/features.py`'s feature
  cache; and `STATS_SYSTEM_PROMPT` now explicitly permits answering other
  well-known, static all-time trivia (most poles, most podiums, GOAT
  debates) that no tool covers, from general knowledge, clearly labeled
  as such — never for anything current-season or dynamic, which stays
  tool-only.
