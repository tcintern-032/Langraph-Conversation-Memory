# Langraph-Conversation-Memory
# LangGraph Conversation Memory Agent

Same goal as your LangChain version — the agent remembers prior turns and
uses them in follow-ups — but memory is now a first-class part of the
**graph's state**, not a list you manage by hand.

## Project layout

```
config.py    LLM setup (reads .env)
state.py     ChatState schema — the memory data structure
tools.py     Two example tools (calculator, get_current_time)
nodes.py     chat_node — reads state, calls the LLM, returns an update
memory.py    Checkpointer setup (SQLite-persisted or in-memory) + reset_session()
graph.py     Wires nodes + checkpointer into a compiled graph
main.py      CLI: chat, switch sessions, view history, reset history
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in OPENAI_API_KEY (and LangSmith keys if wanted)
python main.py
```

## How each topic maps to the code

**LangGraph State** — `state.py`. `ChatState.messages` uses the
`add_messages` reducer: nodes return *partial* updates (e.g. `{"messages": [new_msg]}`)
and LangGraph appends them to the running list automatically, instead of you
overwriting or manually concatenating a history array.

**Message History** — every node (`chat_node` in `nodes.py`, and the built-in
`ToolNode`) reads `state["messages"]` — the full conversation so far — and
returns new messages that get merged back in. No separate memory object to
sync.

**Thread / Session IDs** — `main.py` passes
`config={"configurable": {"thread_id": ...}}` on every `graph.invoke()` call.
The checkpointer uses that id to load/save that thread's state in isolation.
Different `thread_id` = a completely separate conversation, same graph.

**Persistent Conversation Context** — `memory.py`. The graph is compiled
with `checkpointer=` (SQLite by default), so state survives process restarts.
Set `USE_SQLITE_MEMORY=false` in `.env` to fall back to `MemorySaver`
(in-memory only, useful for tests).

**State Across Multiple Nodes** — `graph.py`: `chat -> tools -> chat`. When
the model calls a tool, the tool call *and* its result both get written into
the same `messages` list before control returns to `chat`, so the model
"remembers" what it looked up on the next turn too — not just what it said.

## CLI commands

```
/new <id>      start (or jump to) a session with this id
/switch <id>   switch to an existing session id
/reset         clear history for the CURRENT session only
/history       print the message history for the current session
/sessions      list session ids used so far this run
/exit          quit
```

Try this to see memory + follow-ups in action:

```
(default) You: My name is Sam and I live in Austin.
Assistant: Nice to meet you, Sam! ...

(default) You: What city did I say I live in?
Assistant: You said you live in Austin.

(default) You: /new work
-> Switched to NEW session 'work'

(work) You: What city did I say I live in?
Assistant: I don't have that information yet in this session — could you tell me?
```

That last answer proves session isolation: `work` has no access to
`default`'s history because it's a different `thread_id`.

## Bonus features included

- **Persistent memory (checkpoint/store):** SQLite-backed checkpointer in
  `memory.py` — history survives restarting `python main.py`.
- **Memory + tool calling combined:** `tools.py` + `ToolNode` in `graph.py`.
  Ask "what's 12 * (7+3)?" or "what time is it?" and the tool call/result
  become part of permanent thread history.
- **LangSmith tracing:** just set `LANGCHAIN_TRACING_V2=true`,
  `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` in `.env` — LangChain/LangGraph
  pick these up automatically, no code changes needed. Every node execution,
  state transition, and tool call for a thread will show up as a trace in
  your LangSmith project.

## Notes / things you may want to change

- `tools.py`'s `calculator` uses a restricted `eval` for demo purposes only —
  swap in a real expression parser (e.g. `numexpr`) before using this beyond
  a demo.
- For production-scale persistence, swap `SqliteSaver` for `PostgresSaver`
  or `RedisSaver` from the same `langgraph.checkpoint.*` family — no changes
  needed to `graph.py` or `nodes.py`, since they only depend on the
  checkpointer interface, not the backend.
