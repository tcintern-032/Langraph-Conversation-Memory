"""
THREAD / SESSION IDs
=====================
Every call into the graph passes a `config` dict with
`configurable.thread_id`. That single string is the entire multi-session
mechanism: different thread_id -> completely separate, isolated history,
same graph, same checkpointer. Switching sessions is just switching a
string, not re-architecting anything.

Run:
    python main.py

Commands inside the CLI:
    /new <id>      start (or jump to) a session with this id
    /switch <id>   switch to an existing session id
    /reset         clear history for the CURRENT session only
    /history       print the message history for the current session
    /sessions      list session ids used so far this run
    /help          show commands
    /exit          quit
"""
from langchain_core.messages import HumanMessage

from graph import build_graph
from memory import get_checkpointer, reset_session


def print_help():
    print(
        """
Commands:
  /new <id>      Start/switch to a NEW session with the given id
  /switch <id>   Switch to an existing session id
  /reset         Clear history for the CURRENT session
  /history       Show the message history for the current session
  /sessions      List session ids used this run
  /help          Show this help
  /exit          Quit
"""
    )


def print_history(graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    messages = snapshot.values.get("messages", []) if snapshot else []
    if not messages:
        print("(no history yet)")
        return
    for m in messages:
        role = m.type
        content = m.content if isinstance(m.content, str) else str(m.content)
        print(f"  [{role}] {content}")


def main():
    graph = build_graph()
    checkpointer = get_checkpointer()

    thread_id = "default"
    known_sessions = {thread_id}

    print("LangGraph Memory Agent -- type /help for commands.\n")

    while True:
        try:
            user_input = input(f"({thread_id}) You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input.startswith("/new "):
            thread_id = user_input.split(" ", 1)[1].strip()
            known_sessions.add(thread_id)
            print(f"-> Switched to NEW session '{thread_id}'")
            continue

        if user_input.startswith("/switch "):
            thread_id = user_input.split(" ", 1)[1].strip()
            known_sessions.add(thread_id)
            print(f"-> Switched to session '{thread_id}'")
            continue

        if user_input == "/reset":
            ok = reset_session(checkpointer, thread_id)
            if ok:
                print(f"-> History cleared for session '{thread_id}'.")
            else:
                print(
                    "-> This checkpointer backend doesn't support delete_thread; "
                    "use /new <id> to start a fresh session instead."
                )
            continue

        if user_input == "/history":
            print_history(graph, thread_id)
            continue

        if user_input == "/sessions":
            print("Sessions used this run:", ", ".join(sorted(known_sessions)))
            continue

        # --- normal turn: this is the whole "remember + follow-up" flow ---
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        ai_message = result["messages"][-1]
        print(f"Assistant: {ai_message.content}\n")


if __name__ == "__main__":
    main()
