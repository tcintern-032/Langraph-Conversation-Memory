"""
PERSISTENT CONVERSATION CONTEXT (checkpointer)
================================================
A LangGraph "checkpointer" is what actually stores state between calls,
keyed by thread_id (see graph.py / main.py). Swapping the backend does
NOT require touching your graph or node code -- that's the point of
separating state (what's remembered) from persistence (where it's
stored).

- MemorySaver        -> in-process dict, lost on restart. Good for tests.
- SqliteSaver         -> writes to a local .sqlite file, survives restarts.
                         (This is the bonus "persistent memory" requirement.)

For production you'd swap in PostgresSaver / RedisSaver etc. from the
same `langgraph.checkpoint.*` family with no graph changes.
"""
import os

from langgraph.checkpoint.memory import MemorySaver

USE_SQLITE = os.getenv("USE_SQLITE_MEMORY", "true").lower() == "true"
DB_PATH = os.getenv("MEMORY_DB_PATH", "conversation_memory.sqlite")

_checkpointer = None
_sqlite_cm = None  # keep the context manager alive for the process lifetime


def get_checkpointer():
    """Return a singleton checkpointer instance for the process."""
    global _checkpointer, _sqlite_cm

    if _checkpointer is not None:
        return _checkpointer

    if USE_SQLITE:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            _sqlite_cm = SqliteSaver.from_conn_string(DB_PATH)
            _checkpointer = _sqlite_cm.__enter__()
            return _checkpointer
        except ImportError:
            print(
                "[memory] langgraph-checkpoint-sqlite not installed -- "
                "falling back to in-memory checkpointer (pip install "
                "langgraph-checkpoint-sqlite for persistence)."
            )

    _checkpointer = MemorySaver()
    return _checkpointer


def reset_session(checkpointer, thread_id: str) -> bool:
    """
    Allow conversation history to be reset for one session/thread_id
    without touching any other thread's history.
    """
    try:
        checkpointer.delete_thread(thread_id)
        return True
    except AttributeError:
        # Older checkpointer versions without delete_thread support.
        return False
