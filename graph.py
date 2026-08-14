"""
Wires everything together:

    START -> chat -> (tools_condition) -> tools -> chat -> ... -> END

`tools_condition` inspects the last AI message: if it contains tool
calls, route to the "tools" node; otherwise route straight to END.
Both branches write into the SAME state, so tool results become part
of permanent thread memory just like normal messages.

Compiling with `checkpointer=` is what turns this from a single-call
graph into a multi-turn, multi-session memory system: every `.invoke()`
call automatically loads prior state for the given thread_id, applies
the update, and saves it back.
"""
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from state import ChatState
from nodes import chat_node
from tools import TOOLS
from memory import get_checkpointer


def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("chat", chat_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "chat")
    builder.add_conditional_edges("chat", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "chat")

    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
