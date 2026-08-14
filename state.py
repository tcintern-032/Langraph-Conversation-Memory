"""
LANGGRAPH STATE
===============
In LangChain you usually kept history yourself: a list you appended to
and passed back into the chain on every call.

In LangGraph, history IS the state. The graph carries a typed dict
across every node. The "messages" field below uses the `add_messages`
reducer, which means:

  - New messages returned by a node are APPENDED to the existing list
    (not overwritten).
  - If a returned message has the same `id` as one already in state,
    it REPLACES that message instead of duplicating it (useful for
    streaming/tool-call updates).

Every node in the graph receives this state and returns a partial
update to it -- that's the whole memory mechanism. No manual list
management needed.
"""
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
