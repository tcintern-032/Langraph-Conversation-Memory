"""
STATE ACROSS MULTIPLE NODES
============================
This node is deliberately "dumb": it does not track history itself.
It just reads state["messages"] (everything so far, injected by the
graph/checkpointer) and returns the new AI message. LangGraph merges
that into state via the add_messages reducer defined in state.py.

The same state object is what flows into the "tools" node too (see
graph.py), so a tool call and its result live in the exact same
history the chat node sees on the next turn.
"""
from langchain_core.messages import SystemMessage

from config import llm
from tools import TOOLS

llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant. You have access to the full prior "
        "conversation in this thread -- use it to resolve follow-up "
        "questions (pronouns, 'that', 'the previous one', etc.) instead "
        "of asking the user to repeat themselves. Use tools when they "
        "would make your answer more accurate."
    )
)


def chat_node(state: dict) -> dict:
    messages = state["messages"]

    # Only prepend the system prompt if this thread doesn't already have one
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_PROMPT] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
