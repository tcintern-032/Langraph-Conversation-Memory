"""
Bonus: combine memory with tool calling.

These are plain LangChain tools. The graph will route to them whenever
the LLM emits a tool call, then route back to the chat node -- and the
tool call + tool result both get written into the same shared state,
so the model remembers *what it looked up* on later turns too.
"""
from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains unsupported characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


TOOLS = [get_current_time, calculator]
