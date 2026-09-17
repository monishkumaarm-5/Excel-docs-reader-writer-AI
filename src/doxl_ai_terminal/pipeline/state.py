# state.py
"""
State data structure for the LangGraph multi-agent system.

Fields:
  - file_type:       "docx" or "xlsx" — decides which agent system runs
  - wants_continue:  True if user wants to process another file
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Core state shared across all nodes in the graph."""

    # Current state/node name
    state: str

    # Which agent is currently active
    current_agent: str

    # Whether onboarding is complete
    onboarded: bool

    # API error message (empty = no error)
    api_error: str

    # Gemini model name
    model: str

    # Gemini API key
    api_key: str

    # File path provided by user
    file_path: str

    # Chat messages (LangChain message history)
    messages: Annotated[list, add_messages]

    # Detected file type: "docx" or "xlsx"
    file_type: str

    # Whether user wants to process another file
    wants_continue: bool


def create_initial_state() -> AgentState:
    """Create a fresh initial state."""
    return AgentState(
        state="start",
        current_agent="",
        onboarded=False,
        api_error="",
        model="",
        api_key="",
        file_path="",
        messages=[],
        file_type="",
        wants_continue=False,
    )
