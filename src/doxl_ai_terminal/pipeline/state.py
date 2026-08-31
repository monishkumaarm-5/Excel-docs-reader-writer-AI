# state.py
"""
State data structure for the LangGraph multi-agent system.

New fields added for agent routing:
  - file_type:      "docx" or "xlsx" — decides which agent system runs
  - wants_continue: True if user wants to process another file after finishing
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Core state shared across all agents in the graph"""

    # Current state/node name in the graph
    state: str

    # Task queue — holds pending tasks
    task: list

    # Human-in-the-loop interrupt flag
    interrupt: bool

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

    # Answer from human interrupt
    interrupt_answer: str

    # File path provided by user
    file_path: str

    # Whether the user confirmed the file path at the interrupt checkpoint
    confirmed: bool

    # Chat messages (LangChain message history)
    messages: Annotated[list, add_messages]

    # ── NEW: Agent routing fields ──

    # Detected file type: "docx" or "xlsx"
    file_type: str

    # Whether user wants to process another file after finishing
    wants_continue: bool


def create_initial_state() -> AgentState:
    """Create a fresh initial state"""
    return AgentState(
        state="start",
        task=[],
        interrupt=False,
        current_agent="",
        onboarded=False,
        api_error="",
        model="",
        api_key="",
        interrupt_answer="",
        file_path="",
        confirmed=False,
        messages=[],
        file_type="",
        wants_continue=False,
    )