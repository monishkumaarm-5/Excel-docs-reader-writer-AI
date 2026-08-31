# Graph.py
"""
LangGraph wiring — builds the state graph with all nodes and conditional edges.

Graph Flow:
    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │   START                                                           │
    │     │                                                             │
    │     ▼                                                             │
    │   onboarding ──→ test_credentials                                 │
    │     ▲                  │                                          │
    │     │           ┌──────┴──────┐                                   │
    │     │           │             │                                   │
    │     └── FAIL ◄──┘       PASS ─┘                                   │
    │                           │                                       │
    │                           ▼                                       │
    │               ┌──→ path_request ◄──────────────────┐              │
    │               │         │                          │              │
    │               │         ▼                          │              │
    │               │    interrupt  ← graph PAUSES       │              │
    │               │    │       │    (human-in-the-loop) │              │
    │               │    │       │                        │              │
    │               │ rejected  confirmed                │              │
    │               │    │       │                        │              │
    │               └────┘       ▼                        │              │
    │                       process_file                  │              │
    │                       │         │                   │              │
    │                  ┌────┘         └────┐              │              │
    │                  ▼                   ▼              │              │
    │             docs_agent         excel_agent          │              │
    │                  │                   │              │              │
    │                  └─────────┬─────────┘              │              │
    │                            ▼                        │              │
    │                     continue_prompt                 │              │
    │                      │          │                   │              │
    │                 (yes) │          │ (no)              │              │
    │                      │          ▼                   │              │
    │                      │         END                  │              │
    │                      └──────────────────────────────┘              │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from doxl_ai_terminal.pipeline.state import AgentState
from doxl_ai_terminal.pipeline.nodes import (
    # Existing nodes
    onboarding_node,
    test_credentials_node,
    path_request_node,
    interrupt_node,
    process_file_node,
    # New nodes
    docs_agent_node,
    excel_agent_node,
    continue_node,
    # Routing functions
    route_after_test,
    route_after_interrupt,
    route_after_process,
    route_after_continue,
)


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state graph with agent routing."""

    graph = StateGraph(AgentState)

    # ══════════════════════════════════════
    # ADD NODES
    # ══════════════════════════════════════

    # Onboarding + credentials
    graph.add_node("onboarding", onboarding_node)
    graph.add_node("test_credentials", test_credentials_node)

    # File path collection
    graph.add_node("path_request", path_request_node)
    graph.add_node("interrupt", interrupt_node)

    # File type detection
    graph.add_node("process_file", process_file_node)

    # Document/Excel agent nodes (LangGraph router + specialist subgraphs)
    graph.add_node("docs_agent", docs_agent_node)
    graph.add_node("excel_agent", excel_agent_node)

    # Continue or exit (NEW)
    graph.add_node("continue_prompt", continue_node)

    # ══════════════════════════════════════
    # EDGES
    # ══════════════════════════════════════

    # ── Entry Point ──
    graph.add_edge(START, "onboarding")

    # ── Onboarding → Test Credentials ──
    graph.add_edge("onboarding", "test_credentials")

    # ── Test Credentials → Pass/Fail ──
    graph.add_conditional_edges(
        "test_credentials",
        route_after_test,
        {
            "path_request": "path_request",
            "onboarding": "onboarding",
        }
    )

    # ── Path Request → Interrupt (always goes to human confirmation) ──
    graph.add_edge("path_request", "interrupt")

    # ── Interrupt → confirmed/rejected ──
    graph.add_conditional_edges(
        "interrupt",
        route_after_interrupt,
        {
            "process_file": "process_file",
            "path_request": "path_request",
        }
    )

    # ── Process File → Route to correct agent (NEW) ──
    graph.add_conditional_edges(
        "process_file",
        route_after_process,
        {
            "docs_agent": "docs_agent",
            "excel_agent": "excel_agent",
            "path_request": "path_request",    # fallback for unsupported types
        }
    )

    # ── Agent nodes → Continue prompt (NEW) ──
    graph.add_edge("docs_agent", "continue_prompt")
    graph.add_edge("excel_agent", "continue_prompt")

    # ── Continue prompt → Loop back or End (NEW) ──
    graph.add_conditional_edges(
        "continue_prompt",
        route_after_continue,
        {
            "path_request": "path_request",
            "end": END,
        }
    )

    # ══════════════════════════════════════
    # COMPILE WITH CHECKPOINTER
    # ══════════════════════════════════════

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    return compiled


def print_graph_structure():
    """Print the graph structure for debugging"""
    print("""
    Graph Structure:
    ────────────────
    START
      │
      ▼
    onboarding
      │
      ▼
    test_credentials
      │
      ├── FAIL → onboarding (loop)
      │
      └── PASS → path_request ◄─────────────────────┐
                    │                                 │
                    ▼                                 │
                interrupt  ← graph PAUSES             │
                    │         (human-in-the-loop)     │
                    ├── rejected → path_request       │
                    │                                 │
                    └── confirmed → process_file      │
                                    │        │        │
                              ┌─────┘        └─────┐  │
                              ▼                    ▼  │
                         docs_agent          excel_agent
                              │                    │  │
                              └────────┬───────────┘  │
                                       ▼              │
                                continue_prompt       │
                                  │        │          │
                             (yes) │       │ (no)     │
                                  │       ▼          │
                                  │      END         │
                                  └──────────────────┘
    """)