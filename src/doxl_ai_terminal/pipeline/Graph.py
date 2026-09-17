# Graph.py
"""
LangGraph wiring — builds the state graph with all nodes and conditional edges.

Simplified flow (no interrupt node):

    START → onboarding → test_credentials
              ▲              │
              └── FAIL ◄─────┘
                        PASS ─┘
                          │
                          ▼
              ┌──→ path_request ◄──────────────────┐
              │         │                           │
              │         ▼                           │
              │    process_file                     │
              │    │         │                      │
              │  docx       xlsx                    │
              │    ▼         ▼                      │
              │  docs_agent  excel_agent            │
              │    │         │                      │
              │    └────┬────┘                      │
              │         ▼                           │
              │   continue_prompt                   │
              │    │          │                     │
              │  (yes)       (no)                   │
              │    │          ▼                     │
              │    │         END                    │
              └────┘────────────────────────────────┘
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from doxl_ai_terminal.pipeline.state import AgentState
from doxl_ai_terminal.pipeline.nodes import (
    onboarding_node,
    test_credentials_node,
    path_request_node,
    process_file_node,
    docs_agent_node,
    excel_agent_node,
    continue_node,
    route_after_test,
    route_after_process,
    route_after_continue,
)


def build_graph():
    """Build and compile the LangGraph state graph with agent routing."""
    graph = StateGraph(AgentState)

    # ── Nodes ──
    graph.add_node("onboarding", onboarding_node)
    graph.add_node("test_credentials", test_credentials_node)
    graph.add_node("path_request", path_request_node)
    graph.add_node("process_file", process_file_node)
    graph.add_node("docs_agent", docs_agent_node)
    graph.add_node("excel_agent", excel_agent_node)
    graph.add_node("continue_prompt", continue_node)

    # ── Edges ──
    graph.add_edge(START, "onboarding")
    graph.add_edge("onboarding", "test_credentials")

    graph.add_conditional_edges(
        "test_credentials", route_after_test,
        {"path_request": "path_request", "onboarding": "onboarding"},
    )

    # path_request now goes directly to process_file (no interrupt node)
    graph.add_edge("path_request", "process_file")

    graph.add_conditional_edges(
        "process_file", route_after_process,
        {"docs_agent": "docs_agent", "excel_agent": "excel_agent", "path_request": "path_request"},
    )

    graph.add_edge("docs_agent", "continue_prompt")
    graph.add_edge("excel_agent", "continue_prompt")

    graph.add_conditional_edges(
        "continue_prompt", route_after_continue,
        {"path_request": "path_request", "end": END},
    )

    # ── Compile ──
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
