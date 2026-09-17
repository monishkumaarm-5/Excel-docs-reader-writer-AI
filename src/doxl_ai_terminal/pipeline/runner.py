# runner.py
"""
Graph runner — CLI entry point for the docs-excel command.

Usage:
    $ docs-excel          # after pip install
    $ python -m doxl_ai_terminal.pipeline.runner   # direct

No interrupt-resume loop needed — all human interaction happens inside
graph nodes via terminal_ui prompts.
"""

import os
import sys

# Suppress noisy TensorFlow / oneDNN info messages before anything imports TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from doxl_ai_terminal.pipeline.state import create_initial_state
from doxl_ai_terminal.pipeline.Graph import build_graph
from doxl_ai_terminal.pipeline.terminal_ui import warn


def run_graph():
    """Run the graph — the main CLI entry point ('docs-excel' in pyproject.toml)."""
    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": "session-1"}}
        graph.invoke(create_initial_state(), config)

    except KeyboardInterrupt:
        print("\n")
        warn("Interrupted by user. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    run_graph()
