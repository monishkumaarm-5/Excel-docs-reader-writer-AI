#runner.py
"""
Graph runner — CLI entry point for the docs-excel command.

Usage:
    $ docs-excel          # after pip install
    $ python -m doxl_ai_terminal.pipeline.runner   # direct

Handles the interrupt-resume loop for human-in-the-loop.
When the graph hits an interrupt() call, execution pauses and control
returns to this runner. The runner reads the interrupt payload, prompts
the user in the terminal, then resumes the graph with their answer.
"""

import sys
from langgraph.types import Command
from doxl_ai_terminal.pipeline.state import create_initial_state
from doxl_ai_terminal.pipeline.Graph import build_graph
from doxl_ai_terminal.pipeline.terminal_ui import warn, interrupt_prompt


def run_graph():
    """Run the graph with interrupt-resume support.

    This is the main CLI entry point registered as 'docs-excel' in pyproject.toml.
    """

    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": "session-1"}}
        initial_state = create_initial_state()

        # ── Initial invocation ──
        # This runs until the graph either finishes or hits an interrupt()
        graph.invoke(initial_state, config)

        # ── Interrupt-resume loop ──
        # After the initial run, check if the graph is paused at an interrupt.
        # If so, read the interrupt payload, ask the user, and resume.
        while True:
            snapshot = graph.get_state(config)

            # If no next nodes, the graph has finished
            if not snapshot.next:
                break

            # Read the interrupt payload from the paused task
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                interrupt_value = snapshot.tasks[0].interrupts[0].value

                # Display the interrupt question in the terminal
                question = interrupt_value.get("question", "Please confirm:")
                options = interrupt_value.get("options", "")
                if options:
                    question = f"{question} ({options})"

                answer = interrupt_prompt(question)

                # Resume the graph with the user's answer
                graph.invoke(Command(resume=answer), config)
            else:
                # Safety: graph is waiting but no interrupt data
                warn("Graph is paused but no interrupt data found. Breaking.")
                break

    except KeyboardInterrupt:
        print("\n")
        warn("Interrupted by user. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    run_graph()
