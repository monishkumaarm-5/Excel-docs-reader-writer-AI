#terminal_ui.py
"""
Terminal UI — the app's one design system.

Every screen in doxl-ai-terminal (the onboarding flow in nodes.py, and
the interactive chat loops in excel_agent.py / docs_agent.py) renders
through the helpers in this file rather than calling print() directly,
so status colors, spacing, and phrasing stay identical everywhere.

Design intent: quiet and legible over decorative. One accent color for
emphasis, muted status colors, no boxes/banners beyond a single thin
rule, no emoji — plain monochrome glyphs (✓ ✗ → ⚠) that read correctly
in any terminal, including ones with color disabled.
"""

import itertools
import threading
import time


# --- Palette ---
#
# A small, deliberate set of colors used for STATUS ONLY. Body text
# (agent answers, file contents, etc.) is left in the terminal's
# default foreground — color is a signal here, not decoration.

class Color:
    ACCENT = "\033[38;2;202;122;95m"    # headers, prompts, the app's own voice
    SUCCESS = "\033[38;2;94;160;110m"
    ERROR = "\033[38;2;214;92;92m"
    WARN = "\033[38;2;212;163;80m"
    MUTED = "\033[38;2;140;140;140m"

    # Back-compat aliases — kept because older/external call sites may
    # still reference these names directly.
    GREEN = SUCCESS
    LIGHT_BLUE = ACCENT
    RED = ERROR
    YELLOW = WARN
    CYAN = ACCENT

    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def success(text: str):
    print(f"{Color.SUCCESS}✓ {text}{Color.RESET}")


def error(text: str):
    print(f"{Color.ERROR}✗ {text}{Color.RESET}")


def info(text: str):
    print(f"{Color.MUTED}→ {text}{Color.RESET}")


def warn(text: str):
    print(f"{Color.WARN}⚠ {text}{Color.RESET}")


def agent_say(agent_name: str, text: str):
    print(f"{Color.BOLD}{Color.ACCENT}{agent_name}{Color.RESET}  {text}")


def divider():
    print(f"{Color.MUTED}{'─' * 44}{Color.RESET}")


def header(title: str, subtitle: str = ""):
    """A single clean section header — replaces ad hoc '=' * N banners.

    Usage:
        header("DOXL AI — Excel Agent")
        header("Agents ready", "Reader · Writer · Updater · Deleter · Formula Gen")
    """
    print(f"\n{Color.BOLD}{Color.ACCENT}{title}{Color.RESET}")
    if subtitle:
        print(f"{Color.MUTED}{subtitle}{Color.RESET}")
    divider()


def banner():
    """App wordmark — one bold line + a thin rule. No box-drawing."""
    print(f"\n{Color.BOLD}{Color.ACCENT}DOXL AI{Color.RESET}")
    print(f"{Color.MUTED}Document & Excel terminal agent{Color.RESET}")
    divider()


def prompt_input(label: str) -> str:
    return input(f"{Color.ACCENT}{label}{Color.RESET}").strip()


def agent_result(text: str, label: str = "Agent"):
    """Render a specialist's final answer — the one thing the user
    actually came for. Visually distinct from progress/status lines so
    it can't be mistaken for an intermediate message."""
    print(f"\n{Color.BOLD}{Color.ACCENT}{label}{Color.RESET}")
    print(text)


def interrupt_prompt(question: str) -> str:
    """Human-in-the-loop prompt."""
    divider()
    print(f"{Color.BOLD}{Color.WARN}Input needed{Color.RESET}")
    print(f"  {question}")
    answer = input(f"{Color.ACCENT}  Your answer: {Color.RESET}").strip()
    divider()
    return answer


# --- Spinner (no external dependency) ---

class Spinner:
    """Animated single-line spinner for a step that has no per-item progress
    to report (chunking, embedding, vector-store writes, an LLM call, etc.).

    Usage:
        with Spinner("Loading the file..."):
            do_the_slow_thing()

    The label is overwritten in place (via \\r) so it doesn't spam the
    terminal with step-by-step internals — those still happen, they're
    just no longer printed line-by-line. This is the app's one loading
    indicator; every operation that blocks for more than an instant
    should run inside one instead of leaving the terminal silent or
    printing a static "processing..." line.
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _INTERVAL = 0.08

    def __init__(self, label: str, done_label: str | None = None):
        self.label = label
        self.done_label = done_label or label.rstrip(".")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        for frame in itertools.cycle(self._FRAMES):
            if self._stop_event.is_set():
                break
            print(
                f"\r{Color.ACCENT}{frame} {self.label}{Color.RESET}",
                end="",
                flush=True,
            )
            time.sleep(self._INTERVAL)

    def _clear_line(self):
        width = len(self.label) + 4
        print("\r" + " " * width + "\r", end="", flush=True)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self, ok: bool = True):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._clear_line()
        if ok:
            success(self.done_label)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Only print the success line if nothing blew up inside the block.
        self.stop(ok=exc_type is None)
        return False  # don't swallow exceptions


# --- Native file picker ---

def browse_for_file() -> str:
    """Open a native OS file-picker dialog and return the chosen path.

    Returns "" if the dialog can't be opened (headless environment, no
    tkinter, user cancelled, etc.) — callers should fall back to asking
    for a typed path in that case.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        warn("File picker isn't available here (tkinter not installed).")
        return ""

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a Word or Excel file",
            filetypes=[
                ("Word & Excel files", "*.docx *.xlsx"),
                ("Word documents", "*.docx"),
                ("Excel spreadsheets", "*.xlsx"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path.strip() if path else ""
    except Exception as e:
        warn(f"Couldn't open the file picker ({e}). Paste the path instead.")
        return ""
