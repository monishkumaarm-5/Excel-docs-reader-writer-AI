# terminal_ui.py
"""
Terminal UI — the app's design system, built on Rich.

Every screen in doxl-ai-terminal renders through the helpers in this file
rather than calling print() directly, so styling, spacing, and layout stay
consistent everywhere.

Design intent: clean, modern, and legible. Rich panels for structure,
tables for data, spinners for progress, and a warm accent palette that
works on both dark and light terminals.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
)
from rich.theme import Theme
from rich.markdown import Markdown
from rich.rule import Rule
from rich.prompt import Prompt
from rich import box

# ── Theme ──────────────────────────────────────────────────────────
# A small, deliberate palette. Body text uses the terminal's default;
# color signals status, not decoration.

DOXL_THEME = Theme({
    "accent":   "bold #ca7a5f",
    "success":  "#5ea06e",
    "error":    "bold #d65c5c",
    "warn":     "#d4a350",
    "muted":    "#8c8c8c",
    "info":     "#8c8c8c",
    "agent":    "bold #ca7a5f",
    "heading":  "bold #ca7a5f",
    "key":      "bold white",
    "dim":      "dim",
})

console = Console(theme=DOXL_THEME)

# ── Back-compat color constants (kept for any external call sites) ─

class Color:
    ACCENT   = "\033[38;2;202;122;95m"
    SUCCESS  = "\033[38;2;94;160;110m"
    ERROR    = "\033[38;2;214;92;92m"
    WARN     = "\033[38;2;212;163;80m"
    MUTED    = "\033[38;2;140;140;140m"
    GREEN    = SUCCESS
    LIGHT_BLUE = ACCENT
    RED      = ERROR
    YELLOW   = WARN
    CYAN     = ACCENT
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RESET    = "\033[0m"


# ═══════════════════════════════════════════════════════════════════
# STATUS MESSAGES
# ═══════════════════════════════════════════════════════════════════

def success(text: str):
    console.print(f"  [success]✓[/success] {text}")


def error(text: str):
    console.print(f"  [error]✗ {text}[/error]")


def info(text: str):
    console.print(f"  [muted]→ {text}[/muted]")


def warn(text: str):
    console.print(f"  [warn]⚠ {text}[/warn]")


# ═══════════════════════════════════════════════════════════════════
# AGENT OUTPUT
# ═══════════════════════════════════════════════════════════════════

def agent_say(agent_name: str, text: str):
    """A named agent speaking to the user."""
    console.print(f"  [agent]{agent_name}[/agent]  {text}")


def agent_result(text: str, label: str = "Agent"):
    """Render a specialist's final answer inside a panel — the one
    thing the user actually came for, visually distinct from status."""
    console.print()
    console.print(Panel(
        Text(text),
        title=f"[accent]{label}[/accent]",
        title_align="left",
        border_style="muted",
        padding=(1, 2),
        expand=True,
    ))


# ═══════════════════════════════════════════════════════════════════
# STRUCTURAL ELEMENTS
# ═══════════════════════════════════════════════════════════════════

def divider():
    console.print(Rule(style="muted"))


def header(title: str, subtitle: str = ""):
    """A clean section header with an optional subtitle."""
    console.print()
    console.print(f"  [heading]{title}[/heading]")
    if subtitle:
        console.print(f"  [muted]{subtitle}[/muted]")
    console.print(Rule(style="muted"))


def banner():
    """App wordmark — a styled panel with the app name."""
    title_text = Text()
    title_text.append("DOXL AI", style="bold #ca7a5f")

    subtitle = Text("Document & Spreadsheet Terminal Agent", style="#8c8c8c")

    content = Text()
    content.append("\n")
    content.append("  DOXL AI\n", style="bold #ca7a5f")
    content.append("  Document & Spreadsheet Terminal Agent\n", style="#8c8c8c")
    content.append("\n")
    content.append("  Powered by Google Gemini + LangGraph\n", style="dim")

    console.print(Panel(
        content,
        border_style="#ca7a5f",
        box=box.ROUNDED,
        expand=False,
        padding=(0, 2),
    ))


def key_value(label: str, value: str):
    """Print a key-value pair inline."""
    console.print(f"  [key]{label}:[/key] {value}")


# ═══════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════

def prompt_input(label: str) -> str:
    """Styled input prompt."""
    return console.input(f"[accent]{label}[/accent]").strip()


def interrupt_prompt(question: str) -> str:
    """Human-in-the-loop prompt inside a panel."""
    console.print()
    console.print(Panel(
        f"[warn]Input needed[/warn]\n\n  {question}",
        border_style="warn",
        padding=(1, 2),
        expand=False,
    ))
    answer = console.input("[accent]  Your answer: [/accent]").strip()
    return answer


def choice_prompt(title: str, options: list[str]) -> int:
    """Present a numbered list and return the 0-based index chosen."""
    console.print()
    console.print(f"  [accent]{title}[/accent]")
    for i, opt in enumerate(options, 1):
        console.print(f"    [key]{i}.[/key] {opt}")
    console.print()
    while True:
        raw = console.input("[accent]  Choose (number): [/accent]").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        warn(f"Enter a number between 1 and {len(options)}.")


# ═══════════════════════════════════════════════════════════════════
# TABLES (for displaying structured data)
# ═══════════════════════════════════════════════════════════════════

def make_table(title: str, columns: list[str], rows: list[list[str]],
               show_lines: bool = False) -> Table:
    """Build a Rich Table with the app's styling."""
    table = Table(
        title=title,
        title_style="accent",
        box=box.SIMPLE_HEAVY_HEAD,
        show_lines=show_lines,
        padding=(0, 1),
        border_style="muted",
    )
    for col in columns:
        table.add_column(col, style="key", no_wrap=False)
    for row in rows:
        table.add_row(*row)
    return table


def print_table(title: str, columns: list[str], rows: list[list[str]],
                show_lines: bool = False):
    """Build and print a table immediately."""
    console.print(make_table(title, columns, rows, show_lines))


# ═══════════════════════════════════════════════════════════════════
# SPINNER (Rich-based, context manager)
# ═══════════════════════════════════════════════════════════════════

class Spinner:
    """Animated single-line spinner for blocking operations.

    Usage:
        with Spinner("Loading the file..."):
            do_the_slow_thing()

        # Or manual start/stop:
        spinner = Spinner("Processing...").start()
        ...
        spinner.stop(ok=True)
    """

    def __init__(self, label: str, done_label: str | None = None):
        self.label = label
        self.done_label = done_label or label.rstrip(".").rstrip()
        self._progress: Optional[Progress] = None
        self._task_id = None

    def start(self) -> "Spinner":
        self._progress = Progress(
            SpinnerColumn("dots", style="accent"),
            TextColumn("[accent]{task.description}[/accent]"),
            console=console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(self.label)
        return self

    def stop(self, ok: bool = True):
        if self._progress:
            self._progress.stop()
            self._progress = None
        if ok:
            success(self.done_label)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop(ok=exc_type is None)
        return False


# ═══════════════════════════════════════════════════════════════════
# FILE PICKER
# ═══════════════════════════════════════════════════════════════════

def browse_for_file() -> str:
    """Open a native OS file-picker dialog and return the chosen path.
    Returns "" if unavailable or cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        warn("File picker isn't available (tkinter not installed).")
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
        warn(f"Couldn't open file picker ({e}). Paste the path instead.")
        return ""


# ═══════════════════════════════════════════════════════════════════
# WELCOME SCREEN (Rich layout)
# ═══════════════════════════════════════════════════════════════════

def welcome_screen():
    """Full welcome screen shown on first launch."""
    console.print()
    banner()
    console.print()
    console.print("  [muted]Commands:[/muted]")
    console.print("    [key]sheets[/key]   — list all sheets")
    console.print("    [key]history[/key]  — show change log")
    console.print("    [key]quit[/key]     — exit the session")
    console.print()
    console.print("  [muted]Ask in plain English, e.g.[/muted]")
    console.print("    [dim]\"find rows where revenue > 1000\"[/dim]")
    console.print("    [dim]\"add a SUM formula for column B\"[/dim]")
    console.print("    [dim]\"replace all 'draft' with 'final'\"[/dim]")
    console.print()


def file_loaded_panel(filename: str, details: list[tuple[str, str]]):
    """Show a panel after a file is successfully loaded.
    details: list of (label, value) pairs like [("Sheets", "3"), ("Cells", "450")]
    """
    lines = []
    for label, value in details:
        lines.append(f"  [key]{label}:[/key] {value}")
    body = "\n".join(lines)

    console.print(Panel(
        body,
        title=f"[success]✓ Loaded[/success]  [accent]{filename}[/accent]",
        title_align="left",
        border_style="success",
        padding=(1, 1),
        expand=False,
    ))


def agents_ready_panel(agent_names: list[str]):
    """Show which specialist agents are available."""
    agent_str = " · ".join(f"[accent]{a}[/accent]" for a in agent_names)
    console.print(f"\n  [muted]Agents:[/muted] {agent_str}\n")


# ═══════════════════════════════════════════════════════════════════
# TASK PROGRESS (inline counter — no Rich live display, so it
# never conflicts with a Spinner that may still be running)
# ═══════════════════════════════════════════════════════════════════

class TaskProgress:
    """Lightweight edit counter that overwrites a single terminal line.

    Usage:
        tp = TaskProgress()
        tp.add("add_cell R2C3")   # prints:  ⟳ Edits: 1 queued
        tp.add("add_cell R3C3")   # overwrites: ⟳ Edits: 2 queued
        ...
        tp.finish(saved=True)     # clears line, prints: ✓ Applied 22 edit(s)

    Uses raw print() with \\r to stay on one line — no Rich live display,
    so it coexists with Spinner or any other Progress bar.
    """

    def __init__(self, label: str = "Edits"):
        self._label = label
        self._count = 0

    def add(self, description: str = ""):
        """Record one more queued edit and update the counter in-place."""
        self._count += 1
        # \r returns to start of line, flush ensures it renders immediately
        print(f"\r  ⟳ {self._label}: {self._count} queued  ", end="", flush=True)

    @property
    def count(self) -> int:
        return self._count

    def finish(self, saved: bool = True):
        """Clear the counter line and print a summary."""
        n = self._count
        if n > 0:
            # Clear the in-place counter line
            print("\r" + " " * 40 + "\r", end="", flush=True)
            if saved:
                success(f"Applied {n} edit(s) — saved to disk.")
        self._count = 0
