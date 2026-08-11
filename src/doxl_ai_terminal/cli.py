"""
CLI entry point for doxl-ai-terminal.

Provides the ``doxl-ai`` command after installation::

    $ doxl-ai --help
    $ doxl-ai --version
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="doxl-ai",
        description="AI-powered terminal for reading and writing Excel & Word documents.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- excel sub-command ---------------------------------------------------
    excel_parser = subparsers.add_parser("excel", help="Excel file operations")
    excel_parser.add_argument("file", nargs="?", help="Path to the Excel file")
    excel_parser.add_argument(
        "--read", action="store_true", help="Read the Excel file"
    )
    excel_parser.add_argument(
        "--write", action="store_true", help="Write to the Excel file"
    )

    # --- docs sub-command ----------------------------------------------------
    docs_parser = subparsers.add_parser("docs", help="Word document operations")
    docs_parser.add_argument("file", nargs="?", help="Path to the Word document")
    docs_parser.add_argument(
        "--read", action="store_true", help="Read the Word document"
    )
    docs_parser.add_argument(
        "--write", action="store_true", help="Write to the Word document"
    )

    # --- terminal sub-command ------------------------------------------------
    terminal_parser = subparsers.add_parser("terminal", help="Terminal operations")
    terminal_parser.add_argument("cmd", nargs="?", help="Command to execute")

    return parser


def _get_version() -> str:
    """Return the package version string."""
    from doxl_ai_terminal import __version__

    return __version__


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "excel":
        print(f"[doxl-ai] Excel operation on: {args.file or '(no file specified)'}")
    elif args.command == "docs":
        print(f"[doxl-ai] Docs operation on: {args.file or '(no file specified)'}")
    elif args.command == "terminal":
        print(f"[doxl-ai] Terminal command: {args.cmd or '(no command specified)'}")
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
