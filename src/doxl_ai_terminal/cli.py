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

    # --- process sub-command (read → chunk → vectorize) -----------------------
    process_parser = subparsers.add_parser("process", help="Process a file into vector store")
    process_parser.add_argument("file", help="Path to .docx or .xlsx file")

    # --- search sub-command ---------------------------------------------------
    search_parser = subparsers.add_parser("search", help="Search the vector store")
    search_parser.add_argument("file", help="Path to .docx or .xlsx file")
    search_parser.add_argument("query", help="Search query")

    # --- view sub-command -----------------------------------------------------
    view_parser = subparsers.add_parser("view", help="View file contents")
    view_parser.add_argument("file", help="Path to .docx or .xlsx file")

    # --- info sub-command -----------------------------------------------------
    info_parser = subparsers.add_parser("info", help="Show vector store info")

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

    if args.command == "process":
        from doxl_ai_terminal.pipeline.pipeliner import process_file

        print(f"[doxl-ai] Processing: {args.file}")
        instances = process_file(args.file)
        print(f"\n[doxl-ai] Created {len(instances)} vector store collections:")
        for db in instances:
            print(
                f"  [{db.label}] {db.format_name} "
                f"→ {db.collection_name} ({db.total_chunks} chunks)"
            )

    elif args.command == "search":
        from doxl_ai_terminal.pipeline.pipeliner import process_file
        from doxl_ai_terminal.pipeline.search import search

        print(f"[doxl-ai] Processing: {args.file}")
        instances = process_file(args.file)
        print(f"\n[doxl-ai] Searching for: '{args.query}'")
        search(instances, args.query)

    elif args.command == "view":
        import os
        ext = os.path.splitext(args.file)[1].lower()

        if ext == ".docx":
            from doxl_ai_terminal.Frontier.fileReader import read_word
            from doxl_ai_terminal.Frontier.displayFunction import display_doc
            data = read_word(args.file)
            display_doc(data)
        elif ext == ".xlsx":
            from doxl_ai_terminal.Frontier.fileReader import read_excel
            from doxl_ai_terminal.Frontier.displayFunction import display_excel
            data = read_excel(args.file)
            display_excel(data)
        else:
            print(f"[doxl-ai] Unsupported file type: {ext}")
            return 1

    elif args.command == "info":
        from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager

        mgr = VectorDBManager()
        collections = mgr.list_collections()
        if not collections:
            print("[doxl-ai] No vector store collections found.")
        else:
            print(f"[doxl-ai] {len(collections)} collections:")
            for name in collections:
                mgr.load_collection(name)
                count = mgr.collection_count()
                print(f"  - {name} ({count} documents)")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
