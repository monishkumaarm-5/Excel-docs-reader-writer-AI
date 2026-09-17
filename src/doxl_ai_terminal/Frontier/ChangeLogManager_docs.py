# ChangeLogManager_docs.py
"""Live Word document editing manager — read, edit, and sync a .docx file."""

from typing import List, Dict

from doxl_ai_terminal.Frontier.fileReader import read_word
from doxl_ai_terminal.data_structure.change_log import ChangeLog, ChangeType
from doxl_ai_terminal.data_structure.docs import DocLine
from doxl_ai_terminal.data_structure.docs_write import write_word
from doxl_ai_terminal.pipeline.terminal_ui import info


class LiveDocManager:
    """Read, edit, and sync a Word document in real-time."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.doc_data = read_word(filepath)
        self.changelog = ChangeLog()
        info(f"Loaded: {filepath} ({self.doc_data.total_lines} lines)")

    # ---- View ----

    def view(self) -> str:
        lines = []
        for line in self.doc_data.lines:
            lines.append(f"P{line.paragraph} L{line.line}: {line.line_str}")
        return "\n".join(lines)

    # ---- Search ----

    def search(self, keyword: str) -> List[DocLine]:
        return [l for l in self.doc_data.lines
                if keyword.lower() in l.line_str.lower()]

    # ---- Formatting kwargs accepted by update_line/add_line/format_line ----
    _FORMAT_FIELDS = (
        "font_name", "font_size", "bold", "italic", "underline",
        "strikethrough", "font_color", "highlight_color",
        "superscript", "subscript", "alignment", "style",
    )

    @classmethod
    def _apply_format_kwargs(cls, line_obj: DocLine, format_kwargs: dict) -> dict:
        applied = {}
        for key, val in format_kwargs.items():
            if key in cls._FORMAT_FIELDS and val is not None:
                setattr(line_obj, key, val)
                applied[key] = val
        return applied

    # ---- Update (with auto-sync) ----

    def update_line(self, paragraph: int, line: int, new_text: str, auto_save=True, **format_kwargs):
        """Update a line's text. Optional formatting kwargs may be passed."""
        for l in self.doc_data.lines:
            if l.paragraph == paragraph and l.line == line:
                old = l.line_str
                l.line_str = new_text
                applied = self._apply_format_kwargs(l, format_kwargs)

                self.changelog.add(
                    ChangeType.UPDATE,
                    {"paragraph": paragraph, "line": line},
                    old_value=old,
                    new_value=new_text,
                )

                if auto_save:
                    self._sync_to_file()

                suffix = f" ({applied})" if applied else ""
                return f"Updated P{paragraph} L{line}: '{old}' → '{new_text}'{suffix}"

        return f"Not found: P{paragraph} L{line}"

    # ---- Format line only (with auto-sync) ----

    def format_line(self, paragraph: int, line: int, auto_save=True, **format_kwargs):
        """Change a line's formatting without touching its text."""
        for l in self.doc_data.lines:
            if l.paragraph == paragraph and l.line == line:
                applied = self._apply_format_kwargs(l, format_kwargs)
                if not applied:
                    return f"No formatting fields given for P{paragraph} L{line}"

                self.changelog.add(
                    ChangeType.UPDATE,
                    {"paragraph": paragraph, "line": line, "format": True},
                    new_value=str(applied),
                )

                if auto_save:
                    self._sync_to_file()

                return f"Formatted P{paragraph} L{line}: {applied}"

        return f"Not found: P{paragraph} L{line}"

    # ---- Add (with auto-sync) ----

    def add_line(self, paragraph: int, line: int, text: str, auto_save=True, **format_kwargs):
        for l in self.doc_data.lines:
            if l.paragraph == paragraph and l.line >= line:
                l.line += 1

        new_line = DocLine(paragraph=paragraph, line=line, line_str=text)
        self._apply_format_kwargs(new_line, format_kwargs)
        self.doc_data.lines.append(new_line)
        self.doc_data.total_lines += 1
        if paragraph > self.doc_data.total_paragraphs:
            self.doc_data.total_paragraphs = paragraph

        self.changelog.add(
            ChangeType.ADD,
            {"paragraph": paragraph, "line": line},
            new_value=text,
        )

        if auto_save:
            self._sync_to_file()

        return f"Added P{paragraph} L{line}: '{text}'"

    # ---- Delete (with auto-sync) ----

    def delete_line(self, paragraph: int, line: int, auto_save=True):
        for l in self.doc_data.lines:
            if l.paragraph == paragraph and l.line == line:
                old = l.line_str
                self.doc_data.lines.remove(l)
                self.doc_data.total_lines -= 1

                self.changelog.add(
                    ChangeType.DELETE,
                    {"paragraph": paragraph, "line": line},
                    old_value=old,
                )

                if auto_save:
                    self._sync_to_file()

                return f"Deleted P{paragraph} L{line}: '{old}'"

        return f"Not found: P{paragraph} L{line}"

    # ---- Replace All (with auto-sync) ----

    def replace_all(self, old_text: str, new_text: str, auto_save=True):
        count = 0
        for line in self.doc_data.lines:
            if old_text in line.line_str:
                old = line.line_str
                line.line_str = line.line_str.replace(old_text, new_text)
                count += 1

                self.changelog.add(
                    ChangeType.UPDATE,
                    {"paragraph": line.paragraph, "line": line.line},
                    old_value=old,
                    new_value=line.line_str,
                )

        if auto_save and count > 0:
            self._sync_to_file()

        return f"Replaced '{old_text}' → '{new_text}' in {count} lines"

    # ---- Batch Edit (multiple changes, one save) ----

    def batch_edit(self, edits: List[Dict], auto_save: bool = True):
        """Apply multiple edits and save once.

        Each edit: {"action": "update/add/delete", "paragraph": int, "line": int, "text": str}
        """
        results = []

        for edit in edits:
            action = edit["action"]
            fmt = edit.get("format") or {}

            if action == "update":
                r = self.update_line(edit["paragraph"], edit["line"], edit["text"], auto_save=False, **fmt)
            elif action == "add":
                r = self.add_line(edit["paragraph"], edit["line"], edit["text"], auto_save=False, **fmt)
            elif action == "delete":
                r = self.delete_line(edit["paragraph"], edit["line"], auto_save=False)
            else:
                r = f"Unknown action: {action}"

            results.append(r)

        if auto_save:
            self._sync_to_file()
        return results

    # ---- Sync data structure → file ----

    def _sync_to_file(self):
        write_word(self.doc_data, self.filepath)

    # ---- Save to a different path ----

    def save_as(self, output_path: str):
        write_word(self.doc_data, output_path, source_path=self.filepath)
        return f"Saved copy: {output_path}"

    # ---- Reload from file (discard in-memory changes) ----

    def reload(self):
        self.doc_data = read_word(self.filepath)
        self.changelog.clear()
        return f"Reloaded from: {self.filepath}"

    # ---- Paragraph style / heading inspection ----

    def get_paragraph_style(self, paragraph: int):
        for l in self.doc_data.lines:
            if l.paragraph == paragraph:
                return {"paragraph": paragraph, "style": l.style, "heading_level": l.heading_level}
        return None

    def get_outline(self) -> List[Dict]:
        """The document's heading structure."""
        by_paragraph: Dict[int, List] = {}
        for l in self.doc_data.lines:
            if l.heading_level is not None:
                by_paragraph.setdefault(l.paragraph, []).append(l)

        outline = []
        for p in sorted(by_paragraph.keys()):
            lines = sorted(by_paragraph[p], key=lambda dl: dl.line)
            outline.append({
                "paragraph": p,
                "level": lines[0].heading_level,
                "text": " ".join(l.line_str for l in lines),
            })
        return outline

    # ---- Show change history ----

    def history(self) -> str:
        if not self.changelog.has_changes():
            return "No changes yet."
        return self.changelog.summary()
