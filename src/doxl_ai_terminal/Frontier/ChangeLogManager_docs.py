#ChangeLogManager_docs.py
from typing import List, Dict


from doxl_ai_terminal.Frontier.fileReader import read_word
from doxl_ai_terminal.data_structure.change_log import ChangeLog, ChangeType
from doxl_ai_terminal.data_structure.docs import DocLine
from doxl_ai_terminal.data_structure.docs_write import write_word


class LiveDocManager:
    """Read, edit, and sync a Word document in real-time"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.doc_data = read_word(filepath)
        self.changelog = ChangeLog()
        print(f"Loaded: {filepath} ({self.doc_data.total_lines} lines)")

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
        "font_color", "alignment",
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
        """Update a line's text. Optional formatting kwargs (font_name,
        font_size, bold, italic, underline, font_color, alignment) may be
        passed to restyle the line at the same time; any left unset are
        untouched."""
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
        """Change a line's formatting (font, size, bold, italic, underline,
        color, alignment) without touching its text."""
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
        # Make room: any existing line in this SAME paragraph at or after
        # the requested L shifts down by one first. Without this, "add at
        # L2" would collide with (or, combined with write_word's old
        # insertion-order behavior, silently ignore) whatever was already
        # at L2 — the L number needs to mean "insert here", not just be a
        # label attached to whatever gets appended last.
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

    def batch_edit(self, edits: List[Dict]):
        """Apply multiple edits and save once.

        Each edit: {"action": "update/add/delete", "paragraph": int, "line": int, "text": str}
        """
        results = []

        for edit in edits:
            action = edit["action"]

            if action == "update":
                r = self.update_line(edit["paragraph"], edit["line"], edit["text"], auto_save=False)
            elif action == "add":
                r = self.add_line(edit["paragraph"], edit["line"], edit["text"], auto_save=False)
            elif action == "delete":
                r = self.delete_line(edit["paragraph"], edit["line"], auto_save=False)
            else:
                r = f"Unknown action: {action}"

            results.append(r)

        # Single save after all edits
        self._sync_to_file()
        return results

    # ---- Sync data structure → file ----

    def _sync_to_file(self):
        write_word(self.doc_data, self.filepath)

    # ---- Save to a different path ----

    def save_as(self, output_path: str):
        # Load formatting/structure from the live file (self.filepath),
        # not from output_path — output_path is a brand-new destination
        # and almost never the file that actually holds the current styles.
        write_word(self.doc_data, output_path, source_path=self.filepath)
        return f"Saved copy: {output_path}"

    # ---- Reload from file (discard in-memory changes) ----

    def reload(self):
        self.doc_data = read_word(self.filepath)
        self.changelog.clear()
        return f"Reloaded from: {self.filepath}"

    # ---- Show change history ----

    def history(self) -> str:
        if not self.changelog.has_changes():
            return "No changes yet."
        return self.changelog.summary()