#changelogmanger_excel.py
from typing import List, Dict

from doxl_ai_terminal.Frontier.fileReader import read_excel
from doxl_ai_terminal.data_structure.change_log import ChangeLog, ChangeType
from doxl_ai_terminal.data_structure.excel import ExcelCell
from doxl_ai_terminal.data_structure.excel_write import write_excel


class LiveExcelManager:
    """Read, edit, and sync an Excel file in real-time"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.excel_data = read_excel(filepath)
        self.changelog = ChangeLog()
        total = sum(len(s.cells) for s in self.excel_data.sheets)
        print(f"Loaded: {filepath} ({total} cells)")

    # ---- View ----

    def view(self, sheet_name: str = None) -> str:
        lines = []
        for sheet in self.excel_data.sheets:
            if sheet_name and sheet.sheet_name != sheet_name:
                continue
            lines.append(f"\nSheet: {sheet.sheet_name}")
            for cell in sheet.cells:
                lines.append(f"  [{cell.column}{cell.row}] = {cell.data_excel}")
        return "\n".join(lines)

    # ---- Search ----

    def search(self, keyword: str) -> List[Dict]:
        results = []
        for sheet in self.excel_data.sheets:
            for cell in sheet.cells:
                if keyword.lower() in cell.data_excel.lower():
                    results.append({
                        "sheet": sheet.sheet_name,
                        "cell": f"{cell.column}{cell.row}",
                        "value": cell.data_excel,
                    })
        return results

    # ---- Update Cell (with auto-sync) ----

    def update_cell(self, sheet_name: str, row: str, column: str, new_value: str, auto_save=True):
        for sheet in self.excel_data.sheets:
            if sheet.sheet_name == sheet_name:
                for cell in sheet.cells:
                    if cell.row == row and cell.column == column:
                        old = cell.data_excel
                        cell.data_excel = new_value

                        self.changelog.add(
                            ChangeType.UPDATE,
                            {"sheet": sheet_name, "row": row, "column": column},
                            old_value=old,
                            new_value=new_value,
                        )

                        if auto_save:
                            self._sync_to_file()

                        return f"Updated {sheet_name}[{column}{row}]: '{old}' → '{new_value}'"

        return f"Not found: {sheet_name}[{column}{row}]"

    # ---- Add Cell (with auto-sync) ----

    def add_cell(self, sheet_name: str, row: str, column: str, value: str, auto_save=True):
        for sheet in self.excel_data.sheets:
            if sheet.sheet_name == sheet_name:
                sheet.cells.append(ExcelCell(row=row, column=column, data_excel=value))

                self.changelog.add(
                    ChangeType.ADD,
                    {"sheet": sheet_name, "row": row, "column": column},
                    new_value=value,
                )

                if auto_save:
                    self._sync_to_file()

                return f"Added {sheet_name}[{column}{row}] = '{value}'"

        return f"Sheet not found: {sheet_name}"

    # ---- Add Row (with auto-sync) ----

    def add_row(self, sheet_name: str, row_data: Dict[str, str], auto_save=True):
        """Add a full row. row_data = {"A": "val1", "B": "val2", ...}"""
        for sheet in self.excel_data.sheets:
            if sheet.sheet_name == sheet_name:
                # Find next empty row
                max_row = 0
                for cell in sheet.cells:
                    max_row = max(max_row, int(cell.row))
                new_row = str(max_row + 1)

                for col, val in row_data.items():
                    sheet.cells.append(ExcelCell(row=new_row, column=col, data_excel=val))

                self.changelog.add(
                    ChangeType.ADD,
                    {"sheet": sheet_name, "row": new_row},
                    new_value=str(row_data),
                )

                if auto_save:
                    self._sync_to_file()

                return f"Added row {new_row} to {sheet_name}: {row_data}"

        return f"Sheet not found: {sheet_name}"

    # ---- Delete Cell (with auto-sync) ----

    def delete_cell(self, sheet_name: str, row: str, column: str, auto_save=True):
        for sheet in self.excel_data.sheets:
            if sheet.sheet_name == sheet_name:
                for cell in sheet.cells:
                    if cell.row == row and cell.column == column:
                        old = cell.data_excel
                        sheet.cells.remove(cell)

                        self.changelog.add(
                            ChangeType.DELETE,
                            {"sheet": sheet_name, "row": row, "column": column},
                            old_value=old,
                        )

                        if auto_save:
                            self._sync_to_file()

                        return f"Deleted {sheet_name}[{column}{row}]: '{old}'"

        return f"Not found: {sheet_name}[{column}{row}]"

    # ---- Delete Row (with auto-sync) ----

    def delete_row(self, sheet_name: str, row: str, auto_save=True):
        """Delete all cells in a row"""
        for sheet in self.excel_data.sheets:
            if sheet.sheet_name == sheet_name:
                to_remove = [c for c in sheet.cells if c.row == row]

                if not to_remove:
                    return f"Row {row} is empty"

                for cell in to_remove:
                    sheet.cells.remove(cell)

                self.changelog.add(
                    ChangeType.DELETE,
                    {"sheet": sheet_name, "row": row},
                    old_value=f"{len(to_remove)} cells",
                )

                if auto_save:
                    self._sync_to_file()

                return f"Deleted row {row} ({len(to_remove)} cells)"

        return f"Sheet not found: {sheet_name}"

    # ---- Replace All (with auto-sync) ----

    def replace_all(self, old_text: str, new_text: str, auto_save=True):
        count = 0
        for sheet in self.excel_data.sheets:
            for cell in sheet.cells:
                if old_text in cell.data_excel:
                    old = cell.data_excel
                    cell.data_excel = cell.data_excel.replace(old_text, new_text)
                    count += 1

                    self.changelog.add(
                        ChangeType.UPDATE,
                        {"sheet": sheet.sheet_name, "row": cell.row, "column": cell.column},
                        old_value=old,
                        new_value=cell.data_excel,
                    )

        if auto_save and count > 0:
            self._sync_to_file()

        return f"Replaced '{old_text}' → '{new_text}' in {count} cells"

    # ---- Batch Edit ----

    def batch_edit(self, edits: List[Dict]):
        """Apply multiple edits and save once.

        Each edit: {"action": "update/add/delete", "sheet": str, "row": str, "column": str, "value": str}
        """
        results = []

        for edit in edits:
            action = edit["action"]
            sheet = edit["sheet"]

            if action == "update":
                r = self.update_cell(sheet, edit["row"], edit["column"], edit["value"], auto_save=False)
            elif action == "add":
                r = self.add_cell(sheet, edit["row"], edit["column"], edit["value"], auto_save=False)
            elif action == "delete":
                r = self.delete_cell(sheet, edit["row"], edit["column"], auto_save=False)
            else:
                r = f"Unknown action: {action}"

            results.append(r)

        self._sync_to_file()
        return results

    # ---- Sync / Save / Reload ----

    def _sync_to_file(self):
        write_excel(self.excel_data, self.filepath)

    def save_as(self, output_path: str):
        write_excel(self.excel_data, output_path)
        return f"Saved copy: {output_path}"

    def reload(self):
        self.excel_data = read_excel(self.filepath)
        self.changelog.clear()
        return f"Reloaded from: {self.filepath}"

    def history(self) -> str:
        if not self.changelog.has_changes():
            return "No changes yet."
        return self.changelog.summary()