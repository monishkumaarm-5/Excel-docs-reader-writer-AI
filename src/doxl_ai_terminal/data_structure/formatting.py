#formatting.py
"""Shared formatting helpers for the Excel (openpyxl) and Word (python-docx)
data layers.

These centralize how "rich" formatting -- font name/size, bold/italic/
underline, colors, alignment, number formats -- is read off a live
openpyxl/python-docx object into our plain dataclasses (ExcelCell / DocLine)
and applied back the other direction. Keeping both directions in one place
means read_excel/read_word and excel_write/docs_write always agree on what
a "font_color" string looks like, what alignment values are valid, etc.

Convention: colors are plain 6-digit hex RGB strings, e.g. "FF0000" (no
leading '#', no alpha channel). Alignment is one of "left"/"center"/
"right"/"justify". Every formatting field is Optional and tri-state:
None means "not specified / don't touch", so partial updates never
clobber formatting nobody asked to change.
"""

from typing import Optional


# ============================== Excel ================================== #

def read_excel_font_color(font) -> Optional[str]:
    """Return a 6-digit hex RGB string for an openpyxl Font.color, or None."""
    color = getattr(font, "color", None) if font else None
    if color is None:
        return None
    rgb = getattr(color, "rgb", None)
    if not rgb or not isinstance(rgb, str):
        return None
    # openpyxl stores ARGB (8 hex chars, e.g. "FFFF0000"); keep the trailing
    # 6 (the actual RGB) and drop the alpha byte.
    return rgb[-6:]


def read_excel_fill_color(fill) -> Optional[str]:
    """Return a 6-digit hex RGB string for an openpyxl PatternFill's
    foreground color, or None if there's no solid fill."""
    if fill is None or getattr(fill, "fill_type", None) is None:
        return None
    fg = getattr(fill, "fgColor", None)
    rgb = getattr(fg, "rgb", None) if fg is not None else None
    if not rgb or not isinstance(rgb, str):
        return None
    return rgb[-6:]


def read_excel_cell_format(cell) -> dict:
    """Pull the formatting fields we track off a loaded openpyxl cell
    (works for both normal and read-only worksheets)."""
    font = getattr(cell, "font", None)
    fill = getattr(cell, "fill", None)
    alignment = getattr(cell, "alignment", None)

    underline = getattr(font, "underline", None) if font else None

    return {
        "font_name": getattr(font, "name", None) if font else None,
        "font_size": getattr(font, "size", None) if font else None,
        "bold": getattr(font, "bold", None) if font else None,
        "italic": getattr(font, "italic", None) if font else None,
        "underline": bool(underline) if underline is not None else None,
        "font_color": read_excel_font_color(font),
        "fill_color": read_excel_fill_color(fill),
        "alignment": getattr(alignment, "horizontal", None) if alignment else None,
        "number_format": getattr(cell, "number_format", None),
    }


def apply_excel_cell_format(ws_cell, excel_cell) -> None:
    """Apply any explicitly-set formatting fields from an ExcelCell (our
    data-structure) onto a live openpyxl cell. Fields left as None are
    left completely alone so a plain value edit never strips existing
    formatting, and a formatting-only edit never touches unrelated
    attributes.
    """
    from openpyxl.styles import Font, PatternFill, Alignment

    font_fields = ("font_name", "font_size", "bold", "italic", "underline", "font_color")
    if any(getattr(excel_cell, f, None) is not None for f in font_fields):
        current = ws_cell.font
        underline = excel_cell.underline
        ws_cell.font = Font(
            name=excel_cell.font_name if excel_cell.font_name is not None else current.name,
            size=excel_cell.font_size if excel_cell.font_size is not None else current.size,
            bold=excel_cell.bold if excel_cell.bold is not None else current.bold,
            italic=excel_cell.italic if excel_cell.italic is not None else current.italic,
            underline=("single" if underline else None) if underline is not None else current.underline,
            color=excel_cell.font_color if excel_cell.font_color is not None else current.color,
        )

    if excel_cell.fill_color is not None:
        ws_cell.fill = PatternFill(
            start_color=excel_cell.fill_color,
            end_color=excel_cell.fill_color,
            fill_type="solid",
        )

    if excel_cell.alignment is not None:
        current_align = ws_cell.alignment
        ws_cell.alignment = Alignment(
            horizontal=excel_cell.alignment,
            vertical=current_align.vertical,
            wrap_text=current_align.wrap_text,
        )

    if excel_cell.number_format is not None:
        ws_cell.number_format = excel_cell.number_format


# =============================== Word =================================== #
# (docx is imported lazily inside these functions, matching the rest of the
# project's pattern of not paying for the import until a Word file is
# actually touched.)

def _wd_alignment_map() -> dict:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    return {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }


def read_paragraph_alignment(paragraph) -> Optional[str]:
    to_wd = _wd_alignment_map()
    from_wd = {v: k for k, v in to_wd.items()}
    return from_wd.get(paragraph.alignment)


def alignment_to_wd(value: str):
    """Map one of "left"/"center"/"right"/"justify" to python-docx's
    WD_ALIGN_PARAGRAPH enum member, or None if not recognized."""
    return _wd_alignment_map().get(value)


def read_run_format(run) -> dict:
    font = run.font
    color = None
    try:
        rgb = font.color.rgb if font.color is not None else None
        if rgb is not None:
            color = str(rgb)
    except Exception:
        # Theme colors and a few edge cases raise instead of returning
        # None -- formatting is best-effort, so just skip the color.
        color = None

    return {
        "font_name": font.name,
        "font_size": font.size.pt if font.size is not None else None,
        "bold": font.bold,
        "italic": font.italic,
        "underline": bool(font.underline) if font.underline is not None else None,
        "font_color": color,
    }


def base_format_for_paragraph(paragraph) -> dict:
    """Representative formatting for a paragraph, taken from its first
    non-empty run (falling back to its first run of any kind). Used as the
    fallback style when rewriting lines that don't request their own
    explicit formatting, so editing a line's text doesn't quietly strip
    the paragraph's existing look.
    """
    for run in paragraph.runs:
        if run.text:
            return read_run_format(run)
    if paragraph.runs:
        return read_run_format(paragraph.runs[0])
    return {}


def apply_run_format(run, doc_line, base: Optional[dict] = None) -> None:
    """Apply a DocLine's explicit formatting to a run, falling back to
    `base` (usually the paragraph's original formatting) for anything the
    line didn't specify."""
    base = base or {}
    font = run.font

    name = doc_line.font_name if doc_line.font_name is not None else base.get("font_name")
    if name:
        font.name = name

    size = doc_line.font_size if doc_line.font_size is not None else base.get("font_size")
    if size:
        from docx.shared import Pt
        font.size = Pt(size)

    bold = doc_line.bold if doc_line.bold is not None else base.get("bold")
    if bold is not None:
        font.bold = bold

    italic = doc_line.italic if doc_line.italic is not None else base.get("italic")
    if italic is not None:
        font.italic = italic

    underline = doc_line.underline if doc_line.underline is not None else base.get("underline")
    if underline is not None:
        font.underline = underline

    color = doc_line.font_color if doc_line.font_color is not None else base.get("font_color")
    if color:
        from docx.shared import RGBColor
        try:
            font.color.rgb = RGBColor.from_string(color)
        except Exception:
            pass
