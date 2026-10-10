"""Markdown and HTML table rendering (SPEC.md section 5.3)."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List, Optional

from app.annotator import safe_word_annotation
from app.models import WordTranslation

_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")


class _CellCollector(HTMLParser):
    """Rebuild a table from extracted cell text, dropping source attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        name = tag.lower()
        if name in {"script", "style"}:
            self._ignore_depth += 1
            return
        if name == "tr":
            self._row = []
        elif name in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in {"script", "style"} and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if name in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif name == "tr" and self._row is not None:
            if any(cell.strip() for cell in self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._ignore_depth or self._cell is None:
            return
        self._cell.append(data)


def markdown_table_to_html(
    source: str,
    words: Optional[List[WordTranslation]] = None,
) -> str:
    """Render a table wrapped in ``<div class="table-container">``.

    Markdown pipe tables and an HTML ``<table>`` extracted from the PDF are
    both accepted. Cell text is escaped. Original cells can receive word-hover
    spans; translated cells pass ``words=None``.
    """
    rows = _rows_from_source(source)
    if not rows:
        return (
            '<div class="table-container"><pre>'
            + html.escape(source)
            + "</pre></div>"
        )
    width = max(len(row) for row in rows)
    word_list = list(words or [])
    header = rows[0]
    body = rows[1:]

    def cell(text: str, tag: str) -> str:
        cleaned = _clean_cell(text)
        if word_list:
            inner = safe_word_annotation(cleaned, word_list)
        else:
            inner = html.escape(cleaned)
        inner = inner.replace("\n", "<br>")
        return f"<{tag}>{inner}</{tag}>"

    parts = ['<div class="table-container"><table>', "<thead><tr>"]
    for index in range(width):
        value = header[index] if index < len(header) else ""
        parts.append(cell(value, "th"))
    parts.append("</tr></thead>")
    if body:
        parts.append("<tbody>")
        for row in body:
            parts.append("<tr>")
            for index in range(width):
                value = row[index] if index < len(row) else ""
                parts.append(cell(value, "td"))
            parts.append("</tr>")
        parts.append("</tbody>")
    parts.append("</table></div>")
    return "".join(parts)


def _rows_from_source(source: str) -> list[list[str]]:
    if "<table" in source.lower():
        parser = _CellCollector()
        parser.feed(source)
        if parser.rows:
            return parser.rows
    return _rows_from_markdown(source)


def _rows_from_markdown(source: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_row(line)
        if not cells:
            continue
        if all(_SEPARATOR_CELL.fullmatch(cell.replace(" ", "")) for cell in cells if cell != ""):
            continue
        rows.append(cells)
    return rows


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _clean_cell(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _EMPHASIS.sub(r"\2", text)
    return text.strip()
