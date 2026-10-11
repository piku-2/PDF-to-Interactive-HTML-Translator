"""Markdown and HTML table rendering (SPEC.md section 5.3)."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List, Optional

from app.annotator import safe_word_annotation
from app.models import WordTranslation

_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")


class _CellCollector(HTMLParser):
    """Rebuild a table from extracted cell text, dropping source attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, str]]] = []
        self._row: list[tuple[str, str]] | None = None
        self._cell: list[str] | None = None
        self._cell_tag = "td"
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
            self._cell_tag = name

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in {"script", "style"} and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if name in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append((self._cell_tag, "".join(self._cell).strip()))
            self._cell = None
        elif name == "tr" and self._row is not None:
            if any(text.strip() for _tag, text in self._row):
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
            '<div class="table-container"><pre>' + html.escape(source) + "</pre></div>"
        )
    word_list = list(words or [])

    def cell(text: str, tag: str) -> str:
        cleaned = _clean_cell(text)
        if word_list:
            inner = safe_word_annotation(cleaned, word_list)
        else:
            inner = html.escape(cleaned)
        inner = inner.replace("\n", "<br>")
        return f"<{tag}>{inner}</{tag}>"

    parts = ['<div class="table-container"><table>']
    header = rows[0] if rows and all(tag == "th" for tag, _text in rows[0]) else None
    body = rows[1:] if header is not None else rows
    if header is not None:
        parts.append("<thead><tr>")
        for tag, value in header:
            parts.append(cell(value, tag))
        parts.append("</tr></thead>")
    if body:
        parts.append("<tbody>")
        for row in body:
            parts.append("<tr>")
            for tag, value in row:
                parts.append(cell(value, "td" if tag not in {"th", "td"} else tag))
            parts.append("</tr>")
        parts.append("</tbody>")
    parts.append("</table></div>")
    return "".join(parts)


def _rows_from_source(source: str) -> list[list[tuple[str, str]]]:
    raw = source.strip()
    if raw.lower().startswith("<table"):
        parsed = _rows_from_html(raw)
        if parsed:
            return parsed
    if "|" in raw:
        import markdown

        rendered = markdown.markdown(raw, extensions=["tables"])
        parsed = _rows_from_html(rendered)
        if parsed:
            return parsed
    return _rows_from_html(raw)


def _rows_from_html(source: str) -> list[list[tuple[str, str]]]:
    if "<table" not in source.lower() and "<tr" not in source.lower():
        return []
    parser = _CellCollector()
    parser.feed(source)
    return parser.rows


def _clean_cell(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _EMPHASIS.sub(r"\2", text)
    return text.strip()
